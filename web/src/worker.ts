/// <reference lib="webworker" />
/**
 * Inference worker: keeps transformers.js + ONNX Runtime off the UI thread.
 *
 * Messages in:  { type: 'load', device, dtype }        { type: 'infer', audio, requestId }
 *               { type: 'load-asr' }                   { type: 'transcribe', audio, requestId }
 * Messages out: { type: 'progress', payload }          { type: 'ready', info }
 *               { type: 'asr-ready' }                  { type: 'transcript', payload }
 *               { type: 'result', payload }            { type: 'error', message }
 */
import { AutoProcessor, AutoTokenizer, Wav2Vec2ForCTC, env } from '@huggingface/transformers';
import { createSession, type Device, type Dtype, type ModelProgress, type Session, type SessionInfo } from './core/session';
import { createEmbedder, acousticDistance, type Embedder, type EmbedderDevice } from './core/embeddings';
import { MODEL_DIR_IDS, initModelsSource, resolveModels } from './core/config';

// one global resolution for every model in this worker (phone, ASR, encoder);
// resolved asynchronously so the 's3' source can probe reachability and fall
// back to the HF hub while upload/CORS are pending
const modelsReady = (async () => {
  await initModelsSource();
  const models = resolveModels();
  env.allowLocalModels = models.allowLocalModels;
  env.localModelPath = models.localModelPath;
  env.remoteHost = models.remoteHost;
  env.remotePathTemplate = models.remotePathTemplate;
  if (models.allowLocalModels) {
    // Chromium's Cache.put fails on the big local files ("Unexpected internal
    // error"); local serving is instant anyway, so skip the cache entirely
    env.useBrowserCache = false;
  }
})();

const ASR_MODEL_ID = resolveModels().asrModelId;
const ENCODER_PATH = MODEL_DIR_IDS.encoder;

let session: Session | null = null;
let asr: {
  processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>;
  model: Awaited<ReturnType<typeof Wav2Vec2ForCTC.from_pretrained>>;
  idToToken: string[];
} | null = null;
let embedder: Embedder | null = null;
// the acoustic encoder follows the device/dtype chosen for the phone model
let activeDevice: Device = 'wasm';
let activeDtype: Dtype = 'q8';

type LoadMessage = { type: 'load'; device: Device; dtype: Dtype; requestId?: number };
type InferMessage = { type: 'infer'; audio: Float32Array; requestId: number };
type LoadAsrMessage = { type: 'load-asr'; device?: Device; dtype?: Dtype; requestId?: number };
type TranscribeMessage = { type: 'transcribe'; audio: Float32Array; requestId: number };
type AcousticMessage = { type: 'acoustic'; learner: Float32Array; reference: Float32Array; requestId: number };
type WorkerMessage = LoadMessage | InferMessage | LoadAsrMessage | TranscribeMessage | AcousticMessage;

const post = (message: unknown, transfer?: Transferable[]): void => {
  (self as unknown as Worker).postMessage(message, transfer ?? []);
};

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  await modelsReady;
  const message = event.data;
  try {
    if (message.type === 'load') {
      activeDevice = message.device;
      activeDtype = message.dtype;
      session = await createSession({
        device: message.device,
        dtype: message.dtype,
        onProgress: (progress: ModelProgress) => {
          post({ type: 'progress', model: 'phones', payload: progress });
        },
      });
      const info: SessionInfo = session.info;
      post({ type: 'ready', requestId: message.requestId, payload: info });
      return;
    }

    if (message.type === 'infer') {
      if (!session) {
        throw new Error('model not loaded yet');
      }
      const result = await session.runFull(message.audio);
      post(
        {
          type: 'result',
          requestId: message.requestId,
          phones: result.phones,
          confidences: result.confidences,
          frames: result.frames,
          audioSeconds: result.audioSeconds,
          spans: result.spans,
          vocabSize: result.vocabSize,
          logPosteriors: result.logPosteriors,
          timings: result.timings,
        },
        [result.logPosteriors.buffer],
      );
      return;
    }

    if (message.type === 'load-asr') {
      const reportProgress = (progress: ModelProgress) => {
        post({ type: 'progress', model: 'asr', payload: progress });
      };
      reportProgress({ status: 'initiate', file: 'tokenizer' });
      const processor = await AutoProcessor.from_pretrained(ASR_MODEL_ID);
      const tokenizer = await AutoTokenizer.from_pretrained(ASR_MODEL_ID);
      // TokenizerModel.vocab is the id -> token table
      const tokenizerModel = (tokenizer as unknown as { model: { vocab: string[] } }).model;
      const idToToken = tokenizerModel.vocab;
      const model = await Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_ID, {
        device: message.device ?? activeDevice,
        dtype: message.dtype ?? activeDtype,
        progress_callback: reportProgress,
      });
      asr = { processor, model, idToToken };
      post({ type: 'asr-ready', requestId: message.requestId });
      return;
    }

    if (message.type === 'transcribe') {
      if (!asr) {
        throw new Error('asr model not loaded yet');
      }
      const inputs = await asr.processor(message.audio);
      const output = await asr.model({ input_values: inputs.input_values });
      const logits = output.logits;
      const [, frames, vocabSize] = logits.dims as [number, number, number];
      const data = logits.data as Float32Array;
      let transcript = '';
      let previousId = -1;
      for (let frame = 0; frame < frames; frame++) {
        const base = frame * vocabSize;
        let best = 0;
        let bestValue = -Infinity;
        for (let v = 0; v < vocabSize; v++) {
          const value = data[base + v];
          if (value > bestValue) {
            bestValue = value;
            best = v;
          }
        }
        if (best !== previousId && best !== 0) {
          transcript += asr.idToToken[best] ?? '';
        }
        previousId = best;
      }
      // the 960h vocab marks word boundaries with '|'
      transcript = transcript.replace(/\|/g, ' ');
      post({ type: 'transcript', requestId: message.requestId, transcript });
      return;
    }
    if (message.type === 'acoustic') {
      // the encoder lives in this worker too: a single ORT runtime keeps
      // memory and thread pressure under control; its weights come from S3
      // (createEmbedder redirects transformers.js for this load only)
      if (!embedder) {
        // graceful degradation: webgpu fp16 -> webgpu int8 -> wasm int8
        const attempts: Array<[EmbedderDevice, Dtype]> = activeDevice === 'webgpu'
          ? [['webgpu', 'fp16'], ['webgpu', 'q8'], ['wasm', 'q8']]
          : [['wasm', 'q8']];
        let lastError: unknown = null;
        for (const [device, dtype] of attempts) {
          try {
            embedder = await createEmbedder(ENCODER_PATH, dtype, device, (progress) => {
              post({ type: 'progress', model: 'encoder', payload: progress });
            });
            break;
          } catch (error) {
            lastError = error;
          }
        }
        if (!embedder) {
          throw lastError ?? new Error('encoder could not be loaded');
        }
      }
      const learnerEmbedding = await embedder.embed(message.learner);
      const referenceEmbedding = await embedder.embed(message.reference);
      const acoustic = acousticDistance(learnerEmbedding, referenceEmbedding);
      post({
        type: 'acoustic-result',
        requestId: message.requestId,
        distance: acoustic.distance,
        meanPerStep: acoustic.meanPerStep,
        pathLength: acoustic.pathLength,
        ms: acoustic.ms,
      });
      return;
    }
  } catch (error) {
    post({
      type: 'error',
      requestId: (message as { requestId?: number }).requestId,
      message: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    });
  }
};
