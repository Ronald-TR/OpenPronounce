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
import { AutoProcessor, AutoTokenizer, Wav2Vec2ForCTC } from '@huggingface/transformers';
import { createSession, type Device, type Dtype, type ModelProgress, type Session, type SessionInfo } from './core/session';

const ASR_MODEL_ID = 'onnx-community/wav2vec2-base-960h-ONNX';

let session: Session | null = null;
let asr: {
  processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>;
  model: Awaited<ReturnType<typeof Wav2Vec2ForCTC.from_pretrained>>;
  idToToken: string[];
} | null = null;

type LoadMessage = { type: 'load'; device: Device; dtype: Dtype };
type InferMessage = { type: 'infer'; audio: Float32Array; requestId: number };
type LoadAsrMessage = { type: 'load-asr' };
type TranscribeMessage = { type: 'transcribe'; audio: Float32Array; requestId: number };
type WorkerMessage = LoadMessage | InferMessage | LoadAsrMessage | TranscribeMessage;

const post = (message: unknown, transfer?: Transferable[]): void => {
  (self as unknown as Worker).postMessage(message, transfer ?? []);
};

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  try {
    if (message.type === 'load') {
      session = await createSession({
        device: message.device,
        dtype: message.dtype,
        onProgress: (progress: ModelProgress) => {
          post({ type: 'progress', payload: progress });
        },
      });
      const info: SessionInfo = session.info;
      post({ type: 'ready', payload: info });
      return;
    }

    if (message.type === 'infer') {
      if (!session) {
        throw new Error('model not loaded yet');
      }
      const result = await session.runFull(message.audio);
      post(
        { type: 'result', payload: { requestId: message.requestId, ...result } },
        [result.logPosteriors.buffer],
      );
      return;
    }

    if (message.type === 'load-asr') {
      const processor = await AutoProcessor.from_pretrained(ASR_MODEL_ID);
      const tokenizer = await AutoTokenizer.from_pretrained(ASR_MODEL_ID);
      // TokenizerModel.vocab is the id -> token table
      const tokenizerModel = (tokenizer as unknown as { model: { vocab: string[] } }).model;
      const idToToken = tokenizerModel.vocab;
      const model = await Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_ID, { dtype: 'q8' });
      asr = { processor, model, idToToken };
      post({ type: 'asr-ready' });
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
      post({ type: 'transcript', payload: { requestId: message.requestId, transcript } });
      return;
    }
  } catch (error) {
    post({
      type: 'error',
      message: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    });
  }
};
