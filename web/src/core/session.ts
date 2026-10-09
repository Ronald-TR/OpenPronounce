/**
 * Spike pipeline: load the phone recognizer ONNX model via transformers.js and
 * run audio through it, decoding phones exactly like openpronounce.phones.
 *
 * Shared by the browser Web Worker and the Node benchmark/parity harness.
 */
import { AutoProcessor, Wav2Vec2ForCTC, env, type Tensor } from '@huggingface/transformers';
import { decodeCtc, logSoftmaxLastDim, type PhoneRecognition } from './phones';

export const MODEL_ID = 'onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX';
const VOCAB_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/vocab.json`;

env.allowLocalModels = false;

export type Device = 'wasm' | 'webgpu';
export type Dtype = 'q8' | 'q4' | 'q4f16' | 'fp16' | 'fp32';

export interface ModelProgress {
  status: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

export interface SessionInfo {
  modelId: string;
  device: string;
  dtype: Dtype;
}

export interface SpikeResult {
  phones: string[];
  confidences: number[];
  frames: number;
  audioSeconds: number;
  timings: {
    preprocessMs: number;
    inferenceMs: number;
    logSoftmaxMs: number;
    decodeMs: number;
    totalMs: number;
  };
}

export interface Session {
  info: SessionInfo;
  run: (audio: Float32Array) => Promise<SpikeResult>;
  /** Like run, but also carries the decoder internals needed by comparePhones. */
  runFull: (audio: Float32Array) => Promise<FullSpikeResult>;
  recognition: (audio: Float32Array) => Promise<PhoneRecognition>;
}

export interface FullSpikeResult extends SpikeResult {
  spans: Array<[number, number]>;
  vocabSize: number;
  logPosteriors: Float32Array;
}

async function loadVocab(): Promise<{ vocab: string[]; blankId: number }> {
  const response = await fetch(VOCAB_URL);
  if (!response.ok) {
    throw new Error(`failed to fetch vocab.json: ${response.status}`);
  }
  const tokenToId: Record<string, number> = await response.json();
  const vocab: string[] = [];
  for (const [token, id] of Object.entries(tokenToId)) {
    vocab[id] = token;
  }
  const blankId = tokenToId['<pad>'] ?? 0;
  return { vocab, blankId };
}

let vocabPromise: Promise<{ vocab: string[]; blankId: number }> | null = null;

/** Recognizer vocabulary, shared by the worker and the main-thread compare. */
export function loadRecognizerVocab(): Promise<{ vocab: string[]; blankId: number }> {
  vocabPromise ??= loadVocab();
  return vocabPromise;
}

export interface CreateSessionOptions {
  device?: Device;
  dtype?: Dtype;
  onProgress?: (progress: ModelProgress) => void;
}

interface StepTimings {
  preprocessMs: number;
  inferenceMs: number;
  logSoftmaxMs: number;
  decodeMs: number;
}

const IS_NODE = typeof process !== 'undefined' && Boolean(process.versions?.node);

/** Load the model once and return a reusable inference session. */
export async function createSession(options: CreateSessionOptions = {}): Promise<Session> {
  // transformers.js in Node runs on onnxruntime-node, which calls its device "cpu"
  const device = (options.device ?? 'wasm') === 'webgpu' && !IS_NODE ? 'webgpu' : IS_NODE ? 'cpu' : 'wasm';
  const dtype = options.dtype ?? 'q8';

  const processor = await AutoProcessor.from_pretrained(MODEL_ID);
  const model = await Wav2Vec2ForCTC.from_pretrained(MODEL_ID, {
    device,
    dtype,
    progress_callback: options.onProgress,
  });
  const { vocab, blankId } = await loadVocab();

  const recognize = async (audio: Float32Array): Promise<{ rec: PhoneRecognition; timings: StepTimings }> => {
    const t0 = performance.now();
    const inputs = await processor(audio);
    const t1 = performance.now();
    const output = await model({ input_values: inputs.input_values as Tensor });
    const t2 = performance.now();
    const logits = output.logits;
    const [, frames, vocabSize] = logits.dims as [number, number, number];
    const logPosteriors = logSoftmaxLastDim(logits.data as Float32Array, frames, vocabSize);
    const t3 = performance.now();
    const rec = decodeCtc(logPosteriors, frames, vocabSize, vocab, blankId);
    const t4 = performance.now();

    return {
      rec,
      timings: {
        preprocessMs: t1 - t0,
        inferenceMs: t2 - t1,
        logSoftmaxMs: t3 - t2,
        decodeMs: t4 - t3,
      },
    };
  };

  const runFull = async (audio: Float32Array): Promise<FullSpikeResult> => {
    const start = performance.now();
    const { rec, timings } = await recognize(audio);
    return {
      phones: rec.phones,
      confidences: rec.confidences,
      frames: rec.frameCount,
      audioSeconds: audio.length / 16000,
      spans: rec.spans,
      vocabSize: rec.vocabSize,
      logPosteriors: rec.logPosteriors,
      timings: {
        ...timings,
        totalMs: performance.now() - start,
      },
    };
  };

  const run = async (audio: Float32Array): Promise<SpikeResult> => {
    const start = performance.now();
    const { rec, timings } = await recognize(audio);
    return {
      phones: rec.phones,
      confidences: rec.confidences,
      frames: rec.frameCount,
      audioSeconds: audio.length / 16000,
      timings: {
        ...timings,
        totalMs: performance.now() - start,
      },
    };
  };

  const recognition = async (audio: Float32Array): Promise<PhoneRecognition> => {
    const { rec } = await recognize(audio);
    return rec;
  };

  return {
    info: { modelId: MODEL_ID, device, dtype },
    run,
    runFull,
    recognition,
  };
}
