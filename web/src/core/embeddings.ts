/**
 * Acoustic embeddings: the wav2vec2-large-960h ENCODER (the same module the
 * Python pipeline uses, self-hosted as ONNX) plus the DTW acoustic distance.
 */
import { AutoModel, env, Tensor, type Tensor as TensorType } from '@huggingface/transformers';
import { fastdtw } from './dtw';

export interface Embedding {
  frames: number;
  dim: number;
  data: Float32Array;
  ms: number;
}

export interface Embedder {
  embed: (audio: Float32Array) => Promise<Embedding>;
  modelId: string;
  dtype: string;
}

/** Wav2Vec2Processor do_normalize: (x - mean) / sqrt(var + 1e-7). */
export function normalizeUtterance(audio: Float32Array): Float32Array {
  let mean = 0;
  for (let i = 0; i < audio.length; i++) mean += audio[i];
  mean /= audio.length;
  let variance = 0;
  for (let i = 0; i < audio.length; i++) {
    const d = audio[i] - mean;
    variance += d * d;
  }
  variance /= audio.length;
  const scale = Math.sqrt(variance + 1e-7);
  const out = new Float32Array(audio.length);
  for (let i = 0; i < audio.length; i++) out[i] = (audio[i] - mean) / scale;
  return out;
}

export type EmbedderDevice = 'wasm' | 'webgpu';
export type EmbedderDtype = 'q8' | 'q4' | 'q4f16' | 'fp16' | 'fp32' | 'auto' | 'int8' | 'uint8' | 'bnb4';

export interface CreateEmbedderOptions {
  dtype?: EmbedderDtype;
  device?: EmbedderDevice;
  onProgress?: (progress: { status: string; file?: string; progress?: number; loaded?: number; total?: number }) => void;
}

/**
 * Load the local encoder ONNX via transformers.js. `modelId` is either a
 * directory under `env.localModelPath` (Node) or an origin-relative path
 * (browser, e.g. "models/wav2vec2-960h-encoder" with env.localModelPath="/").
 */
export async function createEmbedder(
  modelId: string,
  dtype: EmbedderDtype = 'q8',
  device: EmbedderDevice = 'wasm',
  onProgress?: CreateEmbedderOptions['onProgress'],
): Promise<Embedder> {
  // transformers.js in Node runs on onnxruntime-node, which calls its device "cpu"
  const resolvedDevice = typeof process !== 'undefined' && Boolean(process.versions?.node)
    ? 'cpu'
    : device;
  const model = await AutoModel.from_pretrained(modelId, { dtype, device: resolvedDevice, progress_callback: onProgress });
  return {
    modelId,
    dtype,
    async embed(audio: Float32Array): Promise<Embedding> {
      const start = performance.now();
      const normalized = normalizeUtterance(audio);
      const output = (await model({
        input_values: new Tensor('float32', normalized, [1, normalized.length]),
      })) as { last_hidden_state: TensorType };
      const hidden = output.last_hidden_state;
      const [, frames, dim] = hidden.dims as [number, number, number];
      return {
        frames,
        dim,
        data: hidden.data as Float32Array,
        ms: performance.now() - start,
      };
    },
  };
}

export interface AcousticDistance {
  distance: number;
  meanPerStep: number;
  pathLength: number;
  ms: number;
}

/** Mean per-step DTW distance between two embeddings (speech.py semantics). */
export function acousticDistance(a: Embedding, b: Embedding): AcousticDistance {
  const start = performance.now();
  const toPoints = (emb: Embedding): Point[] => {
    const points: Point[] = new Array(emb.frames);
    for (let f = 0; f < emb.frames; f++) {
      points[f] = Array.from(emb.data.subarray(f * emb.dim, (f + 1) * emb.dim));
    }
    return points;
  };
  const { distance, path } = fastdtw(toPoints(a), toPoints(b));
  return {
    distance,
    meanPerStep: distance / Math.max(1, path.length),
    pathLength: path.length,
    ms: performance.now() - start,
  };
}

type Point = number[];

export { env };
