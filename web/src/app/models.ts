/**
 * Model manager: what is downloaded, how much space it uses, and a way to
 * reclaim it. transformers.js caches weight files in the Cache API; vits-web
 * keeps piper voices in OPFS.
 */
import { flush as flushPiperVoices, stored as storedPiperVoices } from '@diffusionstudio/vits-web';
import { PIPER_VOICE_ID } from '../core/tts';

export interface ModelInfo {
  id: string;
  label: string;
  sizeMb: number;
  kind: 'worker' | 'piper' | 'encoder';
}

/**
 * First-visit download sizes. Device is webgpu-only, so the fp16 variants are
 * the ones actually loaded; the int8 variants serve as in-worker fallback.
 */
export const MODELS: ModelInfo[] = [
  { id: 'phones', label: 'Phone recognizer (wav2vec2 espeak)', sizeMb: 632, kind: 'worker' },
  { id: 'encoder', label: 'Acoustic encoder (wav2vec2 960h)', sizeMb: 355, kind: 'encoder' },
  { id: 'asr', label: 'Word transcription (wav2vec2 960h base)', sizeMb: 190, kind: 'worker' },
  { id: 'piper', label: 'Reference voice (piper lessac-medium)', sizeMb: 63, kind: 'piper' },
];

/** Substrings that identify each model's entries inside the transformers cache. */
const CACHE_MATCHERS: Record<string, { dirTokens: string[]; file: string }> = {
  phones: { dirTokens: ['espeak-cv-ft', 'phone-recognizer'], file: 'model_quantized.onnx' },
  asr: { dirTokens: ['base-960h', 'word-transcription'], file: 'model_fp16.onnx' },
  encoder: { dirTokens: ['wav2vec2-960h-encoder'], file: 'model_quantized.onnx' },
};

export type ModelId = keyof typeof CACHE_MATCHERS | 'piper';

export interface DownloadStatus {
  phones: boolean;
  asr: boolean;
  encoder: boolean;
  piper: boolean;
}

/** Whether each model's weights are already stored in this browser. */
export async function modelDownloadStatus(): Promise<DownloadStatus> {
  const status: DownloadStatus = { phones: false, asr: false, encoder: false, piper: false };

  try {
    const cache = await caches.open('transformers-cache');
    const urls = (await cache.keys()).map((request) => request.url);
    const entries = Object.entries(CACHE_MATCHERS) as Array<[keyof Omit<DownloadStatus, 'piper'>, typeof CACHE_MATCHERS[string]]>;
    for (const [modelId, matcher] of entries) {
      status[modelId] = urls.some((url) =>
        matcher.dirTokens.some((token) => url.includes(token)) && url.includes(matcher.file),
      );
    }
  } catch {
    // Cache API unavailable — report as not downloaded
  }

  try {
    status.piper = (await storedPiperVoices()).includes(PIPER_VOICE_ID);
  } catch {
    // OPFS unavailable
  }

  return status;
}

export async function storedPiperVoicesList(): Promise<string[]> {
  try {
    return await storedPiperVoices();
  } catch {
    return [];
  }
}

export interface StorageEstimate {
  usageMb: number;
  quotaMb: number;
}

export async function storageEstimate(): Promise<StorageEstimate | null> {
  if (!navigator.storage?.estimate) {
    return null;
  }
  const { usage = 0, quota = 0 } = await navigator.storage.estimate();
  return { usageMb: usage / 1e6, quotaMb: quota / 1e6 };
}

/** Delete every cached model (Cache API entries + piper OPFS files). */
export async function clearStoredModels(): Promise<void> {
  const names = await caches.keys();
  await Promise.all(
    names
      .filter((name) => /transformers|onnx|vits|piper/i.test(name))
      .map((name) => caches.delete(name)),
  );
  try {
    await flushPiperVoices();
  } catch {
    // OPFS unavailable or already empty
  }
}
