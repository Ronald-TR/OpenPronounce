/**
 * Model manager: what is downloaded, how much space it uses, and a way to
 * reclaim it. transformers.js caches weight files in the Cache API; vits-web
 * keeps piper voices in OPFS; the local encoder is served from our origin and
 * cached by transformers.js like the rest.
 */
import { flush as flushPiperVoices, stored as storedPiperVoices } from '@diffusionstudio/vits-web';

export interface ModelInfo {
  id: string;
  label: string;
  sizeMb: number;
  kind: 'worker' | 'piper' | 'encoder';
}

/** First-visit download sizes (int8/medium variants). */
export const MODELS: ModelInfo[] = [
  { id: 'phones', label: 'Phone recognizer (wav2vec2 espeak)', sizeMb: 318, kind: 'worker' },
  { id: 'encoder', label: 'Acoustic encoder (wav2vec2 960h)', sizeMb: 355, kind: 'encoder' },
  { id: 'asr', label: 'Word transcription (wav2vec2 960h base)', sizeMb: 95, kind: 'worker' },
  { id: 'piper', label: 'Reference voice (piper lessac-medium)', sizeMb: 63, kind: 'piper' },
];

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

export async function storedPiperVoicesList(): Promise<string[]> {
  try {
    return await storedPiperVoices();
  } catch {
    return [];
  }
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
