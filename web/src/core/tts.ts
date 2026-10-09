/**
 * Reference TTS in the browser: piper (vits) compiled to WASM via
 * @diffusionstudio/vits-web, running in its own worker with OPFS caching.
 */
import { predict, type Progress } from '@diffusionstudio/vits-web';
import { decodeTo16kMono } from './audio';

/** The voice the Python calibration used (piper en_US lessac, medium quality). */
export const PIPER_VOICE_ID = 'en_US-lessac-medium' as const;

export interface SynthesisResult {
  audio: Float32Array;
  blob: Blob;
}

export async function synthesizeReference(
  text: string,
  onProgress?: (progress: Progress) => void,
  voiceId: string = PIPER_VOICE_ID,
): Promise<SynthesisResult> {
  const blob = await predict(
    { text, voiceId: voiceId as never },
    onProgress,
  );
  const audio = await decodeTo16kMono(blob);
  return { audio, blob };
}
