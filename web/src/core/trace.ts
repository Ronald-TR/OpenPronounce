/**
 * Aligned phoneme trace for the details chart, port of
 * speech.align_sequences_dtw + get_phoneme_embeddings: the codepoints of the
 * space-joined phone strings are DTW-aligned with fastdtw and returned as two
 * same-length sequences (reference, learner) for the "reference vs you" plot.
 */
import { fastdtw } from './dtw';

export interface PhonemeTrace {
  expected: number[];
  heard: number[];
}

function codepoints(phones: string[]): number[][] {
  return phones.join(' ').split('').map((character) => [character.charCodeAt(0)]);
}

export function alignPhonemeTrace(expectedPhones: string[], heardPhones: string[]): PhonemeTrace {
  if (!expectedPhones.length || !heardPhones.length) {
    return { expected: [], heard: [] };
  }
  const expectedSequence = codepoints(expectedPhones);
  const heardSequence = codepoints(heardPhones);
  const { path } = fastdtw(expectedSequence, heardSequence);
  const expected: number[] = new Array(path.length);
  const heard: number[] = new Array(path.length);
  for (let step = 0; step < path.length; step++) {
    const [i, j] = path[step];
    expected[step] = expectedSequence[i][0];
    heard[step] = heardSequence[j][0];
  }
  return { expected, heard };
}
