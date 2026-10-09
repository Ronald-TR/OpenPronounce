/**
 * Port of the scoring parts of openpronounce/speech.py: transcription
 * cleaning, word error rate and the weighted 0-100 pronunciation score.
 */
import { levenshteinDistance } from './levenshtein';

// Threshold above which a word is considered mispronounced:
// ratio of edited phonemes over the number of expected phonemes.
export const WORD_ERROR_THRESHOLD = 0.4;

// Mean per-step DTW distance between Wav2Vec2 frames of the learner and of the
// TTS reference. Calibrated on the bundled samples (see benchmarks/): ~6 for a
// clean native-like reading, ~12-15 for a wrong sentence. On speechocean762 the
// median is 11.5 (9.3 to 13.6 for the middle half).
//
// Phase-4 calibration (500 speechocean762 utterances, see web/SPIKE.md): with
// the browser's int8 encoder these constants score Spearman 0.648 against the
// human ratings — the torch fp32 baseline scores 0.652 with the same numbers.
// The int8 encoder inflates distances by ~+0.9 on average; a grid search finds
// 0.685 with a heavy acoustic weight (0.7/0.3/0, GOOD 8 / BAD 18), matching the
// Python finding that heavier acoustic weights correlate slightly better but
// stop punishing a wrong sentence — so the phone and word terms keep the
// majority of the weight and the shipped constants stay.
export const ACOUSTIC_DISTANCE_BAD = 15.0;
// Per-language "good" distance (languages.py), English calibrated on human readings.
export const ACOUSTIC_GOOD: Record<string, number> = {
  en: 6.0, fr: 9.0, es: 10.0, de: 10.0, it: 9.0, pt: 11.0, nl: 13.0,
};
export const ACOUSTIC_DISTANCE_GOOD_EN = 6.0;
export const ACOUSTIC_DISTANCE_SPAN = ACOUSTIC_DISTANCE_BAD - ACOUSTIC_DISTANCE_GOOD_EN;

// Calibrated on 500 speechocean762 utterances (benchmarks/): Spearman 0.65.
export const SCORE_WEIGHTS = { acoustic: 0.3, phonemes: 0.4, words: 0.3 } as const;

/** Lower-case, strip and keep only letters, apostrophes and single spaces (speech.clean_transcription). */
export function cleanTranscription(text: string): string {
  let cleaned = text.toLowerCase().trim();
  cleaned = cleaned.replace(/[^a-zA-Z' ]+/g, '');
  cleaned = cleaned.replace(/\s+/g, ' ');
  return cleaned.trim();
}

const WORD_RE = /[\p{L}\p{N}_](?:[\p{L}\p{N}_']*[\p{L}\p{N}_])?/gu;

export function words(text: string): string[] {
  return text.match(WORD_RE) ?? [];
}

/** Edited words / expected words (speech.py word_error_rate). */
export function wordErrorRate(expectedWords: string[], heardWords: string[]): number {
  return levenshteinDistance(expectedWords, heardWords) / Math.max(1, expectedWords.length);
}

/**
 * Combine length-independent measures into a 0-100 score
 * (speech.compute_pronunciation_score): acoustic 30 %, phonemes 40 %, words 30 %.
 *
 * With `acousticDistance === null` (encoder unavailable, e.g. on a static
 * deploy without self-hosted models) the acoustic term is dropped and the
 * remaining weights renormalized: phonemes 4/7, words 3/7.
 */
export function computePronunciationScore(
  acousticDistance: number | null,
  phonemeErrorRate: number,
  wordErrorRateValue: number,
  lang = 'en',
): number {
  const clip = (x: number) => Math.min(100.0, Math.max(0.0, x));
  const phonemeScore = clip(100 * (1 - phonemeErrorRate));
  const wordScore = clip(100 * (1 - wordErrorRateValue));

  if (acousticDistance === null) {
    const renormalized = (1 - SCORE_WEIGHTS.acoustic);
    const finalScore =
      (SCORE_WEIGHTS.phonemes / renormalized) * phonemeScore
      + (SCORE_WEIGHTS.words / renormalized) * wordScore;
    return Math.round(clip(finalScore) * 100) / 100;
  }

  const good = ACOUSTIC_GOOD[lang] ?? ACOUSTIC_DISTANCE_GOOD_EN;
  const acousticScore = clip(100 * (1 - (acousticDistance - good) / ACOUSTIC_DISTANCE_SPAN));
  const finalScore =
    SCORE_WEIGHTS.acoustic * acousticScore
    + SCORE_WEIGHTS.phonemes * phonemeScore
    + SCORE_WEIGHTS.words * wordScore;
  return Math.round(clip(finalScore) * 100) / 100;
}
