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
 */
export function computePronunciationScore(
  acousticDistance: number,
  phonemeErrorRate: number,
  wordErrorRateValue: number,
  lang = 'en',
): number {
  const good = ACOUSTIC_GOOD[lang] ?? ACOUSTIC_DISTANCE_GOOD_EN;
  const acousticScore = 100 * (1 - (acousticDistance - good) / ACOUSTIC_DISTANCE_SPAN);
  const phonemeScore = 100 * (1 - phonemeErrorRate);
  const wordScore = 100 * (1 - wordErrorRateValue);

  const clip = (x: number) => Math.min(100.0, Math.max(0.0, x));
  const finalScore =
    SCORE_WEIGHTS.acoustic * clip(acousticScore)
    + SCORE_WEIGHTS.phonemes * clip(phonemeScore)
    + SCORE_WEIGHTS.words * clip(wordScore);
  return Math.round(clip(finalScore) * 100) / 100;
}
