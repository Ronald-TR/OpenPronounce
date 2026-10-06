/**
 * Port of openpronounce/phones.py comparison logic: alignment, near-phone
 * costs, plausibility from frame posteriors, per-phone reports and
 * compare_phones. Line-by-line faithful so outputs can be diffed against the
 * Python fixtures.
 */
import { levenshteinDistance, levenshteinOpcodes, type Opcode } from './levenshtein';
import { normalizePhones, normalizePhone, type PhoneRecognition } from './phones';
import { getExpectedPhones } from './g2p';
import { DEFAULT_LANGUAGE } from './languages';

// A word is reported when its error confidences add up to at least this share
// of its phones...
export const PHONE_ERROR_THRESHOLD = 0.4;
// ...or to at least this many phones, whatever the length of the word.
export const PHONE_ERROR_MIN_EDITS = 2;
// Substituting a phone for a close one counts for this much of a full error.
export const NEAR_PHONE_COST = 0.5;
// Extra phones after the last phone of a word / dropped final consonant.
export const FINAL_EXTRA_COST = 0.25;
export const FINAL_DELETION_COST = 0.5;
// An expected phone whose posterior reaches this value where the recognizer
// heard something else is not a confident error.
export const PHONE_PLAUSIBLE_POSTERIOR = 0.05;

// Groups of phones close enough that substituting one for another is a minor
// error: handled in substitutionCost() below.

const NEAR_PHONES: Record<string, string[][]> = {
  fr: [
    ['e', 'i'], ['ø', 'y'], ['ø', 'e'], ['o', 'u'], ['ɑ̃', 'ɔ̃'], ['ɛ̃', 'ɑ̃'], ['ʁ', 'r', 'ɹ'],
    ['b', 'p'], ['d', 't'], ['ɡ', 'k'], ['z', 's'], ['v', 'f'], ['ʒ', 'ʃ'],
  ],
  en: [
    ['ɪ', 'i'], ['ʊ', 'u'], ['ɛ', 'æ'], ['ʌ', 'ɑ', 'ə', 'a'], ['æ', 'a'], ['ɛ', 'eɪ'],
    ['ð', 'z', 'd'], ['θ', 's', 't'], ['b', 'p'], ['d', 't'], ['ɡ', 'k'], ['z', 's'],
    ['v', 'f'], ['ʒ', 'ʃ'], ['dʒ', 'tʃ'], ['n', 'ŋ'], ['h', 'x'], ['ɹ', 'r'], ['v', 'w'],
  ],
};

// Function words with more than one accepted pronunciation (already normalized), per language.
const ALTERNATE_PRONUNCIATIONS: Record<string, Record<string, string[][]>> = {
  en: {
    a: [['ə'], ['eɪ']],
    an: [['ən'], ['æn']],
    the: [['ð', 'ə'], ['ð', 'i'], ['ð', 'ɪ']],
    to: [['t', 'ə'], ['t', 'u']],
    of: [['ʌ', 'v'], ['ə', 'v']],
    and: [['æ', 'n', 'd'], ['ə', 'n', 'd'], ['ə', 'n']],
    for: [['f', 'ɔ', 'ɹ'], ['f', 'ɚ']],
    you: [['j', 'u'], ['j', 'ə']],
    are: [['ɑ', 'ɹ'], ['ɚ']],
    was: [['w', 'ʌ', 'z'], ['w', 'ə', 'z']],
    that: [['ð', 'æ', 't'], ['ð', 'ə', 't']],
    can: [['k', 'æ', 'n'], ['k', 'ə', 'n']],
    have: [['h', 'æ', 'v'], ['h', 'ə', 'v']],
    or: [['ɔ', 'ɹ'], ['ɚ']],
  },
  fr: {
    je: [['ʒ']],
    le: [['l']],
    de: [['d']],
    ne: [['n']],
    ce: [['s']],
    se: [['s']],
    me: [['m']],
    te: [['t']],
    que: [['k']],
  },
};

/** Map every expected phone index to the heard phone indices it aligns with (insertions go to the previous phone). */
export function align(expected: string[], heard: string[]): Set<number>[] {
  const alignment: Set<number>[] = expected.map(() => new Set<number>());
  for (const op of levenshteinOpcodes(expected, heard)) {
    const { tag, i1, i2, j1, j2 } = op;
    if (tag === 'equal') {
      for (let k = i1, l = j1; k < i2 && l < j2; k++, l++) {
        alignment[k].add(l);
      }
    } else if (tag === 'replace') {
      const lenI = i2 - i1;
      const lenJ = j2 - j1;
      for (let k = i1; k < i2; k++) {
        const start = j1 + Math.floor(((k - i1) * lenJ) / lenI);
        const end = j1 + Math.floor(((k - i1 + 1) * lenJ) / lenI);
        if (start === end && lenJ > 0) {
          alignment[k].add(Math.min(start, j2 - 1));
        } else {
          for (let l = start; l < end; l++) {
            alignment[k].add(l);
          }
        }
      }
    } else if (tag === 'insert') {
      const k = i1 > 0 ? i1 - 1 : i1;
      if (k < alignment.length) {
        for (let l = j1; l < j2; l++) {
          alignment[k].add(l);
        }
      }
    }
    // "delete" (missing expected phones) produces no mapping.
  }
  return alignment;
}

/** Accepted phone sequences for a word: expected, alternates, and cross-word merge. */
export function pronunciations(
  word: string,
  expectedSeg: string[],
  previousLastPhone: string | null,
  lang: string,
): string[][] {
  const alternates = ALTERNATE_PRONUNCIATIONS[lang]?.[word] ?? [];
  const candidates: string[][] = [[...expectedSeg], ...alternates.map((alt) => normalizePhones(alt, lang))];
  if (previousLastPhone !== null && expectedSeg.length > 1 && expectedSeg[0] === previousLastPhone) {
    candidates.push(expectedSeg.slice(1));
  }
  return candidates;
}

/** Cost of hearing `heard` instead of `expected`: 1, or NEAR_PHONE_COST for close phones. */
export function substitutionCost(expected: string, heard: string, lang: string): number {
  for (const group of NEAR_PHONES[lang] ?? []) {
    if (group.includes(expected) && group.includes(heard)) {
      return NEAR_PHONE_COST;
    }
  }
  return 1.0;
}

const tokenIdsCache = new Map<string, Map<string, number[]>>();
function tokenIdsByPhone(vocab: string[], lang: string): Map<string, number[]> {
  const cacheKey = lang + '\u0000' + vocab.length + '\u0000' + vocab[0] + '\u0000' + vocab[vocab.length - 1];
  let ids = tokenIdsCache.get(cacheKey);
  if (!ids) {
    ids = new Map<string, number[]>();
    vocab.forEach((token, index) => {
      if (!isSpecial(token)) {
        const key = normalizePhone(token, lang);
        const list = ids!.get(key);
        if (list) {
          list.push(index);
        } else {
          ids!.set(key, [index]);
        }
      }
    });
    tokenIdsCache.set(cacheKey, ids);
  }
  return ids;
}

function isSpecial(token: string): boolean {
  return token.startsWith('<') && token.endsWith('>');
}

/** Highest posterior of `phone` over frames [start:end), 0 without posteriors. */
export function plausibility(
  recognition: PhoneRecognition | null,
  phone: string,
  start: number,
  end: number,
  lang: string,
): number {
  if (!recognition || end <= start) {
    return 0.0;
  }
  const ids = tokenIdsByPhone(recognition.vocab, lang).get(phone);
  if (!ids) {
    return 0.0;
  }
  let max = -Infinity;
  for (let frame = start; frame < end; frame++) {
    const base = frame * recognition.vocabSize;
    for (const id of ids) {
      if (recognition.logPosteriors[base + id] > max) {
        max = recognition.logPosteriors[base + id];
      }
    }
  }
  return Math.exp(max);
}

export interface PhoneReport {
  expected: string;
  heard: string;
  confidence: number;
}

interface ReportRegion {
  start: number;
  end: number;
}

/** Per-phone report of one word (port of phones._phone_reports). */
export function phoneReports(
  candidate: string[],
  actual: string[],
  matched: number[],
  recognition: PhoneRecognition | null,
  region: ReportRegion,
  lang: string,
): PhoneReport[] {
  const local: Set<number>[] = align(candidate, actual);
  const spans = recognition ? recognition.spans : null;

  const frames = (j: number): [number, number] => (spans ? spans[matched[j]] : [0, 0]);

  const reports: PhoneReport[] = [];
  const last = candidate.length - 1;
  for (let k = 0; k < candidate.length; k++) {
    const expected = candidate[k];
    const heardIndices = [...local[k]].sort((a, b) => a - b);
    const heard = heardIndices.map((j) => actual[j]);
    let confidence: number;
    if (heard.length === 1 && heard[0] === expected) {
      confidence = 0.0;
    } else if (heard.includes(expected)) {
      confidence = k === last ? FINAL_EXTRA_COST : 1.0;
    } else {
      let cost: number;
      let start: number;
      let end: number;
      if (heard.length) {
        cost = Math.max(...heard.map((h) => substitutionCost(expected, h, lang)));
        start = Math.min(...heardIndices.map((j) => frames(j)[0]));
        end = Math.max(...heardIndices.map((j) => frames(j)[1]));
      } else {
        cost = k === last ? FINAL_DELETION_COST : 1.0;
        const previous: number[] = [];
        for (let kk = 0; kk < k; kk++) previous.push(...local[kk]);
        const following: number[] = [];
        for (let kk = k + 1; kk < candidate.length; kk++) following.push(...local[kk]);
        start = previous.length ? frames(Math.max(...previous))[0] : region.start;
        end = following.length ? frames(Math.min(...following))[1] : region.end;
      }
      const p = plausibility(recognition, expected, start, end, lang);
      confidence = cost * (1.0 - Math.min(1.0, p / PHONE_PLAUSIBLE_POSTERIOR));
    }
    reports.push({ expected, heard: heard.join(''), confidence });
  }
  return reports;
}

export interface WordReport {
  position: number;
  word: string;
  expected: string[];
  actual: string[];
  distance: number;
  phones: PhoneReport[];
  weightedEdits: number;
}

/** Compare what was heard with the reference text word by word (port of phones._word_reports). */
export async function wordReports(
  heardRecognition: PhoneRecognition,
  textReference: string,
  lang = DEFAULT_LANGUAGE,
): Promise<WordReport[]> {
  const heard = [...heardRecognition.phones];
  const { words, groups } = await getExpectedPhones(textReference, lang);
  const expected = groups.flat();
  const alignment = align(expected, heard);

  const reports: WordReport[] = [];
  let offset = 0;
  let previousLastPhone: string | null = null;
  for (let position = 0; position < words.length; position++) {
    const word = words[position];
    const group = groups[position];
    if (!group.length) {
      continue;
    }
    const indices: number[] = [];
    for (let k = offset; k < offset + group.length; k++) indices.push(k);
    offset += group.length;
    const matched = [...new Set(indices.flatMap((i) => [...alignment[i]]))].sort((a, b) => a - b);
    const actual = matched.map((j) => heard[j]);

    const candidates = pronunciations(word, group, previousLastPhone, lang);
    const distances = candidates.map((c) => levenshteinDistance(c, actual));
    const distance = Math.min(...distances);
    const candidate = candidates[distances.indexOf(distance)];
    previousLastPhone = group[group.length - 1];

    let region: ReportRegion;
    if (matched.length) {
      region = {
        start: heardRecognition.spans[matched[0]][0],
        end: heardRecognition.spans[matched[matched.length - 1]][1],
      };
    } else {
      const before: number[] = [];
      for (let i = 0; i < indices[0]; i++) before.push(...alignment[i]);
      const after: number[] = [];
      for (let i = indices[indices.length - 1] + 1; i < expected.length; i++) after.push(...alignment[i]);
      region = {
        start: before.length ? heardRecognition.spans[Math.max(...before)][0] : 0,
        end: after.length
          ? heardRecognition.spans[Math.min(...after)][1]
          : heardRecognition.logPosteriors.length / heardRecognition.vocabSize,
      };
    }
    const reports1 = distance ? phoneReports(candidate, actual, matched, heardRecognition, region, lang) : [];
    reports.push({
      position,
      word,
      expected: [...group],
      actual,
      distance,
      phones: reports1,
      weightedEdits: reports1.reduce((sum, p) => sum + p.confidence, 0),
    });
  }
  return reports;
}

export interface PhoneError {
  position: number;
  word: string;
  expected: string;
  actual: string;
  actual_word: string;
  phone_distance: number;
  confidence: number;
  phones: Array<PhoneReport & { confidence: number }>;
}

export interface ComparePhonesResult {
  expected_phones: string[][];
  heard_phones: string[];
  heard_phones_confidence: number[];
  phone_error_rate: number;
  errors: PhoneError[];
  words_with_errors: string[];
}

/**
 * Compare recognized phones with the phones expected for the reference text,
 * word by word — port of phones.compare_phones (async: G2P runs espeak WASM).
 */
export async function comparePhones(
  heardRecognition: PhoneRecognition,
  textReference: string,
  lang = DEFAULT_LANGUAGE,
): Promise<ComparePhonesResult> {
  const heard = [...heardRecognition.phones];
  const heardConfidences = [...heardRecognition.confidences];
  const { groups } = await getExpectedPhones(textReference, lang);
  const expected = groups.flat();
  let phoneErrorRate: number;
  if (expected.length) {
    phoneErrorRate = levenshteinDistance(expected, heard) / expected.length;
  } else {
    phoneErrorRate = heard.length ? 1.0 : 0.0;
  }

  const errors: PhoneError[] = [];
  const wordsWithErrors: string[] = [];
  for (const report of await wordReports(heardRecognition, textReference, lang)) {
    const edits = report.weightedEdits;
    const length = report.expected.length;
    if (report.distance && (edits / length >= PHONE_ERROR_THRESHOLD || edits >= PHONE_ERROR_MIN_EDITS)) {
      errors.push({
        position: report.position,
        word: report.word,
        expected: report.expected.join(''),
        actual: report.actual.join(''),
        actual_word: '',
        phone_distance: report.distance,
        confidence: Math.round(Math.min(1.0, Math.max(edits / length, edits / PHONE_ERROR_MIN_EDITS)) * 1000) / 1000,
        phones: report.phones.map((p) => ({ ...p, confidence: Math.round(p.confidence * 1000) / 1000 })),
      });
      wordsWithErrors.push(report.word);
    }
  }

  return {
    expected_phones: groups.map((g) => [...g]),
    heard_phones: heard,
    heard_phones_confidence: heardConfidences.map((c) => Math.round(c * 1000) / 1000),
    phone_error_rate: Math.round(phoneErrorRate * 10000) / 10000,
    errors,
    words_with_errors: wordsWithErrors,
  };
}

export type { Opcode };
