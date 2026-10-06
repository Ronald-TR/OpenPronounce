/**
 * Port of openpronounce/phones.py: phone normalization and greedy CTC decoding.
 *
 * Kept line-by-line faithful to the Python implementation so the browser output
 * can be compared 1:1 with the server output in the parity fixtures.
 */

// Recognizer tokens that are not phones of any supported language: Mandarin tone
// numbers, aspiration marks and two-letter spellings of English diphthongs.
export const TOKEN_ALIASES: Record<string, string> = {
  th: 't', kh: 'k', ph: 'p', 'tʰ': 't', 'kʰ': 'k', 'pʰ': 'p',
  ai: 'aɪ', ei: 'eɪ', au: 'aʊ', ou: 'oʊ',
};

// Phones that espeak and the recognizer use interchangeably, or that no learner
// should be penalised for (English only for the MVP).
export const PHONE_MAP_EN: Record<string, string> = {
  'ᵻ': 'ɪ',   // espeak's reduced /ɪ/
  'ɐ': 'ə',   // espeak's reduced /a/ (article "a")
  'ɔ': 'ɑ',   // cot-caught merger (American English)
  'ɔɹ': 'oɹ', // horse-hoarse merger ("for" / "four")
  'ɜ': 'ɚ',   // nurse vowel: espeak writes ɜː, the recognizer ɚ
  'ɾ': 't',   // flapped t
  'ɫ': 'l',
};

export function normalizePhone(phone: string, lang = 'en'): string {
  let p = phone.replace(/ː/g, '').replace(/\d/g, '');
  p = TOKEN_ALIASES[p] ?? p;
  if (lang === 'en') {
    p = PHONE_MAP_EN[p] ?? p;
  }
  return p;
}

export interface IndexedPhone {
  phone: string;
  /** Positions of the input phones merged into this one. */
  sources: number[];
}

export function normalizeIndexed(phones: string[], lang = 'en'): IndexedPhone[] {
  const out: IndexedPhone[] = [];
  phones.forEach((raw, index) => {
    const phone = normalizePhone(raw, lang);
    if (!phone) {
      return;
    }
    const last = out[out.length - 1];
    if (last && (last.phone === phone || (last.phone === 'ɚ' && phone === 'ɹ'))) {
      last.sources.push(index);
      return;
    }
    out.push({ phone, sources: [index] });
  });
  return out;
}

export function normalizePhones(phones: string[], lang = 'en'): string[] {
  return normalizeIndexed(phones, lang).map((entry) => entry.phone);
}

export interface PhoneRecognition {
  phones: string[];
  confidences: number[];
  /** (start, end) frame range of each phone. */
  spans: Array<[number, number]>;
  /** Row-major (frames x vocab) log posteriors. */
  logPosteriors: Float32Array;
  frameCount: number;
  vocabSize: number;
  vocab: string[];
}

const isSpecial = (token: string) => token.startsWith('<') && token.endsWith('>');

/**
 * Greedy CTC decoding of row-major log posteriors, port of phones.decode_ctc.
 * Repeated frames are collapsed, blanks and special tokens dropped; each phone
 * gets the peak posterior over the frames it was decoded from. When
 * normalization merges phones, the merged phone keeps the highest confidence
 * and the union of the frames.
 */
export function decodeCtc(
  logPosteriors: Float32Array,
  frameCount: number,
  vocabSize: number,
  vocab: string[],
  blankId = 0,
  lang = 'en',
  normalize = true,
): PhoneRecognition {
  // argmax per frame (first index wins on ties, like numpy)
  const ids = new Int32Array(frameCount);
  for (let f = 0; f < frameCount; f++) {
    const base = f * vocabSize;
    let best = 0;
    let bestValue = logPosteriors[base];
    for (let v = 1; v < vocabSize; v++) {
      const value = logPosteriors[base + v];
      if (value > bestValue) {
        bestValue = value;
        best = v;
      }
    }
    ids[f] = best;
  }

  const tokens: string[] = [];
  const confidences: number[] = [];
  const spans: Array<[number, number]> = [];
  let start = 0;
  for (let end = 1; end <= frameCount; end++) {
    if (end < frameCount && ids[end] === ids[start]) {
      continue;
    }
    const token = vocab[ids[start]];
    if (ids[start] !== blankId && !isSpecial(token)) {
      tokens.push(token);
      let max = -Infinity;
      for (let f = start; f < end; f++) {
        const value = logPosteriors[f * vocabSize + ids[start]];
        if (value > max) {
          max = value;
        }
      }
      confidences.push(Math.exp(max));
      spans.push([start, end]);
    }
    start = end;
  }

  if (!normalize) {
    return { phones: tokens, confidences, spans, logPosteriors, frameCount, vocabSize, vocab };
  }

  const phones: string[] = [];
  const mergedConfidences: number[] = [];
  const mergedSpans: Array<[number, number]> = [];
  for (const entry of normalizeIndexed(tokens, lang)) {
    phones.push(entry.phone);
    let best = -Infinity;
    for (const source of entry.sources) {
      if (confidences[source] > best) {
        best = confidences[source];
      }
    }
    mergedConfidences.push(best);
    mergedSpans.push([
      spans[entry.sources[0]][0],
      spans[entry.sources[entry.sources.length - 1]][1],
    ]);
  }
  return { phones, confidences: mergedConfidences, spans: mergedSpans, logPosteriors, frameCount, vocabSize, vocab };
}

/** Row-major log softmax over the last dimension (torch.log_softmax(dim=-1)). */
export function logSoftmaxLastDim(logits: Float32Array, frameCount: number, vocabSize: number): Float32Array {
  const out = new Float32Array(logits.length);
  for (let f = 0; f < frameCount; f++) {
    const base = f * vocabSize;
    let max = -Infinity;
    for (let v = 0; v < vocabSize; v++) {
      if (logits[base + v] > max) {
        max = logits[base + v];
      }
    }
    let sum = 0;
    for (let v = 0; v < vocabSize; v++) {
      sum += Math.exp(logits[base + v] - max);
    }
    const logSum = max + Math.log(sum);
    for (let v = 0; v < vocabSize; v++) {
      out[base + v] = logits[base + v] - logSum;
    }
  }
  return out;
}
