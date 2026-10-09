/**
 * G2P (grapheme-to-phoneme) via espeak-ng compiled to WASM (npm `phonemizer`),
 * port of the phonemization parts of openpronounce/phones.py.
 *
 * The npm build does not expose espeak's phone separator, so phones are split
 * client-side by maximal munch over the per-language phone inventory (the same
 * tokens espeak emits). Stress marks are dropped, mirroring phonemizer's
 * strip=True. Everything then goes through the same normalization as Python.
 */
import { phonemize } from 'phonemizer';
import { normalizePhones } from './phones';
import { getLanguage } from './languages';

/** Phone tokens espeak emits per language, used by the maximal-munch splitter.
 * Longest tokens must be matched first (enforced at build time below).
 * Supersets of the observed inventories are safe: unknown single codepoints
 * fall back to one-phone-per-codepoint instead of dropping data. */
const PHONE_INVENTORY: Record<string, string[]> = {
  en: [
    // 3-char
    'aɪɚ',
    // 2-char
    'aɪ', 'aʊ', 'eɪ', 'ɔɪ', 'oʊ', 'dʒ', 'tʃ',
    'ɑːɹ', 'ɔːɹ', 'ɑɹ', 'ɪɹ', 'ɛɹ', 'ʊɹ', 'ɜːɹ', 'oʊɹ', 'ɝː', 'əɹ',
    'iː', 'uː', 'ɑː', 'ɔː', 'ɜː', 'eə', 'ɪə', 'ʊə', 'ɛə',
    'aɪə', 'aʊə',
    // French (for future languages; nasal vowels are 2 codepoints)
    'ɑ̃', 'ɔ̃', 'ɛ̃', 'œ̃',
    // 1-char
    'b', 'd', 'f', 'ɡ', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'ɹ', 's', 't', 'v', 'w', 'z',
    'æ', 'ð', 'ŋ', 'ɐ', 'ɔ', 'ə', 'ɚ', 'ɛ', 'ɪ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'θ', 'ᵻ', 'i', 'u',
    'a', 'e', 'o', 'x', 'ç', 'ɲ', 'ʎ', 'β',
  ],
  fr: [
    'ɑ̃', 'ɔ̃', 'ɛ̃', 'œ̃', 'aj', 'ej', 'uj', 'ɔj',
    'i', 'e', 'ɛ', 'a', 'ɑ', 'ɔ', 'o', 'u', 'y', 'ø', 'ə', 'œ', 'ə̃',
    'b', 'd', 'f', 'ɡ', 'k', 'l', 'm', 'n', 'ɲ', 'p', 'ʁ', 's', 't', 'v', 'w', 'x', 'z',
    'ʒ', 'ʃ', 'j', 'ɥ', 'ŋ',
  ],
  es: [
    'b', 'd', 'f', 'ɡ', 'k', 'l', 'm', 'n', 'ɲ', 'p', 'ɾ', 'r', '̄', 's', 't', 'θ',
    'x', 'β', 'ð', 'ɣ', 'j', 'w', 'ʎ', 'ʧ', 'a', 'e', 'i', 'o', 'u',
  ],
  de: [
    'aʊ', 'ɔʏ', 'aɪ', 'ʃ', 'ʒ', 'dʒ', 'tʃ', 'ç', 'x', 'ŋ', 'ɐ', 'ə',
    'a', 'e', 'ɛ', 'ɪ', 'i', 'ɔ', 'o', 'ʊ', 'u', 'y', 'ʏ', 'ø', 'œ',
    'b', 'd', 'f', 'ɡ', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 's', 't', 'v', 'z', 'ʀ',
  ],
  it: [
    'b', 'd', 'dz', 'dʒ', 'f', 'ɡ', 'k', 'l', 'ʎ', 'm', 'n', 'ɲ', 'p', 'r', 'ɾ',
    's', 'ʃ', 't', 'ts', 'tʃ', 'v', 'w', 'j', 'z', 'a', 'e', 'ɛ', 'i', 'ɔ', 'o', 'u',
  ],
  pt: [
    'aɪ', 'aʊ', 'eɪ', 'oj', 'ej', 'ew', 'iw', 'uj', 'ɔj', 'ɐj', 'ẽj',
    'b', 'd', 'dʒ', 'f', 'ɡ', 'j', 'k', 'l', 'ʎ', 'm', 'n', 'ɲ', 'p', 'ɾ', 'ʁ',
    's', 'ʃ', 't', 'tʃ', 'v', 'w', 'x', 'z', 'ʒ', 'a', 'ɐ', 'e', 'ɛ', 'i', 'o', 'ɔ', 'u',
  ],
  nl: [
    'aː', 'ɛː', 'ɪː', 'ɔː', 'uː', 'yː', 'øː', 'eː', 'oː', 'i', 'ɛ', 'ɪ', 'ɔ', 'ʊ',
    'ɑ', 'u', 'y', 'ø', 'ə', 'œy', 'ɑu', 'ɛu', 'ʌi',
    'b', 'd', 'f', 'ɣ', 'ɦ', 'j', 'k', 'l', 'm', 'n', 'ŋ', 'p', 'r', 's', 't', 'v',
    'x', 'z', 'ʒ', 'c', 'ʃ',
  ],
};

const SPLITTERS = new Map<string, string[]>();
function inventoryFor(lang: string): string[] {
  let inventory = SPLITTERS.get(lang);
  if (!inventory) {
    inventory = [...(PHONE_INVENTORY[lang] ?? PHONE_INVENTORY.en)].sort((a, b) => b.length - a.length);
    SPLITTERS.set(lang, inventory);
  }
  return inventory;
}

/** Split an espeak IPA word (phones concatenated, stress marks included) into phone tokens. */
export function splitEspeakPhones(word: string, lang = 'en'): string[] {
  const inventory = inventoryFor(lang);
  const tokens: string[] = [];
  let position = 0;
  while (position < word.length) {
    let matched = '';
    for (const candidate of inventory) {
      if (candidate.length && word.startsWith(candidate, position)) {
        matched = candidate;
        break;
      }
    }
    if (!matched) {
      // Unknown codepoint: keep it as its own token so nothing is silently dropped
      matched = word[position];
    }
    tokens.push(matched);
    position += matched.length;
  }
  return tokens;
}

const WORD_RE = /[\p{L}\p{N}_](?:[\p{L}\p{N}_']*[\p{L}\p{N}_])?/gu;

function words(text: string): string[] {
  return text.match(WORD_RE) ?? [];
}

async function phonemizeOne(text: string, lang: string): Promise<string[][]> {
  const voice = getLanguage(lang).espeak;
  // phonemize returns segments split at punctuation; words are space-separated within each
  const segments = await phonemize(text, voice);
  const joined = segments.join(' ').trim();
  if (!joined) {
    return [];
  }
  return joined
    .split(/\s+/)
    .map((chunk) => splitEspeakPhones(chunk.replace(/[ˈˌ]/g, ''), lang));
}

/**
 * Port of phones.get_expected_phones: words with their normalized phones.
 * Batch phonemization first, per-word fallback when counts drift apart.
 */
export async function getExpectedPhones(
  text: string,
  lang = 'en',
): Promise<{ words: string[]; groups: string[][] }> {
  const wordsList = words(text).map((w) => w.toLowerCase());
  if (!wordsList.length) {
    return { words: [], groups: [] };
  }

  let groups = await phonemizeOne(wordsList.join(' '), lang);
  if (groups.length !== wordsList.length) {
    const fallback: string[][] = [];
    for (const word of wordsList) {
      try {
        const phones = await phonemizeOne(word, lang);
        fallback.push(phones[0] ?? []);
      } catch {
        fallback.push([]);
      }
    }
    groups = fallback;
  }

  return {
    words: wordsList,
    groups: groups.map((group) => normalizePhones(group, lang)),
  };
}
