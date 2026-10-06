/** Port of openpronounce/languages.py (registry only; acoustic constants live in score code). */

export interface Language {
  code: string;
  espeak: string;
  asrModel: string;
  name: string;
}

export const DEFAULT_LANGUAGE = 'en';

export const LANGUAGES: Record<string, Language> = {
  en: { code: 'en', espeak: 'en-us', asrModel: 'facebook/wav2vec2-large-960h', name: 'English' },
  fr: { code: 'fr', espeak: 'fr-fr', asrModel: 'jonatasgrosman/wav2vec2-large-xlsr-53-french', name: 'French' },
  es: { code: 'es', espeak: 'es', asrModel: 'jonatasgrosman/wav2vec2-large-xlsr-53-spanish', name: 'Spanish' },
  de: { code: 'de', espeak: 'de', asrModel: 'jonatasgrosman/wav2vec2-large-xlsr-53-german', name: 'German' },
  it: { code: 'it', espeak: 'it', asrModel: 'jonatasgrosman/wav2vec2-large-xlsr-53-italian', name: 'Italian' },
  pt: { code: 'pt', espeak: 'pt-br', asrModel: 'jonatasgrosman/wav2vec2-large-xlsr-53-portuguese', name: 'Portuguese' },
  nl: { code: 'nl', espeak: 'nl', asrModel: 'jonatasgrosman/wav2vec2-large-xlsr-53-dutch', name: 'Dutch' },
};

export function getLanguage(code: string): Language {
  const language = LANGUAGES[code];
  if (!language) {
    throw new Error(`unsupported language ${code}, expected one of: ${Object.keys(LANGUAGES).join(', ')}`);
  }
  return language;
}
