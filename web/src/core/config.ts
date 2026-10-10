/**
 * Where the client-side models come from.
 *
 * - 'hf'    — Hugging Face hub (default for the GitHub Pages deploy)
 * - 'local' — web/public/models served by the dev server / preview
 * - 's3'    — the self-hosted S3 bucket (set VITE_MODELS_SOURCE=s3 in CI once
 *             the models are uploaded and CORS is configured)
 *
 * All modes share the same on-disk/on-bucket layout, one folder per model:
 *   models/phone-recognizer/      (wav2vec2-lv-60-espeak-cv-ft, ONNX)
 *   models/word-transcription/    (wav2vec2-base-960h, ONNX)
 *   models/wav2vec2-960h-encoder/ (acoustic encoder, ONNX)
 */

export type ModelsSource = 'hf' | 'local' | 's3';

export const MODELS_SOURCE: ModelsSource =
  (import.meta.env.VITE_MODELS_SOURCE as ModelsSource | undefined) ?? 'local';

/** The source actually in use ('s3' falls back to 'hf' until reachable). */
let effectiveSource: ModelsSource = MODELS_SOURCE;

/** Runtime probe for the 's3' source: upload + CORS still pending → HF. */
export async function initModelsSource(): Promise<void> {
  if (MODELS_SOURCE !== 's3') {
    effectiveSource = MODELS_SOURCE;
    return;
  }
  try {
    const response = await fetch(`${S3_MODELS_HOST}/${MODEL_DIR_IDS.phones}/config.json`);
    effectiveSource = response.ok ? 's3' : 'hf';
  } catch {
    effectiveSource = 'hf'; // network or CORS failure
  }
  if (effectiveSource !== MODELS_SOURCE) {
    console.warn('[models] S3 unreachable (upload or CORS pending) — falling back to the HF hub');
  }
}

export function currentModelsSource(): ModelsSource {
  return effectiveSource;
}

export const S3_MODELS_HOST = 'https://meu-bucket-c1f5135e.s3.us-east-1.amazonaws.com';

/** Folder names under models/ (also the S3 keys, one folder per model). */
export const MODEL_DIR_IDS = {
  phones: 'phone-recognizer',
  asr: 'word-transcription',
  encoder: 'wav2vec2-960h-encoder',
} as const;

export interface ModelsResolution {
  allowLocalModels: boolean;
  localModelPath: string;
  remoteHost: string;
  remotePathTemplate: string;
  phoneModelId: string;
  asrModelId: string;
}

const HF_HOST = 'https://huggingface.co';

/** How transformers.js should resolve models in the current mode. */
export function resolveModels(): ModelsResolution {
  return resolveFor(effectiveSource);
}

function resolveFor(source: ModelsSource): ModelsResolution {
  switch (source) {
    case 's3':
      return {
        allowLocalModels: false,
        localModelPath: '',
        remoteHost: S3_MODELS_HOST,
        remotePathTemplate: '{model}/',
        phoneModelId: MODEL_DIR_IDS.phones,
        asrModelId: MODEL_DIR_IDS.asr,
      };
    case 'hf':
      return {
        allowLocalModels: false,
        localModelPath: '',
        remoteHost: HF_HOST,
        remotePathTemplate: '{model}/resolve/{revision}/',
        phoneModelId: 'onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX',
        asrModelId: 'onnx-community/wav2vec2-base-960h-ONNX',
      };
    case 'local':
    default:
      return {
        allowLocalModels: true,
        localModelPath: `${import.meta.env.BASE_URL}models/`,
        remoteHost: HF_HOST,
        remotePathTemplate: '{model}/',
        phoneModelId: MODEL_DIR_IDS.phones,
        asrModelId: MODEL_DIR_IDS.asr,
      };
  }
}

/** URL of the phone recognizer vocabulary (fetched manually by session.ts). */
export function phoneVocabUrl(): string {
  const resolution = resolveModels();
  switch (currentModelsSource()) {
    case 'hf':
      return `${HF_HOST}/onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX/resolve/main/vocab.json`;
    case 's3':
      return `${resolution.remoteHost}/${resolution.remotePathTemplate.replace('{model}', MODEL_DIR_IDS.phones)}vocab.json`;
    case 'local':
    default:
      return `${resolution.localModelPath}${MODEL_DIR_IDS.phones}/vocab.json`;
  }
}
