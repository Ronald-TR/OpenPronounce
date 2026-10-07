/**
 * Spike UI: full client-side pronunciation assessment.
 *
 * 1. Phone recognizer (worker) → heard phones + posteriors
 * 2. Word transcription (worker, wav2vec2-base-960h)
 * 3. G2P of the expected sentence (espeak-ng WASM, main thread)
 * 4. Reference TTS (piper WASM) + acoustic encoder → DTW distance
 * 5. Score = 0.3 acoustic + 0.4 phones + 0.3 words
 */
import './style.css';
import { decodeTo16kMono } from './core/audio';
import { getExpectedPhones } from './core/g2p';
import { env } from '@huggingface/transformers';
import { createEmbedder, acousticDistance, type Embedder } from './core/embeddings';
import { synthesizeReference } from './core/tts';
import { cleanTranscription, computePronunciationScore, wordErrorRate, words as tokenize } from './core/score';
import { comparePhones } from './core/compare';
import type { ComparePhonesResult } from './core/compare';
import { loadRecognizerVocab } from './core/session';

const MODEL_ID = 'onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX';
const ENCODER_PATH = 'models/wav2vec2-960h-encoder';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

interface ProgressEventDetail {
  status: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

interface ResultPayload {
  requestId: number;
  phones: string[];
  confidences: number[];
  frames: number;
  audioSeconds: number;
  spans: Array<[number, number]>;
  vocabSize: number;
  logPosteriors: Float32Array;
  timings: {
    preprocessMs: number;
    inferenceMs: number;
    logSoftmaxMs: number;
    decodeMs: number;
    totalMs: number;
  };
}

let ready = false;
let busy = false;
let requestId = 0;

/** State kept for the score step after an analysis. */
let lastLearner: Float32Array | null = null;
let lastRecognition: {
  phones: string[];
  confidences: number[];
  spans: Array<[number, number]>;
  frames: number;
  vocabSize: number;
  logPosteriors: Float32Array;
} | null = null;
let lastTranscript: string | null = null;
let embedder: Embedder | null = null;
let asrReadyResolve: (() => void) | null = null;

function setStatus(text: string): void {
  $('status').textContent = text;
}

function setBusy(on: boolean): void {
  busy = on;
  ($('run-btn') as HTMLButtonElement).disabled = !ready || on;
  ($('score-btn') as HTMLButtonElement).disabled = !ready || on || !lastRecognition;
}

worker.onmessage = (event: MessageEvent) => {
  const data = event.data;
  if (data.type === 'progress') {
    const p = data.payload as ProgressEventDetail;
    if (p.status === 'progress' && p.file && p.total) {
      const mb = (p.loaded ?? 0) / 1e6;
      const total = p.total / 1e6;
      setStatus(`downloading ${p.file}: ${mb.toFixed(0)} / ${total.toFixed(0)} MB (${Math.round(p.progress ?? 0)}%)`);
    } else if (p.status === 'done' && p.file) {
      setStatus(`downloaded ${p.file}`);
    }
  } else if (data.type === 'ready') {
    ready = true;
    setBusy(false);
    const mem = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
    const heap = mem ? ` · JS heap ${(mem.usedJSHeapSize / 1e6).toFixed(0)} MB` : '';
    setStatus(`phone model ready (${data.payload.device}/${data.payload.dtype})${heap}`);
  } else if (data.type === 'result') {
    renderResult(data as unknown as ResultPayload);
    setBusy(false);
  } else if (data.type === 'asr-ready') {
    asrReadyResolve?.();
  } else if (data.type === 'transcript') {
    lastTranscript = String((data as { transcript?: string }).transcript ?? '');
  } else if (data.type === 'error') {
    setStatus(`error: ${data.message}`);
    setBusy(false);
  }
};

function renderResult(result: ResultPayload): void {
  $('timings').classList.remove('hidden');
  $('t-total').textContent = `${(result.timings.totalMs / 1000).toFixed(2)} s`;
  $('t-inference').textContent = `${(result.timings.inferenceMs / 1000).toFixed(2)} s`;
  $('t-preprocess').textContent = `${result.timings.preprocessMs.toFixed(0)} ms`;
  $('t-decode').textContent = `${result.timings.decodeMs.toFixed(1)} ms`;
  $('t-frames').textContent = `${result.frames} (${result.audioSeconds.toFixed(1)} s audio)`;

  const phonesEl = $('phones');
  phonesEl.innerHTML = '';
  result.phones.forEach((phone, i) => {
    const span = document.createElement('span');
    span.textContent = phone;
    const conf = result.confidences[i];
    span.className = 'phone' + (typeof conf === 'number' && conf < 0.5 ? ' phone-low' : '');
    if (typeof conf === 'number') {
      span.title = `${Math.round(conf * 100)} % confidence`;
    }
    phonesEl.appendChild(span);
  });

  lastRecognition = {
    phones: result.phones,
    confidences: result.confidences,
    spans: result.spans,
    frames: result.frames,
    vocabSize: result.vocabSize,
    logPosteriors: result.logPosteriors,
  };
  setBusy(busy);
}

async function runAudio(blob: Blob): Promise<void> {
  if (!ready || busy) {
    return;
  }
  setBusy(true);
  setStatus('decoding audio to 16 kHz mono...');
  try {
    const audio = await decodeTo16kMono(blob);
    // one private copy for scoring, one transferred to the worker per request
    lastLearner = audio.slice();
    ($('audio-preview') as HTMLAudioElement).src = URL.createObjectURL(blob);
    $('audio-preview').classList.remove('hidden');
    setStatus('running phone recognition...');
    requestId += 1;
    worker.postMessage({ type: 'infer', audio, requestId }, [audio.buffer]);

    // word transcription for the word term of the score
    await ensureAsr();
    const transcriptAudio = lastLearner.slice();
    requestId += 1;
    worker.postMessage({ type: 'transcribe', audio: transcriptAudio, requestId }, [transcriptAudio.buffer]);
  } catch (error) {
    setStatus(`audio decode failed: ${error instanceof Error ? error.message : String(error)}`);
    setBusy(false);
  }
}

async function ensureAsr(): Promise<void> {
  if (!asrReadyResolve) {
    asrReadyResolve = () => { };
  }
  if (!$('asr-loaded').dataset.done) {
    setStatus('loading word transcription model (~95 MB)...');
    const promise = new Promise<void>((resolve) => {
      asrReadyResolve = () => {
        $('asr-loaded').dataset.done = '1';
        resolve();
      };
    });
    worker.postMessage({ type: 'load-asr' });
    await promise;
  }
}

async function ensureEmbedder(): Promise<Embedder> {
  if (!embedder) {
    env.allowLocalModels = true;
    env.localModelPath = '/';
    embedder = await createEmbedder(ENCODER_PATH, 'q8');
  }
  return embedder;
}

async function scoreAgainstReference(): Promise<void> {
  const expectedText = ($('expected-text') as HTMLInputElement).value.trim();
  if (!expectedText || !lastLearner || !lastRecognition) {
    setStatus('analyze some audio first');
    return;
  }
  setBusy(true);
  try {
    // 1. reference TTS (piper, cached in OPFS after the first download)
    setStatus('synthesizing reference with piper (~63 MB first time)...');
    const reference = await synthesizeReference(expectedText, (p) => {
      setStatus(`downloading piper voice: ${(p.loaded / 1e6).toFixed(0)} / ${(p.total / 1e6).toFixed(0)} MB`);
    });
    if (!lastLearner.length || !reference.audio.length) {
      throw new Error(`empty audio (learner ${lastLearner.length} samples, reference ${reference.audio.length})`);
    }
    ($('reference-preview') as HTMLAudioElement).src = URL.createObjectURL(reference.blob);
    $('reference-preview').classList.remove('hidden');

    // 2. acoustic term: encoder embeddings + DTW
    setStatus('computing acoustic distance (encoder + DTW)...');
    const embeddingModel = await ensureEmbedder();
    const learnerEmbedding = await embeddingModel.embed(lastLearner);
    const referenceEmbedding = await embeddingModel.embed(reference.audio);
    const acoustic = acousticDistance(learnerEmbedding, referenceEmbedding);

    // 3. phone term: comparePhones in the browser
    setStatus('comparing phones...');
    const { vocab } = await loadRecognizerVocab();
    const phoneResult: ComparePhonesResult = await comparePhones(
      { ...lastRecognition, frameCount: lastRecognition.frames, vocab },
      expectedText,
      'en',
    );

    // 4. word term
    const transcript = cleanTranscription(lastTranscript ?? '');
    const wer = wordErrorRate(tokenize(expectedText.toLowerCase()), tokenize(transcript));

    // 5. score
    const score = computePronunciationScore(acoustic.meanPerStep, phoneResult.phone_error_rate, wer, 'en');

    $('score-value').textContent = String(score);
    $('score-acoustic').textContent = `${acoustic.meanPerStep.toFixed(2)} (${(acoustic.ms / 1000).toFixed(2)} s DTW)`;
    $('score-phones').textContent =
      `${phoneResult.phone_error_rate} · ${phoneResult.errors.length} flagged word(s): `
      + (phoneResult.errors.map((e) => e.word).join(', ') || 'none');
    $('score-words').textContent = `${wer} ("${transcript || 'nothing recognized'}")`;
    $('score-box').classList.remove('hidden');
    setStatus('done');
  } catch (error) {
    setStatus(`scoring failed: ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    setBusy(false);
  }
}

async function renderExpectedPhones(): Promise<void> {
  const text = ($('expected-text') as HTMLInputElement).value.trim();
  const el = $('expected-phones');
  if (!text) {
    el.textContent = '';
    return;
  }
  try {
    const { groups } = await getExpectedPhones(text, 'en');
    el.innerHTML = '';
    groups.forEach((phones) => {
      if (!phones.length) {
        return;
      }
      const word = document.createElement('span');
      word.textContent = `/${phones.join(' ')}/`;
      word.className = 'inline-block mr-3';
      el.appendChild(word);
    });
  } catch (error) {
    el.textContent = `g2p failed: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function init(): void {
  $('env-info').textContent = `crossOriginIsolated: ${String(self.crossOriginIsolated)} · hardwareConcurrency: ${navigator.hardwareConcurrency} · webgpu: ${String('gpu' in navigator)}`;

  ($('device-select') as HTMLSelectElement).addEventListener('change', () => {
    if (ready) {
      setStatus('reload the page to switch backend (spike limitation)');
    }
  });

  $('load-btn').addEventListener('click', () => {
    if (busy) {
      return;
    }
    setBusy(true);
    const device = ($('device-select') as HTMLSelectElement).value as 'wasm' | 'webgpu';
    const dtype = ($('dtype-select') as HTMLSelectElement).value as 'q8' | 'q4' | 'q4f16' | 'fp16' | 'fp32';
    setStatus(`loading ${MODEL_ID} (${device}/${dtype})...`);
    worker.postMessage({ type: 'load', device, dtype });
  });

  $('file-input').addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      void runAudio(file);
    }
  });

  $('score-btn').addEventListener('click', () => {
    void scoreAgainstReference();
  });

  const zone = document.body;
  zone.addEventListener('dragover', (e) => e.preventDefault());
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      void runAudio(file);
    }
  });

  setBusy(true);
  // Expected phones via espeak-ng WASM on the main thread (fast, no model needed)
  const expectedInput = $('expected-text') as HTMLInputElement;
  expectedInput.addEventListener('input', () => void renderExpectedPhones());
  void renderExpectedPhones();

  // Auto-load with the selected backend; the Load button stays for manual reloads
  const device = ($('device-select') as HTMLSelectElement).value as 'wasm' | 'webgpu';
  const dtype = ($('dtype-select') as HTMLSelectElement).value as 'q8' | 'q4' | 'q4f16' | 'fp16' | 'fp32';
  setStatus(`loading ${MODEL_ID} (${device}/${dtype})...`);
  worker.postMessage({ type: 'load', device, dtype });
}

init();
