/**
 * Spike UI: load the phone model in a worker, run a recording through it and
 * show phones, confidences and timings. Goal: measure feasibility (latency,
 * download, memory) and eyeball parity with the Python pipeline.
 */
import './style.css';
import { decodeTo16kMono } from './core/audio';
import { getExpectedPhones } from './core/g2p';

const MODEL_ID = 'onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX';

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

function setStatus(text: string): void {
  $('status').textContent = text;
}

function setBusy(on: boolean): void {
  busy = on;
  ($('run-btn') as HTMLButtonElement).disabled = !ready || on;
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
    setStatus(`model ready (${data.payload.device}/${data.payload.dtype})${heap}`);
  } else if (data.type === 'result') {
    renderResult(data.payload as ResultPayload);
    setBusy(false);
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
}

async function runAudio(blob: Blob): Promise<void> {
  if (!ready || busy) {
    return;
  }
  setBusy(true);
  setStatus('decoding audio to 16 kHz mono...');
  try {
    const audio = await decodeTo16kMono(blob);
    ($('audio-preview') as HTMLAudioElement).src = URL.createObjectURL(blob);
    $('audio-preview').classList.remove('hidden');
    setStatus('running inference...');
    requestId += 1;
    worker.postMessage({ type: 'infer', audio, requestId }, [audio.buffer]);
  } catch (error) {
    setStatus(`audio decode failed: ${error instanceof Error ? error.message : String(error)}`);
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
