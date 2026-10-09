/**
 * OpenPronounce client-side app.
 *
 * States: models (first download) → practice (record/upload) → loading → result | error.
 * All inference is local: phones + ASR in a worker, G2P + piper + encoder on the main thread.
 */
import '../style.css';
import { AudioRecorder } from './recorder';
import { Pipeline, type AnalysisResult, type InferenceDevice } from './pipeline';
import type { WorkerProgress } from './worker-client';
import { decodeTo16kMono } from '../core/audio';
import { enhanceAudio } from '../core/enhance';
import { synthesizeReference } from '../core/tts';
import { MODELS, storageEstimate, clearStoredModels, storedPiperVoicesList } from './models';
import { Viseme } from './viseme';
import Chart from 'chart.js/auto';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const EXAMPLES = [
  'Hello, how are you?',
  'I would like a cup of coffee.',
  'The weather is lovely today.',
  'She thought three thieves threw the theory out the theatre.',
];

// BCP 47 tags for the browser SpeechSynthesis, keyed by language code
const SPEECH_LANGS: Record<string, string> = {
  en: 'en-US', fr: 'fr-FR', es: 'es-ES', de: 'de-DE', it: 'it-IT', pt: 'pt-BR', nl: 'nl-NL',
};

const MIC_STORAGE_KEY = 'openpronounce.micId';
const AUTO_STOP_STORAGE_KEY = 'openpronounce.autoStop';
const ENHANCE_STORAGE_KEY = 'openpronounce.enhance';
const DEVICE_STORAGE_KEY = 'openpronounce.device';
const SPEAK_SPEED_STORAGE_KEY = 'openpronounce.speakSpeed';
const RING_LENGTH = 2 * Math.PI * 62;
const COLORS = { good: '#10b981', mid: '#f59e0b', bad: '#ef4444' };

const RECORD_ERRORS: Record<string, string> = {
  insecure: 'Recording needs a secure page (https or localhost). You can still upload a file.',
  denied: 'Microphone access was denied. Allow it in your browser, or upload a file.',
  notfound: 'No microphone found. Plug one in, or upload a file.',
  unknown: 'The microphone could not be started. Try again, or upload a file.',
};

const state = {
  micId: localStorage.getItem(MIC_STORAGE_KEY) || null,
  autoStop: localStorage.getItem(AUTO_STOP_STORAGE_KEY) !== '0',
  enhance: localStorage.getItem(ENHANCE_STORAGE_KEY) !== '0',
  device: ((localStorage.getItem(DEVICE_STORAGE_KEY) === 'webgpu' ? 'webgpu' : 'wasm') as InferenceDevice),
  speakSpeed: Number(localStorage.getItem(SPEAK_SPEED_STORAGE_KEY)) || 1,
  blob: null as Blob | null,
  filename: null as string | null,
  needsConfirmation: false,
  result: null as AnalysisResult | null,
  selected: null as number | null,
  busy: false,
  ready: false,
  view: 'idle' as 'idle' | 'loading' | 'result' | 'error',
  charts: {} as Record<string, Chart>,
  chartsDirty: false,
};

const recorder = new AudioRecorder();
let viseme: Viseme | null = null;
let pipelineReady: Promise<Pipeline> | null = null;
let timerInterval: ReturnType<typeof setInterval> | null = null;

document.addEventListener('DOMContentLoaded', init);

function setStatus(text: string): void {
  $('loading-step').textContent = text;
}

function setBusy(on: boolean): void {
  state.busy = on;
  ($('record-btn') as HTMLButtonElement).disabled = !state.ready || on;
  ($('upload-btn') as HTMLButtonElement).disabled = !state.ready || on;
  ($('analyze-btn') as HTMLButtonElement).disabled = on;
}

function setView(name: 'idle' | 'loading' | 'result' | 'error'): void {
  state.view = name;
  $('loading').classList.toggle('hidden', name !== 'loading');
  $('error').classList.toggle('hidden', name !== 'error');
  $('result').classList.toggle('hidden', name !== 'result');
  if (name === 'loading') {
    $('loading').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } else if (name === 'error') {
    $('error').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function init(): void {
  viseme = new Viseme($('viseme-image') as HTMLImageElement);
  renderExamples();
  setupMicSelector();
  setupRecorderEvents();
  setupModelsPanel();

  $('expected-text').addEventListener('input', hideTextHint);
  $('auto-stop-toggle').addEventListener('change', (e) => {
    state.autoStop = (e.target as HTMLInputElement).checked;
    localStorage.setItem(AUTO_STOP_STORAGE_KEY, state.autoStop ? '1' : '0');
    recorder.setSilenceDetection(state.autoStop);
  });
  ($('auto-stop-toggle') as HTMLInputElement).checked = state.autoStop;
  recorder.setSilenceDetection(state.autoStop);
  setupSettingsBar();
  $('details').addEventListener('toggle', () => {
    if (($('details') as HTMLDetailsElement).open && state.chartsDirty) {
      renderCharts();
    }
  });
  ($('enhance-toggle') as HTMLInputElement).checked = state.enhance;
  $('enhance-toggle').addEventListener('change', (e) => {
    state.enhance = (e.target as HTMLInputElement).checked;
    localStorage.setItem(ENHANCE_STORAGE_KEY, state.enhance ? '1' : '0');
  });

  $('record-btn').addEventListener('click', onRecordClick);
  $('listen-btn').addEventListener('click', () => void playReference());
  $('upload-btn').addEventListener('click', () => ($('file-input') as HTMLInputElement).click());
  $('file-input').addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      setAudio(file, file.name, true);
    }
  });
  setupDragAndDrop();

  $('analyze-btn').addEventListener('click', () => void analyze());
  $('detail-listen').addEventListener('click', () => {
    if (state.selected !== null && state.result) {
      speakWord(state.selected, state.result);
    }
  });
  $('retry-btn').addEventListener('click', () => setView('idle'));
  $('again-btn').addEventListener('click', resetForNewTake);
  $('replay-btn').addEventListener('click', () => ($('audio-player') as HTMLAudioElement).play().catch(() => { }));

  void bootstrap();
}

async function bootstrap(): Promise<void> {
  setStatus('loading local AI models (~830 MB on the first visit)...');
  const callbacks = {
    onStatus: (status: string) => {
      if (state.view === 'loading' || !state.ready) {
        setStatus(status);
      }
    },
    onProgress: (event: WorkerProgress) => trackDownload(event),
  };
  try {
    pipelineReady = Pipeline.create(callbacks, { device: state.device });
    await pipelineReady;
    state.ready = true;
    setBusy(false);
    ($('mic-select-btn') as HTMLButtonElement).disabled = false;
    $('record-hint').textContent = 'Tap to record, then say the sentence';
    void refreshStorageInfo();
  } catch (error) {
    if (state.device === 'webgpu') {
      // WebGPU failed (unavailable, out of memory, driver): fall back to wasm once
      console.warn('webgpu load failed, falling back to wasm', error);
      state.device = 'wasm';
      localStorage.setItem(DEVICE_STORAGE_KEY, 'wasm');
      syncSettingsUi();
      setStatus('WebGPU unavailable — falling back to wasm...');
      try {
        pipelineReady = Pipeline.create(callbacks, { device: 'wasm' as const });
        await pipelineReady;
        state.ready = true;
        setBusy(false);
        ($('mic-select-btn') as HTMLButtonElement).disabled = false;
        $('record-hint').textContent = 'Tap to record, then say the sentence';
        ($('device-note') as HTMLElement).textContent = 'WebGPU failed on this device; using wasm.';
        void refreshStorageInfo();
        return;
      } catch (wasmError) {
        error = wasmError;
      }
    }
    $('record-hint').textContent = 'Could not load the local models.';
    setStatus(`model load failed: ${error instanceof Error ? error.message : String(error)}`);
    setView('error');
    ($('error-message') as HTMLParagraphElement).textContent =
      'The local AI models could not be loaded. Check your connection and reload the page.';
  }
}

// ---------------------------------------------------------------- recording

function onRecordClick(): void {
  if (recorder.isRecording()) {
    recorder.stop();
    return;
  }
  if (state.busy || !requireText()) {
    return;
  }
  hideRecordError();
  recorder.setDeviceId(state.micId);
  void recorder.start();
}

function setupRecorderEvents(): void {
  document.addEventListener('record:start', () => {
    const btn = $('record-btn');
    btn.classList.add('is-recording');
    btn.setAttribute('aria-label', 'Stop recording');
    $('mic-icon').classList.add('hidden');
    $('stop-icon').classList.remove('hidden');
    $('record-hint').textContent = state.autoStop
      ? 'Listening. Tap to stop, or pause for two seconds.'
      : 'Listening. Tap to stop when you are done.';
    $('record-timer').classList.remove('hidden');
    $('audio-review').classList.add('hidden');
    ($('mic-select-btn') as HTMLButtonElement).disabled = true;
    setView('idle');

    const startedAt = Date.now();
    timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startedAt) / 1000);
      const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const ss = String(elapsed % 60).padStart(2, '0');
      $('record-timer').textContent = `${mm}:${ss}`;
    }, 200);
  });

  document.addEventListener('record:stop', () => {
    const btn = $('record-btn');
    btn.classList.remove('is-recording');
    btn.setAttribute('aria-label', 'Record');
    $('mic-icon').classList.remove('hidden');
    $('stop-icon').classList.add('hidden');
    $('record-hint').textContent = 'Tap to record, then say the sentence';
    $('record-timer').classList.add('hidden');
    $('record-timer').textContent = '00:00';
    ($('mic-select-btn') as HTMLButtonElement).disabled = false;
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  });

  document.addEventListener('record:silence', () => {
    new Audio('/assets/sounds/notification.mp3').play().catch(() => { });
  });

  document.addEventListener('record:ready', (event) => {
    const detail = (event as CustomEvent).detail as { blob: Blob };
    setAudio(detail.blob, 'recording.webm', false);
    void analyze();
  });

  document.addEventListener('record:error', (event) => {
    const detail = (event as CustomEvent).detail as { reason: string };
    showRecordError(RECORD_ERRORS[detail.reason] ?? RECORD_ERRORS.unknown);
  });
}

function showRecordError(message: string): void {
  const el = $('record-error');
  el.textContent = message;
  el.classList.remove('hidden');
}

function hideRecordError(): void {
  $('record-error').classList.add('hidden');
}

// ---------------------------------------------------------------- microphone selector

function setupMicSelector(): void {
  const btn = $('mic-select-btn');
  if (!navigator.mediaDevices?.enumerateDevices) {
    btn.classList.add('hidden');
    return;
  }

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const menu = $('mic-select-menu');
    if (menu.classList.contains('hidden')) {
      void openMicMenu();
    } else {
      hideMicMenu();
    }
  });
  document.addEventListener('click', (e) => {
    const menu = $('mic-select-menu');
    if (!menu.classList.contains('hidden') && !menu.contains(e.target as Node) && !btn.contains(e.target as Node)) {
      hideMicMenu();
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      hideMicMenu();
    }
  });
  navigator.mediaDevices.addEventListener?.('devicechange', () => {
    if (!$('mic-select-menu').classList.contains('hidden')) {
      void renderMicMenu();
    }
  });
  void updateMicLabel();
}

async function openMicMenu(): Promise<void> {
  $('mic-select-menu').classList.remove('hidden');
  $('mic-select-btn').setAttribute('aria-expanded', 'true');
  await renderMicMenu();
}

function hideMicMenu(): void {
  $('mic-select-menu').classList.add('hidden');
  $('mic-select-btn').setAttribute('aria-expanded', 'false');
}

async function micInputs(): Promise<MediaDeviceInfo[]> {
  let inputs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput');
  if (inputs.length && !inputs.some((d) => d.label) && navigator.mediaDevices.getUserMedia) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
    } catch {
      // Permission denied: keep the anonymous list
    }
    inputs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput');
  }
  return inputs;
}

async function renderMicMenu(): Promise<void> {
  const menu = $('mic-select-menu');
  const list = document.createElement('div');
  list.className = 'flex flex-col gap-0.5';
  menu.innerHTML = '';
  menu.appendChild(list);

  let devices: MediaDeviceInfo[] = [];
  try {
    devices = await micInputs();
  } catch {
    // enumerateDevices unavailable
  }
  if (!devices.length) {
    const empty = document.createElement('p');
    empty.className = 'px-3 py-2 text-sm text-neutral-500';
    empty.textContent = 'No microphone found.';
    list.appendChild(empty);
    return;
  }
  devices.forEach((device, index) => list.appendChild(micMenuItem(device, index)));
  void updateMicLabel();
}

function micMenuItem(device: MediaDeviceInfo, index: number): HTMLButtonElement {
  const item = document.createElement('button');
  item.type = 'button';
  item.setAttribute('role', 'option');
  const selected = device.deviceId === state.micId || (state.micId === null && device.deviceId === 'default');
  item.className = 'w-full flex items-center justify-between gap-3 px-3 py-2 rounded-lg text-sm text-left '
    + (selected ? 'bg-accent-50 text-accent-700 font-medium' : 'text-neutral-700 hover:bg-neutral-50');
  const name = device.label
    || (device.deviceId === 'default' ? 'System default microphone' : `Microphone ${index + 1}`);
  const label = document.createElement('span');
  label.textContent = name;
  label.className = 'truncate';
  item.title = name;
  item.appendChild(label);
  if (selected) {
    const check = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    check.setAttribute('viewBox', '0 0 24 24');
    check.setAttribute('fill', 'none');
    check.setAttribute('stroke', 'currentColor');
    check.setAttribute('stroke-width', '2.5');
    check.setAttribute('class', 'w-4 h-4 shrink-0');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    path.setAttribute('d', 'm4.5 12.75 6 6 9-13.5');
    check.appendChild(path);
    item.appendChild(check);
  }
  item.addEventListener('click', (e) => {
    e.stopPropagation();
    state.micId = device.deviceId === 'default' ? null : device.deviceId;
    localStorage.setItem(MIC_STORAGE_KEY, state.micId || '');
    hideMicMenu();
  });
  return item;
}

function updateMicLabel(): void {
  const el = $('mic-name');
  if (!el || !navigator.mediaDevices?.enumerateDevices) {
    return;
  }
  navigator.mediaDevices.enumerateDevices().then((devices) => {
    const inputs = devices.filter((d) => d.kind === 'audioinput');
    const current = state.micId
      ? inputs.find((d) => d.deviceId === state.micId)
      : inputs.find((d) => d.deviceId === 'default');
    el.textContent = current?.label ? `Microphone: ${current.label}` : '';
  }).catch(() => { });
}

// ---------------------------------------------------------------- audio input

function setAudio(blob: Blob, filename: string, needsConfirmation: boolean): void {
  state.blob = blob;
  state.filename = filename;
  state.needsConfirmation = needsConfirmation;
  const player = $('audio-player') as HTMLAudioElement;
  if (player.dataset.url) {
    URL.revokeObjectURL(player.dataset.url);
  }
  const url = URL.createObjectURL(blob);
  player.src = url;
  player.dataset.url = url;
  $('file-name').textContent = needsConfirmation ? filename : '';
  $('analyze-btn').classList.toggle('hidden', !needsConfirmation);
  $('audio-review').classList.remove('hidden');
  hideRecordError();
}

function setupDragAndDrop(): void {
  const zone = $('practice');
  const overlay = $('drop-overlay');
  let depth = 0;
  const show = (on: boolean) => {
    overlay.classList.toggle('hidden', !on);
    overlay.classList.toggle('flex', on);
  };
  zone.addEventListener('dragenter', (e) => { e.preventDefault(); depth++; show(true); });
  zone.addEventListener('dragover', (e) => e.preventDefault());
  zone.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) show(false); });
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    show(false);
    const file = e.dataTransfer?.files?.[0];
    if (file && state.ready) {
      setAudio(file, file.name, true);
    }
  });
}

// ---------------------------------------------------------------- reference playback

let referenceAudio: HTMLAudioElement | null = null;

/**
 * Synthesize the expected sentence with piper (in-browser, voice cached in
 * OPFS after the first download) and play it.
 */
async function playReference(): Promise<void> {
  if (recorder.isRecording()) {
    return;
  }
  const expectedText = ($('expected-text') as HTMLInputElement).value.trim();
  if (!tokenizeWords(expectedText).length) {
    showTextHint('Type the sentence first, or pick one of the examples.');
    $('expected-text').focus();
    return;
  }

  const btn = $('listen-btn') as HTMLButtonElement;
  const label = btn.querySelector('[data-role="listen.label"]') as HTMLElement;
  const original = label.textContent;
  btn.disabled = true;

  const restore = () => {
    btn.disabled = false;
    label.textContent = original;
  };

  try {
    label.textContent = 'Synthesizing…';
    const reference = await synthesizeReference(expectedText, (p) => {
      label.textContent = `Downloading voice ${(p.loaded / 1e6).toFixed(0)} / ${(p.total / 1e6).toFixed(0)} MB`;
      trackDownload({
        model: 'piper',
        status: 'progress',
        file: 'piper-voice.onnx',
        loaded: p.loaded,
        total: p.total,
      });
    });
    trackDownload({ model: 'piper', status: 'done', file: 'piper-voice.onnx' });

    if (referenceAudio) {
      referenceAudio.pause();
      if (referenceAudio.dataset.url) {
        URL.revokeObjectURL(referenceAudio.dataset.url);
      }
    }
    const url = URL.createObjectURL(reference.blob);
    referenceAudio = new Audio(url);
    referenceAudio.dataset.url = url;
    // speak speed from the settings bar; preservesPitch keeps the voice natural
    referenceAudio.playbackRate = state.speakSpeed;
    referenceAudio.preservesPitch = true;
    (referenceAudio as HTMLAudioElement & { webkitPreservesPitch?: boolean }).webkitPreservesPitch = true;
    referenceAudio.onended = restore;
    referenceAudio.onerror = restore;

    label.textContent = 'Playing…';
    await referenceAudio.play();
  } catch (error) {
    label.textContent = 'Reference unavailable right now';
    setTimeout(restore, 2500);
  }
}

// ---------------------------------------------------------------- analysis

function requireText(): boolean {
  const textarea = $('expected-text') as HTMLInputElement;
  if (!tokenizeWords(textarea.value.trim()).length) {
    showTextHint('Type the sentence first, or pick one of the examples.');
    textarea.focus();
    return false;
  }
  return true;
}

function showTextHint(message: string): void {
  const hint = $('text-hint');
  hint.textContent = message;
  hint.classList.remove('hidden');
}

function hideTextHint(): void {
  $('text-hint').classList.add('hidden');
}

async function analyze(): Promise<void> {
  if (state.busy || !state.blob) {
    return;
  }
  const expectedText = ($('expected-text') as HTMLInputElement).value.trim();
  if (!tokenizeWords(expectedText).length) {
    showTextHint('Type the sentence first, or pick one of the examples.');
    return;
  }
  setBusy(true);
  setView('loading');
  setStatus('decoding audio...');

  try {
    let audio = await decodeTo16kMono(state.blob);
    if (state.enhance) {
      setStatus('reducing noise & enhancing voice...');
      audio = await enhanceAudio(audio);
    }
    const currentPipeline = await pipelineReady!;
    const result = await currentPipeline.analyze(audio, expectedText);
    state.result = result;
    renderResult(result);
  } catch (error) {
    ($('error-message') as HTMLParagraphElement).textContent =
      error instanceof Error ? error.message : 'The analysis failed. Try again in a moment.';
    setView('error');
  } finally {
    setBusy(false);
  }
}

function resetForNewTake(): void {
  viseme?.rest();
  setView('idle');
  $('audio-review').classList.add('hidden');
  state.blob = null;
  state.filename = null;
  state.result = null;
  $('practice').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------------------------------------------------------------- details charts

const CHART_COLORS = { you: '#ea580c', reference: '#2f6fb3' };

function renderCharts(): void {
  const result = state.result;
  if (!result) {
    return;
  }
  state.chartsDirty = false;

  const steps = result.trace.expected.length;
  lineChart('phoneme-chart', Array.from({ length: steps }, (_, i) => i + 1), [
    { label: 'Reference', data: result.trace.expected, borderColor: CHART_COLORS.reference },
    { label: 'You', data: result.trace.heard, borderColor: CHART_COLORS.you },
  ], { legend: true, yTicks: false });

  const f0Steps = result.prosody.f0.length;
  lineChart('f0-chart', Array.from({ length: f0Steps }, (_, i) => i), [
    { label: 'Pitch (Hz)', data: result.prosody.f0, borderColor: CHART_COLORS.you },
  ], { legend: false, yTicks: true });

  const energySteps = result.prosody.energy.length;
  lineChart('energy-chart', Array.from({ length: energySteps }, (_, i) => i), [
    { label: 'Energy', data: result.prosody.energy, borderColor: CHART_COLORS.you },
  ], { legend: false, yTicks: true });
}

function lineChart(
  canvasId: string,
  labels: number[],
  datasets: Array<{ label: string; data: number[]; borderColor: string }>,
  { legend, yTicks }: { legend: boolean; yTicks: boolean },
): void {
  state.charts[canvasId]?.destroy();
  const canvas = $(canvasId) as HTMLCanvasElement;
  state.charts[canvasId] = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: datasets.map((d) => ({
        ...d,
        borderWidth: 2,
        pointRadius: 0,
        pointHitRadius: 8,
        tension: 0.3,
        fill: false,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { display: false },
        y: {
          display: true,
          grid: { color: '#f0efed' },
          border: { display: false },
          ticks: { display: yTicks, color: '#8a8782', font: { size: 11 }, maxTicksLimit: 4 },
        },
      },
      plugins: {
        legend: {
          display: legend,
          position: 'bottom',
          labels: { boxWidth: 10, boxHeight: 10, usePointStyle: true, color: '#57534e', font: { size: 11 } },
        },
      },
    },
  });
}

// ---------------------------------------------------------------- result

function tokenizeWords(text: string): string[] {
  return text.match(/[\p{L}\p{N}_](?:[\p{L}\p{N}_']*[\p{L}\p{N}_])?/gu) ?? [];
}

function renderResult(result: AnalysisResult): void {
  state.selected = null;
  renderScore(result);
  renderChips(result);
  renderHeard(result);
  $('word-detail').classList.add('hidden');
  state.chartsDirty = true;
  if (($('details') as HTMLDetailsElement).open) {
    renderCharts();
  }
  setView('result');
  const section = $('result');
  section.classList.remove('reveal');
  void section.offsetWidth;
  section.classList.add('reveal');
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const flagged = result.phones.errors.map((e) => e.position);
  const firstFlagged = flagged.length ? flagged[0] : -1;
  if (firstFlagged >= 0) {
    selectWord(firstFlagged, result, false);
  } else {
    viseme?.rest();
  }
}

function renderScore(result: AnalysisResult): void {
  const score = Math.round(result.score);
  const ring = $('score-ring');
  delete ($('score-value') as HTMLElement).dataset.final;
  ring.style.stroke = result.score >= 80 ? COLORS.good : result.score >= 50 ? COLORS.mid : COLORS.bad;
  ring.style.strokeDashoffset = String(RING_LENGTH);
  requestAnimationFrame(() => {
    ring.style.strokeDashoffset = String(RING_LENGTH * (1 - Math.max(0, Math.min(100, result.score)) / 100));
  });
  countUp($('score-value'), score, 900);

  const flagged = result.phones.errors.length;
  const total = result.expectedWords.length;
  let verdict: string;
  let sub: string;
  if (total === 0) {
    verdict = 'Nothing to compare';
    sub = 'The sentence is empty.';
  } else if (flagged === 0 && result.score < 70) {
    verdict = 'No single word stands out';
    sub = `But the sentence as a whole was hard to follow: we heard "${result.transcript || 'nothing'}". Check that the recording started before you spoke.`;
  } else if (flagged === 0) {
    verdict = 'Every word came through clearly';
    sub = 'Tap a word to see its sounds anyway.';
  } else if (flagged === total) {
    verdict = 'None of the words came through';
    sub = 'Tap a word to see which sounds went wrong. Was it the right sentence?';
  } else {
    verdict = `${flagged} of ${total} word${total > 1 ? 's' : ''} need${flagged === 1 ? 's' : ''} work`;
    sub = 'Tap a red word to see which sounds went wrong.';
  }
  $('verdict').textContent = verdict;
  $('verdict-sub').textContent = sub;
}

function countUp(el: HTMLElement, target: number, duration: number): void {
  const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  if (reduced || duration <= 0) {
    el.textContent = String(target);
    el.dataset.final = '1';
    return;
  }
  const start = performance.now();
  const step = (now: number) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = String(Math.round(target * eased));
    if (t < 1) {
      requestAnimationFrame(step);
    } else {
      el.dataset.final = '1';
    }
  };
  requestAnimationFrame(step);
}

function renderChips(result: AnalysisResult): void {
  const container = $('word-chips');
  container.innerHTML = '';
  const errorsByPosition = new Map(result.phones.errors.map((e) => [e.position, e]));
  const expectedByPosition = new Map(result.phones.expected_phones.map((phones, i) => [i, phones]));

  result.expectedWords.forEach((word, index) => {
    const error = errorsByPosition.get(index);
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.dataset.index = String(index);
    chip.textContent = word;
    chip.setAttribute('aria-pressed', 'false');
    chip.className = 'chip px-3 py-1.5 rounded-lg border text-base font-medium ' + (error
      ? 'bg-red-50 border-red-200 text-red-700 hover:bg-red-100'
      : 'bg-emerald-50 border-emerald-200 text-emerald-800 hover:bg-emerald-100');
    const phones = expectedByPosition.get(index) ?? [];
    if (error) {
      const conf = `${Math.round(error.confidence * 100)} % sure`;
      chip.title = error.actual === ''
        ? `Not heard (${conf}). Expected /${error.expected}/`
        : `Flagged, ${conf}. Expected /${error.expected}/, heard /${error.actual}/`;
    } else {
      chip.title = phones.length ? `Sounds fine: /${phones.join(' ')}/` : 'Sounds fine';
    }
    chip.addEventListener('click', () => selectWord(index, result, true));
    container.appendChild(chip);
  });
}

function selectWord(index: number, result: AnalysisResult, speak: boolean): void {
  state.selected = index;
  $('word-chips').querySelectorAll('.chip').forEach((chip) => {
    chip.setAttribute('aria-pressed', String(Number((chip as HTMLElement).dataset.index) === index));
  });

  const error = result.phones.errors.find((e) => e.position === index);
  const word = result.expectedWords[index];
  const detail = $('word-detail');
  const status = $('detail-status');
  const expectedEl = $('detail-expected');
  const heardEl = $('detail-heard');

  $('detail-word').textContent = word;
  expectedEl.innerHTML = '';
  heardEl.innerHTML = '';
  $('detail-note').textContent = '';

  if (!error) {
    status.textContent = 'Sounds fine';
    status.className = 'text-sm font-medium text-emerald-700';
    const expectedPhones = result.phones.expected_phones[index] ?? [];
    expectedEl.textContent = expectedPhones.length ? `/${expectedPhones.join(' ')}/` : '/?/';
    heardEl.textContent = 'the same';
    heardEl.className = 'text-base text-neutral-500';
  } else {
    status.textContent = `Flagged, ${Math.round(error.confidence * 100)} % sure`;
    status.className = 'text-sm font-medium text-red-600';

    expectedEl.appendChild(document.createTextNode('/'));
    error.phones.forEach((phone, i) => {
      const span = document.createElement('span');
      span.textContent = phone.expected;
      if (phone.confidence >= 0.5) {
        span.className = 'text-red-600 font-semibold underline decoration-2 decoration-red-300 underline-offset-4';
        span.title = `heard /${phone.heard || '∅'}/ (${Math.round(phone.confidence * 100)} %)`;
      }
      expectedEl.appendChild(span);
      if (i < error.phones.length - 1) {
        expectedEl.appendChild(document.createTextNode(' '));
      }
    });
    expectedEl.appendChild(document.createTextNode('/'));
    expectedEl.className = 'font-mono text-lg tracking-wide text-neutral-800';

    if (error.actual === '') {
      heardEl.textContent = 'nothing';
      heardEl.className = 'text-base text-neutral-500';
    } else {
      heardEl.textContent = `/${error.actual}/`;
      heardEl.className = 'font-mono text-lg tracking-wide text-neutral-800';
    }
  }

  detail.classList.remove('hidden');
  if (speak) {
    speakWord(index, result);
  } else {
    viseme?.rest();
  }
}

/** Speak the word with the browser TTS while the mouth animates its phones. */
function speakWord(index: number, result: AnalysisResult): void {
  const word = result.expectedWords[index];
  const error = result.phones.errors.find((e) => e.position === index);
  const expectedPhones = result.phones.expected_phones[index] ?? [];
  const phones = expectedPhones.length ? expectedPhones : (error && error.expected ? [error.expected] : []);

  if (phones.length && viseme) {
    const duration = viseme.estimateWordDuration(phones) * 1.5;
    void viseme.play(phones, duration);
  }
  if ('speechSynthesis' in window) {
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(word);
    utterance.lang = SPEECH_LANGS.en;
    utterance.rate = 0.8;
    speechSynthesis.speak(utterance);
  }
}

function renderHeard(result: AnalysisResult): void {
  const transcriptEl = $('transcript');
  transcriptEl.textContent = result.transcript ? result.transcript.toLowerCase() : 'nothing we could recognize';

  const container = $('heard-phones');
  container.innerHTML = '';
  if (!result.heardPhones.length) {
    container.textContent = 'no phones recognized';
    container.className = 'text-sm text-neutral-500';
    return;
  }
  container.className = 'font-mono text-base leading-relaxed';
  result.heardPhones.forEach((phone, i) => {
    const span = document.createElement('span');
    span.textContent = phone;
    const conf = result.phoneConfidences[i];
    const sure = typeof conf !== 'number' || conf >= 0.5;
    span.className = 'inline-block mr-1.5 ' + (sure ? 'text-neutral-800' : 'text-neutral-400');
    if (typeof conf === 'number') {
      span.title = `${Math.round(conf * 100)} % confidence`;
    }
    container.appendChild(span);
  });

  const t = result.timings;
  $('breakdown').textContent =
    `acoustic ${result.acoustic.meanPerStep.toFixed(2)} · phones ${Math.round((1 - result.phones.phone_error_rate) * 100)} % · words ${Math.round((1 - result.wordErrorRate) * 100)} %`
    + ` — ${('total ' + (t.totalMs / 1000).toFixed(1) + 's')}`;
}

// ---------------------------------------------------------------- download popup

const DOWNLOAD_LABELS: Record<string, string> = {
  phones: 'Phone recognizer (wav2vec2 espeak)',
  asr: 'Word transcription (wav2vec2 960h base)',
  encoder: 'Acoustic encoder (wav2vec2 960h)',
  piper: 'Reference voice (piper lessac-medium)',
};

interface ActiveDownload {
  label: string;
  file: string;
  loaded: number;
  total: number;
}

const activeDownloads = new Map<string, ActiveDownload>();

function trackDownload(event: WorkerProgress): void {
  // worker events nest the transformers.js progress inside `payload`;
  // pipeline-generated events (piper) are flat
  const raw = event as WorkerProgress & { payload?: WorkerProgress };
  const progress = raw.payload ?? raw;
  const model = raw.model ?? progress.model ?? 'model';

  const key = `${model}:${progress.file ?? ''}`;
  const total = progress.total ?? 0;
  const loaded = progress.loaded ?? 0;
  if (progress.status === 'done' || (total > 0 && loaded >= total)) {
    activeDownloads.delete(key);
  } else if (total > 0) {
    activeDownloads.set(key, {
      label: DOWNLOAD_LABELS[model] ?? 'Model',
      file: (progress.file ?? '').split('/').pop() || '',
      loaded,
      total,
    });
  } else {
    return; // initiate/download events without sizes carry no useful info
  }
  renderDownloadPopup();
}

function renderDownloadPopup(): void {
  const popup = $('download-popup');
  // ignore tiny files (configs, tokenizers): only meaningful downloads count
  const active = [...activeDownloads.values()].filter((d) => d.total > 1024 * 1024);
  if (!active.length) {
    popup.classList.add('hidden');
    return;
  }

  // show the file with the most bytes left to fetch
  const current = active.reduce((a, b) => (b.total - b.loaded > a.total - a.loaded ? b : a));
  const pct = Math.min(100, (current.loaded / current.total) * 100);
  ($('download-model') as HTMLElement).textContent = current.label;
  ($('download-file') as HTMLElement).textContent = `${current.file} · ${active.length} file${active.length > 1 ? 's' : ''} left`;
  ($('download-bar') as HTMLElement).style.width = `${pct.toFixed(1)}%`;
  ($('download-pct') as HTMLElement).textContent =
    `${pct.toFixed(0)} % · ${(current.loaded / 1e6).toFixed(0)} / ${(current.total / 1e6).toFixed(0)} MB`;
  popup.classList.remove('hidden');
}

// ---------------------------------------------------------------- settings bar

function setupSettingsBar(): void {
  const toggle = $('settings-toggle') as HTMLButtonElement;
  const body = $('settings-body');

  const applyCollapsed = (collapsed: boolean) => {
    body.classList.toggle('hidden', collapsed);
    toggle.setAttribute('aria-expanded', String(!collapsed));
  };
  // collapsed by default on narrow screens
  applyCollapsed(window.innerWidth < 1280);
  toggle.addEventListener('click', () => {
    applyCollapsed(body.classList.contains('hidden'));
  });

  const deviceSelect = $('device-setting') as HTMLSelectElement;
  deviceSelect.value = state.device;
  deviceSelect.addEventListener('change', () => {
    state.device = deviceSelect.value as InferenceDevice;
    localStorage.setItem(DEVICE_STORAGE_KEY, state.device);
    ($('device-note') as HTMLElement).textContent =
      state.device === 'webgpu'
        ? 'webgpu loads fp16 weights (~1.3 GB extra).'
        : 'webgpu loads fp16 weights (~1.3 GB extra).';
    // the worker models must be recreated; a reload applies the new device
    window.location.reload();
  });

  const speedSelect = $('speak-speed-setting') as HTMLSelectElement;
  speedSelect.value = String(state.speakSpeed);
  speedSelect.addEventListener('change', () => {
    state.speakSpeed = Number(speedSelect.value) || 1;
    localStorage.setItem(SPEAK_SPEED_STORAGE_KEY, String(state.speakSpeed));
  });
}

function syncSettingsUi(): void {
  ($('device-setting') as HTMLSelectElement).value = state.device;
}

// ---------------------------------------------------------------- examples & models panel

function renderExamples(): void {
  const container = $('examples');
  container.querySelectorAll('button').forEach((b) => b.remove());
  EXAMPLES.forEach((text) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.textContent = text;
    chip.className = 'text-xs px-2.5 py-1 rounded-full border border-neutral-200 bg-white text-neutral-600 '
      + 'hover:border-accent-500 hover:text-accent-700 transition-colors';
    chip.addEventListener('click', () => {
      ($('expected-text') as HTMLInputElement).value = text;
      hideTextHint();
    });
    container.appendChild(chip);
  });
}

function setupModelsPanel(): void {
  const list = $('models-list');
  list.innerHTML = '';
  for (const model of MODELS) {
    const item = document.createElement('li');
    item.className = 'flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-neutral-50';
    const name = document.createElement('span');
    name.textContent = model.label;
    name.className = 'text-neutral-700 truncate';
    const size = document.createElement('span');
    size.textContent = `~${model.sizeMb} MB`;
    size.className = 'text-xs text-neutral-400 shrink-0 tabular-nums';
    item.append(name, size);
    list.appendChild(item);
  }

  $('models-clear').addEventListener('click', async () => {
    const button = $('models-clear') as HTMLButtonElement;
    button.disabled = true;
    button.textContent = 'Deleting…';
    try {
      await clearStoredModels();
      button.textContent = 'Deleted. Reload to re-download.';
    } catch {
      button.textContent = 'Could not delete — try the browser settings.';
    }
    void refreshStorageInfo();
  });

  void refreshStorageInfo();
}

async function refreshStorageInfo(): Promise<void> {
  const estimate = await storageEstimate();
  const voices = await storedPiperVoicesList().catch(() => [] as string[]);
  const info = $('storage-info');
  if (estimate) {
    info.textContent = `using ${estimate.usageMb.toFixed(0)} MB of ${estimate.quotaMb.toFixed(0)} MB quota`
      + (voices.length ? ` · piper voices stored: ${voices.length}` : '');
  } else {
    info.textContent = voices.length ? `piper voices stored: ${voices.length}` : '';
  }
  const first = !localStorage.getItem('openpronounce.seenModels');
  if (first && state.ready) {
    localStorage.setItem('openpronounce.seenModels', '1');
    ($('models-details') as HTMLDetailsElement).open = true;
  }
}
