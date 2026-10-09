/**
 * Full client-side analysis pipeline: phones (worker) + word transcription
 * (worker) + G2P + piper reference + acoustic embeddings + score.
 */
import { WorkerClient, type WorkerProgress } from './worker-client';
import { comparePhones, type ComparePhonesResult } from '../core/compare';
import { getExpectedPhones } from '../core/g2p';
import { loadRecognizerVocab } from '../core/session';
import { synthesizeReference } from '../core/tts';
import { cleanTranscription, computePronunciationScore, wordErrorRate, words as tokenize } from '../core/score';
import { extractF0, extractEnergy, interpolateF0 } from '../core/prosody';
import { alignPhonemeTrace } from '../core/trace';


export interface AnalysisResult {
  score: number;
  acoustic: { meanPerStep: number; ms: number };
  phones: ComparePhonesResult;
  transcript: string;
  wordErrorRate: number;
  expectedWords: string[];
  heardWords: string[];
  heardPhones: string[];
  phoneConfidences: number[];
  /** Aligned codepoint traces of the expected vs transcribed phones (details chart). */
  trace: { expected: number[]; heard: number[] };
  /** Prosody of the learner audio (details charts). */
  prosody: { f0: number[]; energy: number[] };
  timings: {
    phonesMs: number;
    asrMs: number;
    referenceMs: number;
    acousticMs: number;
    compareMs: number;
    totalMs: number;
  };
}

export interface PipelineCallbacks {
  onStatus?: (status: string) => void;
  onProgress?: (progress: WorkerProgress) => void;
}

interface PhonesResponse {
  type: 'result';
  phones: string[];
  confidences: number[];
  frames: number;
  audioSeconds: number;
  spans: Array<[number, number]>;
  vocabSize: number;
  logPosteriors: Float32Array;
  timings: Record<string, number>;
}

interface TranscriptResponse {
  type: 'transcript';
  transcript: string;
}

interface AcousticResponse {
  type: 'acoustic-result';
  distance: number;
  meanPerStep: number;
  pathLength: number;
  ms: number;
}

export type InferenceDevice = 'wasm' | 'webgpu';

export interface PipelineOptions {
  device?: InferenceDevice;
  /** defaults to fp16 on webgpu and q8 on wasm */
  dtype?: string;
}

export function dtypeForDevice(device: InferenceDevice): string {
  return device === 'webgpu' ? 'fp16' : 'q8';
}

export class Pipeline {
  private worker: WorkerClient;
  private callbacks: PipelineCallbacks;
  readonly device: InferenceDevice;

  private constructor(worker: WorkerClient, callbacks: PipelineCallbacks, device: InferenceDevice) {
    this.worker = worker;
    this.callbacks = callbacks;
    this.device = device;
  }

  /** Create the pipeline and load the worker-side models (phone recognizer + ASR). */
  static async create(callbacks: PipelineCallbacks = {}, options: PipelineOptions = {}): Promise<Pipeline> {
    const device = options.device ?? 'wasm';
    const dtype = options.dtype ?? dtypeForDevice(device);
    const worker = new WorkerClient(
      new Worker(new URL('../worker.ts', import.meta.url), { type: 'module' }),
      (message) => {
        if (message.type === 'progress') {
          callbacks.onProgress?.(message as unknown as WorkerProgress);
        } else if (message.type === 'error') {
          callbacks.onStatus?.(`error: ${String(message.message)}`);
        }
      },
    );

    const pipeline = new Pipeline(worker, callbacks, device);
    await pipeline.worker.request({ type: 'load', device, dtype });
    await pipeline.worker.request({ type: 'load-asr', device, dtype });
    return pipeline;
  }

  private status(text: string): void {
    this.callbacks.onStatus?.(text);
  }

  async analyze(learnerAudio: Float32Array, expectedText: string): Promise<AnalysisResult> {
    const start = performance.now();

    // private copies: each worker request transfers the buffer it receives
    const asrAudio = learnerAudio.slice();
    const acousticAudio = learnerAudio.slice();
    const prosodyAudio = learnerAudio.slice();

    // 1. phones
    this.status('recognizing phones...');
    const phonesStart = performance.now();
    const phonesResponse = await this.worker.request<PhonesResponse>(
      { type: 'infer', audio: learnerAudio },
      [learnerAudio.buffer],
    );
    const phonesMs = performance.now() - phonesStart;

    // 2. word transcription
    this.status('transcribing words...');
    const asrStart = performance.now();
    const transcriptResponse = await this.worker.request<TranscriptResponse>(
      { type: 'transcribe', audio: asrAudio },
      [asrAudio.buffer],
    );
    const asrMs = performance.now() - asrStart;
    const transcript = cleanTranscription(transcriptResponse.transcript);

    // 3. reference TTS
    this.status('synthesizing reference (piper, ~63 MB first time)...');
    const referenceStart = performance.now();
    const reference = await synthesizeReference(expectedText, (p) => {
      this.status(`downloading piper voice: ${(p.loaded / 1e6).toFixed(0)} / ${(p.total / 1e6).toFixed(0)} MB`);
    });
    const referenceMs = performance.now() - referenceStart;

    // 4. acoustic term (encoder + DTW in the worker)
    this.status('computing acoustic distance (encoder + DTW)...');
    const acousticStart = performance.now();
    const acousticResponse = await this.worker.request<AcousticResponse>(
      { type: 'acoustic', learner: acousticAudio, reference: reference.audio },
      [acousticAudio.buffer, reference.audio.buffer],
    );
    const acousticMs = performance.now() - acousticStart;

    // 5. phone term
    this.status('comparing phones...');
    const compareStart = performance.now();
    const { vocab } = await loadRecognizerVocab();
    const phones = await comparePhones(
      {
        phones: phonesResponse.phones,
        confidences: phonesResponse.confidences,
        spans: phonesResponse.spans,
        frameCount: phonesResponse.frames,
        vocabSize: phonesResponse.vocabSize,
        logPosteriors: phonesResponse.logPosteriors,
        vocab,
      },
      expectedText,
      'en',
    );
    const compareMs = performance.now() - compareStart;

    // 5b. details data: prosody contours + aligned phoneme trace
    const f0 = Array.from(interpolateF0(extractF0(prosodyAudio, 16000)));
    const energy = Array.from(extractEnergy(prosodyAudio));
    const expectedG2P = await getExpectedPhones(expectedText, 'en');
    const heardG2P = transcript ? await getExpectedPhones(transcript, 'en') : { words: [], groups: [] };
    const trace = alignPhonemeTrace(
      expectedG2P.groups.flat(),
      heardG2P.groups.flat(),
    );

    // 6. word term + score
    const expectedWords = tokenize(expectedText.toLowerCase());
    const heardWords = tokenize(transcript);
    const wer = wordErrorRate(expectedWords, heardWords);
    const score = computePronunciationScore(acousticResponse.meanPerStep, phones.phone_error_rate, wer, 'en');
    this.status('done');

    return {
      score,
      acoustic: { meanPerStep: acousticResponse.meanPerStep, ms: acousticResponse.ms },
      phones,
      transcript,
      wordErrorRate: wer,
      expectedWords,
      heardWords,
      heardPhones: phonesResponse.phones,
      phoneConfidences: phonesResponse.confidences,
      trace,
      prosody: { f0, energy },
      timings: {
        phonesMs,
        asrMs,
        referenceMs,
        acousticMs,
        compareMs,
        totalMs: performance.now() - start,
      },
    };
  }
}
