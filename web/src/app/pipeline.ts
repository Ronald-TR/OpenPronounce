/**
 * Full client-side analysis pipeline: phones (worker) + word transcription
 * (worker) + G2P + piper reference + acoustic embeddings + score.
 */
import { WorkerClient, type WorkerProgress } from './worker-client';
import { comparePhones, type ComparePhonesResult } from '../core/compare';import { loadRecognizerVocab } from '../core/session';
import { synthesizeReference } from '../core/tts';
import { cleanTranscription, computePronunciationScore, wordErrorRate, words as tokenize } from '../core/score';


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

export class Pipeline {
  private worker: WorkerClient;
  private callbacks: PipelineCallbacks;

  private constructor(worker: WorkerClient, callbacks: PipelineCallbacks) {
    this.worker = worker;
    this.callbacks = callbacks;
  }

  /** Create the pipeline and load the worker-side models (phone recognizer + ASR). */
  static async create(callbacks: PipelineCallbacks = {}): Promise<Pipeline> {
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

    const pipeline = new Pipeline(worker, callbacks);
    await pipeline.worker.request({ type: 'load', device: 'wasm', dtype: 'q8' });
    await pipeline.worker.request({ type: 'load-asr' });
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
