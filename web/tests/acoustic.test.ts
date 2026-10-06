/**
 * Phase-2 gated test: acoustic embeddings from the quantized ONNX encoder and
 * the full score pipeline, validated against Python fixtures.
 *
 * Gated behind OPENPRONOUNCE_PARITY=1 (downloads/loads models):
 *   cd web && OPENPRONOUNCE_PARITY=1 npx vitest run tests/acoustic.test.ts
 */
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { env } from '@huggingface/transformers';
import { createEmbedder, acousticDistance, type Embedding } from '../src/core/embeddings';
import { parseWav } from '../src/core/audio';
import { createSession } from '../src/core/session';
import { comparePhones } from '../src/core/compare';
import { cleanTranscription, wordErrorRate, computePronunciationScore } from '../src/core/score';

const FIXTURES_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures');
const run = process.env.OPENPRONOUNCE_PARITY ? it : it.skip;

interface AcousticCase {
  learner: string;
  referenceText: string;
  distance: number;
  pathLength: number;
  meanPerStep: number;
  learnerFrames: number;
  referenceFrames: number;
  dim: number;
}

async function loadWav(name: string): Promise<Float32Array> {
  const buffer = await readFile(path.join(FIXTURES_DIR, name));
  const { samples } = parseWav(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
  return samples;
}

describe('acoustic embeddings + score (gated)', () => {
  let embedder: Awaited<ReturnType<typeof createEmbedder>>;
  let embeddingCache: Map<string, Embedding>;

  beforeAll(async () => {
    env.allowLocalModels = true; // session.ts disables it; the encoder is local
    env.localModelPath = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models');
    embedder = await createEmbedder('wav2vec2-960h-encoder', 'q8');
    embeddingCache = new Map();
  }, 10 * 60_000);

  async function embeddingFor(name: string): Promise<Embedding> {
    if (!embeddingCache.has(name)) {
      embeddingCache.set(name, await embedder.embed(await loadWav(name)));
    }
    return embeddingCache.get(name)!;
  }

  run('matches Python DTW distances within int8 drift', { timeout: 10 * 60_000 }, async () => {
    const fixture = JSON.parse(
      await readFile(path.join(FIXTURES_DIR, 'acoustic.json'), 'utf8'),
    ) as { cases: AcousticCase[] };

    for (const acousticCase of fixture.cases) {
      // reference wavs are not stored (gTTS/piper synthesized at fixture time);
      // re-derive them from the text with the browser-side reference path is
      // phase 2's piper work — here we compare the self-distance (identical
      // audio) and the wrong-sentence pair, which uses stored fixtures only.
      if (acousticCase.referenceText.includes('piper')) {
        continue;
      }
      const referenceName = acousticCase.referenceText.toLowerCase().startsWith('hello')
        ? 'en-hello.wav'
        : acousticCase.referenceText.toLowerCase().startsWith('the weather')
          ? 'en-weather.wav'
          : 'en-coffee.wav';
      const learner = await embeddingFor(acousticCase.learner);
      const reference = await embeddingFor(referenceName);
      const result = acousticDistance(learner, reference);

      expect(result.pathLength !== 0).toBe(true);
      // identical audio pairs must stay near zero; the wrong-sentence pair
      // (~14.3 in fp32) is checked relatively to allow int8 drift
      if (acousticCase.meanPerStep === 0) {
        expect(result.meanPerStep, `${acousticCase.learner} self-distance`).toBeLessThan(0.75);
      } else {
        expect(result.meanPerStep, `${acousticCase.learner} wrong-sentence distance`)
          .toBeGreaterThan(acousticCase.meanPerStep * 0.75);
        expect(result.meanPerStep).toBeLessThan(acousticCase.meanPerStep * 1.25);
      }
    }
  });

  run('reproduces Python word error rates and scores', async () => {
    const fixture = JSON.parse(
      await readFile(path.join(FIXTURES_DIR, 'score.json'), 'utf8'),
    ) as {
      cases: Array<{
        reference: string;
        transcription: string;
        expectedWords: string[];
        heardWords: string[];
        wordErrorRate: number;
        acousticDistance: number;
        score: number;
      }>;
    };

    for (const scoreCase of fixture.cases) {
      const heard = cleanTranscription(scoreCase.transcription);
      const heardWords = heard.match(/[\p{L}\p{N}_](?:[\p{L}\p{N}_']*[\p{L}\p{N}_])?/gu)?.map((w) => w.toLowerCase()) ?? [];
      expect(heardWords).toEqual(scoreCase.heardWords);

      const expectedWords = scoreCase.expectedWords;
      const wer = wordErrorRate(expectedWords, heardWords);
      expect(wer).toBeCloseTo(scoreCase.wordErrorRate, 4);

      const score = computePronunciationScore(scoreCase.acousticDistance, 0.15, wer, 'en');
      expect(score, `${scoreCase.transcription} @ ${scoreCase.acousticDistance}`).toBeCloseTo(scoreCase.score, 2);
    }
  });

  run('end-to-end: fixture audio scored against its own text', { timeout: 10 * 60_000 }, async () => {
    const phoneSession = await createSession({ device: 'wasm', dtype: 'q8' });
    const text = 'I would like a cup of coffee, please.';
    const audio = await loadWav('en-coffee.wav');

    const recognition = await phoneSession.recognition(audio);
    const phoneResult = await comparePhones(recognition, text, 'en');
    const transcription = cleanTranscription('I would like a cup of coffee please');

    const expectedWords = text.match(/[\p{L}\p{N}_](?:[\p{L}\p{N}_']*[\p{L}\p{N}_])?/gu)?.map((w) => w.toLowerCase()) ?? [];
    const heardWords = transcription.match(/[\p{L}\p{N}_](?:[\p{L}\p{N}_']*[\p{L}\p{N}_])?/gu) ?? [];
    const wer = wordErrorRate(expectedWords, heardWords);

    const learner = await embeddingFor('en-coffee.wav');
    const reference = await embeddingFor('en-coffee.wav');
    const acoustic = acousticDistance(learner, reference);

    const score = computePronunciationScore(
      acoustic.meanPerStep,
      phoneResult.phone_error_rate,
      wer,
      'en',
    );
    console.log(`end-to-end score: ${score} (acoustic ${acoustic.meanPerStep.toFixed(3)}, `
      + `phones ${phoneResult.phone_error_rate}, words ${wer})`);
    expect(score).toBeGreaterThan(70); // clean TTS reading against itself
  });
});
