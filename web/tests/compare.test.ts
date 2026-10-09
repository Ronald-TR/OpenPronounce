import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { comparePhones } from '../src/core/compare';
import type { PhoneRecognition } from '../src/core/phones';

/** compare_phones parity: run the TypeScript port on the exact recognition
 * (phones + posteriors + vocab) produced by the Python pipeline and diff the
 * outputs component by component. */
const FIXTURES_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures');

interface CompareCase {
  wav: string;
  text: string;
  recognition: {
    phones: string[];
    confidences: number[];
    spans: number[][];
    frames: number;
    vocabSize: number;
    logPosteriorsB64: string;
  };
  compare: {
    expected_phones: string[][];
    heard_phones: string[];
    heard_phones_confidence: number[];
    phone_error_rate: number;
    errors: Array<{
      position: number;
      word: string;
      expected: string;
      actual: string;
      phone_distance: number;
      confidence: number;
      phones: Array<{ expected: string; heard: string; confidence: number }>;
    }>;
    words_with_errors: string[];
  };
}

function base64ToFloat32(base64: string): Float32Array {
  const buffer = Buffer.from(base64, 'base64');
  return new Float32Array(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
}

describe('compare_phones parity vs Python', () => {
  let vocab: string[];

  beforeAll(async () => {
    vocab = JSON.parse(await readFile(path.join(FIXTURES_DIR, 'vocab.json'), 'utf8')) as string[];
  });

  it('reproduces errors, confidences and rates on every fixture', async () => {
    const raw = await readFile(path.join(FIXTURES_DIR, 'compare.json'), 'utf8');
    const fixture = JSON.parse(raw) as { cases: CompareCase[] };

    for (const compareCase of fixture.cases) {
      const recognition: PhoneRecognition = {
        phones: compareCase.recognition.phones,
        confidences: compareCase.recognition.confidences,
        spans: compareCase.recognition.spans.map(([s, e]) => [s, e] as [number, number]),
        logPosteriors: base64ToFloat32(compareCase.recognition.logPosteriorsB64),
        frameCount: compareCase.recognition.frames,
        vocabSize: compareCase.recognition.vocabSize,
        vocab,
      };

      const result = await comparePhones(recognition, compareCase.text, 'en');
      const expected = compareCase.compare;
      const label = `${compareCase.wav} vs "${compareCase.text}"`;

      expect(result.heard_phones, `${label}: heard`).toEqual(expected.heard_phones);
      expect(result.expected_phones, `${label}: expected phones`).toEqual(expected.expected_phones);
      expect(result.phone_error_rate, `${label}: error rate`).toBeCloseTo(expected.phone_error_rate, 4);
      expect(result.words_with_errors, `${label}: flagged words`).toEqual(expected.words_with_errors);
      expect(result.errors.length, `${label}: error count`).toBe(expected.errors.length);

      result.errors.forEach((error, index) => {
        const ref = expected.errors[index];
        const errorLabel = `${label}: error[${index}] (${ref.word})`;
        expect(error.position, `${errorLabel} position`).toBe(ref.position);
        expect(error.word, `${errorLabel} word`).toBe(ref.word);
        expect(error.expected, `${errorLabel} expected phones`).toBe(ref.expected);
        expect(error.actual, `${errorLabel} actual phones`).toBe(ref.actual);
        expect(error.phone_distance, `${errorLabel} distance`).toBe(ref.phone_distance);
        expect(error.confidence, `${errorLabel} confidence`).toBeCloseTo(ref.confidence, 2);

        expect(error.phones.length, `${errorLabel} phone report count`).toBe(ref.phones.length);
        error.phones.forEach((phone, phoneIndex) => {
          const refPhone = ref.phones[phoneIndex];
          expect(phone.expected, `${errorLabel} phone[${phoneIndex}] expected`).toBe(refPhone.expected);
          expect(phone.heard, `${errorLabel} phone[${phoneIndex}] heard`).toBe(refPhone.heard);
          expect(phone.confidence, `${errorLabel} phone[${phoneIndex}] confidence`).toBeCloseTo(refPhone.confidence, 2);
        });
      });
    }
  });
});
