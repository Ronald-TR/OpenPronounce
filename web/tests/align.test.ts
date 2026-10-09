import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { levenshteinDistance, levenshteinOpcodes } from '../src/core/levenshtein';
import { fastdtw } from '../src/core/dtw';

/** Levenshtein + FastDTW parity vs the Python packages on the real phone sequences. */
const FIXTURES_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures');

interface SeqCase {
  wav: string;
  text: string;
  expected: string[];
  heard: string[];
  levenshtein_distance: number;
  opcodes: Array<[string, number, number, number, number]>;
  fastdtw_distance: number;
  fastdtw_path_length: number;
}

interface SeqFixture {
  cases: SeqCase[];
  small_dtw: { distance: number; path: number[][] };
}

describe('levenshtein + fastdtw parity vs Python', () => {
  it('matches distances and opcodes on the fixture phone sequences', async () => {
    const raw = await readFile(path.join(FIXTURES_DIR, 'sequences.json'), 'utf8');
    const fixture = JSON.parse(raw) as SeqFixture;

    for (const seqCase of fixture.cases) {
      const { expected, heard } = seqCase;

      expect(levenshteinDistance(expected, heard), `distance ${seqCase.wav}`).toBe(seqCase.levenshtein_distance);
      expect(levenshteinDistance(heard, expected), `distance symmetric ${seqCase.wav}`).toBe(seqCase.levenshtein_distance);

      const opcodes = levenshteinOpcodes(expected, heard).map(
        (op) => [op.tag, op.i1, op.i2, op.j1, op.j2] as [string, number, number, number, number],
      );
      expect(opcodes, `opcodes ${seqCase.wav}`).toEqual(seqCase.opcodes);

      const codepoints = (phones: string[]) => phones.join(' ').split('').map((c) => [c.charCodeAt(0)]);
      const { distance, path } = fastdtw(codepoints(expected), codepoints(heard));
      expect(distance, `fastdtw ${seqCase.wav}`).toBeCloseTo(seqCase.fastdtw_distance, 4);
      expect(path.length, `fastdtw path ${seqCase.wav}`).toBe(seqCase.fastdtw_path_length);
    }
  });

  it('matches the small exact DTW case', async () => {
    const raw = await readFile(path.join(FIXTURES_DIR, 'sequences.json'), 'utf8');
    const fixture = JSON.parse(raw) as SeqFixture;

    const { distance, path: dtwPath } = fastdtw(
      [[1], [2], [3], [4], [5]],
      [[2], [3], [4]],
    );
    expect(distance).toBeCloseTo(fixture.small_dtw.distance, 6);
    expect(dtwPath).toEqual(fixture.small_dtw.path);
  });
});
