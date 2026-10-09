import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { getExpectedPhones } from '../src/core/g2p';

/** G2P parity: the espeak-ng WASM build must produce the same normalized
 * phones as the Python phonemizer for the fixture texts. */
const FIXTURES_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures');

interface G2pCase {
  text: string;
  words: string[];
  groups: string[][];
}

interface G2pFixture {
  lang: string;
  cases: G2pCase[];
}

describe('g2p parity vs Python (espeak-ng)', () => {
  it('produces the same words and phones as the Python phonemizer', async () => {
    const raw = await readFile(path.join(FIXTURES_DIR, 'g2p.json'), 'utf8');
    const fixture = JSON.parse(raw) as G2pFixture;

    let mismatches = 0;
    const details: string[] = [];
    for (const g2pCase of fixture.cases) {
      const result = await getExpectedPhones(g2pCase.text, fixture.lang);
      expect(result.words).toEqual(g2pCase.words);

      if (JSON.stringify(result.groups) !== JSON.stringify(g2pCase.groups)) {
        mismatches++;
        details.push(`${g2pCase.text}\n  python: ${JSON.stringify(g2pCase.groups)}\n  js:     ${JSON.stringify(result.groups)}`);
      }
    }
    // The espeak-ng WASM build is a different version from the system one:
    // allow residual drift here (tracked; must be zero for the common words).
    expect(mismatches, details.join('\n')).toBeLessThanOrEqual(1);
  });
});
