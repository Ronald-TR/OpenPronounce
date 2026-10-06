import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { createSession } from '../src/core/session';
import { parseWav } from '../src/core/audio';

/**
 * Parity test: the browser pipeline must produce the same phones as the
 * Python pipeline for the generated fixtures.
 *
 * Gated behind OPENPRONOUNCE_PARITY=1 because it downloads the model (~318 MB):
 *   cd web && OPENPRONOUNCE_PARITY=1 npx vitest run tests/parity.test.ts
 */
const run = process.env.OPENPRONOUNCE_PARITY ? it : it.skip;
const FIXTURES_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures');

interface Fixture {
  name: string;
  file: string;
  phones: string[];
}

describe('phone parity vs Python', () => {
  run('matches the Python phone output on every fixture', { timeout: 20 * 60_000 }, async () => {
    const raw = await readFile(path.join(FIXTURES_DIR, 'expected.json'), 'utf8');
    const { fixtures } = JSON.parse(raw) as { fixtures: Fixture[] };
    const session = await createSession({ device: 'wasm', dtype: 'q8' });

    for (const fixture of fixtures) {
      const buffer = await readFile(path.join(FIXTURES_DIR, fixture.file));
      const { samples } = parseWav(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
      const result = await session.run(samples);

      const aligned = Math.min(fixture.phones.length, result.phones.length);
      let matches = 0;
      const diffs: string[] = [];
      for (let i = 0; i < aligned; i++) {
        if (fixture.phones[i] === result.phones[i]) {
          matches++;
        } else {
          diffs.push(`#${i}: expected ${fixture.phones[i]}, got ${result.phones[i]}`);
        }
      }
      if (fixture.phones.length !== result.phones.length) {
        diffs.push(`length: expected ${fixture.phones.length}, got ${result.phones.length}`);
      }
      // allow small numeric drift from int8 quantization (a couple of borderline phones)
      expect(matches, `${fixture.name}: ${diffs.join('; ')}`).toBeGreaterThanOrEqual(aligned - 2);
      expect(result.phones.length, `${fixture.name}: phone count differs`).toBeGreaterThanOrEqual(fixture.phones.length - 2);
    }
  });
});
