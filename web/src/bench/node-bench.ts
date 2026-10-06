/**
 * Node benchmark + parity harness: runs the same ONNX model through
 * transformers.js in Node and prints phones + timings per fixture.
 *
 * Usage: npm run bench [-- --parity]
 */
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createSession, type Session } from '../core/session';
import { parseWav } from '../core/audio';
import type { SpikeResult } from '../core/session';

const FIXTURES_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'tests', 'fixtures');

interface Fixture {
  name: string;
  text: string;
  file: string;
  phones: string[];
  confidences: number[];
}

async function loadFixtures(): Promise<Fixture[]> {
  const raw = await readFile(path.join(FIXTURES_DIR, 'expected.json'), 'utf8');
  return (JSON.parse(raw) as { fixtures: Fixture[] }).fixtures;
}

function phoneAccuracy(expected: string[], actual: string[]): { matches: number; total: number; aligned: number } {
  const total = Math.max(expected.length, actual.length);
  const aligned = Math.min(expected.length, actual.length);
  let matches = 0;
  for (let i = 0; i < aligned; i++) {
    if (expected[i] === actual[i]) {
      matches++;
    }
  }
  return { matches, total, aligned };
}

async function evaluate(session: Session, fixture: Fixture, runs: number): Promise<{ result: SpikeResult; matches: number; total: number }> {
  const buffer = await readFile(path.join(FIXTURES_DIR, fixture.file));
  const { samples } = parseWav(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));

  let result: SpikeResult | null = null;
  for (let i = 0; i < runs; i++) {
    result = await session.run(samples);
  }
  result = result!;

  const { matches, total } = phoneAccuracy(fixture.phones, result.phones);
  return { result, matches, total };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const runs = args.includes('--runs') ? Number(args[args.indexOf('--runs') + 1]) || 3 : 3;

  const fixtures = await loadFixtures();
  console.log(`fixtures: ${fixtures.length}, runs each: ${runs}`);
  console.log(`rss: ${(process.memoryUsage().rss / 1e6).toFixed(0)} MB\n`);

  const session = await createSession({ device: 'wasm', dtype: 'q8' });
  console.log(`model loaded (${session.info.device}/${session.info.dtype})`);
  console.log(`rss: ${(process.memoryUsage().rss / 1e6).toFixed(0)} MB\n`);

  const summary: Array<Record<string, unknown>> = [];
  let totalMatches = 0;
  let totalPhones = 0;

  for (const fixture of fixtures) {
    const { result, matches, total } = await evaluate(session, fixture, runs);
    totalMatches += matches;
    totalPhones += total;

    console.log(`— ${fixture.name} (${fixture.file}, ${result.audioSeconds.toFixed(1)}s)`);
    console.log(`  expected: ${fixture.phones.join(' ')}`);
    console.log(`  heard:    ${result.phones.join(' ')}`);
    console.log(`  match: ${matches}/${total} aligned phones`);
    console.log(`  timings: total ${(result.timings.totalMs / 1000).toFixed(2)}s`
      + ` · inference ${(result.timings.inferenceMs / 1000).toFixed(2)}s`
      + ` · preprocess ${result.timings.preprocessMs.toFixed(0)}ms`
      + ` · decode ${result.timings.decodeMs.toFixed(1)}ms`);
    console.log(`  rss: ${(process.memoryUsage().rss / 1e6).toFixed(0)} MB\n`);

    summary.push({
      name: fixture.name,
      audioSeconds: result.audioSeconds,
      expected: fixture.phones,
      heard: result.phones,
      match: `${matches}/${total}`,
      timingsMs: result.timings,
    });
  }

  console.log(`TOTAL phone match: ${totalMatches}/${totalPhones} (${((totalMatches / Math.max(1, totalPhones)) * 100).toFixed(1)} %)`);
  const report = { backend: session.info, summary, totalMatch: `${totalMatches}/${totalPhones}` };
  await writeFile(path.join(FIXTURES_DIR, 'bench-report.json'), JSON.stringify(report, null, 2));
  console.log('report written to web/tests/fixtures/bench-report.json');

  if (args.includes('--parity')) {
    await writeFile(path.join(FIXTURES_DIR, 'heard.json'), JSON.stringify(report, null, 2));
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
