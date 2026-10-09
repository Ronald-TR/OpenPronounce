/**
 * Phase-4 calibration harness: recompute the acoustic term of the
 * speechocean762 sample with the quantized ONNX encoder (the browser's model)
 * and score each utterance with the Python pipeline's phone/word components.
 *
 * Output: tests/fixtures/calibration/calibration-results.csv
 *
 * Usage: npm run calibrate [-- --limit 200]
 */
import { readFile, writeFile, appendFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { env } from '@huggingface/transformers';
import { createEmbedder, acousticDistance, type Embedding } from '../core/embeddings';
import { parseWav } from '../core/audio';
import { computePronunciationScore } from '../core/score';

const CAL_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'tests', 'fixtures', 'calibration');
const OUT_CSV = path.join(CAL_DIR, 'calibration-results.csv');

interface ManifestEntry {
  utt: string;
  text: string;
  humanTotal: number;
  pythonAcoustic: number;
  per: number;
  wer: number;
  learnerFile: string;
  referenceFile: string;
}

interface CalibrationManifest {
  entries: ManifestEntry[];
}

async function loadWav(dir: string, name: string): Promise<Float32Array> {
  const buffer = await readFile(path.join(dir, name));
  const { samples } = parseWav(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
  return samples;
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const limitIndex = args.indexOf('--limit');
  const limit = limitIndex >= 0 ? Number(args[limitIndex + 1]) : Infinity;

  const manifest = JSON.parse(
    await readFile(path.join(CAL_DIR, 'manifest.json'), 'utf8'),
  ) as CalibrationManifest;
  const entries = manifest.entries.slice(0, Number.isFinite(limit) ? limit : undefined);

  // resume: skip utterances already in the output CSV
  const done = new Set<string>();
  try {
    const existing = await readFile(OUT_CSV, 'utf8');
    for (const line of existing.split('\n').slice(1)) {
      const utt = line.split(',')[0];
      if (utt) {
        done.add(utt);
      }
    }
  } catch {
    await writeFile(OUT_CSV, 'utt,human_total,python_acoustic,browser_acoustic,per,wer,browser_score\n');
  }

  env.allowLocalModels = true;
  env.localModelPath = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'public', 'models');
  const embedder = await createEmbedder('wav2vec2-960h-encoder', 'q8');

  const refCache = new Map<string, Embedding>();
  const wavDir = path.join(CAL_DIR, 'wav');
  const refDir = path.join(CAL_DIR, 'ref');

  let count = 0;
  for (const entry of entries) {
    if (done.has(entry.utt)) {
      continue;
    }
    const learner = await embedder.embed(await loadWav(wavDir, entry.learnerFile));
    let reference = refCache.get(entry.referenceFile);
    if (!reference) {
      reference = await embedder.embed(await loadWav(refDir, entry.referenceFile));
      refCache.set(entry.referenceFile, reference);
    }
    const acoustic = acousticDistance(learner, reference);
    const score = computePronunciationScore(acoustic.meanPerStep, entry.per, entry.wer, 'en');

    await appendFile(
      OUT_CSV,
      `${entry.utt},${entry.humanTotal},${entry.pythonAcoustic},${acoustic.meanPerStep.toFixed(4)},${entry.per},${entry.wer},${score}\n`,
    );
    count += 1;
    if (count % 25 === 0) {
      console.log(`${count} processed (${entry.utt}: browser ${acoustic.meanPerStep.toFixed(2)} vs python ${entry.pythonAcoustic.toFixed(2)})`);
    }
  }

  const outStat = await stat(OUT_CSV);
  console.log(`done: ${count} new rows, ${outStat.size} bytes -> ${OUT_CSV}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
