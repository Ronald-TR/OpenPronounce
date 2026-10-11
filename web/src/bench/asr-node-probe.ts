import { env, AutoProcessor, Wav2Vec2ForCTC } from '@huggingface/transformers';
import { readFileSync } from 'node:fs';
import { parseWav } from '/Users/ronaldfarias/OpenPronounce/web/src/core/audio';

async function main() {
  env.allowLocalModels = true;
  env.localModelPath = '/Users/ronaldfarias/OpenPronounce/web/public/models';
  const ID = 'word-transcription';
  const processor = await AutoProcessor.from_pretrained(ID);
  const model = await Wav2Vec2ForCTC.from_pretrained(ID, { dtype: 'q8', device: 'cpu' });
  const buf = readFileSync('/Users/ronaldfarias/OpenPronounce/web/tests/fixtures/en-coffee.wav');
  const { samples } = parseWav(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const inputs = await processor(samples);
  const inData = inputs.input_values.data as Float32Array;
  let inMin = Infinity, inMax = -Infinity, inSum = 0;
  for (let i = 0; i < inData.length; i++) { inMin = Math.min(inMin, inData[i]); inMax = Math.max(inMax, inData[i]); inSum += inData[i]; }
  console.log('[node] input_values: len', inData.length, '· min', inMin.toFixed(4), '· max', inMax.toFixed(4), '· mean', (inSum / inData.length).toFixed(5));
  const out = await model({ input_values: inputs.input_values });
  const data = out.logits.data as Float32Array;
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i];
  const first8 = Array.from(data.slice(0, 8)).map((v) => v.toFixed(4)).join(',');
  let blankCount = 0;
  const dims = out.logits.dims as number[];
  for (let f = 0; f < dims[1]; f++) {
    let best = 0, bv = -Infinity;
    for (let v = 0; v < dims[2]; v++) { const val = data[f*dims[2]+v]; if (val > bv) { bv = val; best = v; } }
    if (best === 0) blankCount++;
  }
  console.log('[node] sum', sum.toFixed(2), '· first8', first8, '· blank frames', blankCount, '/', dims[1]);
}
main().catch(e => { console.error('ERR', e.message); process.exit(1); });
