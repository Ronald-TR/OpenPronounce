/**
 * Audio helpers: browser decoding to 16 kHz mono and a minimal WAV parser
 * (PCM 16-bit) for Node tests and benchmarks.
 */

export const SAMPLING_RATE = 16000;

/** Decode any browser-supported audio file (wav, webm, mp3...) to 16 kHz mono Float32. */
export async function decodeTo16kMono(blob: Blob): Promise<Float32Array> {
  const arrayBuffer = await blob.arrayBuffer();
  const context = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await context.decodeAudioData(arrayBuffer);
  } finally {
    void context.close();
  }
  const frames = Math.max(1, Math.ceil(decoded.duration * SAMPLING_RATE));
  const target = new OfflineAudioContext(1, frames, SAMPLING_RATE);
  const source = target.createBufferSource();
  source.buffer = decoded;
  source.connect(target.destination);
  source.start();
  const rendered = await target.startRendering();
  return rendered.getChannelData(0).slice();
}

/** Parse a RIFF/WAVE file (PCM 16-bit or float 32) into a Float32Array. */
export function parseWav(buffer: ArrayBuffer): { samples: Float32Array; sampleRate: number } {
  const view = new DataView(buffer);
  if (view.getUint32(0, false) !== 0x52494646) { // "RIFF"
    throw new Error('not a RIFF file');
  }
  if (view.getUint32(8, false) !== 0x57415645) { // "WAVE"
    throw new Error('not a WAVE file');
  }

  let offset = 12;
  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataLength = 0;

  while (offset + 8 <= view.byteLength) {
    const chunkId = view.getUint32(offset, false);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkData = offset + 8;
    if (chunkId === 0x666d7420) { // "fmt "
      format = view.getUint16(chunkData, true);
      channels = view.getUint16(chunkData + 2, true);
      sampleRate = view.getUint32(chunkData + 4, true);
      bitsPerSample = view.getUint16(chunkData + 14, true);
    } else if (chunkId === 0x64617461) { // "data"
      dataOffset = chunkData;
      dataLength = Math.min(chunkSize, view.byteLength - chunkData);
    }
    offset = chunkData + chunkSize + (chunkSize % 2);
  }

  if (dataOffset < 0) {
    throw new Error('WAV file has no data chunk');
  }

  const bytesPerSample = bitsPerSample / 8;
  const frameCount = Math.floor(dataLength / bytesPerSample / channels);
  const samples = new Float32Array(frameCount);

  for (let frame = 0; frame < frameCount; frame++) {
    const base = dataOffset + frame * bytesPerSample * channels;
    let sum = 0;
    for (let channel = 0; channel < channels; channel++) {
      const pos = base + channel * bytesPerSample;
      if (format === 1 && bitsPerSample === 16) {
        sum += view.getInt16(pos, true) / 32768;
      } else if (format === 3 && bitsPerSample === 32) {
        sum += view.getFloat32(pos, true);
      } else {
        throw new Error(`unsupported WAV format ${format} (${bitsPerSample}-bit)`);
      }
    }
    samples[frame] = sum / channels;
  }

  if (sampleRate !== SAMPLING_RATE) {
    throw new Error(`expected 16 kHz fixtures, got ${sampleRate} Hz`);
  }
  return { samples, sampleRate };
}
