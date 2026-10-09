/**
 * Audio enhancement for learner recordings: noise gate (spectral subtraction)
 * plus voice-emphasis filtering (high-pass, presence EQ, compression).
 *
 * `spectralGate` is pure TypeScript (runs anywhere); `enhanceAudio` additionally
 * uses the Web Audio API and therefore needs a browser. The gate estimates the
 * noise floor from the quietest frames (classic Boll spectral subtraction with
 * over-subtraction and a spectral floor to avoid musical noise), which is
 * effective for stationary noise: fans, hum, hiss.
 */

const FFT_SIZE = 512;
const HOP = FFT_SIZE / 2;
/** Over-subtraction factor: how aggressively the noise estimate is removed. */
const OVER_SUBTRACTION = 2.0;
/** Per-bin gain floor: keeps a little of every bin to avoid musical noise. */
const GAIN_FLOOR = 0.1;
/** Temporal smoothing applied to the per-bin gains (0 = none, 1 = frozen). */
const GAIN_SMOOTHING = 0.5;
/** Fraction of quietest frames used to estimate the noise floor. */
const NOISE_QUANTILE = 0.2;

function hannWindow(size: number): Float32Array {
  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return window;
}

/** In-place iterative radix-2 FFT (decimation in time). Length must be a power of two. */
function fft(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      const tRe = re[i]; re[i] = re[j]; re[j] = tRe;
      const tIm = im[i]; im[i] = im[j]; im[j] = tIm;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const stepRe = Math.cos(ang);
    const stepIm = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < half; k++) {
        const aRe = re[i + k];
        const aIm = im[i + k];
        const bRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
        const bIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
        re[i + k] = aRe + bRe;
        im[i + k] = aIm + bIm;
        re[i + k + half] = aRe - bRe;
        im[i + k + half] = aIm - bIm;
        const nextRe = curRe * stepRe - curIm * stepIm;
        curIm = curRe * stepIm + curIm * stepRe;
        curRe = nextRe;
      }
    }
  }
}

function ifft(re: Float32Array, im: Float32Array): void {
  fft(im, re); // conjugate trick: swapping re/im computes the inverse
  const n = re.length;
  for (let i = 0; i < n; i++) {
    re[i] /= n;
    im[i] /= n;
  }
}

/**
 * Spectral-subtraction noise gate. Estimates the noise magnitude spectrum
 * from the quietest frames, then attenuates every bin below
 * `OVER_SUBTRACTION * noise` down to `GAIN_FLOOR` of its original magnitude,
 * with temporal smoothing on the gains.
 */
export function spectralGate(audio: Float32Array, sampleRate = 16000): Float32Array {
  void sampleRate;
  const frames = Math.floor((audio.length - FFT_SIZE) / HOP) + 1;
  if (frames < 4) {
    return audio;
  }

  const bins = FFT_SIZE / 2 + 1;
  const analysisWindow = hannWindow(FFT_SIZE);
  const frameRe = new Float32Array(frames * FFT_SIZE);
  const frameIm = new Float32Array(frames * FFT_SIZE);
  const magnitudes = new Float32Array(frames * bins);
  const frameEnergy = new Float32Array(frames);

  // analysis: STFT
  for (let f = 0; f < frames; f++) {
    const offset = f * HOP;
    const re = frameRe.subarray(f * FFT_SIZE, (f + 1) * FFT_SIZE);
    const im = frameIm.subarray(f * FFT_SIZE, (f + 1) * FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) {
      re[i] = audio[offset + i] * analysisWindow[i];
    }
    fft(re, im);
    let energy = 0;
    for (let k = 0; k < bins; k++) {
      const magnitude = Math.hypot(re[k], im[k]);
      magnitudes[f * bins + k] = magnitude;
      energy += magnitude * magnitude;
    }
    frameEnergy[f] = energy;
  }

  // noise floor: mean magnitude spectrum of the quietest frames
  const sorted = Array.from(frameEnergy).sort((a, b) => a - b);
  const energyThreshold = sorted[Math.max(1, Math.floor(sorted.length * NOISE_QUANTILE)) - 1];
  const noiseFloor = new Float32Array(bins);
  let quietCount = 0;
  for (let f = 0; f < frames; f++) {
    if (frameEnergy[f] <= energyThreshold) {
      quietCount += 1;
      for (let k = 0; k < bins; k++) {
        noiseFloor[k] += magnitudes[f * bins + k];
      }
    }
  }
  if (quietCount === 0) {
    return audio;
  }
  for (let k = 0; k < bins; k++) {
    noiseFloor[k] /= quietCount;
  }

  // gains with frequency (3-tap) and temporal smoothing
  const gains = new Float32Array(frames * bins);
  const previous = new Float32Array(bins).fill(1);
  const raw = new Float32Array(bins);
  for (let f = 0; f < frames; f++) {
    for (let k = 0; k < bins; k++) {
      const magnitude = magnitudes[f * bins + k];
      if (magnitude > 1e-9) {
        raw[k] = Math.max(
          (magnitude - OVER_SUBTRACTION * noiseFloor[k]) / magnitude,
          GAIN_FLOOR,
        );
      } else {
        raw[k] = GAIN_FLOOR;
      }
    }
    for (let k = 0; k < bins; k++) {
      const left = k > 0 ? raw[k - 1] : raw[k];
      const right = k < bins - 1 ? raw[k + 1] : raw[k];
      const frequencySmoothed = (left + raw[k] + right) / 3;
      const finalGain = GAIN_SMOOTHING * previous[k] + (1 - GAIN_SMOOTHING) * frequencySmoothed;
      gains[f * bins + k] = finalGain;
      previous[k] = finalGain;
    }
  }

  // synthesis: scaled inverse STFT with overlap-add
  const output = new Float32Array(audio.length);
  const windowSum = new Float32Array(audio.length);
  for (let f = 0; f < frames; f++) {
    const offset = f * HOP;
    const re = frameRe.subarray(f * FFT_SIZE, (f + 1) * FFT_SIZE);
    const im = frameIm.subarray(f * FFT_SIZE, (f + 1) * FFT_SIZE);
    for (let k = 0; k < bins; k++) {
      const gain = gains[f * bins + k];
      re[k] *= gain;
      im[k] *= gain;
      // Hermitian symmetry: mirror to the negative frequencies
      if (k > 0 && k < FFT_SIZE / 2) {
        re[FFT_SIZE - k] = re[k];
        im[FFT_SIZE - k] = -im[k];
      }
    }
    ifft(re, im);
    for (let i = 0; i < FFT_SIZE; i++) {
      const synthesized = re[i] * analysisWindow[i];
      output[offset + i] += synthesized;
      windowSum[offset + i] += analysisWindow[i] * analysisWindow[i];
    }
  }
  for (let i = 0; i < audio.length; i++) {
    output[i] = windowSum[i] > 1e-6 ? output[i] / windowSum[i] : audio[i];
    if (!Number.isFinite(output[i])) {
      return audio;
    }
  }
  return output;
}

/** Peak-normalize to `target` (0-1). */
export function normalizePeak(audio: Float32Array, target = 0.9): Float32Array {
  let peak = 0;
  for (let i = 0; i < audio.length; i++) {
    const value = Math.abs(audio[i]);
    if (value > peak) {
      peak = value;
    }
  }
  if (peak < 1e-6) {
    return audio;
  }
  const scale = target / peak;
  const out = new Float32Array(audio.length);
  for (let i = 0; i < audio.length; i++) {
    out[i] = audio[i] * scale;
  }
  return out;
}

/**
 * Voice-emphasis chain over Web Audio: high-pass at 80 Hz (rumble/hum),
 * low-shelf cut at 250 Hz (mud), presence boost at 2.5 kHz, gentle
 * compression. Falls back to the input unchanged when Web Audio is not
 * available (Node tests).
 */
async function filterAndCompress(audio: Float32Array, sampleRate: number): Promise<Float32Array> {
  const OfflineAudioContextCtor = (globalThis as {
    OfflineAudioContext?: typeof OfflineAudioContext;
  }).OfflineAudioContext;
  if (!OfflineAudioContextCtor) {
    return audio;
  }
  const context = new OfflineAudioContextCtor(1, audio.length, sampleRate);
  const buffer = context.createBuffer(1, audio.length, sampleRate);
  buffer.getChannelData(0).set(audio);
  const source = context.createBufferSource();
  source.buffer = buffer;

  const highPass = context.createBiquadFilter();
  highPass.type = 'highpass';
  highPass.frequency.value = 80;
  highPass.Q.value = 0.707;

  const lowShelf = context.createBiquadFilter();
  lowShelf.type = 'lowshelf';
  lowShelf.frequency.value = 250;
  lowShelf.gain.value = -3;

  const presence = context.createBiquadFilter();
  presence.type = 'peaking';
  presence.frequency.value = 2500;
  presence.Q.value = 1;
  presence.gain.value = 4;

  const compressor = context.createDynamicsCompressor();
  compressor.threshold.value = -24;
  compressor.knee.value = 12;
  compressor.ratio.value = 3;
  compressor.attack.value = 0.01;
  compressor.release.value = 0.25;

  source.connect(highPass);
  highPass.connect(lowShelf);
  lowShelf.connect(presence);
  presence.connect(compressor);
  compressor.connect(context.destination);
  source.start(0);

  const rendered = await context.startRendering();
  return rendered.getChannelData(0).slice();
}

/**
 * Full enhancement chain: voice filters + noise gate + peak normalization.
 * Short clips (< 0.3 s) are returned unchanged.
 */
export async function enhanceAudio(audio: Float32Array, sampleRate = 16000): Promise<Float32Array> {
  if (audio.length < sampleRate * 0.3) {
    return audio;
  }
  try {
    const filtered = await filterAndCompress(audio, sampleRate);
    const denoised = spectralGate(filtered, sampleRate);
    return normalizePeak(denoised);
  } catch {
    return audio; // enhancement must never break the analysis
  }
}
