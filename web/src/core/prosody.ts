/**
 * Port of the prosody parts of openpronounce/speech.py: fundamental frequency
 * (librosa.pyin replaced by a plain YIN tracker — visual grade), RMS energy
 * (librosa.feature.rms + MinMaxScaler to 0-250) and the unvoiced-frame
 * interpolation of the pitch contour.
 */

const FRAME_LENGTH = 2048;
const HOP_LENGTH = 512;
/** YIN search range: 50 Hz to 400 Hz at 16 kHz. */
const TAU_MIN = 40;
const TAU_MAX = 320;
const YIN_THRESHOLD = 0.15;

/**
 * Fundamental frequency contour in Hz, unvoiced frames set to 0
 * (speech.extract_f0 semantics).
 */
export function extractF0(audio: Float32Array, sampleRate = 16000): Float32Array {
  const frames = Math.max(1, Math.ceil(audio.length / HOP_LENGTH));
  const f0 = new Float32Array(frames);
  // center-pad like librosa (zeros; edge frames contribute a little less)
  const padded = new Float32Array(audio.length + FRAME_LENGTH);
  padded.set(audio, FRAME_LENGTH / 2);

  for (let frame = 0; frame < frames; frame++) {
    const offset = frame * HOP_LENGTH;
    const tau = yinPitch(padded.subarray(offset, offset + FRAME_LENGTH));
    f0[frame] = tau > 0 ? sampleRate / tau : 0;
  }
  return f0;
}

/** YIN pitch estimate for one window; returns the period in samples or 0 when unvoiced. */
function yinPitch(window: Float32Array): number {
  const size = window.length;
  const maxTau = Math.min(TAU_MAX, size >> 1);
  if (maxTau <= TAU_MIN) {
    return 0;
  }

  // difference function over the search range
  const diff = new Float32Array(maxTau + 1);
  for (let tau = TAU_MIN; tau <= maxTau; tau++) {
    let sum = 0;
    for (let i = 0; i + tau < size; i++) {
      const delta = window[i] - window[i + tau];
      sum += delta * delta;
    }
    diff[tau] = sum;
  }

  // cumulative mean normalized difference
  const normalized = new Float32Array(maxTau + 1);
  normalized[TAU_MIN] = 1;
  let runningSum = 0;
  for (let tau = TAU_MIN + 1; tau <= maxTau; tau++) {
    runningSum += diff[tau - 1] + diff[tau];
    normalized[tau] = runningSum > 0 ? (diff[tau] * (tau - TAU_MIN + 1)) / runningSum : 1;
  }

  // first dip below the threshold, else the global minimum
  let bestTau = 0;
  let bestValue = Infinity;
  for (let tau = TAU_MIN + 1; tau <= maxTau; tau++) {
    if (normalized[tau] < bestValue) {
      bestValue = normalized[tau];
      bestTau = tau;
    }
    if (normalized[tau] < YIN_THRESHOLD && normalized[tau + 1] >= normalized[tau]) {
      return tau;
    }
  }
  return bestValue < YIN_THRESHOLD ? bestTau : 0;
}

/**
 * RMS energy per frame, min-max scaled to 0-250 so it can share an axis with
 * F0 (speech.extract_energy; constant frames map to 0 like sklearn).
 */
export function extractEnergy(audio: Float32Array): Float32Array {
  const frames = Math.max(1, Math.ceil(audio.length / HOP_LENGTH));
  const padded = new Float32Array(audio.length + FRAME_LENGTH);
  padded.set(audio, FRAME_LENGTH / 2);

  const rms = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame++) {
    const offset = frame * HOP_LENGTH;
    let sum = 0;
    for (let i = 0; i < FRAME_LENGTH; i++) {
      const value = padded[offset + i];
      sum += value * value;
    }
    rms[frame] = Math.sqrt(sum / FRAME_LENGTH);
  }

  let min = Infinity;
  let max = -Infinity;
  for (const value of rms) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const scaled = new Float32Array(frames);
  const span = max - min;
  for (let i = 0; i < frames; i++) {
    scaled[i] = span > 0 ? ((rms[i] - min) / span) * 250 : 0;
  }
  return scaled;
}

/** Linearly interpolate unvoiced (0) frames so the pitch curve has no gaps. */
export function interpolateF0(f0: Float32Array): Float32Array {
  const out = new Float32Array(f0.length);
  const voiced: number[] = [];
  for (let i = 0; i < f0.length; i++) {
    if (f0[i] > 0) {
      voiced.push(i);
    }
  }
  if (!voiced.length) {
    return out;
  }
  let next = 0;
  for (let i = 0; i < f0.length; i++) {
    if (f0[i] > 0) {
      out[i] = f0[i];
      continue;
    }
    while (next < voiced.length && voiced[next] < i) {
      next += 1;
    }
    const previousVoiced = next > 0 ? voiced[next - 1] : voiced[0];
    const followingVoiced = next < voiced.length ? voiced[next] : voiced[voiced.length - 1];
    if (followingVoiced === previousVoiced) {
      out[i] = f0[previousVoiced];
      continue;
    }
    const t = (i - previousVoiced) / (followingVoiced - previousVoiced);
    out[i] = f0[previousVoiced] + t * (f0[followingVoiced] - f0[previousVoiced]);
  }
  return out;
}
