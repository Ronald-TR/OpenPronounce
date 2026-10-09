import { describe, expect, it } from 'vitest';
import { spectralGate, normalizePeak } from '../src/core/enhance';

/** RMS of a signal segment. */
function rms(samples: Float32Array, start = 0, end = samples.length): number {
  let sum = 0;
  for (let i = start; i < end; i++) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / Math.max(1, end - start));
}

/** Pseudo-random white noise, deterministic. */
function noise(length: number, amplitude: number, seed = 7): Float32Array {
  let state = seed;
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    out[i] = ((state / 0x3fffffff) - 1) * amplitude;
  }
  return out;
}

function tone(length: number, frequency: number, amplitude: number, sampleRate = 16000): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = amplitude * Math.sin((2 * Math.PI * frequency * i) / sampleRate);
  }
  return out;
}

function add(...signals: Float32Array[]): Float32Array {
  const out = new Float32Array(signals[0].length);
  for (const signal of signals) {
    for (let i = 0; i < out.length; i++) {
      out[i] += signal[i];
    }
  }
  return out;
}

describe('spectralGate', () => {
  it('suppresses noise-only segments while keeping the tone', () => {
    const sampleRate = 16000;
    const length = sampleRate; // 1 s
    const half = length / 2;
    const toneSignal = tone(length, 440, 0.3, sampleRate);
    const noiseSignal = noise(length, 0.12);
    // first half: tone + noise (speech); second half: noise only (pause)
    const noisy = add(toneSignal, noiseSignal);
    for (let i = half; i < length; i++) {
      noisy[i] -= toneSignal[i];
    }

    const processed = spectralGate(noisy, sampleRate);
    expect(processed.length).toBe(length);

    const noiseBefore = rms(noisy, half);
    const noiseAfter = rms(processed, half);
    const speechBefore = rms(noisy, 0, half);
    const speechAfter = rms(processed, 0, half);

    // noise-only segment strongly attenuated
    expect(noiseAfter / noiseBefore).toBeLessThan(0.6);
    // speech segment mostly preserved
    expect(speechAfter / speechBefore).toBeGreaterThan(0.55);
  });

  it('is near-transparent on speech-like signals (bursts + pauses)', () => {
    const sampleRate = 16000;
    const length = sampleRate;
    const clean = new Float32Array(length);
    // 100 ms tone bursts alternating with 100 ms pauses, like speech rhythm
    for (let i = 0; i < length; i++) {
      const burst = Math.floor(i / (sampleRate / 10)) % 2 === 0;
      if (burst) {
        clean[i] = 0.3 * Math.sin((2 * Math.PI * 440 * i) / sampleRate);
      }
    }
    const processed = spectralGate(clean, sampleRate);
    // measure inside the first burst, away from the edges
    const before = rms(clean, 400, 1200);
    const after = rms(processed, 400, 1200);
    expect(after / before).toBeGreaterThan(0.6);
    expect(after / before).toBeLessThan(1.5);
  });

  it('preserves length and produces finite samples', () => {
    const noisy = add(tone(8000, 300, 0.2), noise(8000, 0.15));
    const processed = spectralGate(noisy);
    expect(processed.length).toBe(noisy.length);
    for (const value of processed) {
      expect(Number.isFinite(value)).toBe(true);
    }
  });

  it('returns short input unchanged', () => {
    const tiny = new Float32Array([0.1, -0.1, 0.2]);
    expect(spectralGate(tiny)).toBe(tiny);
  });
});

describe('normalizePeak', () => {
  it('scales the peak to the target', () => {
    const signal = tone(1000, 440, 0.25);
    const processed = normalizePeak(signal, 0.9);
    let peak = 0;
    for (const value of processed) {
      peak = Math.max(peak, Math.abs(value));
    }
    expect(peak).toBeCloseTo(0.9, 5);
  });

  it('returns silent input unchanged', () => {
    const silent = new Float32Array(100);
    expect(normalizePeak(silent)).toBe(silent);
  });
});
