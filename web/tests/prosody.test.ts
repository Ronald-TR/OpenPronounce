import { describe, expect, it } from 'vitest';
import { extractF0, extractEnergy, interpolateF0 } from '../src/core/prosody';
import { alignPhonemeTrace } from '../src/core/trace';

function sine(length: number, frequency: number, amplitude: number, sampleRate = 16000): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = amplitude * Math.sin((2 * Math.PI * frequency * i) / sampleRate);
  }
  return out;
}

describe('prosody', () => {
  it('tracks the pitch of a 220 Hz sine', () => {
    const sampleRate = 16000;
    const signal = sine(sampleRate / 2, 220, 0.4, sampleRate);
    const f0 = extractF0(signal, sampleRate);
    const voiced = Array.from(f0).filter((value) => value > 0);
    expect(voiced.length).toBeGreaterThan(f0.length * 0.5);
    const mean = voiced.reduce((sum, value) => sum + value, 0) / voiced.length;
    expect(mean).toBeGreaterThan(200);
    expect(mean).toBeLessThan(240);
  });

  it('returns zeros for silence', () => {
    const f0 = extractF0(new Float32Array(8000));
    for (const value of f0) {
      expect(value).toBe(0);
    }
  });

  it('interpolates unvoiced frames between voiced ones', () => {
    const f0 = new Float32Array([110, 0, 0, 0, 130]);
    const interpolated = interpolateF0(f0);
    expect(interpolated[1]).toBeCloseTo(115, 5);
    expect(interpolated[2]).toBeCloseTo(120, 5);
    expect(interpolated[3]).toBeCloseTo(125, 5);
  });

  it('scales energy to 0-250 with loud frames at the top', () => {
    // half quiet, half loud
    const length = 16000;
    const signal = new Float32Array(length);
    for (let i = length / 2; i < length; i++) {
      signal[i] = 0.5 * Math.sin((2 * Math.PI * 300 * i) / 16000);
    }
    for (let i = 0; i < length / 2; i++) {
      signal[i] = 0.05 * Math.sin((2 * Math.PI * 300 * i) / 16000);
    }
    const energy = Array.from(extractEnergy(signal));
    expect(Math.max(...energy)).toBeCloseTo(250, 5);
    expect(Math.min(...energy)).toBeCloseTo(0, 5);
  });
});

describe('alignPhonemeTrace', () => {
  it('produces equal-length traces along the DTW path', () => {
    const trace = alignPhonemeTrace(['h', 'ʌ', 'l', 'oʊ'], ['h', 'ʌ', 'l', 'oʊ']);
    expect(trace.expected).toEqual(trace.heard);
    expect(trace.expected.length).toBeGreaterThan(0);
  });

  it('handles differing sequences', () => {
    const trace = alignPhonemeTrace(['k', 'æ', 't'], ['k', 'ɑ']);
    expect(trace.expected.length).toBe(trace.heard.length);
    expect(trace.expected.length).toBeGreaterThan(0);
  });

  it('returns empty traces when either side is missing', () => {
    expect(alignPhonemeTrace([], ['a'])).toEqual({ expected: [], heard: [] });
    expect(alignPhonemeTrace(['a'], [])).toEqual({ expected: [], heard: [] });
  });
});
