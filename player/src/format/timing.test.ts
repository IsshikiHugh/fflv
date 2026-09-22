import { describe, expect, it } from 'vitest';
import { formatTime, frameAtUs, ptsUs } from './timing';

describe('ptsUs', () => {
  it('matches the reference values of crates/lvf/src/timing.rs', () => {
    expect([0, 1, 2, 3].map((f) => ptsUs(f, { num: 30, den: 1 }))).toEqual([0, 33333, 66667, 100000]);
    expect(ptsUs(1, { num: 30000, den: 1001 })).toBe(33367);
    expect(ptsUs(2, { num: 30000, den: 1001 })).toBe(66733);
    expect(ptsUs(30000, { num: 30000, den: 1001 })).toBe(1_001_000_000);
  });
  it('rounds half up', () => {
    expect(ptsUs(1, { num: 128, den: 1 })).toBe(7813);
    expect(ptsUs(3, { num: 128, den: 1 })).toBe(23438);
  });
  it('stays exact beyond 2^53 intermediate products', () => {
    // 100 hours of 30000/1001 video: f * 1e6 * den overflows doubles, BigInt does not.
    const f = 10_789_200;
    expect(ptsUs(f, { num: 30000, den: 1001 })).toBe(Number((2n * BigInt(f) * 1_000_000n * 1001n + 30000n) / 60000n));
  });
});

describe('frameAtUs', () => {
  const fps = { num: 30, den: 1 };
  it('returns the last frame whose pts <= t', () => {
    expect(frameAtUs(0, fps)).toBe(0);
    expect(frameAtUs(33332, fps)).toBe(0);
    expect(frameAtUs(33333, fps)).toBe(1);
    expect(frameAtUs(66666, fps)).toBe(1);
    expect(frameAtUs(66667, fps)).toBe(2);
    expect(frameAtUs(-5, fps)).toBe(0);
  });
  it('is consistent with ptsUs for NTSC rates', () => {
    const ntsc = { num: 30000, den: 1001 };
    for (let f = 0; f < 5000; f += 7) {
      expect(frameAtUs(ptsUs(f, ntsc), ntsc)).toBe(f);
      expect(frameAtUs(ptsUs(f + 1, ntsc) - 1, ntsc)).toBe(f);
    }
  });
});

describe('formatTime', () => {
  it('formats mm:ss.mmm and h:mm:ss.mmm', () => {
    expect(formatTime(0)).toBe('00:00.000');
    expect(formatTime(61_234_000)).toBe('01:01.234');
    expect(formatTime(3_600_000_000)).toBe('1:00:00.000');
  });
});
