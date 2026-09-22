/**
 * Frame ↔ time arithmetic. Must agree bit-for-bit with crates/lvf/src/timing.rs:
 * pts_us(f) = round(f * 1_000_000 * den / num), rounding half up, in exact integer math.
 */
export interface Fps {
  num: number;
  den: number;
}

export function ptsUs(frame: number, fps: Fps): number {
  const n = BigInt(fps.num);
  return Number((2n * BigInt(frame) * 1_000_000n * BigInt(fps.den) + n) / (2n * n));
}

/** The last frame whose pts is <= us (clamped to >= 0). */
export function frameAtUs(us: number, fps: Fps): number {
  if (us <= 0) return 0;
  let f = Math.floor((us * fps.num) / (1_000_000 * fps.den));
  while (f > 0 && ptsUs(f, fps) > us) f--;
  while (ptsUs(f + 1, fps) <= us) f++;
  return f;
}

export function formatTime(us: number): string {
  const totalMs = Math.max(0, Math.round(us / 1000));
  const ms = totalMs % 1000;
  const s = Math.floor(totalMs / 1000) % 60;
  const m = Math.floor(totalMs / 60000) % 60;
  const h = Math.floor(totalMs / 3600000);
  const mmss = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
  return h > 0 ? `${h}:${mmss}` : mmss;
}
