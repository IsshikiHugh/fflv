/**
 * VideoFrame bookkeeping. Every VideoFrame that enters the player goes through `adopt` and leaves
 * through `release`, so the number of live frames is always known (debug panel, leak test).
 */
export const frameStats = {
  live: 0,
  peak: 0,
  adopted: 0,
};

export function adopt(frame: VideoFrame): VideoFrame {
  frameStats.live++;
  frameStats.adopted++;
  if (frameStats.live > frameStats.peak) frameStats.peak = frameStats.live;
  return frame;
}

export function release(frame: VideoFrame | null | undefined): void {
  if (!frame) return;
  frame.close();
  frameStats.live--;
}

/** Raw 8-bit luma of an alpha plane (limited range, as coded), copied out of its VideoFrame. */
export interface LumaPlane {
  data: Uint8Array;
  width: number;
  height: number;
  timestamp: number;
}

export interface LayerPlanes {
  color: VideoFrame | null;
  /** Alpha plane kept as a VideoFrame (only when its pixel format cannot be copied as luma). */
  alpha: VideoFrame | null;
  alphaLuma: LumaPlane | null;
}

const LUMA_FIRST_FORMATS = new Set(['I420', 'I420A', 'I422', 'I444', 'NV12']);

export function hasCopyableLuma(frame: VideoFrame): boolean {
  return frame.format !== null && LUMA_FIRST_FORMATS.has(frame.format);
}

/**
 * Copy the Y plane of an 8-bit YUV frame. The browser's own YUV→RGB conversion is not exact enough
 * for alpha (Chrome/Edge map Y=235 to 253, not 255), so alpha is taken from the coded luma and
 * range-mapped in the shader instead.
 */
export async function extractLuma(frame: VideoFrame): Promise<LumaPlane> {
  const rect = frame.visibleRect!;
  const w = rect.width;
  const h = rect.height;
  const buf = new Uint8Array(frame.allocationSize());
  const layout = await frame.copyTo(buf);
  const { offset, stride } = layout[0];
  let data: Uint8Array;
  if (stride === w) {
    data = buf.subarray(offset, offset + w * h);
  } else {
    data = new Uint8Array(w * h);
    for (let r = 0; r < h; r++) data.set(buf.subarray(offset + r * stride, offset + r * stride + w), r * w);
  }
  return { data, width: w, height: h, timestamp: frame.timestamp };
}

/**
 * A complete composite frame: every plane of every FRAME entry has been decoded (P1).
 * It is displayed, and released, only as a whole (P2, P4).
 */
export class CompositeFrame {
  private closed = false;

  constructor(
    readonly frameIndex: number,
    readonly ptsUs: number,
    readonly planes: Map<number, LayerPlanes>,
  ) {}

  get isClosed(): boolean {
    return this.closed;
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const p of this.planes.values()) {
      release(p.color);
      release(p.alpha);
    }
    this.planes.clear();
  }
}
