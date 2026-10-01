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

/**
 * Raw 8-bit luma of an alpha plane (limited range, as coded), copied out of its VideoFrame. `data`
 * holds the whole copied frame (a pooled buffer): row r of the luma starts at offset + r · stride.
 */
export interface LumaPlane {
  data: Uint8Array;
  offset: number;
  stride: number;
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
 * Buffers of released luma planes, reused for the next copies (copyTo writes every plane, so each
 * copy needs a whole frame's allocationSize; reusing them avoids that much garbage per frame).
 */
const lumaPool: Uint8Array[] = [];
const LUMA_POOL_MAX = 32;

function takeBuffer(size: number): Uint8Array {
  const i = lumaPool.findIndex((b) => b.byteLength === size);
  return i >= 0 ? lumaPool.splice(i, 1)[0] : new Uint8Array(size);
}

function recycle(buf: Uint8Array): void {
  if (lumaPool.length >= LUMA_POOL_MAX) lumaPool.shift();
  lumaPool.push(buf);
}

/** Give a luma plane's buffer back to the pool; the plane must not be used afterwards. */
export function releaseLuma(luma: LumaPlane | null): void {
  if (luma) recycle(luma.data);
}

/**
 * Copy the Y plane of an 8-bit YUV frame. The browser's own YUV→RGB conversion is not exact enough
 * for alpha (Chrome/Edge map Y=235 to 253, not 255), so alpha is taken from the coded luma and
 * range-mapped in the shader instead. The rows stay at the frame's stride (the upload skips the
 * padding with UNPACK_ROW_LENGTH).
 */
export async function extractLuma(frame: VideoFrame): Promise<LumaPlane> {
  const rect = frame.visibleRect!;
  const buf = takeBuffer(frame.allocationSize());
  let layout: PlaneLayout[];
  try {
    layout = await frame.copyTo(buf);
  } catch (e) {
    recycle(buf);
    throw e;
  }
  const { offset, stride } = layout[0];
  return { data: buf, offset, stride, width: rect.width, height: rect.height, timestamp: frame.timestamp };
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
      releaseLuma(p.alphaLuma);
    }
    this.planes.clear();
  }
}
