/**
 * LVF v1 binary structures (LVF_SPEC.md sections 3–7). All little-endian.
 * Parsing is strict: anything that does not match the spec throws LvfFormatError,
 * because a composite frame we cannot trust is one we must not display.
 */
import type { Fps } from './timing';

export const HEADER_SIZE = 64;
export const CAU_HEADER_SIZE = 20;
export const VIDEO_ENTRY_HEADER_SIZE = 12;
export const AUDIO_PACKET_HEADER_SIZE = 16;
export const INDEX_HEADER_SIZE = 8;
export const INDEX_ENTRY_SIZE = 16;

export const ENTRY_EMPTY = 0;
export const ENTRY_FRAME = 1;
export const CAU_FLAG_RAP = 1;
export const FRAME_FLAG_KEY = 1;

export class LvfFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LvfFormatError';
  }
}

const ascii = (u8: Uint8Array, off: number, len: number) => String.fromCharCode(...u8.subarray(off, off + len));

function u64(dv: DataView, off: number): number {
  const v = dv.getBigUint64(off, true);
  if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new LvfFormatError(`64-bit offset ${v} is out of range`);
  return Number(v);
}

// ---------------------------------------------------------------------------------------------
// File header
// ---------------------------------------------------------------------------------------------
export interface LvfHeader {
  version: number;
  flags: number;
  metaOffset: number;
  metaLength: number;
  resourcesOffset: number;
  cauOffset: number;
  indexOffset: number;
}

export function parseHeader(buf: ArrayBuffer, fileSize: number): LvfHeader {
  if (buf.byteLength < HEADER_SIZE) throw new LvfFormatError('file is shorter than the 64-byte header');
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  if (ascii(u8, 0, 4) !== 'LVF1') throw new LvfFormatError('not an LVF file (magic is not "LVF1")');
  const h: LvfHeader = {
    version: dv.getUint16(4, true),
    flags: dv.getUint16(6, true),
    metaOffset: u64(dv, 8),
    metaLength: dv.getUint32(16, true),
    resourcesOffset: u64(dv, 24),
    cauOffset: u64(dv, 32),
    indexOffset: u64(dv, 40),
  };
  if (h.version !== 1) throw new LvfFormatError(`unsupported LVF version ${h.version}`);
  if (
    h.metaOffset < HEADER_SIZE ||
    h.metaOffset + h.metaLength > h.resourcesOffset ||
    h.resourcesOffset > h.cauOffset ||
    h.cauOffset > h.indexOffset ||
    h.indexOffset > fileSize
  ) {
    throw new LvfFormatError('file header offsets are inconsistent (truncated or corrupt file?)');
  }
  return h;
}

// ---------------------------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------------------------
export type BlendMode = 'normal' | 'add' | 'multiply' | 'screen';

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface LayerBase {
  id: string;
  name: string;
  z: number;
  rect: Rect;
  start_frame: number;
  end_frame: number;
  blend: BlendMode;
  opacity: number;
  visible: boolean;
}

export interface VideoLayerMeta extends LayerBase {
  kind: 'video';
  codec: string;
  coded_width: number;
  coded_height: number;
  has_alpha: boolean;
  alpha_codec: string | null;
  /** Lossless layer: profile-1 RGB color plane, full-range alpha (appendix B). */
  lossless?: boolean;
  /** How alpha is coded in the luma of the alpha plane (default "limited": Y 16..235). */
  alpha_range?: 'limited' | 'full';
  /** Valid pixels in the coded frame when it was padded to even size (default: the coded size). */
  content_size?: [number, number];
}

export interface StillLayerMeta extends LayerBase {
  kind: 'still';
  resource: { offset: number; length: number; mime: string };
}

export type LayerMeta = VideoLayerMeta | StillLayerMeta;

export interface AudioMeta {
  codec: 'opus';
  sample_rate: number;
  channels: number;
  description_b64: string | null;
  /** Opus pre-skip in 48 kHz samples (LVF addition, see LVF_SPEC.md appendix A). */
  pre_skip?: number;
}

export interface LvfMeta {
  format: 'LVF';
  version: 1;
  generator?: string;
  canvas: { width: number; height: number; background: string };
  fps: Fps;
  frame_count: number;
  max_rap_interval: number;
  layers: LayerMeta[];
  audio: AudioMeta | null;
}

const isInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v);

export function parseMeta(bytes: Uint8Array, resourcesSize: number): LvfMeta {
  let m: LvfMeta;
  try {
    m = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch (e) {
    throw new LvfFormatError(`metadata is not valid UTF-8 JSON: ${(e as Error).message}`);
  }
  const fail = (msg: string): never => {
    throw new LvfFormatError(`metadata: ${msg}`);
  };
  if (m?.format !== 'LVF' || m.version !== 1) fail('format/version must be "LVF"/1');
  if (!isInt(m.canvas?.width) || !isInt(m.canvas?.height) || m.canvas.width <= 0 || m.canvas.height <= 0)
    fail('canvas width/height must be positive integers');
  if (!/^#[0-9a-fA-F]{6}$/.test(m.canvas.background ?? '')) fail('canvas.background must be #RRGGBB');
  if (!isInt(m.fps?.num) || !isInt(m.fps?.den) || m.fps.num <= 0 || m.fps.den <= 0) fail('fps must be {num, den}');
  if (!isInt(m.frame_count) || m.frame_count <= 0) fail('frame_count must be a positive integer');
  if (!isInt(m.max_rap_interval) || m.max_rap_interval <= 0) fail('max_rap_interval must be positive');
  if (!Array.isArray(m.layers)) fail('layers must be an array');
  m.layers.forEach((L, i) => {
    const tag = `layer ${i} (${L?.id})`;
    if (L.kind !== 'video' && L.kind !== 'still') fail(`${tag}: unknown kind ${(L as { kind: unknown }).kind}`);
    if (!isInt(L.start_frame) || !isInt(L.end_frame) || !(0 <= L.start_frame && L.start_frame < L.end_frame && L.end_frame <= m.frame_count))
      fail(`${tag}: bad frame interval`);
    if (!L.rect || ![L.rect.x, L.rect.y, L.rect.w, L.rect.h].every(isInt) || L.rect.w <= 0 || L.rect.h <= 0)
      fail(`${tag}: bad rect`);
    if (!['normal', 'add', 'multiply', 'screen'].includes(L.blend)) fail(`${tag}: bad blend mode ${L.blend}`);
    if (typeof L.opacity !== 'number' || L.opacity < 0 || L.opacity > 1) fail(`${tag}: opacity must be in [0, 1]`);
    if (typeof L.visible !== 'boolean') fail(`${tag}: visible must be boolean`);
    if (typeof L.z !== 'number') fail(`${tag}: z must be a number`);
    if (L.kind === 'video') {
      if (typeof L.codec !== 'string' || !isInt(L.coded_width) || !isInt(L.coded_height)) fail(`${tag}: bad codec info`);
      if (L.has_alpha && typeof L.alpha_codec !== 'string') fail(`${tag}: has_alpha without alpha_codec`);
      if (L.alpha_range !== undefined && L.alpha_range !== 'limited' && L.alpha_range !== 'full') fail(`${tag}: bad alpha_range`);
      const cs = L.content_size;
      if (cs !== undefined && !(Array.isArray(cs) && cs.length === 2 && cs.every(isInt) && cs[0] > 0 && cs[1] > 0 && cs[0] <= L.coded_width && cs[1] <= L.coded_height))
        fail(`${tag}: bad content_size`);
    } else {
      const r = L.resource;
      if (!r || !isInt(r.offset) || !isInt(r.length) || r.offset < 0 || r.offset + r.length > resourcesSize)
        fail(`${tag}: resource lies outside the resource region`);
    }
  });
  if (m.audio !== null) {
    if (m.audio?.codec !== 'opus' || m.audio.sample_rate !== 48000 || !isInt(m.audio.channels)) fail('audio must be 48 kHz Opus or null');
  }
  return m;
}

export function videoLayerIndices(meta: LvfMeta): number[] {
  return meta.layers.flatMap((L, i) => (L.kind === 'video' ? [i] : []));
}

export function isActive(L: LayerMeta, frame: number): boolean {
  return L.start_frame <= frame && frame < L.end_frame;
}

// ---------------------------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------------------------
export class IndexTable {
  constructor(
    readonly offsets: Float64Array,
    readonly rap: Uint8Array,
    readonly rapFrames: Int32Array,
    readonly endOffset: number,
  ) {}

  get count(): number {
    return this.offsets.length;
  }

  /** Byte range [start, end) of composite frame f. */
  range(f: number): [number, number] {
    return [this.offsets[f], f + 1 < this.offsets.length ? this.offsets[f + 1] : this.endOffset];
  }

  /** The nearest RAP at or before frame f. */
  rapAtOrBefore(f: number): number {
    const r = this.rapFrames;
    let lo = 0;
    let hi = r.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (r[mid] <= f) lo = mid;
      else hi = mid - 1;
    }
    return r[lo];
  }
}

export function parseIndex(buf: ArrayBuffer, header: LvfHeader, frameCount: number): IndexTable {
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  if (buf.byteLength < INDEX_HEADER_SIZE || ascii(u8, 0, 4) !== 'IDX1') throw new LvfFormatError('index table magic is not "IDX1"');
  const count = dv.getUint32(4, true);
  if (count !== frameCount) throw new LvfFormatError(`index has ${count} entries, frame_count is ${frameCount}`);
  if (buf.byteLength < INDEX_HEADER_SIZE + count * INDEX_ENTRY_SIZE) throw new LvfFormatError('index table is truncated');
  const offsets = new Float64Array(count);
  const rap = new Uint8Array(count);
  const raps: number[] = [];
  let prev = header.cauOffset - 1;
  for (let i = 0; i < count; i++) {
    const p = INDEX_HEADER_SIZE + i * INDEX_ENTRY_SIZE;
    if (dv.getUint32(p, true) !== i) throw new LvfFormatError(`index entry ${i} has frame_index ${dv.getUint32(p, true)}`);
    rap[i] = dv.getUint8(p + 4) & CAU_FLAG_RAP;
    offsets[i] = u64(dv, p + 8);
    if (offsets[i] <= prev || offsets[i] + CAU_HEADER_SIZE > header.indexOffset)
      throw new LvfFormatError(`index entry ${i} points outside the composite-frame region`);
    prev = offsets[i];
    if (rap[i]) raps.push(i);
  }
  if (offsets[0] !== header.cauOffset) throw new LvfFormatError('index entry 0 does not point at the first composite frame');
  if (!rap[0]) throw new LvfFormatError('frame 0 is not a RAP (I7)');
  return new IndexTable(offsets, rap, Int32Array.from(raps), header.indexOffset);
}

// ---------------------------------------------------------------------------------------------
// Composite frame (CAU)
// ---------------------------------------------------------------------------------------------
export interface VideoEntry {
  layerIndex: number;
  type: number;
  key: boolean;
  color: Uint8Array;
  alpha: Uint8Array | null;
}

export interface AudioPacket {
  ptsUs: number;
  durationUs: number;
  data: Uint8Array;
}

export interface ParsedCau {
  frameIndex: number;
  rap: boolean;
  entries: VideoEntry[];
  audio: AudioPacket[];
  byteLength: number;
}

/**
 * Parse the composite frame occupying exactly `u8[start, end)` and check it against what the index
 * and metadata promise (frame number, one entry per video layer in order, I3 activity).
 */
export function parseCau(u8: Uint8Array, start: number, end: number, expectFrame: number, meta: LvfMeta, videoLayers: number[]): ParsedCau {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const bad = (msg: string): never => {
    throw new LvfFormatError(`composite frame ${expectFrame}: ${msg}`);
  };
  if (end - start < CAU_HEADER_SIZE) bad('truncated header');
  if (ascii(u8, start, 4) !== 'CAUF') bad('bad magic');
  const payload = dv.getUint32(start + 4, true);
  if (start + 8 + payload !== end) bad(`payload_size ${payload} disagrees with the index`);
  const frameIndex = dv.getUint32(start + 8, true);
  if (frameIndex !== expectFrame) bad(`frame_index is ${frameIndex}`);
  const flags = dv.getUint8(start + 12);
  const nVideo = dv.getUint16(start + 14, true);
  const nAudio = dv.getUint16(start + 16, true);
  if (nVideo !== videoLayers.length) bad(`${nVideo} video entries, expected ${videoLayers.length} (I2)`);
  let p = start + CAU_HEADER_SIZE;
  const entries: VideoEntry[] = [];
  for (let k = 0; k < nVideo; k++) {
    if (p + VIDEO_ENTRY_HEADER_SIZE > end) bad('video entry overruns the frame');
    const layerIndex = dv.getUint16(p, true);
    const type = dv.getUint8(p + 2);
    const fflags = dv.getUint8(p + 3);
    const colorLen = dv.getUint32(p + 4, true);
    const alphaLen = dv.getUint32(p + 8, true);
    p += VIDEO_ENTRY_HEADER_SIZE;
    if (p + colorLen + alphaLen > end) bad(`layer ${layerIndex} data overruns the frame`);
    if (layerIndex !== videoLayers[k]) bad(`entry ${k} is for layer ${layerIndex}, expected ${videoLayers[k]} (I2)`);
    const L = meta.layers[layerIndex] as VideoLayerMeta;
    const active = isActive(L, frameIndex);
    if (type !== ENTRY_EMPTY && type !== ENTRY_FRAME) bad(`layer ${layerIndex} has unsupported entry type ${type}`);
    if ((type === ENTRY_FRAME) !== active) bad(`layer ${layerIndex} entry type ${type} contradicts its active range (I3)`);
    if (type === ENTRY_FRAME && (colorLen === 0 || (L.has_alpha ? alphaLen === 0 : alphaLen !== 0)))
      bad(`layer ${layerIndex} planes do not match has_alpha`);
    entries.push({
      layerIndex,
      type,
      key: (fflags & FRAME_FLAG_KEY) !== 0,
      color: u8.subarray(p, p + colorLen),
      alpha: alphaLen ? u8.subarray(p + colorLen, p + colorLen + alphaLen) : null,
    });
    p += colorLen + alphaLen;
  }
  const audio: AudioPacket[] = [];
  for (let k = 0; k < nAudio; k++) {
    if (p + AUDIO_PACKET_HEADER_SIZE > end) bad('audio packet overruns the frame');
    const ptsUs = Number(dv.getBigInt64(p, true));
    const durationUs = dv.getUint32(p + 8, true);
    const len = dv.getUint32(p + 12, true);
    p += AUDIO_PACKET_HEADER_SIZE;
    if (p + len > end) bad('audio data overruns the frame');
    audio.push({ ptsUs, durationUs, data: u8.subarray(p, p + len) });
    p += len;
  }
  if (p !== end) bad(`${end - p} unaccounted bytes`);
  return { frameIndex, rap: (flags & CAU_FLAG_RAP) !== 0, entries, audio, byteLength: end - start };
}
