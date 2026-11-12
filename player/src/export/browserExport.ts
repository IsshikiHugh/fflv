/**
 * Export in the browser, without fflv: a range of composite frames is decoded by a pipeline of its
 * own, drawn by a compositor of its own (on a canvas off the page) with the given layer settings
 * and order, so the result is what the player shows, then encoded with WebCodecs and written as an
 * MP4 (mediabunny). The player goes on playing meanwhile.
 *
 * Like `fflv render`, frames become BT.709 limited-range 4:2:0, and an odd width / height is padded
 * to even by repeating the last column / row. The conversion is done here (not by the encoder) so
 * the matrix is known and written into the file.
 */
import { BufferTarget, EncodedPacket, EncodedVideoPacketSource, Mp4OutputFormat, Output, StreamTarget, type StreamTargetChunk } from 'mediabunny';
import { DecodePipeline } from '../decode/pipeline';
import type { CompositeFrame } from '../decode/frames';
import type { StillLayerMeta } from '../format/lvf';
import type { LvfSource } from '../format/source';
import { ptsUs, type Fps } from '../format/timing';
import { Compositor, type LayerState } from '../render/compositor';

export interface BrowserExportJob {
  source: LvfSource;
  /** Per layer index, as in the player. */
  states: LayerState[];
  /** Layer indices, bottom first. */
  order: number[];
  alphaRangeFix: boolean;
  /** First and last frame, both included. */
  from: number;
  to: number;
  /** Where the MP4 goes: a file chosen with the save picker, or memory (the result's blob). */
  file: FileSystemFileHandle | null;
  signal: AbortSignal;
  onProgress: (done: number, total: number) => void;
}

export interface BrowserExportResult {
  /** The file, when it was kept in memory. */
  blob: Blob | null;
  frames: number;
  /** Frames that could not be decoded (left out, the previous one stays on screen longer). */
  lost: number;
}

/** Whether this browser can encode video at all. */
export const canExportInBrowser = () => typeof VideoEncoder !== 'undefined';

/** A key frame at least this often (seconds). */
const KEY_INTERVAL_S = 2;
/** H.264 / VP9 quantizer when the encoder takes one (about `fflv render`'s crf 18). */
const QP = { avc: 18, vp9: 24 } as const;
/** Frames queued in the encoder before drawing waits. */
const MAX_ENCODE_QUEUE = 4;
/** Packets handed to the muxer and not yet written before encoding waits. */
const MAX_MUX_PENDING = 16;

const BT709: VideoColorSpaceInit = { matrix: 'bt709', primaries: 'bt709', transfer: 'bt709', fullRange: false };

type Codec = 'avc' | 'vp9';
interface EncoderChoice {
  codec: Codec;
  config: VideoEncoderConfig;
  quantizer: boolean;
}

/** H.264 levels: [level_idc, max macroblocks per frame, max macroblocks per second]. */
const AVC_LEVELS: [number, number, number][] = [
  [0x1f, 3600, 108000],
  [0x20, 5120, 216000],
  [0x28, 8192, 245760],
  [0x2a, 8704, 522240],
  [0x32, 22080, 589824],
  [0x33, 36864, 983040],
  [0x34, 36864, 2073600],
  [0x3c, 139264, 4177920],
  [0x3d, 139264, 8355840],
  [0x3e, 139264, 16711680],
];

function avcCodec(w: number, h: number, fps: number): string | null {
  const mbs = Math.ceil(w / 16) * Math.ceil(h / 16);
  const level = AVC_LEVELS.find(([, fs, rate]) => mbs <= fs && mbs * fps <= rate);
  return level ? `avc1.6400${level[0].toString(16).padStart(2, '0')}` : null; // High profile
}

function vp9Codec(w: number, h: number): string {
  const samples = w * h;
  const level = samples <= 2228224 ? '41' : samples <= 8912896 ? '51' : '61';
  return `vp09.00.${level}.08`;
}

/** The first configuration this browser can encode: H.264, else VP9 (both fit in MP4); a quantizer when supported. */
async function pickEncoder(width: number, height: number, fps: number): Promise<EncoderChoice> {
  const base = { width, height, framerate: fps, latencyMode: 'quality' as const };
  const bitrate = Math.round(Math.min(80e6, Math.max(2e6, width * height * fps * 0.2)));
  const codecs: [Codec, string | null][] = [
    ['avc', avcCodec(width, height, fps)],
    ['vp9', vp9Codec(width, height)],
  ];
  for (const [codec, string] of codecs) {
    if (!string) continue;
    const config: VideoEncoderConfig = { ...base, codec: string, ...(codec === 'avc' ? { avc: { format: 'avc' as const } } : {}) };
    for (const quantizer of [true, false]) {
      const c: VideoEncoderConfig = quantizer ? { ...config, bitrateMode: 'quantizer' } : { ...config, bitrate, bitrateMode: 'variable' };
      const res = await VideoEncoder.isConfigSupported(c).catch(() => null);
      if (res?.supported) return { codec, config: c, quantizer };
    }
  }
  throw new Error(`this browser cannot encode H.264 or VP9 video at ${width}x${height}`);
}

/**
 * RGBA rows read back from WebGL (bottom row first, W×H) → I420 of EW×EH (EW, EH even, ≥ W, H),
 * BT.709 limited range; the last column / row is repeated into the padding. Chroma is the average
 * of each 2×2 block.
 */
function rgbaToI420(rgba: Uint8Array, W: number, H: number, EW: number, EH: number, out: Uint8Array): void {
  const cw = EW >> 1;
  const uOff = EW * EH;
  const vOff = uOff + cw * (EH >> 1);
  const px = (x: number, y: number) => ((H - 1 - Math.min(y, H - 1)) * W + Math.min(x, W - 1)) * 4;
  for (let y = 0; y < EH; y += 2) {
    for (let x = 0; x < EW; x += 2) {
      let rs = 0;
      let gs = 0;
      let bs = 0;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const i = px(x + dx, y + dy);
          const r = rgba[i];
          const g = rgba[i + 1];
          const b = rgba[i + 2];
          out[(y + dy) * EW + x + dx] = Math.round(16 + (219 * (0.2126 * r + 0.7152 * g + 0.0722 * b)) / 255);
          rs += r;
          gs += g;
          bs += b;
        }
      }
      const r = rs / 4;
      const g = gs / 4;
      const b = bs / 4;
      const c = (y >> 1) * cw + (x >> 1);
      out[uOff + c] = Math.round(128 + (224 * (-0.114572 * r - 0.385428 * g + 0.5 * b)) / 255);
      out[vOff + c] = Math.round(128 + (224 * (0.5 * r - 0.454153 * g - 0.045847 * b)) / 255);
    }
  }
}

/**
 * The file as a stream for mediabunny. Closing it saves the file only after `commit()`; a cancelled
 * or failed export aborts the stream instead, so no partial video is left in the file.
 */
async function fileStream(handle: FileSystemFileHandle): Promise<{ stream: WritableStream<StreamTargetChunk>; commit: () => void; abort: () => Promise<void> }> {
  const w = await handle.createWritable();
  let committed = false;
  let ended = false;
  const abort = async () => {
    if (ended) return;
    ended = true;
    await w.abort().catch(() => {});
  };
  const stream = new WritableStream<StreamTargetChunk>({
    write: (chunk) => w.write({ type: 'write', data: chunk.data, position: chunk.position }),
    close: async () => {
      if (!committed) return abort();
      ended = true;
      await w.close();
    },
    abort: () => abort(),
  });
  return { stream, commit: () => (committed = true), abort };
}

/** Run an export; rejects with an AbortError when `job.signal` is aborted. */
export async function exportInBrowser(job: BrowserExportJob): Promise<BrowserExportResult> {
  const { source, signal } = job;
  const meta = source.meta;
  const fps: Fps = meta.fps;
  const fpsValue = fps.num / fps.den;
  const W = meta.canvas.width;
  const H = meta.canvas.height;
  const EW = W + (W & 1);
  const EH = H + (H & 1);
  const total = job.to - job.from + 1;
  const base = ptsUs(job.from, fps);
  const keyEvery = Math.max(1, Math.round(KEY_INTERVAL_S * fpsValue));

  const choice = await pickEncoder(EW, EH, fpsValue);
  signal.throwIfAborted();

  const canvas = document.createElement('canvas');
  const compositor = new Compositor(canvas);
  let pipeline: DecodePipeline | null = null;
  let encoder: VideoEncoder | null = null;
  let output: Output | null = null;
  let file: Awaited<ReturnType<typeof fileStream>> | null = null;
  let failure: Error | null = null;
  let wake: (() => void) | null = null;
  const poke = () => {
    const w = wake;
    wake = null;
    w?.();
  };
  // Lost frames, an error or an abort do not always come with a "ready" call: look again now and then.
  const timer = window.setInterval(poke, 50);
  signal.addEventListener('abort', poke);
  const check = () => {
    signal.throwIfAborted();
    if (failure) throw failure;
  };
  /** Resolves when `ready()` is true (checked at every wake-up). */
  const until = async (ready: () => boolean) => {
    for (;;) {
      check();
      if (ready()) return;
      await new Promise<void>((r) => (wake = r));
    }
  };

  try {
    await compositor.load(meta, (i) => source.stillBlob(meta.layers[i] as StillLayerMeta));
    compositor.setOrder(job.order);
    compositor.alphaRangeFix = job.alphaRangeFix;
    const gl = compositor.gl;

    let target: BufferTarget | StreamTarget;
    if (job.file) {
      file = await fileStream(job.file);
      target = new StreamTarget(file.stream, { chunked: true });
    } else {
      target = new BufferTarget();
    }
    output = new Output({ format: new Mp4OutputFormat({ fastStart: job.file ? false : 'in-memory' }), target });
    const track = new EncodedVideoPacketSource(choice.codec);
    output.addVideoTrack(track, { frameRate: fpsValue });
    await output.start();

    let muxPending = 0;
    let mux: Promise<void> = Promise.resolve();
    encoder = new VideoEncoder({
      output: (chunk, md) => {
        if (md?.decoderConfig) md = { ...md, decoderConfig: { ...md.decoderConfig, colorSpace: BT709 } };
        const packet = EncodedPacket.fromEncodedChunk(chunk);
        muxPending++;
        mux = mux
          .then(() => track.add(packet, md))
          .catch((e: Error) => {
            failure ??= e;
          })
          .finally(() => {
            muxPending--;
            poke();
          });
      },
      error: (e) => {
        failure ??= new Error(`encoder: ${e.message}`);
        poke();
      },
    });
    encoder.addEventListener('dequeue', poke);
    encoder.configure(choice.config);

    pipeline = await DecodePipeline.create(source, null);
    pipeline.onReady = poke;
    pipeline.onError = (e) => {
      failure ??= e;
      poke();
    };
    pipeline.seek(job.from);

    const rgba = new Uint8Array(W * H * 4);
    const yuv = new Uint8Array((EW * EH * 3) / 2);
    let done = 0;
    let lost = 0;
    let expected = job.from;
    const p = pipeline;
    const enc = encoder;
    for (;;) {
      await until(() => p.ready.length > 0 || p.finished || !!p.error);
      if (p.error) throw p.error;
      const cf: CompositeFrame | undefined = p.shift();
      if (!cf) break; // finished: the rest of the file could not be decoded
      const f = cf.frameIndex;
      if (f > job.to) {
        cf.close();
        break;
      }
      lost += f - expected;
      expected = f + 1;
      compositor.draw(cf, job.states);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      cf.close();
      rgbaToI420(rgba, W, H, EW, EH, yuv);
      const frame = new VideoFrame(yuv, {
        format: 'I420',
        codedWidth: EW,
        codedHeight: EH,
        timestamp: ptsUs(f, fps) - base,
        duration: ptsUs(f + 1, fps) - ptsUs(f, fps),
        colorSpace: BT709,
      });
      const options: VideoEncoderEncodeOptions = { keyFrame: done % keyEvery === 0 };
      if (choice.quantizer) Object.assign(options, choice.codec === 'avc' ? { avc: { quantizer: QP.avc } } : { vp9: { quantizer: QP.vp9 } });
      enc.encode(frame, options);
      frame.close();
      done++;
      job.onProgress(done + lost, total);
      if (f >= job.to) break;
      await until(() => enc.encodeQueueSize <= MAX_ENCODE_QUEUE && muxPending <= MAX_MUX_PENDING);
    }
    lost += job.to + 1 - expected;
    if (done === 0) throw new Error('no frame of the range could be decoded');
    await enc.flush();
    await mux;
    check();
    file?.commit();
    await output.finalize();
    const blob = target instanceof BufferTarget && target.buffer ? new Blob([target.buffer], { type: 'video/mp4' }) : null;
    return { blob, frames: done, lost };
  } catch (e) {
    if (output && output.state !== 'finalized' && output.state !== 'canceled') await output.cancel().catch(() => {});
    await file?.abort();
    throw e;
  } finally {
    window.clearInterval(timer);
    signal.removeEventListener('abort', poke);
    pipeline?.dispose();
    if (encoder && encoder.state !== 'closed') encoder.close();
    compositor.release();
    compositor.gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
