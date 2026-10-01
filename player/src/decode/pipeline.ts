/**
 * Reader → dispatcher → per-plane decoders → composite-frame assembler → ready queue (spec 9.2).
 *
 *  - Composite frames are read ahead (compressed) in bounded batches; their audio packets go to the
 *    audio sink immediately, so audio is decoded well ahead of the playhead. Reading starts when
 *    the read-ahead falls to half of its limit and then refills it, so steady playback makes a few
 *    large range requests instead of one per frame.
 *  - Every FRAME entry of a composite frame is sent to its layer's decoders, hidden or not (P5).
 *  - A composite frame becomes "ready" only when every plane it needs has been decoded (P1), and
 *    frames are released strictly in frame order.
 *  - Back-pressure: ready + in-flight composite frames are capped (spec 9.4).
 */
import { CompositeFrame, extractLuma, hasCopyableLuma, release, releaseLuma, type LayerPlanes } from './frames';
import { PlaneDecoder, type PlaneKind, type PlaneSink } from './planeDecoder';
import { ENTRY_FRAME, type AudioPacket, type ParsedCau, type VideoLayerMeta } from '../format/lvf';
import type { LvfSource } from '../format/source';
import { ptsUs } from '../format/timing';

export interface AudioSink {
  feed(packet: AudioPacket): void;
  /** Drop everything queued; the next packets start a new (seeked) stream at `targetUs`. */
  reset(targetUs: number): void;
}

export interface PipelineOptions {
  /** ready + in-flight composite frames (spec 9.4 suggests 8). */
  maxQueued: number;
  /** In-flight limit while nothing is ready, for decoders that hold a few frames before output. */
  hardCap: number;
  /** Per-decoder decodeQueueSize above which dispatch pauses. */
  maxDecodeQueue: number;
  readAheadFrames: number;
  readAheadBytes: number;
  batchBytes: number;
  hardwareAcceleration: HardwareAcceleration;
  /** Testing aid: extra latency per read, to simulate slow storage. */
  readDelayMs: number;
}

export const defaultPipelineOptions: PipelineOptions = {
  maxQueued: 8,
  hardCap: 16,
  maxDecodeQueue: 8,
  readAheadFrames: 45,
  readAheadBytes: 32 << 20,
  batchBytes: 4 << 20,
  hardwareAcceleration: 'no-preference',
  readDelayMs: 0,
};

interface Pending {
  remaining: number;
  broken: boolean;
  planes: Map<number, LayerPlanes>;
}

export interface PipelineStats {
  released: number;
  prerollDiscarded: number;
  lostFrames: number;
  staleOutputs: number;
}

export class DecodePipeline implements PlaneSink {
  readonly ready: CompositeFrame[] = [];
  readonly stats: PipelineStats = { released: 0, prerollDiscarded: 0, lostFrames: 0, staleOutputs: 0 };
  error: Error | null = null;
  onReady: () => void = () => {};
  onError: (e: Error) => void = () => {};

  private readonly decoders = new Map<number, { color: PlaneDecoder; alpha: PlaneDecoder | null }>();
  private pending = new Map<number, Pending>();
  private parsed: ParsedCau[] = [];
  private parsedBytes = 0;
  private readPos = 0;
  private nextRelease = 0;
  private discardBefore = 0;
  private gen = 0;
  /** Generation whose read loop is running (a superseded loop is aborted, not waited for). */
  private readingGen = -1;
  private abort = new AbortController();
  private disposed = false;

  private constructor(
    private readonly source: LvfSource,
    private readonly audio: AudioSink | null,
    private readonly opts: PipelineOptions,
  ) {}

  /** Checks every decoder configuration with isConfigSupported before creating anything (9.4). */
  static async create(source: LvfSource, audio: AudioSink | null, opts: Partial<PipelineOptions> = {}): Promise<DecodePipeline> {
    const o = { ...defaultPipelineOptions, ...opts };
    const configs: { layer: number; plane: PlaneKind; config: VideoDecoderConfig }[] = [];
    for (const li of source.videoLayers) {
      const L = source.meta.layers[li] as VideoLayerMeta;
      const base = { codedWidth: L.coded_width, codedHeight: L.coded_height, hardwareAcceleration: o.hardwareAcceleration, optimizeForLatency: true };
      configs.push({ layer: li, plane: 'color', config: { ...base, codec: L.codec } });
      if (L.has_alpha) configs.push({ layer: li, plane: 'alpha', config: { ...base, codec: L.alpha_codec! } });
    }
    for (const c of configs) {
      const res = await VideoDecoder.isConfigSupported(c.config);
      if (!res.supported) {
        const L = source.meta.layers[c.layer];
        throw new Error(`layer ${c.layer} (${L.id}) ${c.plane}: this browser cannot decode ${c.config.codec} at ${c.config.codedWidth}x${c.config.codedHeight}`);
      }
    }
    const p = new DecodePipeline(source, audio, o);
    for (const c of configs) {
      const dec = new PlaneDecoder(c.layer, c.plane, c.config, source.meta.fps, p);
      const slot = p.decoders.get(c.layer) ?? { color: dec, alpha: null };
      if (c.plane === 'alpha') slot.alpha = dec;
      p.decoders.set(c.layer, slot);
    }
    return p;
  }

  // ----------------------------------------------------------------------------------------------
  // Consumer API (render loop)
  // ----------------------------------------------------------------------------------------------
  peek(): CompositeFrame | undefined {
    return this.ready[0];
  }

  shift(): CompositeFrame | undefined {
    const f = this.ready.shift();
    this.kick();
    return f;
  }

  get inFlight(): number {
    return this.pending.size;
  }

  get parsedCount(): number {
    return this.parsed.length;
  }

  /** Every composite frame up to the end of the file has been released (or discarded). */
  get finished(): boolean {
    return this.readPos >= this.source.frameCount && this.parsed.length === 0 && this.pending.size === 0;
  }

  /** True when back-pressure, not decoding speed, is what holds the pipeline back. */
  get saturated(): boolean {
    return this.ready.length + this.pending.size >= this.opts.maxQueued;
  }

  decodeQueueSizes(): number[] {
    const out: number[] = [];
    for (const d of this.decoders.values()) {
      out.push(d.color.queueSize);
      if (d.alpha) out.push(d.alpha.queueSize);
    }
    return out;
  }

  /**
   * Restart decoding so that `target` is the first frame to become ready (spec 9.7): drop every
   * queued frame, reset + reconfigure all decoders, reset audio, and start reading at the nearest
   * RAP at or before `target`; complete frames before `target` are discarded as preroll.
   */
  seek(target: number): void {
    this.newGeneration();
    for (const f of this.ready) f.close();
    this.ready.length = 0;
    for (const p of this.pending.values()) closePlanes(p.planes);
    this.pending.clear();
    this.parsed = [];
    this.parsedBytes = 0;
    this.error = null;
    for (const d of this.decoders.values()) {
      d.color.reset();
      d.alpha?.reset();
    }
    this.audio?.reset(ptsUs(target, this.source.meta.fps));
    const rap = this.source.index.rapAtOrBefore(target);
    this.readPos = rap;
    this.nextRelease = rap;
    this.discardBefore = target;
    this.kick();
  }

  dispose(): void {
    this.disposed = true;
    this.newGeneration();
    for (const f of this.ready) f.close();
    this.ready.length = 0;
    for (const p of this.pending.values()) closePlanes(p.planes);
    this.pending.clear();
    for (const d of this.decoders.values()) {
      d.color.close();
      d.alpha?.close();
    }
  }

  /** Start a new generation: outputs, reads and callbacks of the previous one are ignored. */
  private newGeneration(): void {
    this.gen++;
    this.abort.abort();
    this.abort = new AbortController();
  }

  // ----------------------------------------------------------------------------------------------
  // Reading and dispatch
  // ----------------------------------------------------------------------------------------------
  kick(): void {
    if (this.disposed || this.error) return;
    this.feed();
    if (this.readingGen !== this.gen && this.readPos < this.source.frameCount && this.readAheadLow()) void this.readLoop(this.gen);
  }

  private readAheadFull(): boolean {
    return this.parsed.length >= this.opts.readAheadFrames || this.parsedBytes >= this.opts.readAheadBytes;
  }

  /** Low-water mark: below it a read starts (and goes on until the read-ahead is full). */
  private readAheadLow(): boolean {
    return this.parsed.length <= this.opts.readAheadFrames >> 1 && this.parsedBytes <= this.opts.readAheadBytes / 2;
  }

  private async readLoop(gen: number): Promise<void> {
    this.readingGen = gen;
    const signal = this.abort.signal;
    try {
      while (gen === this.gen && !this.error && this.readPos < this.source.frameCount && !this.readAheadFull()) {
        const want = this.opts.readAheadFrames - this.parsed.length;
        if (this.opts.readDelayMs) await new Promise((r) => setTimeout(r, this.opts.readDelayMs));
        if (gen !== this.gen) break;
        const batch = await this.source.readCaus(this.readPos, want, this.opts.batchBytes, signal);
        if (gen !== this.gen) break;
        for (const cau of batch) {
          this.parsed.push(cau);
          this.parsedBytes += cau.byteLength;
          for (const a of cau.audio) this.audio?.feed(a);
        }
        this.readPos += batch.length;
        this.feed();
      }
    } catch (e) {
      if (gen === this.gen) this.fail(e as Error); // a superseded read ends with an AbortError
    } finally {
      if (this.readingGen === gen) this.readingGen = -1;
    }
  }

  private canDispatch(): boolean {
    const limit = this.ready.length === 0 ? this.opts.hardCap : this.opts.maxQueued;
    if (this.ready.length + this.pending.size >= limit) return false;
    for (const d of this.decoders.values()) {
      if (d.color.queueSize >= this.opts.maxDecodeQueue) return false;
      if (d.alpha && d.alpha.queueSize >= this.opts.maxDecodeQueue) return false;
    }
    return true;
  }

  private feed(): void {
    while (this.parsed.length && !this.error && this.canDispatch()) {
      const cau = this.parsed.shift()!;
      this.parsedBytes -= cau.byteLength;
      this.dispatch(cau);
    }
  }

  private dispatch(cau: ParsedCau): void {
    const f = cau.frameIndex;
    const pend: Pending = { remaining: 0, broken: false, planes: new Map() };
    this.pending.set(f, pend);
    try {
      for (const e of cau.entries) {
        if (e.type !== ENTRY_FRAME) continue;
        const d = this.decoders.get(e.layerIndex)!;
        pend.remaining++;
        d.color.decode(f, e.key, e.color);
        if (d.alpha) {
          pend.remaining++;
          d.alpha.decode(f, e.key, e.alpha!);
        }
        // The layer's last frame (at the end of the file too): no more input follows to push
        // frames a decoder holds back out of it, so flush. Its next chunk comes after a seek
        // (reset), which starts with a key frame.
        if (f === this.source.meta.layers[e.layerIndex].end_frame - 1) {
          d.color.flush();
          d.alpha?.flush();
        }
      }
    } catch (err) {
      this.fail(err as Error);
      return;
    }
    if (pend.remaining === 0) this.releaseComplete();
  }

  // ----------------------------------------------------------------------------------------------
  // Assembly (PlaneSink)
  // ----------------------------------------------------------------------------------------------
  onFrame(layer: number, plane: PlaneKind, frameIndex: number, frame: VideoFrame): void {
    const pend = this.pending.get(frameIndex);
    if (!pend) {
      release(frame);
      this.stats.staleOutputs++;
      return;
    }
    let slot = pend.planes.get(layer);
    if (!slot) {
      slot = { color: null, alpha: null, alphaLuma: null };
      pend.planes.set(layer, slot);
    }
    if (plane === 'alpha' && hasCopyableLuma(frame)) {
      // The plane counts as decoded once its luma is copied out (P1); the VideoFrame goes back to
      // the decoder right away.
      const s = slot;
      extractLuma(frame).then(
        (luma) => {
          release(frame);
          if (this.pending.get(frameIndex) !== pend || s.alphaLuma) return releaseLuma(luma); // seeked away meanwhile
          s.alphaLuma = luma;
          if (--pend.remaining === 0) this.releaseComplete();
        },
        (err) => {
          release(frame);
          console.error('[lvf] alpha copy failed:', err);
          if (this.pending.get(frameIndex) !== pend) return;
          pend.broken = true;
          if (--pend.remaining === 0) this.releaseComplete();
        },
      );
      return;
    }
    if (slot[plane]) {
      release(slot[plane]);
      this.stats.staleOutputs++;
    } else {
      pend.remaining--;
    }
    slot[plane] = frame;
    if (pend.remaining === 0) this.releaseComplete();
  }

  onLost(_layer: number, _plane: PlaneKind, frameIndex: number): void {
    const pend = this.pending.get(frameIndex);
    if (!pend) return;
    pend.broken = true;
    pend.remaining--;
    if (pend.remaining === 0) this.releaseComplete();
  }

  onStale(): void {
    this.stats.staleOutputs++;
  }

  onDecodeError(layer: number, plane: PlaneKind, error: Error): void {
    const L = this.source.meta.layers[layer];
    this.fail(new Error(`decoder error in layer ${layer} (${L.id}) ${plane}: ${error.message}`));
  }

  onDequeue(): void {
    this.feed();
  }

  /** Move complete composite frames, in frame order, to the ready queue. */
  private releaseComplete(): void {
    let added = false;
    for (;;) {
      const pend = this.pending.get(this.nextRelease);
      if (!pend || pend.remaining > 0) break;
      const f = this.nextRelease++;
      this.pending.delete(f);
      const cf = new CompositeFrame(f, ptsUs(f, this.source.meta.fps), pend.planes);
      if (pend.broken) {
        cf.close(); // a plane never arrived: drop the whole composite frame, never a partial one
        this.stats.lostFrames++;
      } else if (f < this.discardBefore) {
        cf.close();
        this.stats.prerollDiscarded++;
      } else {
        this.ready.push(cf);
        this.stats.released++;
        added = true;
      }
    }
    this.kick();
    if (added) this.onReady();
  }

  private fail(e: Error): void {
    if (this.error) return;
    this.error = e;
    console.error('[lvf] pipeline error:', e);
    this.onError(e);
  }
}

function closePlanes(planes: Map<number, LayerPlanes>): void {
  for (const p of planes.values()) {
    release(p.color);
    release(p.alpha);
    releaseLuma(p.alphaLuma);
  }
  planes.clear();
}
