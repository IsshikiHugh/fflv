/**
 * The player: owns the file, the decode pipeline, the master clock and the compositor, and runs
 * the render loop (spec 9.3 / 9.5 / 9.7).
 *
 * Invariants enforced here:
 *  P2  what is on screen is always exactly one CompositeFrame (`current`); layer toggles re-draw
 *      that same composite frame.
 *  P3  when the next composite frame is due but not ready, `current` stays on screen as a whole;
 *      after BUFFER_AFTER_MS the clock is stopped (buffering) until frames are ready again.
 *  P4  when late, whole composite frames are skipped (closed) — never single layers.
 *  P6  before a composite frame is shown, every VideoFrame's timestamp is checked against
 *      pts_us(frame_index) (and every active video layer must have its planes, P1).
 */
import { AudioClock } from './clock/audioClock';
import { PerformanceClock, type MediaClock } from './clock/clock';
import { DecodePipeline, type PipelineOptions } from './decode/pipeline';
import { CompositeFrame, frameStats } from './decode/frames';
import { HttpByteSource, SourceChangedError } from './format/bytes';
import { isActive, type StillLayerMeta } from './format/lvf';
import { LvfSource } from './format/source';
import { frameAtUs, ptsUs } from './format/timing';
import { Compositor, type LayerState } from './render/compositor';

export type PlayerMode = 'empty' | 'loading' | 'seeking' | 'paused' | 'playing' | 'buffering' | 'ended' | 'error';

/** Next frame overdue by this long → stop the clock and buffer (P3). */
const BUFFER_AFTER_MS = 100;
/** Leave buffering once this many composite frames are ready (or the file is fully decoded). */
const RESUME_READY = 4;
/** A forward seek this short waits for the running pipeline instead of resetting the decoders. */
const SOFT_SEEK_FRAMES = 8;
/**
 * Video this far behind the clock (e.g. the tab was hidden while audio kept playing) is not
 * caught up frame by frame: the player seeks to where the clock is.
 */
const LATE_JUMP_US = 2_000_000;

export interface PlayerStats {
  shown: number;
  dropped: number;
  bufferingEvents: number;
  seeks: number;
  lateJumps: number;
  p6Failures: number;
  p1Failures: number;
}

export interface PlayerOptions {
  pipeline?: Partial<PipelineOptions>;
  /** Do not create an AudioContext even if the file has audio. */
  noAudio?: boolean;
}

/**
 * What survives a reload of a changed file: position, play state, and the layer settings changed in
 * the UI (by layer id). Settings left at the file's defaults follow the new file.
 */
interface KeptState {
  frame: number;
  playing: boolean;
  overrides: Map<string, Partial<LayerState>>;
}

type Origin = { kind: 'blob'; blob: Blob; name?: string } | { kind: 'url'; url: string; name?: string };

/** How often a file opened from a URL is checked for a new version. */
const WATCH_INTERVAL_MS = 1000;

export class Player extends EventTarget {
  readonly compositor: Compositor;
  source: LvfSource | null = null;
  pipeline: DecodePipeline | null = null;
  audio: AudioClock | null = null;
  clock: MediaClock = new PerformanceClock();
  mode: PlayerMode = 'empty';
  current: CompositeFrame | null = null;
  layerStates: LayerState[] = [];
  loop = false;
  error: Error | null = null;
  notice: string | null = null;
  readonly stats: PlayerStats = { shown: 0, dropped: 0, bufferingEvents: 0, seeks: 0, lateJumps: 0, p6Failures: 0, p1Failures: 0 };
  lastSyncFailure: string | null = null;

  private seekTarget: number | null = null;
  private pendingSeek: number | null = null;
  private resumeAfterSeek = false;
  private waitingSince: number | null = null;
  private dirty = false;
  private raf = 0;
  private origin: Origin | null = null;
  private reloading = false;
  private watchTimer = 0;
  reloads = 0;
  private lastRafMs = 0;
  private refreshMs = 1000 / 60;
  private loadToken = 0;

  constructor(
    canvas: HTMLCanvasElement,
    private readonly opts: PlayerOptions = {},
  ) {
    super();
    this.compositor = new Compositor(canvas);
    this.raf = requestAnimationFrame(this.tick);
  }

  // ---------------------------------------------------------------------------------------------
  // Loading
  // ---------------------------------------------------------------------------------------------
  /** Open a local File/Blob, or a URL served with range requests (`fflv view`). */
  async open(input: Blob | string, name?: string, keep?: KeptState): Promise<void> {
    const token = ++this.loadToken;
    this.close();
    this.origin = typeof input === 'string' ? { kind: 'url', url: input, name } : { kind: 'blob', blob: input, name };
    this.setMode('loading');
    try {
      const bytes = typeof input === 'string' ? await HttpByteSource.open(input, name) : input;
      const source = await LvfSource.open(bytes, name);
      const audio = source.meta.audio && !this.opts.noAudio ? await AudioClock.create(source.meta.audio) : null;
      if (token !== this.loadToken) return audio?.dispose();
      await this.compositor.load(source.meta, (i) => source.stillBlob(source.meta.layers[i] as StillLayerMeta));
      if (token !== this.loadToken) {
        audio?.dispose();
        return;
      }
      const pipeline = await DecodePipeline.create(source, audio, this.opts.pipeline);
      if (token !== this.loadToken) {
        pipeline.dispose();
        audio?.dispose();
        return;
      }
      this.source = source;
      this.audio = audio;
      this.clock = audio ?? new PerformanceClock();
      this.notice = source.meta.audio && !audio ? 'audio track present but not decodable here; playing without audio' : null;
      this.pipeline = pipeline;
      pipeline.onError = (e) => this.fail(e);
      this.layerStates = source.meta.layers.map((L) => ({ visible: L.visible, opacity: L.opacity, ...keep?.overrides.get(L.id) }));
      this.emit('loaded');
      this.startSeek(Math.min(keep?.frame ?? 0, source.frameCount - 1), keep?.playing ?? false);
    } catch (e) {
      if (token !== this.loadToken) return;
      if (e instanceof SourceChangedError) {
        window.setTimeout(() => void this.reload(), 200); // replaced while opening: try again
        return;
      }
      this.fail(e as Error);
    }
  }

  /** Re-open the same file (e.g. after it was rewritten), keeping position and layer settings. */
  async reload(): Promise<void> {
    if (!this.origin || this.reloading) return;
    this.reloading = true;
    try {
      const overrides = new Map<string, Partial<LayerState>>();
      (this.source?.meta.layers ?? []).forEach((L, i) => {
        const s = this.layerStates[i];
        const o: Partial<LayerState> = {};
        if (s.visible !== L.visible) o.visible = s.visible;
        if (s.opacity !== L.opacity) o.opacity = s.opacity;
        if (Object.keys(o).length) overrides.set(L.id, o);
      });
      const keep: KeptState = { frame: this.targetFrame, playing: this.isPlaying, overrides };
      const o = this.origin;
      await this.open(o.kind === 'url' ? o.url : o.blob, o.name, keep);
      this.reloads++;
      this.emit('reloaded');
    } finally {
      this.reloading = false;
    }
  }

  /** Poll the URL's ETag and reload when the file changes (files opened from a URL only). */
  watch(enable = true): void {
    window.clearInterval(this.watchTimer);
    if (!enable) return;
    this.watchTimer = window.setInterval(async () => {
      const o = this.origin;
      const bytes = this.source?.bytes;
      if (o?.kind !== 'url' || !(bytes instanceof HttpByteSource) || this.reloading || this.mode === 'loading') return;
      const etag = await HttpByteSource.currentEtag(o.url);
      if (etag && bytes.etag && etag !== bytes.etag && this.source?.bytes === bytes) void this.reload();
    }, WATCH_INTERVAL_MS);
  }

  close(): void {
    this.pipeline?.dispose();
    this.pipeline = null;
    if (this.clock !== this.audio) this.clock.dispose();
    this.audio?.dispose();
    this.audio = null;
    this.clock = new PerformanceClock();
    this.current?.close();
    this.current = null;
    this.source = null;
    this.error = null;
    this.seekTarget = this.pendingSeek = null;
    this.compositor.release();
    Object.assign(this.stats, { shown: 0, dropped: 0, bufferingEvents: 0, seeks: 0, lateJumps: 0, p6Failures: 0, p1Failures: 0 });
    this.lastSyncFailure = null;
    this.dirty = true;
    this.setMode('empty');
  }

  get frameCount(): number {
    return this.source?.frameCount ?? 0;
  }

  get currentFrame(): number {
    return this.current?.frameIndex ?? 0;
  }

  /** The frame the player is heading to (seek target) or showing. */
  get targetFrame(): number {
    return this.pendingSeek ?? this.seekTarget ?? this.currentFrame;
  }

  // ---------------------------------------------------------------------------------------------
  // Transport
  // ---------------------------------------------------------------------------------------------
  play(): void {
    if (!this.pipeline) return;
    switch (this.mode) {
      case 'ended':
        this.seek(0, true);
        return;
      case 'seeking':
        this.resumeAfterSeek = true;
        this.emit('state');
        return;
      case 'paused':
        if (this.currentFrame >= this.frameCount - 1) {
          this.seek(0, true);
          return;
        }
        this.setMode('playing');
        this.startClock();
        return;
      case 'error':
        // e.g. a decoder error: restart decoding where we are
        this.startSeek(this.currentFrame, true);
        return;
      default:
        return;
    }
  }

  pause(): void {
    if (this.mode === 'playing' || this.mode === 'buffering') {
      this.clock.stop();
      this.setMode('paused');
    } else if (this.mode === 'seeking') {
      this.resumeAfterSeek = false;
      this.emit('state');
    }
  }

  togglePlay(): void {
    if (this.isPlaying) this.pause();
    else this.play();
  }

  get isPlaying(): boolean {
    return this.mode === 'playing' || this.mode === 'buffering' || (this.mode === 'seeking' && this.resumeAfterSeek);
  }

  /** Jump to frame `target`. Requests during a seek are coalesced (the latest wins). */
  seek(target: number, resume?: boolean): void {
    if (!this.pipeline || this.mode === 'loading' || this.mode === 'empty') return;
    const t = Math.max(0, Math.min(this.frameCount - 1, Math.round(target)));
    if (t === this.currentFrame && this.current && (this.mode === 'paused' || this.mode === 'ended') && !resume) return;
    this.resumeAfterSeek = resume ?? this.isPlaying;
    if (this.mode === 'seeking') {
      this.pendingSeek = t === this.seekTarget ? null : t;
      this.emit('state');
      return;
    }
    this.startSeek(t, this.resumeAfterSeek);
  }

  /** Frame step (spec 9.8): +1 shows the next composite frame, -1 seeks to T-1. Pauses playback. */
  step(delta: number): void {
    if (!this.pipeline) return;
    if (this.mode === 'playing' || this.mode === 'buffering') this.pause();
    this.seek(this.targetFrame + delta, false);
  }

  setLayerVisible(i: number, visible: boolean): void {
    this.layerStates[i].visible = visible;
    this.dirty = true;
    this.emit('layers');
  }

  setLayerOpacity(i: number, opacity: number): void {
    this.layerStates[i].opacity = Math.max(0, Math.min(1, opacity));
    this.dirty = true;
    this.emit('layers');
  }

  // ---------------------------------------------------------------------------------------------
  // Internals
  // ---------------------------------------------------------------------------------------------
  private startSeek(target: number, resume: boolean): void {
    const p = this.pipeline!;
    this.clock.stop();
    this.seekTarget = target;
    this.resumeAfterSeek = resume;
    this.waitingSince = null;
    // Frames after `current` keep arriving in order, so a short hop forward just waits for them.
    // Restart the decoders only to go backwards, or when decoding from a RAP is shorter.
    const cur = this.currentFrame;
    const soft =
      this.current !== null &&
      !p.error &&
      target > cur &&
      (this.source!.index.rapAtOrBefore(target) <= cur || target - cur <= SOFT_SEEK_FRAMES);
    if (!soft) {
      p.seek(target);
      this.stats.seeks++;
    }
    this.setMode('seeking');
  }

  private startClock(): void {
    const clock = this.clock;
    clock.start().catch((e: Error) => {
      if (clock !== this.clock || this.mode !== 'playing') return;
      // Autoplay policy or no output device: fall back to a silent performance.now() clock.
      const t = clock.nowUs();
      this.clock = new PerformanceClock();
      this.clock.setTime(t);
      void this.clock.start();
      this.notice = `${e.message}; playing without sound`;
      this.emit('state');
    });
  }

  private present(cf: CompositeFrame): void {
    this.checkSync(cf);
    if (this.current && this.current !== cf) this.current.close();
    this.current = cf;
    this.stats.shown++;
    this.dirty = true;
    this.emit('frame');
  }

  /** P6 (timestamps) and P1 (every active video layer has all of its planes). */
  private checkSync(cf: CompositeFrame): void {
    const meta = this.source!.meta;
    const pts = ptsUs(cf.frameIndex, meta.fps);
    const problems: string[] = [];
    if (cf.ptsUs !== pts) problems.push(`composite pts ${cf.ptsUs} ≠ ${pts}`);
    for (const [li, pl] of cf.planes) {
      if (pl.color && pl.color.timestamp !== pts) problems.push(`layer ${li} color ts ${pl.color.timestamp}`);
      if (pl.alpha && pl.alpha.timestamp !== pts) problems.push(`layer ${li} alpha ts ${pl.alpha.timestamp}`);
      if (pl.alphaLuma && pl.alphaLuma.timestamp !== pts) problems.push(`layer ${li} alpha ts ${pl.alphaLuma.timestamp}`);
    }
    if (problems.length) this.stats.p6Failures++;
    let p1 = false;
    for (const [li, L] of meta.layers.entries()) {
      if (L.kind !== 'video' || !isActive(L, cf.frameIndex)) continue;
      const pl = cf.planes.get(li);
      if (!pl?.color || (L.has_alpha && !pl.alpha && !pl.alphaLuma)) {
        problems.push(`layer ${li} is missing a plane`);
        p1 = true;
      }
    }
    if (p1) this.stats.p1Failures++;
    if (problems.length) {
      this.lastSyncFailure = `frame ${cf.frameIndex} (expected ${pts} us): ${problems.join('; ')}`;
      console.error(`[lvf] SYNC ASSERTION FAILED at ${this.lastSyncFailure}`);
      this.emit('sync-failure');
    }
  }

  /**
   * What is drawn now reaches the screen about one refresh later, so frames are picked for that
   * moment: the clock is read this far ahead (keeps picture and sound aligned at the display).
   */
  get displayLeadUs(): number {
    return Math.min(this.refreshMs, 50) * 1000;
  }

  private tick = (now: number): void => {
    this.raf = requestAnimationFrame(this.tick);
    const dt = now - this.lastRafMs;
    if (this.lastRafMs && dt > 2 && dt < 100) this.refreshMs += (dt - this.refreshMs) * 0.1;
    this.lastRafMs = now;
    const p = this.pipeline;
    if (p && !p.error) {
      if (this.mode === 'seeking') this.tickSeeking(p);
      else if (this.mode === 'playing') this.tickPlaying(p, now);
      else if (this.mode === 'buffering') this.tickBuffering(p);
    }
    if (this.dirty) {
      this.dirty = false;
      this.compositor.draw(this.current, this.layerStates);
      this.emit('draw');
    }
  };

  private tickSeeking(p: DecodePipeline): void {
    const target = this.seekTarget!;
    for (let f = p.peek(); f; f = p.peek()) {
      if (f.frameIndex < target) {
        p.shift()!.close(); // soft seek: skip the frames between current and target
        continue;
      }
      this.present(p.shift()!);
      this.seekTarget = null;
      this.clock.setTime(f.ptsUs);
      if (this.pendingSeek !== null) {
        const next = this.pendingSeek;
        this.pendingSeek = null;
        this.startSeek(next, this.resumeAfterSeek);
      } else if (this.resumeAfterSeek) {
        this.setMode('playing');
        this.startClock();
      } else {
        this.setMode('paused');
      }
      return;
    }
    if (p.finished && !p.peek()) {
      // Target beyond what could be decoded (e.g. frames dropped as broken): stay where we are.
      this.seekTarget = null;
      this.setMode(this.current ? 'paused' : 'error');
    }
  }

  private tickPlaying(p: DecodePipeline, now: number): void {
    const n = this.frameCount;
    const fps = this.source!.meta.fps;
    const t = this.clock.nowUs() + this.displayLeadUs;
    const behind = t - ptsUs(Math.min(this.currentFrame + 1, n - 1), fps);
    if (behind > LATE_JUMP_US) {
      this.stats.lateJumps++;
      this.startSeek(Math.min(frameAtUs(t, fps), n - 1), true);
      return;
    }
    let pick: CompositeFrame | null = null;
    for (let f = p.peek(); f && f.ptsUs <= t; f = p.peek()) {
      if (pick) {
        pick.close(); // P4: late — skip the whole composite frame
        this.stats.dropped++;
      }
      pick = p.shift()!;
    }
    if (pick) {
      this.present(pick);
      this.waitingSince = null;
      return;
    }
    const next = this.currentFrame + 1;
    if (next >= n || p.finished) {
      // last frame shown (or nothing more will ever be released): end after its duration
      if (t >= ptsUs(next, fps)) this.onEnd();
      return;
    }
    if (ptsUs(next, fps) <= t) {
      // P3: due but not decoded yet; everything stays on the current composite frame.
      this.waitingSince ??= now;
      if (now - this.waitingSince > BUFFER_AFTER_MS) {
        this.clock.stop();
        this.stats.bufferingEvents++;
        this.waitingSince = null;
        this.setMode('buffering');
      }
    }
  }

  private tickBuffering(p: DecodePipeline): void {
    if (p.ready.length >= RESUME_READY || p.finished) {
      this.setMode('playing');
      this.startClock();
    }
  }

  private onEnd(): void {
    if (this.loop) {
      this.seek(0, true);
      return;
    }
    this.clock.stop();
    this.setMode('ended');
  }

  private fail(e: Error): void {
    if (e instanceof SourceChangedError || e.name === 'SourceChangedError') {
      void this.reload(); // the file was rewritten under us: pick up the new version
      return;
    }
    this.error = e;
    this.clock.stop();
    this.setMode('error');
  }

  private setMode(m: PlayerMode): void {
    if (this.mode === m) return;
    this.mode = m;
    this.emit('state');
  }

  private emit(type: string): void {
    this.dispatchEvent(new Event(type));
  }

  // ---------------------------------------------------------------------------------------------
  // Diagnostics
  // ---------------------------------------------------------------------------------------------
  debugInfo(): Record<string, string | number> {
    const p = this.pipeline;
    return {
      mode: this.mode,
      frame: this.current ? this.current.frameIndex : '—',
      clock: this.source ? `${(this.clock.nowUs() / 1e6).toFixed(3)} s (${this.clock.kind})` : '—',
      ready: p ? p.ready.length : 0,
      inFlight: p ? p.inFlight : 0,
      readAhead: p ? p.parsedCount : 0,
      decodeQueues: p ? p.decodeQueueSizes().join(' ') : '',
      dropped: this.stats.dropped,
      bufferingEvents: this.stats.bufferingEvents,
      p6Failures: this.stats.p6Failures,
      p1Failures: this.stats.p1Failures,
      seeks: this.stats.seeks,
      lateJumps: this.stats.lateJumps,
      prerollDiscarded: p ? p.stats.prerollDiscarded : 0,
      lostFrames: p ? p.stats.lostFrames : 0,
      staleOutputs: p ? p.stats.staleOutputs : 0,
      liveVideoFrames: frameStats.live,
      peakVideoFrames: frameStats.peak,
      audioAhead: this.audio ? `${(this.audio.bufferedAheadUs() / 1000).toFixed(0)} ms` : 'no audio',
      displayLead: `${(this.displayLeadUs / 1000).toFixed(1)} ms`,
      audioLateStarts: this.audio ? this.audio.stats.lateStarts : 0,
    };
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.watch(false);
    this.close();
  }
}
