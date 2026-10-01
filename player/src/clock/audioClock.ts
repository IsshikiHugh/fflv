/**
 * Audio output and the audio master clock (spec 9.5).
 *
 * Opus packets are decoded with WebCodecs AudioDecoder; each AudioData becomes an AudioBuffer that is
 * scheduled on the AudioContext timeline through a fixed mapping
 *
 *     context time = anchorCtx + (media time − anchorUs)
 *
 * and the clock reads that mapping backwards at the sample currently reaching the speakers
 * (AudioContext.getOutputTimestamp), so what is shown is what is heard.
 *
 * Pausing stops the scheduled sources and freezes the clock; resuming re-anchors at the frozen time
 * and reschedules the decoded audio that is still ahead, so pause/resume is exact to the sample.
 *
 * Opus pre-skip: packets are stored on the decoder timeline (first packet at 0) and
 * metadata.audio.pre_skip says how many leading samples to drop. Chromium's AudioDecoder applies
 * the pre-skip itself when given the OpusHead description and then reports presentation timestamps;
 * a decoder that does not is detected from the size of its first output and compensated here.
 */
import type { AudioSink } from '../decode/pipeline';
import type { AudioMeta, AudioPacket } from '../format/lvf';
import type { MediaClock } from './clock';

const SAMPLE_RATE = 48000;
/** How far ahead of "now" playback (re)starts, so the first buffers are scheduled in time. */
const START_LEAD_S = 0.08;
/** Decoded audio kept behind the playhead (lets a buffering resume re-schedule seamlessly). */
const KEEP_BEHIND_US = 1_000_000;

interface Buffered {
  startUs: number;
  endUs: number;
  buf: AudioBuffer;
  node: AudioBufferSourceNode | null;
}

function b64(s: string): Uint8Array {
  return Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
}

export class AudioClock implements MediaClock, AudioSink {
  readonly kind = 'audio' as const;
  readonly ctx: AudioContext;
  running = false;
  error: Error | null = null;
  readonly stats = { decoded: 0, scheduled: 0, lateStarts: 0, decoderTrimsPreSkip: null as boolean | null };

  private readonly gain: GainNode;
  private readonly config: AudioDecoderConfig;
  private readonly preSkipSamples: number;
  private decoder: AudioDecoder;
  private tsOffsetUs = 0;
  private firstChunk: { durationUs: number } | null = null;
  private firstOutputSeen = false;
  private buffers: Buffered[] = [];
  private anchorCtx = 0;
  private anchorUs = 0;
  private frozenUs = 0;
  private lastNowUs = 0;
  private startToken = 0;

  private constructor(meta: AudioMeta, config: AudioDecoderConfig, head: Uint8Array | null) {
    this.config = config;
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: 'interactive' });
    this.gain = this.ctx.createGain();
    this.gain.connect(this.ctx.destination);
    let preSkip = meta.pre_skip ?? 0;
    if (meta.pre_skip === undefined && head && head.length >= 12) preSkip = head[10] | (head[11] << 8);
    this.preSkipSamples = preSkip;
    this.decoder = this.createDecoder();
  }

  /** Returns null when the browser cannot decode the track (the player then runs without audio). */
  static async create(meta: AudioMeta): Promise<AudioClock | null> {
    if (typeof AudioDecoder === 'undefined') return null;
    const config: AudioDecoderConfig = { codec: 'opus', sampleRate: meta.sample_rate, numberOfChannels: meta.channels };
    let head: Uint8Array | null = null;
    if (meta.description_b64) {
      try {
        head = b64(meta.description_b64);
      } catch {
        return null; // not base64: an OpusHead we cannot use, so play without audio
      }
      config.description = head;
    }
    const res = await AudioDecoder.isConfigSupported(config).catch(() => null); // TypeError: invalid config
    return res?.supported ? new AudioClock(meta, config, head) : null;
  }

  private createDecoder(): AudioDecoder {
    const dec: AudioDecoder = new AudioDecoder({
      output: (ad) => {
        if (dec === this.decoder) this.onOutput(ad);
        else ad.close();
      },
      error: (e) => {
        if (dec !== this.decoder) return;
        this.error = e;
        console.error('[lvf] audio decoder error:', e);
      },
    });
    dec.configure(this.config);
    this.firstChunk = null;
    this.firstOutputSeen = false;
    return dec;
  }

  // ---------------------------------------------------------------------------------------------
  // AudioSink
  // ---------------------------------------------------------------------------------------------
  feed(p: AudioPacket): void {
    if (this.decoder.state !== 'configured') return;
    if (!this.firstChunk) this.firstChunk = { durationUs: p.durationUs };
    this.decoder.decode(new EncodedAudioChunk({ type: 'key', timestamp: p.ptsUs, duration: p.durationUs, data: p.data }));
  }

  /**
   * Called on seek: a fresh decoder (so no stale output can leak in) and no decoded audio. The
   * frozen clock moves to the seek target right away: decoded audio is pruned relative to it, and
   * audio for the new position arrives before the player sets the clock to the first shown frame.
   */
  reset(targetUs: number): void {
    this.setTime(targetUs);
    this.buffers = [];
    if (this.decoder.state !== 'closed') this.decoder.close();
    this.error = null;
    this.decoder = this.createDecoder();
  }

  private onOutput(ad: AudioData): void {
    if (!this.firstOutputSeen) {
      this.firstOutputSeen = true;
      // Did the decoder drop the pre-skip itself? Then its timestamps are presentation times.
      const expected = this.firstChunk ? Math.round((this.firstChunk.durationUs * ad.sampleRate) / 1e6) : ad.numberOfFrames;
      const trims = this.preSkipSamples > 0 && ad.numberOfFrames <= expected - this.preSkipSamples;
      this.stats.decoderTrimsPreSkip = trims;
      this.tsOffsetUs = trims || this.preSkipSamples === 0 ? 0 : Math.round((this.preSkipSamples * 1e6) / SAMPLE_RATE);
    }
    const n = ad.numberOfFrames;
    const buf = this.ctx.createBuffer(ad.numberOfChannels, n, ad.sampleRate);
    const tmp = new Float32Array(n);
    for (let c = 0; c < ad.numberOfChannels; c++) {
      ad.copyTo(tmp, { planeIndex: c, format: 'f32-planar' });
      buf.copyToChannel(tmp, c);
    }
    const startUs = ad.timestamp - this.tsOffsetUs;
    ad.close();
    this.stats.decoded++;
    const b: Buffered = { startUs, endUs: startUs + (n * 1e6) / buf.sampleRate, buf, node: null };
    let i = this.buffers.length;
    while (i > 0 && this.buffers[i - 1].startUs > startUs) i--;
    this.buffers.splice(i, 0, b);
    // drop audio that is long past
    const horizon = (this.running ? this.nowUs() : this.frozenUs) - KEEP_BEHIND_US;
    while (this.buffers.length && this.buffers[0].endUs < horizon && !this.buffers[0].node) this.buffers.shift();
    if (this.running) this.schedule(b);
  }

  // ---------------------------------------------------------------------------------------------
  // Scheduling
  // ---------------------------------------------------------------------------------------------
  private schedule(b: Buffered): void {
    if (b.node) return;
    const when = this.anchorCtx + (b.startUs - this.anchorUs) / 1e6;
    const at = Math.max(when, this.anchorCtx, this.ctx.currentTime + 0.005);
    const offset = at - when;
    if (offset >= b.buf.duration) return; // entirely before the playhead
    if (offset > 0 && when >= this.anchorCtx) this.stats.lateStarts++;
    const node = this.ctx.createBufferSource();
    node.buffer = b.buf;
    node.connect(this.gain);
    node.onended = () => {
      if (b.node === node) b.node = null;
      node.disconnect();
    };
    node.start(at, offset);
    b.node = node;
    this.stats.scheduled++;
  }

  private stopSources(): void {
    for (const b of this.buffers) {
      if (!b.node) continue;
      b.node.onended = null;
      try {
        b.node.stop();
      } catch {
        /* not started yet */
      }
      b.node.disconnect();
      b.node = null;
    }
  }

  // ---------------------------------------------------------------------------------------------
  // MediaClock
  // ---------------------------------------------------------------------------------------------
  /** Context time of the sample that is reaching the output device right now. */
  private outputContextTime(): number {
    const ts = this.ctx.getOutputTimestamp();
    if (ts.contextTime !== undefined && ts.performanceTime !== undefined && ts.contextTime > 0) {
      return ts.contextTime + (performance.now() - ts.performanceTime) / 1000;
    }
    return this.ctx.currentTime - (this.ctx.outputLatency || 0) - this.ctx.baseLatency;
  }

  nowUs(): number {
    if (!this.running) return this.frozenUs;
    const us = this.anchorUs + (this.outputContextTime() - this.anchorCtx) * 1e6;
    this.lastNowUs = Math.max(this.lastNowUs, this.anchorUs, us);
    return this.lastNowUs;
  }

  /** Rejects if the AudioContext cannot run (autoplay policy without a user gesture). */
  async start(): Promise<void> {
    if (this.running) return;
    const token = ++this.startToken;
    const state = () => this.ctx.state as AudioContextState;
    if (state() !== 'running') {
      await Promise.race([this.ctx.resume(), new Promise((r) => setTimeout(r, 1500))]);
      if (state() !== 'running') throw new Error('audio output is blocked (AudioContext did not start)');
    }
    if (token !== this.startToken || this.running) return;
    this.anchorUs = this.frozenUs;
    this.lastNowUs = this.frozenUs;
    this.anchorCtx = this.ctx.currentTime + START_LEAD_S;
    this.running = true;
    for (const b of this.buffers) if (b.endUs > this.anchorUs) this.schedule(b);
  }

  stop(): void {
    this.startToken++;
    if (!this.running) return;
    this.frozenUs = this.nowUs();
    this.running = false;
    this.stopSources();
  }

  setTime(us: number): void {
    this.stop();
    this.frozenUs = us;
  }

  setVolume(v: number): void {
    this.gain.gain.value = v;
  }

  /** Decoded audio available ahead of the playhead, in microseconds. */
  bufferedAheadUs(): number {
    const now = this.nowUs();
    const last = this.buffers[this.buffers.length - 1];
    return last ? Math.max(0, last.endUs - now) : 0;
  }

  /** For tests: route the mixed output into an analysis node as well. */
  tap(node: AudioNode): void {
    this.gain.connect(node);
  }

  /** For tests: the presentation-time mapping of decoded audio. */
  debugBuffers(): { startUs: number; endUs: number; data: Float32Array }[] {
    return this.buffers.map((b) => ({ startUs: b.startUs, endUs: b.endUs, data: b.buf.getChannelData(0) }));
  }

  dispose(): void {
    this.stop();
    this.buffers = [];
    if (this.decoder.state !== 'closed') this.decoder.close();
    void this.ctx.close();
  }
}
