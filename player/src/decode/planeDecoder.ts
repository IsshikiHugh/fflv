/**
 * One VideoDecoder for one plane (color or alpha) of one layer.
 *
 * Output frames are mapped back to their composite frame by timestamp: every chunk is sent with
 * timestamp = pts_us(frame_index), and the decoder must hand frames back in the order they were
 * fed (VP9 without alt-ref has no reordering). The FIFO of fed frames makes that checkable:
 *  - an output matching the head of the FIFO is the expected frame;
 *  - an output matching a later FIFO entry means the earlier ones were lost (reported, so the
 *    affected composite frames are dropped whole instead of waiting forever);
 *  - an output matching nothing is stale (e.g. produced before a reset) and is discarded.
 */
import { adopt, release } from './frames';
import { ptsUs, type Fps } from '../format/timing';

export type PlaneKind = 'color' | 'alpha';

export interface PlaneSink {
  onFrame(layer: number, plane: PlaneKind, frameIndex: number, frame: VideoFrame): void;
  onLost(layer: number, plane: PlaneKind, frameIndex: number): void;
  onStale(): void;
  onDecodeError(layer: number, plane: PlaneKind, error: Error): void;
  /** The decoder's queue shrank; more chunks may be sent. */
  onDequeue(): void;
}

export class PlaneDecoder {
  private decoder: VideoDecoder;
  private fifo: { frame: number; ts: number }[] = [];
  private needKey = true;

  constructor(
    readonly layer: number,
    readonly plane: PlaneKind,
    readonly config: VideoDecoderConfig,
    private readonly fps: Fps,
    private readonly sink: PlaneSink,
  ) {
    this.decoder = this.create();
  }

  private create(): VideoDecoder {
    const dec: VideoDecoder = new VideoDecoder({
      output: (vf) => this.onOutput(dec, vf),
      error: (e) => {
        if (dec === this.decoder) this.sink.onDecodeError(this.layer, this.plane, e);
      },
    });
    dec.addEventListener('dequeue', () => this.sink.onDequeue());
    dec.configure(this.config);
    return dec;
  }

  /** Throws if the first chunk after a reset is not a key frame (the file lied about a RAP). */
  decode(frameIndex: number, key: boolean, data: Uint8Array): void {
    if (this.needKey && !key) {
      throw new Error(`layer ${this.layer} ${this.plane}: frame ${frameIndex} must be a key frame after a reset`);
    }
    this.needKey = false;
    const timestamp = ptsUs(frameIndex, this.fps);
    this.fifo.push({ frame: frameIndex, ts: timestamp });
    this.decoder.decode(new EncodedVideoChunk({ type: key ? 'key' : 'delta', timestamp, data }));
  }

  private onOutput(dec: VideoDecoder, vf: VideoFrame): void {
    adopt(vf);
    if (dec !== this.decoder) {
      release(vf);
      this.sink.onStale();
      return;
    }
    const k = this.fifo.findIndex((e) => e.ts === vf.timestamp);
    if (k < 0) {
      release(vf);
      this.sink.onStale();
      return;
    }
    const done = this.fifo.splice(0, k + 1);
    const hit = done.pop()!;
    for (const lost of done) this.sink.onLost(this.layer, this.plane, lost.frame);
    this.sink.onFrame(this.layer, this.plane, hit.frame, vf);
  }

  /** Drop all queued work (spec 9.7: reset() then configure()); a closed decoder is recreated. */
  reset(): void {
    this.fifo = [];
    this.needKey = true;
    if (this.decoder.state === 'closed') {
      this.decoder = this.create();
    } else {
      this.decoder.reset();
      this.decoder.configure(this.config);
    }
  }

  get queueSize(): number {
    return this.decoder.decodeQueueSize;
  }

  get inFlight(): number {
    return this.fifo.length;
  }

  close(): void {
    this.fifo = [];
    if (this.decoder.state !== 'closed') this.decoder.close();
  }
}
