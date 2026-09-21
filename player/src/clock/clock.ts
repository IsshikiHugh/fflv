/**
 * Master clock (spec 9.5). Media time is in microseconds on the file's timeline.
 * `stop()` freezes it, `start()` continues from the frozen value, `setTime()` moves it while stopped.
 */
export interface MediaClock {
  readonly kind: 'audio' | 'performance';
  readonly running: boolean;
  nowUs(): number;
  start(): Promise<void>;
  stop(): void;
  setTime(us: number): void;
  dispose(): void;
}

/** Used when the file has no audio (or audio output is unavailable). */
export class PerformanceClock implements MediaClock {
  readonly kind = 'performance' as const;
  running = false;
  private anchorUs = 0;
  private anchorMs = 0;

  nowUs(): number {
    return this.running ? this.anchorUs + (performance.now() - this.anchorMs) * 1000 : this.anchorUs;
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.anchorMs = performance.now();
    this.running = true;
  }

  stop(): void {
    if (!this.running) return;
    this.anchorUs = this.nowUs();
    this.running = false;
  }

  setTime(us: number): void {
    this.anchorUs = us;
    this.anchorMs = performance.now();
  }

  dispose(): void {
    this.running = false;
  }
}
