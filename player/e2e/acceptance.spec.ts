/**
 * Acceptance tests of LVF_SPEC.md 11.2 (items 3–8) plus seek precision and P5, run against
 * test_assets/test.lvd. Every video layer of the test material carries a barcode of its own frame
 * number; "in sync" is checked by reading all of them back from the rendered canvas.
 */
import { expect, test, type Page } from '@playwright/test';
import {
  assetsAvailable,
  barcodeProbe,
  barcodes,
  loadProbes,
  minFrames,
  mismatches,
  openTestFile,
  REAL_HARDWARE,
  startRecording,
  stats,
  stopRecording,
  waitForMode,
  type Probes,
} from './helpers';

test.skip(!assetsAvailable(), 'run `fflv testsrc` first');

const P: Probes = assetsAvailable() ? loadProbes() : (null as never);
const probe = P ? barcodeProbe(P) : (null as never);
const N = P?.frame_count ?? 0;

/** Deterministic PRNG so failures are reproducible. */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

async function settle(page: Page): Promise<void> {
  await page.waitForFunction(() => ['paused', 'ended', 'error'].includes(window.__lvf.player.mode), null, { timeout: 30_000 });
  const mode = await page.evaluate(() => window.__lvf.player.mode);
  expect(mode).not.toBe('error');
}

async function expectInSync(page: Page, expectFrame?: number) {
  const b = await barcodes(page, probe);
  if (expectFrame !== undefined) expect(b.frame).toBe(expectFrame);
  for (const [id, n] of Object.entries(b.codes)) expect(n, `layer ${id} at frame ${b.frame}`).toBe(b.frame);
  // UI readout shows the same frame
  await expect(page.locator('#frame')).toHaveText(String(b.frame));
  return b;
}

/** Is the still layer drawn? (checked with every other layer hidden, then restored) */
async function logoShown(page: Page): Promise<boolean> {
  const [x, y] = P.logo.rect;
  const px = await page.evaluate(
    ({ x, y }) => {
      const { player } = window.__lvf;
      const saved = player.layerStates.map((s) => ({ ...s }));
      player.layerStates.forEach((s, i) => (s.visible = player.source!.meta.layers[i].id === 'logo'));
      player.compositor.draw(player.current, player.layerStates);
      const out = Array.from(player.compositor.readPixels(x, y, 1, 1));
      saved.forEach((s, i) => Object.assign(player.layerStates[i], s));
      player.setLayerOpacity(0, player.layerStates[0].opacity); // redraw with the real state
      return out;
    },
    { x: x + 30, y: y + 50 },
  );
  return px[0] > 100;
}

test('11.2-3 continuous playback: every composite frame in sync, to the last frame', async ({ page }) => {
  await openTestFile(page);
  await startRecording(page, probe);
  await page.keyboard.press('Space');
  await waitForMode(page, 'ended', 60_000);
  const samples = await stopRecording(page);
  const st = await stats(page);
  console.log(`  drawn ${samples.length} composite frames; stats ${JSON.stringify(st)}`);
  expect(mismatches(samples)).toEqual([]);
  expect(st.p6Failures + st.p1Failures).toBe(0);
  expect(samples.at(-1)!.frame).toBe(N - 1);
  for (let i = 1; i < samples.length; i++) expect(samples[i].frame).toBeGreaterThanOrEqual(samples[i - 1].frame);
  expect(new Set(samples.map((s) => s.frame)).size).toBeGreaterThan(minFrames(N * 0.9));
});

test('11.2-3 the red line stays centred on the white line', async ({ page }) => {
  await openTestFile(page);
  const y = P.sync_band.rect[1] + 50;
  await page.evaluate((y) => {
    const { player } = window.__lvf;
    player.source!.meta.layers.forEach((L, i) => player.setLayerVisible(i, L.id === 'sync_a' || L.id === 'sync_b'));
    const out: { frame: number; white: number[]; red: number[] }[] = [];
    Object.assign(window, { __lines: out });
    player.addEventListener('draw', () => {
      const w = player.compositor.canvas.width;
      const row = player.compositor.readPixels(0, y, w, 1);
      const white: number[] = [];
      const red: number[] = [];
      for (let x = 0; x < w; x++) {
        const [r, g, b] = [row[4 * x], row[4 * x + 1], row[4 * x + 2]];
        if (r > 200 && g > 200 && b > 200) white.push(x);
        else if (r > 150 && g < 110 && b < 110) red.push(x);
      }
      out.push({ frame: player.currentFrame, white, red });
    });
  }, y);
  await page.keyboard.press('Space');
  await page.waitForTimeout(5000);
  await page.keyboard.press('Space');
  const lines = (await page.evaluate(() => (window as unknown as { __lines: { frame: number; white: number[]; red: number[] }[] }).__lines)) ?? [];
  const W = P.canvas[0];
  let checked = 0;
  const bad: string[] = [];
  for (const s of lines) {
    const cx = (s.frame * 16) % W;
    if (cx < 10 || cx > W - 10) continue; // line wraps around the edge
    const mid = (a: number[]) => (a[0] + a[a.length - 1]) / 2;
    if (!s.red.length || !s.white.length) {
      bad.push(`frame ${s.frame}: line missing (white ${s.white.length}px, red ${s.red.length}px)`);
      continue;
    }
    checked++;
    const r = mid(s.red);
    const whiteAll = [...s.white, ...s.red].sort((a, b) => a - b);
    const w = mid(whiteAll);
    if (Math.abs(r - w) > 0.5 || Math.abs(r - (cx - 0.5)) > 1) bad.push(`frame ${s.frame}: red centre ${r}, white centre ${w}, expected ${cx - 0.5}`);
  }
  console.log(`  checked ${checked} frames`);
  expect(bad).toEqual([]);
  expect(checked).toBeGreaterThan(minFrames(100));
});

test('11.2-4 toggling layers, scrubbing and stepping never mixes frames', async ({ page }) => {
  await openTestFile(page);
  await startRecording(page, probe);
  const rand = rng(20260921);
  const pick = (n: number) => Math.floor(rand() * n);
  const layerCount = await page.evaluate(() => window.__lvf.player.layerStates.length);
  const toggle = async () => {
    const i = pick(layerCount);
    await page.locator(`#layers li[data-index="${i}"] input[type=checkbox]`).click();
  };
  const log: string[] = [];
  for (let it = 0; it < 45; it++) {
    const before = await page.evaluate(() => window.__lvf.player.currentFrame);
    const op = pick(7);
    let expectFrame: number | undefined;
    if (op === 0) {
      await toggle();
      log.push('toggle');
    } else if (op === 1) {
      const t = pick(N);
      await page.locator('#seek').fill(String(t));
      expectFrame = t;
      log.push(`seek ${t}`);
    } else if (op === 2) {
      let t = 0;
      for (let k = 0; k < 10; k++) {
        t = pick(N);
        await page.locator('#seek').fill(String(t));
        await page.waitForTimeout(15);
      }
      expectFrame = t;
      log.push(`scrub →${t}`);
    } else if (op === 3 || op === 4) {
      const k = 1 + pick(5);
      const key = op === 3 ? 'ArrowRight' : 'ArrowLeft';
      for (let j = 0; j < k; j++) await page.keyboard.press(key);
      expectFrame = op === 3 ? Math.min(N - 1, before + k) : Math.max(0, before - k);
      log.push(`${key} ×${k}`);
    } else {
      await page.keyboard.press('Space');
      for (let j = 0; j < 3; j++) {
        await page.waitForTimeout(80 + pick(300));
        if (rand() < 0.6) await toggle();
      }
      await page.keyboard.press('Space');
      log.push('play+toggles');
    }
    await settle(page);
    const b = await expectInSync(page, expectFrame);
    const lf = P.logo;
    expect(await logoShown(page), `still layer at frame ${b.frame}`).toBe(b.frame >= lf.start_frame && b.frame < lf.end_frame);
  }
  const samples = await stopRecording(page);
  const st = await stats(page);
  console.log(`  ${log.join(', ')}`);
  console.log(`  drawn ${samples.length} composite frames; stats ${JSON.stringify(st)}`);
  expect(mismatches(samples)).toEqual([]);
  expect(st.p6Failures + st.p1Failures).toBe(0);
});

test('11.2-5 with the CPU slowed 6x frames may drop, layers never desync', async ({ page }) => {
  await openTestFile(page);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 6 });
  try {
    await startRecording(page, probe);
    await page.keyboard.press('Space');
    await page.waitForTimeout(6000);
    await page.locator('#layers li[data-index="2"] input[type=checkbox]').click(); // hide sync_b
    await page.waitForTimeout(2000);
    await page.locator('#layers li[data-index="2"] input[type=checkbox]').click(); // show it again
    await page.locator('#seek').fill('333');
    await page.waitForTimeout(6000);
    await page.keyboard.press('Space');
    await settle(page);
    await expectInSync(page);
    const samples = await stopRecording(page);
    const st = await stats(page);
    console.log(`  drawn ${samples.length}, shown ${st.shown}, dropped ${st.dropped}, buffering events ${st.bufferingEvents}`);
    expect(mismatches(samples)).toEqual([]);
    expect(st.p6Failures + st.p1Failures).toBe(0);
    expect(samples.length).toBeGreaterThan(30);
  } finally {
    await cdp.send('Emulation.setCPUThrottlingRate', { rate: 1 });
  }
});

test('P3/P4 starved pipeline and a busy main thread: frames drop and playback buffers, layers never desync', async ({ page }) => {
  await openTestFile(page, '?queue=2');
  await startRecording(page, probe);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 6 });
  // Hog the main thread (40 of every 50 ms): decoder outputs and rAF arrive late and bunched.
  await page.evaluate(() => {
    const id = setInterval(() => {
      const t = performance.now();
      while (performance.now() - t < 40) {
        /* busy */
      }
    }, 50);
    Object.assign(window, { __hog: id });
  });
  await page.keyboard.press('Space');
  await page.waitForTimeout(8000);
  await page.evaluate(() => clearInterval((window as unknown as { __hog: number }).__hog));
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 1 });
  await page.keyboard.press('Space');
  await settle(page);
  await expectInSync(page);
  const samples = await stopRecording(page);
  const st = await stats(page);
  console.log(`  drawn ${samples.length}, shown ${st.shown}, dropped ${st.dropped}, buffering events ${st.bufferingEvents}`);
  expect(mismatches(samples)).toEqual([]);
  expect(st.p6Failures + st.p1Failures).toBe(0);
  expect(st.dropped + st.bufferingEvents).toBeGreaterThan(0); // the stress actually happened
});

test('P3 slow storage: playback stops as a whole, buffers, resumes — layers never desync', async ({ page }) => {
  // 3 composite frames per read, 150 ms per read: ~20 fps of input for 30 fps playback.
  await openTestFile(page, '?slowread=150&readahead=3');
  await startRecording(page, probe);
  const modes: string[] = [];
  await page.exposeFunction('__mode', (m: string) => modes.push(m));
  await page.evaluate(() => window.__lvf.player.addEventListener('state', () => (window as unknown as { __mode: (m: string) => void }).__mode(window.__lvf.player.mode)));
  await page.keyboard.press('Space');
  await page.waitForTimeout(8000);
  await page.keyboard.press('Space');
  await settle(page);
  await expectInSync(page);
  const samples = await stopRecording(page);
  const st = await stats(page);
  console.log(`  drawn ${samples.length}, shown ${st.shown}, dropped ${st.dropped}, buffering events ${st.bufferingEvents}`);
  expect(mismatches(samples)).toEqual([]);
  expect(st.p6Failures + st.p1Failures).toBe(0);
  expect(st.bufferingEvents).toBeGreaterThan(0);
  expect(modes).toContain('buffering');
  // after buffering, playback resumed
  expect(modes.lastIndexOf('playing')).toBeGreaterThan(modes.indexOf('buffering'));
});

/** Onset times (µs, presentation timeline) of the beeps in the decoded audio held by the player. */
async function decodedOnsets(page: Page): Promise<number[]> {
  return page.evaluate(() => {
    const bufs = window.__lvf.player.audio!.debugBuffers();
    const out: number[] = [];
    let quiet = 0;
    let prevEnd = -1;
    for (const b of bufs) {
      if (Math.abs(b.startUs - prevEnd) > 50) quiet = 0; // gap: do not count silence across it
      prevEnd = b.endUs;
      const dt = (b.endUs - b.startUs) / b.data.length;
      for (let i = 0; i < b.data.length; i++) {
        if (Math.abs(b.data[i]) > 0.02) {
          if (quiet >= 480) out.push(b.startUs + i * dt);
          quiet = 0;
        } else quiet++;
      }
    }
    return out;
  });
}

test('11.2-6 audio timeline: every beep starts on its whole second (pre-skip handled)', async ({ page }) => {
  await openTestFile(page);
  await page.waitForFunction(() => window.__lvf.player.audio!.debugBuffers().some((b) => b.endUs > 1_300_000));
  const a = await decodedOnsets(page);
  const trims = await page.evaluate(() => window.__lvf.player.audio!.stats.decoderTrimsPreSkip);
  console.log(`  decoder applies pre-skip itself: ${trims}; onsets from start: ${a.map((t) => (t / 1e3).toFixed(2) + ' ms').join(', ')}`);
  const near = (list: number[], t: number) => list.some((x) => Math.abs(x - t) < 1500);
  expect(near(a, 1_000_000)).toBe(true);

  // after a seek, decoding restarts at the RAP at 2.0 s
  await page.evaluate(() => window.__lvf.player.seek(75));
  await settle(page);
  await page.waitForFunction(() => window.__lvf.player.audio!.debugBuffers().some((b) => b.endUs > 3_300_000));
  const b = await decodedOnsets(page);
  console.log(`  onsets after seek: ${b.map((t) => (t / 1e3).toFixed(2) + ' ms').join(', ')}`);
  expect(near(b, 3_000_000)).toBe(true);
  expect(b.every((t) => Math.abs(t / 1e6 - Math.round(t / 1e6)) < 0.0015)).toBe(true);
});

test('11.2-6 audio/video: the beep reaches the output when the white flash is shown', async ({ page }) => {
  test.skip(!REAL_HARDWARE, 'measures audio output against the display: needs an audio device and real-time rendering');
  await openTestFile(page);
  await page.evaluate(async () => {
    const { player } = window.__lvf;
    const ctx = player.audio!.ctx;
    const code = `class Onset extends AudioWorkletProcessor {
      constructor() { super(); this.quiet = 0; }
      process(inputs) {
        const ch = inputs[0][0];
        if (ch) for (let i = 0; i < ch.length; i++) {
          if (Math.abs(ch[i]) > 0.02) { if (this.quiet > 4800) this.port.postMessage(currentTime + i / sampleRate); this.quiet = 0; }
          else this.quiet++;
        }
        return true;
      }
    }
    registerProcessor('onset', Onset);`;
    await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([code], { type: 'text/javascript' })));
    const node = new AudioWorkletNode(ctx, 'onset');
    const sink = ctx.createGain();
    sink.gain.value = 0;
    node.connect(sink).connect(ctx.destination);
    player.audio!.tap(node);
    const rec = { onsets: [] as number[], flashes: [] as { f: number; ctx: number }[] };
    Object.assign(window, { __av: rec });
    node.port.onmessage = (e) => rec.onsets.push(e.data as number);
    player.addEventListener('frame', () => {
      const f = player.currentFrame;
      if (f > 0 && f % 30 === 0 && player.mode === 'playing') {
        const ts = ctx.getOutputTimestamp();
        rec.flashes.push({ f, ctx: ts.contextTime! + (performance.now() - ts.performanceTime!) / 1000 });
      }
    });
  });
  await page.keyboard.press('Space');
  await page.waitForTimeout(6500);
  await page.keyboard.press('Space');
  const rec = await page.evaluate(() => (window as unknown as { __av: { onsets: number[]; flashes: { f: number; ctx: number }[] } }).__av);
  const leadMs = await page.evaluate(() => window.__lvf.player.displayLeadUs / 1000);
  // The flash is drawn now and reaches the screen about one refresh (the display lead) later.
  const offsets = rec.flashes.map((fl) => {
    const nearest = rec.onsets.reduce((best, o) => (Math.abs(o - fl.ctx) < Math.abs(best - fl.ctx) ? o : best), Infinity);
    return (fl.ctx - nearest) * 1000 + leadMs; // ms; > 0 means the picture comes after the sound
  });
  console.log(`  flash-vs-beep at the display (ms, lead ${leadMs.toFixed(1)}): ${offsets.map((o) => o.toFixed(1)).join(', ')}`);
  expect(offsets.length).toBeGreaterThanOrEqual(4);
  for (const o of offsets) expect(Math.abs(o)).toBeLessThan(45);
});

test('11.2-7 alpha calibration ramp: transparent at the left, opaque at the right', async ({ page }) => {
  await openTestFile(page);
  const [, cy] = P.calib.rect;
  // Draw the layer over a black and over a white background: out = c·a + bg·(1 − a), so
  // a = 1 − (out_white − out_black) / 255 no matter what the color plane holds.
  const rows = await page.evaluate((y) => {
    const { player, readRow } = window.__lvf;
    player.source!.meta.layers.forEach((L, i) => player.setLayerVisible(i, L.id === 'calib'));
    player.compositor.backgroundOverride = [0, 0, 0];
    const black = readRow(player, y);
    player.compositor.backgroundOverride = [1, 1, 1];
    const white = readRow(player, y);
    player.compositor.backgroundOverride = null;
    return { black, white };
  }, cy + 20);
  const W = rows.black.length / 4;
  const v = Array.from({ length: W }, (_, x) => {
    let s = 0;
    for (let c = 0; c < 3; c++) s += rows.white[4 * x + c] - rows.black[4 * x + c];
    return 255 - s / 3; // alpha in 0..255
  });
  console.log(`  alpha left ${v.slice(0, 4).join(',')} … mid ${v[W >> 1]} … right ${v.slice(-4).join(',')}`);
  expect(v[0]).toBe(0); // alpha 0 → fully transparent
  expect(v[W - 1]).toBe(255); // alpha 255 → fully opaque
  const err = v.map((x, i) => Math.abs(x - (i * 255) / (W - 1)));
  console.log(`  max deviation from the ideal ramp: ${Math.max(...err).toFixed(2)} levels`);
  expect(Math.max(...err)).toBeLessThanOrEqual(4);
});

test('9.6 blend modes and opacity follow their formulas (straight alpha)', async ({ page }) => {
  await openTestFile(page);
  await page.evaluate(() => window.__lvf.player.seek(100)); // the still logo is shown in [90, 360)
  await settle(page);
  const [lx, ly] = P.logo.rect;
  const got = await page.evaluate(
    ({ x, y }) => {
      const { player } = window.__lvf;
      const layers = player.source!.meta.layers;
      const logo = layers.findIndex((L) => L.id === 'logo');
      layers.forEach((_, i) => (player.layerStates[i].visible = i === logo));
      player.compositor.backgroundOverride = [0.5, 0.5, 0.5];
      const out: Record<string, number[]> = {};
      for (const [mode, opacity] of [['normal', 1], ['normal', 0.5], ['add', 1], ['multiply', 1], ['screen', 1]] as const) {
        layers[logo].blend = mode;
        player.layerStates[logo].opacity = opacity;
        player.compositor.draw(player.current, player.layerStates);
        out[`${mode}@${opacity}`] = Array.from(player.compositor.readPixels(x, y, 1, 1)).slice(0, 3);
      }
      player.compositor.backgroundOverride = null;
      return out;
    },
    { x: lx + 30, y: ly + 50 },
  );
  // logo fill: color (230, 60, 120), alpha 200; background 0.5 gray
  const c = [230, 60, 120].map((v) => v / 255);
  const d = 128 / 255; // 0.5 stored in an 8-bit framebuffer
  const expected: Record<string, (ci: number, a: number) => number> = {
    'normal@1': (ci, a) => ci * a + d * (1 - a),
    'normal@0.5': (ci, a) => ci * a * 0.5 + d * (1 - a * 0.5),
    'add@1': (ci, a) => Math.min(1, d + ci * a),
    'multiply@1': (ci, a) => d * (1 - a + ci * a),
    'screen@1': (ci, a) => d + ci * a - d * ci * a,
  };
  const a = 200 / 255;
  for (const [k, f] of Object.entries(expected)) {
    const want = c.map((ci) => Math.round(f(ci, a) * 255));
    console.log(`  ${k.padEnd(11)} got ${got[k].join(',')}  want ${want.join(',')}`);
    got[k].forEach((v, i) => expect(Math.abs(v - want[i]), `${k} channel ${i}`).toBeLessThanOrEqual(2));
  }
});

test('9.7 seeking and stepping land exactly on the requested frame', async ({ page }) => {
  await openTestFile(page);
  for (const t of [0, 1, 45, 59, 60, 61, 119, 333, 554, 555, 598, 599, 300, 12, 44]) {
    await page.evaluate((t) => window.__lvf.player.seek(t), t);
    await settle(page);
    await expectInSync(page, t);
  }
  await page.evaluate(() => window.__lvf.player.seek(60));
  await settle(page);
  await page.keyboard.press('ArrowLeft'); // across a RAP: decode from frame 0
  await settle(page);
  await expectInSync(page, 59);
  await page.keyboard.press('ArrowRight');
  await settle(page);
  await expectInSync(page, 60);
  expect((await stats(page)).p6Failures).toBe(0);
});

test('P5 hidden layers keep decoding and reappear in sync immediately', async ({ page }) => {
  await openTestFile(page);
  await page.evaluate(() => {
    const { player } = window.__lvf;
    player.source!.meta.layers.forEach((L, i) => {
      if (L.id !== 'bg') player.setLayerVisible(i, false);
    });
  });
  await page.keyboard.press('Space');
  await page.waitForTimeout(2500);
  await page.keyboard.press('Space');
  await settle(page);
  const b = await page.evaluate((pr) => {
    const { player, readBarcodes } = window.__lvf;
    player.layerStates.forEach((_, i) => player.setLayerVisible(i, true));
    return { frame: player.currentFrame, codes: readBarcodes(player, pr) };
  }, probe);
  expect(b.frame).toBeGreaterThan(45);
  expect(Object.keys(b.codes).sort()).toEqual(['bg', 'calib', 'exact', 'square', 'sync_a', 'sync_b']);
  for (const n of Object.values(b.codes)) expect(n).toBe(b.frame);
});

test('11.2-8 long playback: live VideoFrames and memory stay flat', async ({ page }) => {
  const seconds = Number(process.env.LVF_SOAK_SECONDS ?? 60);
  test.setTimeout((seconds + 120) * 1000);
  const leaks: string[] = [];
  page.on('console', (m) => {
    if (/garbage collected|without (being )?closed|not closed/i.test(m.text())) leaks.push(m.text());
  });
  await openTestFile(page);
  const cdp = await page.context().newCDPSession(page);
  await page.locator('#loop').check();
  await page.keyboard.press('Space');
  const samples: { t: number; live: number; adopted: number; heap: number; backing: number; shown: number }[] = [];
  const step = Math.max(2, Math.round(seconds / 30));
  for (let t = step; t <= seconds; t += step) {
    await page.waitForTimeout(step * 1000);
    // a real full GC, then the exact heap figures (performance.memory mostly measures garbage)
    await cdp.send('HeapProfiler.collectGarbage');
    const u = (await cdp.send('Runtime.getHeapUsage')) as { usedSize: number; backingStorageSize?: number };
    const s = await page.evaluate(() => ({
      live: window.__lvf.frameStats.live,
      adopted: window.__lvf.frameStats.adopted,
      shown: window.__lvf.player.stats.shown,
    }));
    samples.push({ t, heap: u.usedSize, backing: u.backingStorageSize ?? 0, ...s });
  }
  await page.keyboard.press('Space');
  const st = await stats(page);
  const half = Math.max(1, Math.floor(samples.length / 2));
  const first = samples.slice(0, half);
  const second = samples.slice(half);
  const maxOf = (a: typeof samples, k: 'heap' | 'backing' | 'live') => Math.max(...a.map((s) => s[k]));
  const mb = (x: number) => (x / 1e6).toFixed(2);
  console.log(
    `  ${seconds}s: shown ${st.shown} frames (${Math.floor(st.shown / N)} loops), VideoFrames decoded ${samples.at(-1)!.adopted}, ` +
      `live VideoFrames max ${maxOf(samples, 'live')}`,
  );
  console.log(`  JS heap after GC: max ${mb(maxOf(first, 'heap'))} MB (1st half) → ${mb(maxOf(second, 'heap'))} MB (2nd half); ` +
    `ArrayBuffers: ${mb(maxOf(first, 'backing'))} → ${mb(maxOf(second, 'backing'))} MB`);
  expect(leaks).toEqual([]);
  expect(st.p6Failures + st.p1Failures).toBe(0);
  expect(maxOf(samples, 'live')).toBeLessThanOrEqual(9 * 18); // 9 planes × (16 in flight + ready + shown)
  expect(samples.at(-1)!.shown).toBeGreaterThan(minFrames(seconds * 20));
  expect(maxOf(second, 'heap')).toBeLessThan(maxOf(first, 'heap') + 1e6);
  expect(maxOf(second, 'backing')).toBeLessThan(maxOf(first, 'backing') + 4e6);
});
