import { expect, type Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Player } from '../src/player';
import type { BarcodeProbe, DrawSample } from '../src/testing';
import type * as testing from '../src/testing';
import type { frameStats } from '../src/decode/frames';

export const ASSETS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../test_assets');
export const TEST_FILE = path.join(ASSETS, 'test.lvd');
/** The fflv command (`cargo build --release` builds it; FFLV_BIN overrides). */
export const FFLV = process.env.FFLV_BIN ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../target/release/fflv');

export interface Probes {
  canvas: [number, number];
  fps: number;
  frame_count: number;
  gop: number;
  barcode: { cell: number; bits: number; cells: number; layers: Record<string, { x: number; y: number; start_frame: number; end_frame: number }> };
  sync_band: { rect: [number, number, number, number]; line_rows: [number, number] };
  calib: { rect: [number, number, number, number]; ramp_rows: [number, number] };
  logo: { rect: [number, number, number, number]; start_frame: number; end_frame: number };
  exact: { rect: [number, number, number, number]; pattern_rows: [number, number]; ramp_rows: [number, number] };
  flash: { frames_per_second: number; border: number };
  beep: { hz: number; seconds: number };
}

export function loadProbes(): Probes {
  return JSON.parse(fs.readFileSync(path.join(ASSETS, 'barcodes.json'), 'utf8'));
}

export function barcodeProbe(p: Probes): BarcodeProbe {
  return { cell: p.barcode.cell, cells: p.barcode.cells, layers: p.barcode.layers };
}

export function assetsAvailable(): boolean {
  return fs.existsSync(TEST_FILE) && fs.existsSync(path.join(ASSETS, 'barcodes.json'));
}

declare global {
  interface Window {
    __lvf: { player: Player; frameStats: typeof frameStats } & typeof testing;
    __rec?: () => DrawSample[];
  }
}

/** Open the page, load test.lvd through the file input and wait for frame 0. */
export async function openTestFile(page: Page, query = ''): Promise<void> {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(`/${query}`);
  await page.setInputFiles('#file-input', TEST_FILE);
  await page.waitForFunction(() => window.__lvf.player.mode === 'paused' && window.__lvf.player.current !== null, null, {
    timeout: 30_000,
  });
  expect(errors).toEqual([]);
}

/** Load the file the way `fflv view` does: ?src=/media/<file> (HTTP range requests). */
export async function openViaServer(page: Page, extra = ''): Promise<void> {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(`/?src=/media/test.lvd${extra}`);
  await page.waitForFunction(() => window.__lvf.player.mode === 'paused' && window.__lvf.player.current !== null, null, {
    timeout: 30_000,
  });
  expect(errors).toEqual([]);
}

export async function waitForMode(page: Page, mode: string, timeout = 30_000): Promise<void> {
  await page.waitForFunction((m) => window.__lvf.player.mode === m, mode, { timeout });
}

export async function barcodes(page: Page, probe: BarcodeProbe) {
  return page.evaluate((pr) => {
    const { player, readBarcodes } = window.__lvf;
    return { frame: player.currentFrame, codes: readBarcodes(player, pr), mode: player.mode };
  }, probe);
}

export async function startRecording(page: Page, probe: BarcodeProbe): Promise<void> {
  await page.evaluate((pr) => {
    window.__rec = window.__lvf.recordDraws(window.__lvf.player, pr);
  }, probe);
}

export async function stopRecording(page: Page): Promise<DrawSample[]> {
  return page.evaluate(() => window.__rec!());
}

export async function stats(page: Page) {
  return page.evaluate(() => ({ ...window.__lvf.player.stats, live: window.__lvf.frameStats.live, peak: window.__lvf.frameStats.peak }));
}

/** Every barcode on screen must equal the composite frame's index. Returns the mismatches. */
export function mismatches(samples: DrawSample[]): string[] {
  const bad: string[] = [];
  for (const s of samples) {
    for (const [id, n] of Object.entries(s.codes)) {
      if (n !== s.frame) bad.push(`frame ${s.frame}: layer ${id} shows ${n}`);
    }
  }
  return bad;
}
