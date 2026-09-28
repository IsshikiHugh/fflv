/**
 * The debugging workflow around `fflv view`: loading over HTTP, lossless layers, number keys, and
 * following edits to the file while it is open.
 */
import { expect, test, type Page } from '@playwright/test';
import { execFileSync, spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  ASSETS,
  FFLV,
  TEST_FILE,
  assetsAvailable,
  barcodeProbe,
  barcodes,
  loadProbes,
  minFrames,
  mismatches,
  openViaServer,
  startRecording,
  stats,
  stopRecording,
  type Probes,
} from './helpers';

test.skip(!assetsAvailable(), 'run `fflv testsrc` first');

const P: Probes = assetsAvailable() ? loadProbes() : (null as never);
const probe = P ? barcodeProbe(P) : (null as never);

async function settle(page: Page) {
  await page.waitForFunction(() => ['paused', 'ended'].includes(window.__lvf.player.mode), null, { timeout: 30_000 });
}

async function seek(page: Page, f: number) {
  await page.evaluate((f) => window.__lvf.player.seek(f), f);
  await settle(page);
  expect(await page.evaluate(() => window.__lvf.player.currentFrame)).toBe(f);
}

test('fflv view: plays the file through HTTP range requests, in sync', async ({ page }) => {
  await openViaServer(page);
  // read over HTTP (an HttpByteSource has a url), not from a local File
  expect(await page.evaluate(() => (window.__lvf.player.source!.bytes as unknown as { url?: string }).url)).toBe('/media/test.lvd');
  expect((await barcodes(page, probe)).codes).toEqual({ bg: 0, sync_a: 0, sync_b: 0, calib: 0, exact: 0 });
  await startRecording(page, probe);
  await page.keyboard.press('Space');
  await page.waitForTimeout(2500);
  await page.keyboard.press('Space');
  const samples = await stopRecording(page);
  expect(samples.length).toBeGreaterThan(minFrames(50));
  expect(mismatches(samples)).toEqual([]);
  expect((await stats(page)).p6Failures).toBe(0);
  await expect(page).toHaveTitle('test.lvd — LVF');
});

test('lossless layer: every pixel arrives exactly (RGB and alpha)', async ({ page }) => {
  await openViaServer(page);
  for (const f of [0, 77, 301, 599]) {
    await seek(page, f);
    const res = await page.evaluate(
      ({ f, rect }) => {
        const { player } = window.__lvf;
        const [x0, y0] = rect;
        // RGB pattern rows (alpha 255): the canvas must hold the pattern bit for bit
        player.compositor.draw(player.current, player.layerStates);
        const px = player.compositor.readPixels(x0, y0, 256, 64);
        let rgbErrors = 0;
        for (let y = 0; y < 64; y++)
          for (let x = 0; x < 256; x++) {
            const i = (y * 256 + x) * 4;
            const want = [(x * 7 + f) & 255, (y * 13 + 3 * f) & 255, (x ^ y ^ f) & 255];
            if (px[i] !== want[0] || px[i + 1] !== want[1] || px[i + 2] !== want[2]) rgbErrors++;
          }
        // alpha ramp rows: white with alpha = x, over black with everything else hidden → value = x
        const saved = player.layerStates.map((s) => ({ ...s }));
        player.layerStates.forEach((s, i) => (s.visible = player.source!.meta.layers[i].id === 'exact'));
        player.compositor.backgroundOverride = [0, 0, 0];
        player.compositor.draw(player.current, player.layerStates);
        const ramp = player.compositor.readPixels(x0, y0 + 70, 256, 1);
        player.compositor.backgroundOverride = null;
        saved.forEach((s, i) => Object.assign(player.layerStates[i], s));
        let alphaErrors = 0;
        for (let x = 0; x < 256; x++) if (ramp[x * 4] !== x) alphaErrors++;
        return { rgbErrors, alphaErrors };
      },
      { f, rect: P.exact.rect },
    );
    expect(res, `frame ${f}`).toEqual({ rgbErrors: 0, alphaErrors: 0 });
  }
});

test('number keys toggle, solo and restore layers', async ({ page }) => {
  await openViaServer(page);
  const visible = () => page.evaluate(() => window.__lvf.player.layerStates.map((s) => s.visible));
  const ids = await page.evaluate(() => window.__lvf.player.source!.meta.layers.map((L) => L.id));
  // panel order is top of the stack first: exact (z 6), logo (5), calib (4), ...
  await expect(page.locator('#layers li .name .key').first()).toHaveText('1');
  await page.keyboard.press('1');
  expect((await visible())[ids.indexOf('exact')]).toBe(false);
  await expect(page.locator(`#layers li[data-index="${ids.indexOf('exact')}"] input[type=checkbox]`)).not.toBeChecked();
  await page.keyboard.press('Shift+3'); // solo calib
  expect(await visible()).toEqual(ids.map((id) => id === 'calib'));
  expect(Object.keys((await barcodes(page, probe)).codes)).toEqual(['calib']);
  await page.keyboard.press('Shift+3'); // again: everything back
  expect((await visible()).every(Boolean)).toBe(true);
  await page.keyboard.press('2');
  await page.keyboard.press('0');
  expect((await visible()).every(Boolean)).toBe(true);
});

test.describe('following edits to the file', () => {
  let server: ChildProcess;
  let dir: string;
  let file: string;
  const port = 4792;

  test.beforeEach(async () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fflv-e2e-'));
    file = path.join(dir, 'watched.lvd');
    fs.copyFileSync(TEST_FILE, file);
    server = spawn(FFLV, ['view', file, '--port', String(port), '--no-open'], { stdio: 'ignore' });
    for (let i = 0; i < 100; i++) {
      try {
        if ((await fetch(`http://127.0.0.1:${port}/`)).ok) return;
      } catch {
        /* not up yet */
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    throw new Error('fflv view did not start');
  });

  test.afterEach(() => {
    server.kill();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  const fflv = (...args: string[]) => execFileSync(FFLV, args, { stdio: 'pipe' });
  const waitReload = (page: Page, n: number) =>
    page.waitForFunction((n) => window.__lvf.player.reloads >= n && window.__lvf.player.mode === 'paused', n, { timeout: 30_000 });

  test('reloads on change, keeps frame and UI settings, picks up new defaults', async ({ page }) => {
    await page.goto(`http://127.0.0.1:${port}/?src=/media/watched.lvd&watch=1`);
    await settle(page);
    await seek(page, 123);
    await page.locator('#layers li[data-index="0"] input[type=checkbox]').click(); // hide bg in the UI

    fflv('set', file, 'calib', 'opacity=0.5', 'name=changed');
    await waitReload(page, 1);
    let s = await page.evaluate(() => {
      const p = window.__lvf.player;
      const L = p.source!.meta.layers;
      return { frame: p.currentFrame, bg: p.layerStates[0].visible, calib: p.layerStates[L.findIndex((l) => l.id === 'calib')] };
    });
    expect(s).toEqual({ frame: 123, bg: false, calib: { visible: true, opacity: 0.5 } });
    await expect(page.locator('#layers')).toContainText('changed');

    fflv('add', file, '--still', path.join(ASSETS, 'logo.png'), '--id', 'extra', '--rect', '0,600,220,100');
    await waitReload(page, 2);
    await expect(page.locator('#layers')).toContainText('extra');

    fflv('rm', file, 'sync_b');
    await waitReload(page, 3);
    const b = await barcodes(page, probe);
    expect(b.frame).toBe(123);
    // bg is still hidden (UI setting kept); calib is at opacity 0.5 now, so its barcode is not read
    expect(Object.keys(b.codes).sort()).toEqual(['exact', 'square', 'sync_a']);
    for (const n of Object.values(b.codes)) expect(n).toBe(123);
  });

  test('keeps watching after a failed load and recovers when the file is complete again', async ({ page }) => {
    await page.goto(`http://127.0.0.1:${port}/?src=/media/watched.lvd&watch=1`);
    await settle(page);
    await seek(page, 77);
    await page.locator('#layers li[data-index="0"] input[type=checkbox]').click(); // hide bg in the UI

    // A writer that rewrites the file in place (not atomically) is caught half-way ...
    const good = fs.readFileSync(TEST_FILE);
    fs.writeFileSync(file, good.subarray(0, good.length >> 1));
    await page.waitForFunction(() => window.__lvf.player.mode === 'error', null, { timeout: 15_000 });
    // ... and then finishes: the page must pick the file up again, where it was.
    fs.writeFileSync(file, good);
    await waitReload(page, 1);
    const b = await barcodes(page, probe);
    expect(b.frame).toBe(77);
    expect(Object.keys(b.codes)).not.toContain('bg'); // UI setting from before the failure kept
    for (const n of Object.values(b.codes)) expect(n).toBe(77);
  });
});
