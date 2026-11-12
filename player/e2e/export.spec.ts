/**
 * Export in the browser (WebCodecs → MP4): for a file opened in the page, where no fflv can render,
 * and as a choice under `fflv view`. The video must match what `fflv render` writes.
 */
import { expect, test, type Page } from '@playwright/test';
import { execFileSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { FFLV, TEST_FILE, assetsAvailable, openTestFile, openViaServer } from './helpers';

test.skip(!assetsAvailable(), 'run `fflv testsrc` first');

let dir: string | null = null;
const tmp = (name: string) => path.join((dir ??= fs.mkdtempSync(path.join(os.tmpdir(), 'fflv-e2e-browser-export-'))), name);
test.afterAll(() => {
  if (dir) fs.rmSync(dir, { recursive: true, force: true });
});

/** codec, width, height and frame count of a video. */
function probe(file: string): string {
  const out = execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=codec_name,width,height,nb_read_frames', '-of', 'csv=p=0', file]);
  return out.toString().trim();
}

/** Mean PSNR (dB) of `a` against `b`, frame by frame. */
function psnr(a: string, b: string): number {
  const r = spawnSync('ffmpeg', ['-hide_banner', '-i', a, '-i', b, '-lavfi', 'psnr', '-f', 'null', '-'], { encoding: 'utf8' });
  const m = /average:([\d.]+|inf)/.exec(r.stderr);
  return m ? (m[1] === 'inf' ? Infinity : Number(m[1])) : NaN;
}

/** Without a save picker the video comes as a download: no file dialog to drive in a test. */
async function noPicker(page: Page) {
  await page.addInitScript(() => delete (window as { showSaveFilePicker?: unknown }).showSaveFilePicker);
}

test('a file opened in the page: exported in the browser, as fflv render does', async ({ page }) => {
  await noPicker(page);
  await openTestFile(page);
  await expect(page.locator('#export')).toBeVisible();
  await page.locator('#btn-export').click();
  // no fflv for this file: no choice, no path, MP4 only
  await expect(page.locator('#export-engine')).toBeHidden();
  await expect(page.locator('#export-path')).toBeHidden();
  expect(await page.locator('#export-format option').allTextContents()).toEqual(['mp4']);
  await page.locator('#export-start').fill('100');
  await page.locator('#export-end').fill('149');
  await page.locator('#export-end').press('Tab');
  const download = page.waitForEvent('download', { timeout: 120_000 });
  await page.locator('#export-go').click();
  const d = await download;
  expect(d.suggestedFilename()).toBe('test_100-149.mp4');
  await expect(page.locator('#export-status')).toHaveText('saved test_100-149.mp4');
  const got = tmp('browser.mp4');
  await d.saveAs(got);

  const meta = await page.evaluate(() => window.__lvf.player.source!.meta.canvas);
  const [codec, w, h, n] = probe(got).split(',');
  expect(['h264', 'vp9']).toContain(codec);
  expect([Number(w), Number(h), Number(n)]).toEqual([meta.width + (meta.width & 1), meta.height + (meta.height & 1), 50]);

  const ref = tmp('fflv.mp4');
  execFileSync(FFLV, ['render', TEST_FILE, '-f', '100:150', '-o', ref], { stdio: 'ignore' });
  expect(probe(ref).split(',')[3]).toBe('50');
  const db = psnr(got, ref);
  console.log(`browser export (${codec}) vs fflv render: PSNR ${db} dB`);
  expect(db).toBeGreaterThan(35);
});

test('fflv view: the browser is a choice; cancelling leaves the page usable', async ({ page }) => {
  await noPicker(page);
  await page.addInitScript(() => localStorage.removeItem('lvf.exportEngine'));
  await openViaServer(page);
  await page.locator('#btn-export').click();
  await expect(page.locator('#export-engine')).toBeVisible();
  await expect(page.locator('#export-engine')).toHaveValue('fflv'); // the default stays fflv
  await expect(page.locator('#export-path')).toBeVisible();
  await page.locator('#export-engine').selectOption('browser');
  await expect(page.locator('#export-path')).toBeHidden();
  expect(await page.locator('#export-format option').allTextContents()).toEqual(['mp4']);
  await page.locator('#export-go').click(); // the whole file
  await expect(page.locator('#export-status')).toContainText('exporting');
  await expect(page.locator('#btn-export')).toHaveText('Cancel export');
  await page.locator('#btn-export').click();
  await expect(page.locator('#export-status')).toHaveText('export cancelled');
  await expect(page.locator('#btn-export')).toHaveText('Export…');
  // back to fflv: its formats and path again
  await page.locator('#btn-export').click();
  await page.locator('#export-engine').selectOption('fflv');
  await expect(page.locator('#export-path')).toBeVisible();
  expect((await page.locator('#export-format option').allTextContents()).length).toBeGreaterThan(1);
});
