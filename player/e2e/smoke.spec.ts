import { expect, test } from '@playwright/test';
import { assetsAvailable, barcodeProbe, barcodes, loadProbes, minFrames, mismatches, openTestFile, startRecording, stats, stopRecording } from './helpers';

test.skip(!assetsAvailable(), 'run `fflv testsrc` first');

test('loads the test file and plays in sync', async ({ page }) => {
  const probe = barcodeProbe(loadProbes());
  await openTestFile(page);
  const first = await barcodes(page, probe);
  expect(first.frame).toBe(0);
  expect(first.codes).toEqual({ bg: 0, sync_a: 0, sync_b: 0, calib: 0, exact: 0 });

  await startRecording(page, probe);
  await page.evaluate(() => window.__lvf.player.play());
  await page.waitForTimeout(3000);
  const samples = await stopRecording(page);
  await page.evaluate(() => window.__lvf.player.pause());
  const st = await stats(page);
  console.log(`drawn ${samples.length} frames, last ${samples.at(-1)?.frame}, stats ${JSON.stringify(st)}`);
  console.log(JSON.stringify(await page.evaluate(() => window.__lvf.player.debugInfo())));
  expect(mismatches(samples)).toEqual([]);
  expect(samples.length).toBeGreaterThan(minFrames(60));
  expect(st.p6Failures + st.p1Failures).toBe(0);
});

test('plays without audio on the performance.now() clock', async ({ page }) => {
  const probe = barcodeProbe(loadProbes());
  await openTestFile(page, '?noaudio');
  expect(await page.evaluate(() => window.__lvf.player.clock.kind)).toBe('performance');
  await startRecording(page, probe);
  await page.evaluate(() => window.__lvf.player.play());
  await page.waitForTimeout(2000);
  const samples = await stopRecording(page);
  await page.evaluate(() => window.__lvf.player.pause());
  expect(mismatches(samples)).toEqual([]);
  expect(samples.at(-1)!.frame).toBeGreaterThan(45);
  expect((await stats(page)).p6Failures).toBe(0);
});

test('a hidden tab does not make the video crawl through a backlog', async ({ page }) => {
  await openTestFile(page);
  await page.evaluate(() => window.__lvf.player.play());
  await page.waitForTimeout(500);
  // Simulate rAF being suspended for 4 s while the audio clock keeps running.
  await page.evaluate(() => {
    const t = performance.now();
    while (performance.now() - t < 4000) {
      /* block the main thread: no rAF, audio keeps playing */
    }
  });
  await page.waitForFunction(() => window.__lvf.player.stats.lateJumps > 0 && window.__lvf.player.mode === 'playing', null, { timeout: 5000 });
  const st = await page.evaluate(() => ({ frame: window.__lvf.player.currentFrame, clock: window.__lvf.player.clock.nowUs() }));
  expect(Math.abs(st.frame / 30 - st.clock / 1e6)).toBeLessThan(0.5);
});
