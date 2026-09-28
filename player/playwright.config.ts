import { defineConfig } from '@playwright/test';

const port = 4791;

export default defineConfig({
  testDir: 'e2e',
  timeout: 180_000,
  workers: 1,
  // CI runners are slower and noisier than a laptop: give a timing-sensitive case a second chance there.
  retries: process.env.CI ? 1 : 0,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    launchOptions: { args: ['--autoplay-policy=no-user-gesture-required'] },
  },
  // `fflv view` serves the player compiled into it and the test file with range requests.
  // Build it first (`cargo build --release` after `npm run build`), or point FFLV_BIN elsewhere.
  webServer: {
    command: `${process.env.FFLV_BIN ?? '../target/release/fflv'} view ../test_assets/test.lvd --port ${port} --no-open`,
    url: `http://127.0.0.1:${port}/`,
    reuseExistingServer: false,
  },
  projects: [
    { name: 'chromium', use: { browserName: 'chromium' } },
    { name: 'msedge', use: { browserName: 'chromium', channel: 'msedge' } },
  ],
});
