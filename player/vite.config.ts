import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  // The built player is compiled into the fflv binary (crates/fflv/src/view.rs), which serves it
  // for `fflv view`; rebuild fflv after `npm run build`.
  build: { target: 'es2022', sourcemap: true, outDir: '../crates/fflv/viewer', emptyOutDir: true },
  test: { include: ['src/**/*.test.ts'] },
});
