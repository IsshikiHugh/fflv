import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  // The built player ships inside the Python package, where `fflv view` serves it.
  build: { target: 'es2022', sourcemap: true, outDir: '../fflv/viewer', emptyOutDir: true },
  test: { include: ['src/**/*.test.ts'] },
});
