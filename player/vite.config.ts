import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  // Building fflv builds the player into the binary (crates/fflv/build.rs); `npm run build` writes
  // it here, for packaging (release builds, the sdist) or for building fflv without Node.
  build: { target: 'es2022', sourcemap: true, outDir: '../crates/fflv/viewer', emptyOutDir: true },
  test: { include: ['src/**/*.test.ts'] },
});
