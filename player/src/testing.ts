/**
 * Instrumentation used by the end-to-end tests (and handy from the console): reads the frame-number
 * barcodes that `fflv testsrc` (fflv/devtools/testsrc.py) draws into every video layer back from
 * the canvas, right after the compositor has drawn a composite frame.
 */
import type { Player } from './player';

export interface BarcodeProbe {
  cell: number;
  cells: number;
  /** canvas position of each layer's barcode, keyed by layer id */
  layers: Record<string, { x: number; y: number }>;
}

export interface DrawSample {
  frame: number;
  /** decoded barcode per visible, active, fully opaque layer (null = unreadable) */
  codes: Record<string, number | null>;
  syncFailures: number;
}

function decode(px: Uint8Array, cell: number, cells: number): number | null {
  const stride = cells * cell * 4;
  const row = Math.floor(cell / 2) * stride;
  const bits: number[] = [];
  for (let i = 0; i < cells; i++) {
    const p = row + (i * cell + Math.floor(cell / 2)) * 4;
    bits.push((px[p] + px[p + 1] + px[p + 2]) / 3 > 128 ? 1 : 0);
  }
  const data = bits.slice(1, -1);
  const parity = data.reduce((a, b) => a + b, 0) & 1;
  if (bits[0] !== 1 || parity !== bits[cells - 1]) return null;
  return data.reduce((a, b) => a * 2 + b, 0);
}

/** Barcodes of the layers that are visible, fully opaque and active in the frame on screen. */
export function readBarcodes(player: Player, probe: BarcodeProbe, redraw = true): Record<string, number | null> {
  const meta = player.source?.meta;
  const cur = player.current;
  if (!meta || !cur) return {};
  if (redraw) player.compositor.draw(cur, player.layerStates);
  const out: Record<string, number | null> = {};
  meta.layers.forEach((L, i) => {
    const pos = probe.layers[L.id];
    const st = player.layerStates[i];
    if (!pos || L.kind !== 'video' || !st.visible || st.opacity < 1) return;
    if (cur.frameIndex < L.start_frame || cur.frameIndex >= L.end_frame) return;
    const px = player.compositor.readPixels(pos.x, pos.y, probe.cells * probe.cell, probe.cell);
    out[L.id] = decode(px, probe.cell, probe.cells);
  });
  return out;
}

/** Record the barcodes of every composite frame the player draws. */
export function recordDraws(player: Player, probe: BarcodeProbe): () => DrawSample[] {
  const samples: DrawSample[] = [];
  const onDraw = () => {
    if (!player.current) return;
    samples.push({
      frame: player.current.frameIndex,
      codes: readBarcodes(player, probe, false),
      syncFailures: player.stats.p6Failures + player.stats.p1Failures,
    });
  };
  player.addEventListener('draw', onDraw);
  return () => {
    player.removeEventListener('draw', onDraw);
    return samples;
  };
}

/** Pixels of one canvas row (RGBA), read right after a redraw of the current frame. */
export function readRow(player: Player, y: number, x0 = 0, width?: number): number[] {
  if (!player.current) return [];
  const w = width ?? player.compositor.canvas.width - x0;
  player.compositor.draw(player.current, player.layerStates);
  return Array.from(player.compositor.readPixels(x0, y, w, 1));
}
