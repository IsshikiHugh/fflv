/**
 * Layer draw order: layer indices, bottom of the stack first.
 */
import type { LayerMeta } from './lvf';

/** The file's order: ascending z, then file order (spec 9.6). */
export function zOrder(layers: readonly LayerMeta[]): number[] {
  return layers.map((_, i) => i).sort((a, b) => layers[a].z - layers[b].z || a - b);
}

/**
 * The order after a reload of a changed file. `kept` is the order changed in the UI (layer ids,
 * bottom first): those layers keep it among themselves, in the places the file's order gives that
 * group; layers that are new keep their place in the file's order.
 */
export function mergeOrder(fileOrder: readonly number[], ids: readonly string[], kept: readonly string[]): number[] {
  const rank = new Map(kept.map((id, k) => [id, k]));
  const known = fileOrder.filter((i) => rank.has(ids[i])).sort((a, b) => rank.get(ids[a])! - rank.get(ids[b])!);
  let k = 0;
  return fileOrder.map((i) => (rank.has(ids[i]) ? known[k++] : i));
}

/** `order` with `layer` moved to position `to` (in the same bottom-first terms). */
export function moveTo(order: readonly number[], layer: number, to: number): number[] {
  const rest = order.filter((i) => i !== layer);
  rest.splice(Math.max(0, Math.min(rest.length, to)), 0, layer);
  return rest;
}

export const sameOrder = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((v, i) => v === b[i]);
