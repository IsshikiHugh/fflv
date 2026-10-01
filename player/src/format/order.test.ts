import { describe, expect, it } from 'vitest';
import type { LayerMeta } from './lvf';
import { mergeOrder, moveTo, sameOrder, zOrder } from './order';

describe('zOrder', () => {
  it('sorts by z, ties in file order', () => {
    const layers = [{ z: 2 }, { z: 0 }, { z: 2 }, { z: -1 }] as LayerMeta[];
    expect(zOrder(layers)).toEqual([3, 1, 0, 2]);
  });
});

describe('mergeOrder', () => {
  const ids = ['a', 'b', 'c', 'd'];
  it('applies the kept order', () => {
    expect(mergeOrder([0, 1, 2, 3], ids, ['d', 'c', 'b', 'a'])).toEqual([3, 2, 1, 0]);
  });
  it('keeps new layers where the file puts them', () => {
    // c is new: it stays in slot 2, a/b/d take the other slots in the kept order
    expect(mergeOrder([0, 1, 2, 3], ids, ['d', 'b', 'a'])).toEqual([3, 1, 2, 0]);
  });
  it('ignores removed layers', () => {
    expect(mergeOrder([0, 1], ['a', 'b'], ['b', 'gone', 'a'])).toEqual([1, 0]);
  });
});

describe('moveTo', () => {
  it('moves one layer to a position', () => {
    expect(moveTo([0, 1, 2, 3], 0, 3)).toEqual([1, 2, 3, 0]);
    expect(moveTo([0, 1, 2, 3], 3, 0)).toEqual([3, 0, 1, 2]);
    expect(moveTo([0, 1, 2, 3], 1, 2)).toEqual([0, 2, 1, 3]);
    expect(moveTo([0, 1, 2], 1, 99)).toEqual([0, 2, 1]);
    expect(sameOrder(moveTo([0, 1, 2], 1, 1), [0, 1, 2])).toBe(true);
  });
});
