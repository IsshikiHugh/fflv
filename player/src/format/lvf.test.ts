import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { LvfFormatError, parseHeader } from './lvf';
import { LvfSource } from './source';

const TEST_FILE = path.resolve(__dirname, '../../../test_assets/test.lvd');
const have = fs.existsSync(TEST_FILE);

describe('parseHeader', () => {
  it('rejects a file that is not LVF', () => {
    const buf = new Uint8Array(64);
    buf.set([0x4e, 0x4f, 0x50, 0x45]);
    expect(() => parseHeader(buf.buffer, 64)).toThrow(LvfFormatError);
  });
  it('rejects a truncated header', () => {
    expect(() => parseHeader(new ArrayBuffer(10), 10)).toThrow(LvfFormatError);
  });
});

describe.skipIf(!have)('LvfSource on test_assets/test.lvd', () => {
  const bytes = have ? fs.readFileSync(TEST_FILE) : Buffer.alloc(0);

  it('reads metadata and index', async () => {
    const src = await LvfSource.open(new Blob([bytes]), 'test.lvd');
    expect(src.meta.frame_count).toBe(600);
    expect(Array.from(src.index.rapFrames)).toEqual([0, 60, 120, 180, 240, 300, 360, 420, 480, 540]);
    expect(src.index.rapAtOrBefore(0)).toBe(0);
    expect(src.index.rapAtOrBefore(59)).toBe(0);
    expect(src.index.rapAtOrBefore(60)).toBe(60);
    expect(src.index.rapAtOrBefore(599)).toBe(540);
    expect(src.videoLayers).toEqual([0, 1, 2, 3, 4, 6]); // 5 is the still logo
  });

  it('reads composite frames in bounded batches', async () => {
    const src = await LvfSource.open(new Blob([bytes]), 'test.lvd');
    const caus = await src.readCaus(40, 10, 1 << 30);
    expect(caus.map((c) => c.frameIndex)).toEqual([40, 41, 42, 43, 44, 45, 46, 47, 48, 49]);
    for (const c of caus) {
      expect(c.entries.map((e) => e.layerIndex)).toEqual([0, 1, 2, 3, 4, 6]);
      const square = c.entries[3];
      expect(square.type).toBe(c.frameIndex >= 45 ? 1 : 0);
    }
    expect(caus[5].entries[3].key).toBe(true); // the late layer starts with a key frame
    const small = await src.readCaus(0, 100, 1); // byte limit still returns one frame
    expect(small).toHaveLength(1);
  });

  it('refuses a composite frame whose magic is damaged', async () => {
    const damaged = Buffer.from(bytes);
    const src = await LvfSource.open(new Blob([bytes]), 'test.lvd');
    const [start] = src.index.range(5);
    damaged[start] = 0x58;
    const bad = await LvfSource.open(new Blob([damaged]), 'bad.lvd');
    await expect(bad.readCaus(0, 10, 1 << 30)).rejects.toThrow(/composite frame 5: bad magic/);
  });
});
