/**
 * Random access to an .lvd file: only the header, metadata and index are held in memory;
 * composite frames are read on demand in bounded batches (files may be GBs).
 */
import { BlobByteSource, type ByteSource } from './bytes';
import {
  HEADER_SIZE,
  INDEX_ENTRY_SIZE,
  INDEX_HEADER_SIZE,
  IndexTable,
  LvfFormatError,
  parseCau,
  parseHeader,
  parseIndex,
  parseMeta,
  videoLayerIndices,
  type LvfHeader,
  type LvfMeta,
  type ParsedCau,
  type StillLayerMeta,
} from './lvf';

export class LvfSource {
  private constructor(
    readonly bytes: ByteSource,
    readonly header: LvfHeader,
    readonly meta: LvfMeta,
    readonly index: IndexTable,
    readonly videoLayers: number[],
  ) {}

  static async open(src: ByteSource | Blob, name?: string): Promise<LvfSource> {
    const bytes = src instanceof Blob ? new BlobByteSource(src, name) : src;
    const header = parseHeader(await bytes.read(0, HEADER_SIZE), bytes.size);
    const metaBytes = new Uint8Array(await bytes.read(header.metaOffset, header.metaOffset + header.metaLength));
    const meta = parseMeta(metaBytes, header.cauOffset - header.resourcesOffset);
    const indexEnd = header.indexOffset + INDEX_HEADER_SIZE + meta.frame_count * INDEX_ENTRY_SIZE;
    if (indexEnd > bytes.size) throw new LvfFormatError('index table is truncated');
    const index = parseIndex(await bytes.read(header.indexOffset, indexEnd), header, meta.frame_count);
    return new LvfSource(bytes, header, meta, index, videoLayerIndices(meta));
  }

  get name(): string {
    return this.bytes.name;
  }

  get frameCount(): number {
    return this.meta.frame_count;
  }

  stillBlob(L: StillLayerMeta): Promise<Blob> {
    const a = this.header.resourcesOffset + L.resource.offset;
    return this.bytes.slice(a, a + L.resource.length, L.resource.mime);
  }

  /**
   * Read consecutive composite frames starting at `first`: at least one, then as many as fit in
   * `maxFrames` / `maxBytes`. Each is parsed and checked against the index and metadata.
   * Aborting `signal` rejects with an AbortError.
   */
  async readCaus(first: number, maxFrames: number, maxBytes: number, signal?: AbortSignal): Promise<ParsedCau[]> {
    const n = this.frameCount;
    if (first >= n) return [];
    const [start] = this.index.range(first);
    let last = first;
    while (last + 1 < n && last + 1 - first < maxFrames && this.index.range(last + 1)[1] - start <= maxBytes) last++;
    const end = this.index.range(last)[1];
    const u8 = new Uint8Array(await this.bytes.read(start, end, signal));
    if (u8.byteLength !== end - start) throw new LvfFormatError(`short read at offset ${start}`);
    const out: ParsedCau[] = [];
    for (let f = first; f <= last; f++) {
      const [a, b] = this.index.range(f);
      const cau = parseCau(u8, a - start, b - start, f, this.meta, this.videoLayers);
      if (cau.rap !== (this.index.rap[f] === 1)) throw new LvfFormatError(`frame ${f}: RAP flag differs from the index (I10)`);
      out.push(cau);
    }
    return out;
  }
}
