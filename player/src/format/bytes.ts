/**
 * Where the bytes of an .lvd come from: a local File/Blob (Blob.slice) or a URL served by
 * `fflv view` (HTTP range requests). Either way only the requested ranges are read.
 */
export interface ByteSource {
  readonly name: string;
  readonly size: number;
  read(start: number, end: number): Promise<ArrayBuffer>;
  slice(start: number, end: number, type?: string): Promise<Blob>;
}

/** The file behind a URL changed while we were reading it (a new version was written). */
export class SourceChangedError extends Error {
  constructor(readonly url: string) {
    super(`${url} changed on the server`);
    this.name = 'SourceChangedError';
  }
}

export class BlobByteSource implements ByteSource {
  constructor(
    private readonly blob: Blob,
    readonly name: string = (blob as File).name ?? 'file.lvd',
  ) {}

  get size(): number {
    return this.blob.size;
  }

  read(start: number, end: number): Promise<ArrayBuffer> {
    return this.blob.slice(start, end).arrayBuffer();
  }

  async slice(start: number, end: number, type = ''): Promise<Blob> {
    return this.blob.slice(start, end, type);
  }
}

/**
 * Range reads over HTTP. The ETag seen when the source was opened is sent with every request
 * (If-Match), so a file replaced on the server is detected (412) instead of silently mixing old
 * and new bytes.
 */
export class HttpByteSource implements ByteSource {
  private constructor(
    readonly url: string,
    readonly name: string,
    readonly size: number,
    readonly etag: string | null,
  ) {}

  static async open(url: string, name?: string): Promise<HttpByteSource> {
    const res = await fetch(url, { method: 'HEAD', cache: 'no-store' });
    if (!res.ok) throw new Error(`cannot open ${url}: HTTP ${res.status}`);
    const size = Number(res.headers.get('Content-Length'));
    if (!Number.isFinite(size) || size <= 0) throw new Error(`${url}: server did not report a size`);
    if (res.headers.get('Accept-Ranges') !== 'bytes') throw new Error(`${url}: server does not support range requests`);
    const fallback = decodeURIComponent(new URL(url, location.href).pathname.split('/').pop() || 'file.lvd');
    return new HttpByteSource(url, name ?? fallback, size, res.headers.get('ETag'));
  }

  /** Current ETag on the server (null if the request fails). */
  static async currentEtag(url: string): Promise<string | null> {
    try {
      const res = await fetch(url, { method: 'HEAD', cache: 'no-store' });
      return res.ok ? res.headers.get('ETag') : null;
    } catch {
      return null;
    }
  }

  async read(start: number, end: number): Promise<ArrayBuffer> {
    if (end <= start) return new ArrayBuffer(0);
    const headers: Record<string, string> = { Range: `bytes=${start}-${end - 1}` };
    if (this.etag) headers['If-Match'] = this.etag;
    const res = await fetch(this.url, { headers, cache: 'no-store' });
    if (res.status === 412) throw new SourceChangedError(this.url);
    if (res.status !== 206 && res.status !== 200) throw new Error(`${this.url}: HTTP ${res.status} for bytes ${start}-${end - 1}`);
    const buf = await res.arrayBuffer();
    if (res.status === 200) return buf.slice(start, end); // server ignored Range
    if (buf.byteLength !== end - start) throw new Error(`${this.url}: short read (${buf.byteLength} of ${end - start} bytes)`);
    return buf;
  }

  async slice(start: number, end: number, type = ''): Promise<Blob> {
    return new Blob([await this.read(start, end)], { type });
  }
}
