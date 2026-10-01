/**
 * Where the bytes of an .lvd come from: a local File/Blob (Blob.slice) or a URL served by
 * `fflv view` (HTTP range requests). Either way only the requested ranges are read.
 */
export interface ByteSource {
  readonly name: string;
  readonly size: number;
  /** Bytes [start, end). Aborting `signal` rejects with an AbortError. */
  read(start: number, end: number, signal?: AbortSignal): Promise<ArrayBuffer>;
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

  async read(start: number, end: number, signal?: AbortSignal): Promise<ArrayBuffer> {
    signal?.throwIfAborted();
    const buf = await this.blob.slice(start, end).arrayBuffer();
    signal?.throwIfAborted(); // a local read cannot be cancelled, but its result can be dropped
    return buf;
  }

  async slice(start: number, end: number, type = ''): Promise<Blob> {
    return this.blob.slice(start, end, type);
  }
}

/**
 * Bytes fetched by the first request when a URL is opened. The header and (normally) the metadata
 * lie at the start of the file, so opening takes two round trips: this one and the index.
 */
const PREFIX_BYTES = 64 << 10;

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
    /** The first bytes of the file, fetched by open(); reads inside it are served from memory. */
    private readonly prefix: ArrayBuffer,
  ) {}

  /** The size comes from the Content-Range of a first range request, which also fetches the prefix. */
  static async open(url: string, name?: string): Promise<HttpByteSource> {
    const ctl = new AbortController();
    const res = await fetch(url, { headers: { Range: `bytes=0-${PREFIX_BYTES - 1}` }, cache: 'no-store', signal: ctl.signal });
    if (res.status !== 206) {
      ctl.abort(); // a 200 would be the whole (possibly huge) file
      if (!res.ok) throw new Error(`cannot open ${url}: HTTP ${res.status}`);
      throw new Error(`${url}: server does not support range requests`);
    }
    const m = /^bytes \d+-\d+\/(\d+)$/.exec(res.headers.get('Content-Range')?.trim() ?? '');
    const size = m ? Number(m[1]) : NaN;
    if (!Number.isSafeInteger(size) || size <= 0) {
      ctl.abort();
      throw new Error(`${url}: server did not report a size`);
    }
    const prefix = await res.arrayBuffer();
    if (prefix.byteLength !== Math.min(PREFIX_BYTES, size)) throw new Error(`${url}: short read (${prefix.byteLength} bytes)`);
    const fallback = decodeURIComponent(new URL(url, location.href).pathname.split('/').pop() || 'file.lvd');
    return new HttpByteSource(url, name ?? fallback, size, res.headers.get('ETag'), prefix);
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

  async read(start: number, end: number, signal?: AbortSignal): Promise<ArrayBuffer> {
    if (end <= start) return new ArrayBuffer(0);
    if (end <= this.prefix.byteLength) return this.prefix.slice(start, end);
    const headers: Record<string, string> = { Range: `bytes=${start}-${end - 1}` };
    if (this.etag) headers['If-Match'] = this.etag;
    const res = await fetch(this.url, { headers, cache: 'no-store', signal });
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
