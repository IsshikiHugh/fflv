//! Streaming writer, random-access reader, copy-on-write metadata edits, atomic publishing.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::binary::{pack_index, peek_cau_size, unpack_index, Cau, CauRef, FileHeader, IndexEntry};
use crate::constants::*;
use crate::error::{format_err, Error, Result};
use crate::meta::Meta;

/// Standard JSON (RFC 8259, spec B.12), 2-space indentation, UTF-8.
pub fn encode_meta<T: Serialize>(meta: &T) -> Result<Vec<u8>> {
    let v = serde_json::to_value(meta).map_err(|e| Error::Value(e.to_string()))?;
    serde_json::to_vec_pretty(&v).map_err(|e| Error::Value(e.to_string()))
}

/// Room reserved for the metadata: twice its size (at least 4 KiB), in 4 KiB steps, so an edited
/// copy fits beside the current one (spec B.5).
pub fn meta_capacity_for(meta_len: usize) -> usize {
    let need = (2 * meta_len).max(4096);
    need.div_ceil(4096) * 4096
}

// ------------------------------------------------------------------------------------------------
// Writer
// ------------------------------------------------------------------------------------------------
/// Writes header placeholder → meta → resources → CAUs → index, then back-patches the header.
/// It writes exactly what it is given; invariants are the caller's job (and the validator's).
pub struct LvfWriter {
    path: PathBuf,
    f: Option<BufWriter<File>>,
    pos: u64,
    pub header: FileHeader,
    pub index: Vec<IndexEntry>,
}

impl LvfWriter {
    pub fn create(path: impl AsRef<Path>) -> Result<LvfWriter> {
        let path = path.as_ref().to_path_buf();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            fs::create_dir_all(dir)?;
        }
        let f = BufWriter::with_capacity(1 << 20, File::create(&path)?);
        Ok(LvfWriter { path, f: Some(f), pos: 0, header: FileHeader::default(), index: Vec::new() })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn out(&mut self) -> &mut BufWriter<File> {
        self.f.as_mut().expect("writer already finished")
    }

    fn put(&mut self, data: &[u8]) -> Result<()> {
        self.out().write_all(data)?;
        self.pos += data.len() as u64;
        Ok(())
    }

    /// `meta_capacity`: bytes reserved for the metadata (None: meta_capacity_for; Some(0): none).
    pub fn begin(&mut self, meta: &[u8], resources: &[u8], meta_capacity: Option<usize>) -> Result<()> {
        let cap = meta_capacity.unwrap_or_else(|| meta_capacity_for(meta.len())).max(meta.len());
        self.put(&[0u8; HEADER_SIZE])?;
        self.header.meta_offset = self.pos;
        self.header.meta_length = u32::try_from(meta.len())
            .map_err(|_| Error::Value(format!("metadata of {} bytes is too large", meta.len())))?;
        self.put(meta)?;
        self.put(&vec![b' '; cap - meta.len()])?;
        self.header.resources_offset = self.pos;
        self.put(resources)?;
        self.header.cau_offset = self.pos;
        Ok(())
    }

    /// Append one composite frame; returns its offset. A frame whose counts or sizes do not fit
    /// their fields is an error, and nothing is written.
    pub fn write_cau(&mut self, cau: &Cau) -> Result<u64> {
        let offset = self.pos;
        let n = cau.write_to(self.out())?;
        self.pos += n as u64;
        self.push_index(offset, cau.frame_index, cau.flags);
        Ok(offset)
    }

    /// Append raw composite-frame bytes (tests); `frame_index`/`flags` go into the index.
    pub fn write_raw(&mut self, data: &[u8], frame_index: u32, flags: u8) -> Result<u64> {
        let offset = self.pos;
        self.put(data)?;
        self.push_index(offset, frame_index, flags);
        Ok(offset)
    }

    fn push_index(&mut self, offset: u64, frame_index: u32, flags: u8) {
        let idx_flags = if flags & CAU_FLAG_RAP != 0 { INDEX_FLAG_RAP } else { 0 };
        self.index.push(IndexEntry { frame_index, flags: idx_flags, cau_offset: offset });
    }

    /// Write the index (or `index_bytes`) and back-patch the header. `meta` replaces the metadata
    /// written by begin() (it must fit in the reserved region) — for streaming writers that only
    /// know the final frame count at the end. On an error the file is removed.
    pub fn finish(mut self, index_bytes: Option<Vec<u8>>, meta: Option<&[u8]>) -> Result<()> {
        let result = self.finish_inner(index_bytes, meta);
        if result.is_err() {
            self.f.take(); // closed before it is removed (Windows)
            let _ = fs::remove_file(&self.path);
        }
        result
    }

    fn finish_inner(&mut self, index_bytes: Option<Vec<u8>>, meta: Option<&[u8]>) -> Result<()> {
        self.header.index_offset = self.pos;
        let idx = index_bytes.unwrap_or_else(|| pack_index(&self.index, MAGIC_INDEX, None));
        self.put(&idx)?;
        let mut f = self.f.take().unwrap().into_inner().map_err(|e| Error::Io(e.into_error()))?;
        if let Some(data) = meta {
            let cap = (self.header.resources_offset - self.header.meta_offset) as usize;
            let len = u32::try_from(data.len()).ok().filter(|_| data.len() <= cap);
            let Some(len) = len else {
                return Err(Error::Value(format!("metadata ({} bytes) exceeds the reserved {cap} bytes", data.len())));
            };
            f.seek(SeekFrom::Start(self.header.meta_offset))?;
            f.write_all(data)?;
            f.write_all(&vec![b' '; cap - data.len()])?;
            self.header.meta_length = len;
        }
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&self.header.pack())?;
        f.sync_all()?; // durable before it is renamed into place (spec B.11)
        Ok(())
    }

    /// Discard the file.
    pub fn abort(mut self) {
        self.f.take();
        let _ = fs::remove_file(&self.path);
    }
}

impl Drop for LvfWriter {
    /// A writer dropped before finish() (an error or a panic on the way) removes its file.
    fn drop(&mut self) {
        if self.f.take().is_some() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Reader
// ------------------------------------------------------------------------------------------------
/// Random-access reader. Reads are positional (no shared file cursor), so one reader can be shared
/// between threads, e.g. behind an `Arc`.
pub struct LvfReader {
    f: File,
    pub path: PathBuf,
    pub file_size: u64,
    pub header: FileHeader,
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn read_at(f: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match f.seek_read(buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

impl LvfReader {
    pub fn open(path: impl AsRef<Path>) -> Result<LvfReader> {
        let path = path.as_ref().to_path_buf();
        let f = File::open(&path)?;
        LvfReader::from_file(f, path)
    }

    /// Read an already open file (its size and bytes come from this one open file).
    pub fn from_file(f: File, path: PathBuf) -> Result<LvfReader> {
        let file_size = f.metadata()?.len();
        let mut hb = [0u8; HEADER_SIZE];
        if file_size < HEADER_SIZE as u64 {
            return format_err(format!("file header needs {HEADER_SIZE} bytes, file has {file_size}"));
        }
        read_at(&f, &mut hb, 0)?;
        Ok(LvfReader { f, path, file_size, header: FileHeader::unpack(&hb)? })
    }

    pub fn file(&self) -> &File {
        &self.f
    }

    pub fn read(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.check_range(offset, len)?;
        let mut buf = vec![0u8; len];
        read_at(&self.f, &mut buf, offset)?;
        Ok(buf)
    }

    /// Fill `buf` with the bytes at `offset` (no allocation).
    pub fn read_into(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.check_range(offset, buf.len())?;
        Ok(read_at(&self.f, buf, offset)?)
    }

    fn check_range(&self, offset: u64, len: usize) -> Result<()> {
        if offset.checked_add(len as u64).is_none_or(|end| end > self.file_size) {
            return format_err(format!("short read: wanted {len} bytes at {offset}, file has {}", self.file_size));
        }
        Ok(())
    }

    pub fn meta_bytes(&self) -> Result<Vec<u8>> {
        let (o, l) = (self.header.meta_offset, self.header.meta_length as usize);
        self.read(o, l)
    }

    pub fn meta_json(&self) -> Result<serde_json::Value> {
        let b = self.meta_bytes()?;
        serde_json::from_slice(&b).map_err(|e| Error::Format(format!("metadata is not valid JSON: {e}")))
    }

    pub fn meta(&self) -> Result<Meta> {
        let b = self.meta_bytes()?;
        serde_json::from_slice(&b).map_err(|e| Error::Format(format!("metadata does not match the LVF schema: {e}")))
    }

    pub fn resource(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        match (self.header.resources_offset.checked_add(offset), usize::try_from(len)) {
            (Some(o), Ok(len)) => self.read(o, len),
            _ => format_err(format!("resource at {offset} ({len} bytes) is outside the file")),
        }
    }

    pub fn resources(&self) -> Result<Vec<u8>> {
        let (a, b) = (self.header.resources_offset, self.header.cau_offset);
        match b.checked_sub(a) {
            Some(n) => self.read(a, n as usize),
            None => format_err(format!("cau_offset {b} is before resources_offset {a}")),
        }
    }

    /// (magic, declared count, entries). Reads exactly the table its header declares (bytes after
    /// it are ignored; the validator reports them).
    pub fn index(&self) -> Result<([u8; 4], u32, Vec<IndexEntry>)> {
        let o = self.header.index_offset;
        let Some(n) = self.file_size.checked_sub(o) else {
            return format_err(format!("index_offset {o} is past the end of the file ({} bytes)", self.file_size));
        };
        if n < INDEX_HEADER_SIZE as u64 {
            return format_err("truncated index header");
        }
        let mut head = [0u8; INDEX_HEADER_SIZE];
        self.read_into(o, &mut head)?;
        let count = u32::from_le_bytes(head[4..8].try_into().unwrap());
        let need = INDEX_HEADER_SIZE as u64 + count as u64 * INDEX_ENTRY_SIZE as u64;
        if need > n {
            return format_err(format!("index declares {count} entries but only {n} bytes are present"));
        }
        let b = self.read(o, need as usize)?;
        unpack_index(&b)
    }

    /// The composite frame at `offset` (two reads: its size, then its bytes; the reader keeps no
    /// index to derive the size from).
    pub fn cau_at(&self, offset: u64) -> Result<(Cau, usize)> {
        let head = self.read(offset, 8)?;
        let size = peek_cau_size(&head, 0)?;
        let b = self.read(offset, size)?;
        Cau::unpack(&b, 0)
    }

    /// Walk the CAU region [start, end) sequentially (independent of the index), in large reads.
    pub fn caus(&self, start: Option<u64>, end: Option<u64>) -> CauIter<&LvfReader> {
        CauIter::new(self, start, end)
    }
}

/// Yields (offset, cau, size); an error ends the iteration. `R` is any handle to a reader
/// (`&LvfReader`, `Arc<LvfReader>`, ...). The region is read in large chunks into one reused
/// buffer; [`CauIter::next_ref`] borrows each frame from it instead of copying the payloads.
pub struct CauIter<R: Deref<Target = LvfReader>> {
    r: R,
    pos: u64,
    end: u64,
    buf: Vec<u8>,
    buf_start: u64,
    failed: bool,
}

impl<R: Deref<Target = LvfReader>> CauIter<R> {
    const CHUNK: usize = 4 << 20;

    pub fn new(r: R, start: Option<u64>, end: Option<u64>) -> CauIter<R> {
        let start = start.unwrap_or(r.header.cau_offset);
        let end = end.unwrap_or(r.header.index_offset);
        CauIter { r, pos: start, end, buf: Vec::new(), buf_start: start, failed: false }
    }

    /// Make [pos, pos+len) available in the buffer (the caller checked pos+len <= end). Bytes
    /// already read past `pos` are moved to the front and the buffer is refilled in place.
    fn ensure(&mut self, len: usize) -> Result<()> {
        let have_end = self.buf_start + self.buf.len() as u64;
        if self.pos >= self.buf_start && self.pos + len as u64 <= have_end {
            return Ok(());
        }
        let want = (len.max(Self::CHUNK) as u64).min(self.end - self.pos) as usize;
        let keep = if self.pos >= self.buf_start && self.pos < have_end {
            let rel = (self.pos - self.buf_start) as usize;
            self.buf.copy_within(rel.., 0);
            self.buf.len() - rel
        } else {
            0
        };
        self.buf.resize(want, 0); // only grows beyond `keep` (want >= len > keep)
        self.buf_start = self.pos;
        if let Err(e) = self.r.read_into(self.pos + keep as u64, &mut self.buf[keep..]) {
            self.buf.clear();
            return Err(e);
        }
        Ok(())
    }

    /// Check the next frame's size and bring it into the buffer: (offset, start in buf, size).
    fn locate(&mut self) -> Result<(u64, usize, usize)> {
        if self.pos + 8 > self.end {
            return format_err(format!("{} stray bytes before the index at offset {}", self.end - self.pos, self.pos));
        }
        self.ensure(8)?;
        let rel = (self.pos - self.buf_start) as usize;
        let size = peek_cau_size(&self.buf, rel)?;
        if size as u64 > self.end - self.pos {
            return format_err(format!(
                "CAU at offset {} ({size} bytes) runs past index_offset {}",
                self.pos, self.end
            ));
        }
        self.ensure(size)?;
        Ok((self.pos, (self.pos - self.buf_start) as usize, size))
    }

    /// Like [`Iterator::next`], but the frame borrows its payloads from the iterator's buffer (valid
    /// until the next call), so nothing is copied; [`CauRef::raw`] is the whole frame as stored.
    pub fn next_ref(&mut self) -> Option<Result<(u64, CauRef<'_>, usize)>> {
        if self.failed || self.pos >= self.end {
            return None;
        }
        let (off, rel, size) = match self.locate() {
            Ok(x) => x,
            Err(e) => {
                self.failed = true;
                return Some(Err(e));
            }
        };
        match Cau::unpack_ref(&self.buf[..rel + size], rel) {
            Ok((cau, size)) => {
                self.pos += size as u64;
                Some(Ok((off, cau, size)))
            }
            Err(e) => {
                self.failed = true;
                Some(Err(e))
            }
        }
    }
}

impl<R: Deref<Target = LvfReader>> Iterator for CauIter<R> {
    type Item = Result<(u64, Cau, usize)>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_ref().map(|r| r.map(|(off, cau, size)| (off, cau.to_cau(), size)))
    }
}

// ------------------------------------------------------------------------------------------------
// Copy-on-write metadata edits (spec B.5)
// ------------------------------------------------------------------------------------------------
/// Offset of `size` bytes inside `region` not overlapping `current` (both [start, end)).
fn free_slot(region: (u64, u64), current: (u64, u64), size: u64) -> Option<u64> {
    let (lo, hi) = region;
    if current.0 >= lo && current.0 - lo >= size {
        return Some(lo);
    }
    if hi < size {
        return None;
    }
    let end = (hi - size) / 8 * 8; // right-aligned, so copies alternate between the two ends
    (end >= current.1.max(lo)).then_some(end)
}

/// Replace the metadata without touching anything else. Copy-on-write, so a crash at any point
/// leaves a valid file: the new JSON goes to free space in [end of header, resources_offset) and is
/// synced; only then is the 64-byte header switched to it. Returns false (nothing changed) when
/// there is no room beside the current copy.
pub fn rewrite_meta_in_place(path: impl AsRef<Path>, data: &[u8]) -> Result<bool> {
    rewrite_meta_in_place_hooked(path, data, &mut |_| Ok(()))
}

/// Like [`rewrite_meta_in_place`], with the new metadata computed from the current one, read
/// through the same open file that is then rewritten (so a file renamed over `path` in between
/// cannot get another file's metadata).
pub fn rewrite_meta_with(path: impl AsRef<Path>, edit: impl FnOnce(&[u8]) -> Result<Vec<u8>>) -> Result<bool> {
    let mut edit = Some(edit);
    rewrite(path, &mut |current| (edit.take().expect("called once"))(current), &mut |_| Ok(()))
}

/// Test hook: `before_switch` runs after the new copy is synced, before the header switches.
pub fn rewrite_meta_in_place_hooked(
    path: impl AsRef<Path>,
    data: &[u8],
    before_switch: &mut dyn FnMut(&mut File) -> Result<()>,
) -> Result<bool> {
    rewrite(path, &mut |_| Ok(data.to_vec()), before_switch)
}

type MetaEdit<'a> = &'a mut dyn FnMut(&[u8]) -> Result<Vec<u8>>;

fn rewrite(
    path: impl AsRef<Path>,
    edit: MetaEdit,
    before_switch: &mut dyn FnMut(&mut File) -> Result<()>,
) -> Result<bool> {
    // Held until `f` is closed: concurrent edits of one file are serialized, never interleaved.
    let mut f = open_locked(path.as_ref())?;
    let mut hb = [0u8; HEADER_SIZE];
    f.read_exact(&mut hb)?;
    let mut header = FileHeader::unpack(&hb)?;
    if &header.magic != MAGIC_FILE || header.version != VERSION {
        return format_err(format!(
            "not an LVF v{VERSION} file (magic {:?}, version {})",
            String::from_utf8_lossy(&header.magic),
            header.version
        ));
    }
    let size = f.metadata()?.len();
    let meta_end = header.meta_offset.checked_add(header.meta_length as u64);
    if header.resources_offset > size || meta_end.is_none_or(|e| e > header.resources_offset) {
        return format_err("the header's metadata region is outside the file");
    }
    let mut current = vec![0u8; header.meta_length as usize];
    f.seek(SeekFrom::Start(header.meta_offset))?;
    f.read_exact(&mut current)?;
    let data = edit(&current)?;
    let data = data.as_slice();
    let Ok(data_len) = u32::try_from(data.len()) else {
        return Err(Error::Value(format!("metadata of {} bytes is too large", data.len())));
    };
    let region = (HEADER_SIZE as u64, header.resources_offset);
    let current = (header.meta_offset, header.meta_offset + header.meta_length as u64);
    let Some(offset) = free_slot(region, current, data.len() as u64) else { return Ok(false) };
    f.seek(SeekFrom::Start(offset))?;
    f.write_all(data)?;
    f.sync_all()?;
    before_switch(&mut f)?;
    header.meta_offset = offset;
    header.meta_length = data_len;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&header.pack())?;
    f.sync_all()?;
    Ok(true)
}

/// Take an exclusive advisory lock on `file` (std [`File::lock`]: blocks until it is free; released
/// when the file is closed). The in-place metadata edits hold it for their whole read, choose slot,
/// write, switch sequence; a writer that replaces an LVF file (rewrite + rename over it) can hold
/// it on the source file to serialize with them. Plain readers do not lock.
pub fn lock_exclusive(file: &File) -> Result<()> {
    Ok(file.lock()?)
}

/// Open `path` for reading and writing with [`lock_exclusive`] held. If the file was replaced
/// (renamed over) while waiting for the lock, the new file is opened instead, so the lock is on
/// the file `path` names when this returns.
pub fn open_locked(path: &Path) -> Result<File> {
    for _ in 0..100 {
        let f = OpenOptions::new().read(true).write(true).open(path)?;
        lock_exclusive(&f)?;
        if still_at(&f, path)? {
            return Ok(f);
        }
    }
    Err(Error::Io(std::io::Error::other(format!("{} keeps being replaced", path.display()))))
}

#[cfg(unix)]
fn still_at(f: &File, path: &Path) -> Result<bool> {
    use std::os::unix::fs::MetadataExt;
    let (a, b) = (f.metadata()?, fs::metadata(path)?);
    Ok(a.dev() == b.dev() && a.ino() == b.ino())
}

#[cfg(not(unix))]
fn still_at(_: &File, _: &Path) -> Result<bool> {
    Ok(true) // no stable file identity in std; renaming over an open file is rare there
}

// ------------------------------------------------------------------------------------------------
// Publishing (spec B.11)
// ------------------------------------------------------------------------------------------------
/// A hidden sibling of `dst` (same directory, so the final rename is atomic), unique per call,
/// so concurrent writers to one destination never share a temporary file.
pub fn temp_path_for(dst: impl AsRef<Path>) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dst = dst.as_ref();
    let name = dst.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "out.lvd".into());
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos());
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    dst.with_file_name(format!(".{name}.{}-{n}-{nanos:x}.fflv-tmp", std::process::id()))
}
