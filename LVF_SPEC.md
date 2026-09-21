# LVF Layered Video Format v1 — Implementation Specification

> This document is the complete set of requirements and specifications handed to the implementer (Claude Code). Implement it exactly as written, except where it says "the implementer may decide".
>
> Clarifications and additions made during implementation are in **Appendix A: Implementation notes** at the end.
>
> The tooling was later consolidated into the command-line tool / Python package **`fflv`**, the file extension became **`.lvd`** (layered video document), and extensions such as lossless layers were added — see **Appendix B**. The scripts `lvf_pack.py` / `lvf_inspect.py` / `make_test_assets.py` named in sections 8, 10 and 11 correspond to `fflv pack` / `fflv info` / `fflv testsrc`.

---

## 0. Background and goals

I need a **video format with layers** for personal **interactive playback**: while playing, any layer can be switched on or off at any time and its opacity adjusted.

**The single, non-negotiable core requirement: all layers are always strictly synchronized.**

Existing approaches (MP4 with multiple tracks, several `<video>` elements stacked on top of each other) cannot meet it, because every track / every player has its own timeline and decoding pipeline and can only align on a "best effort" basis. The LVF design principle is:

> **Synchronization is a structural property of the format, not an effort made by the player.**
> The basic unit of storage and decoding is "one frame of every layer at a given moment" (called a **composite frame**), not "one frame of one layer".

### Goals
- Any number of layers, each with its own size and position
- Layers support transparency (alpha)
- Static layers (images) are not stored repeatedly
- Layer visibility and opacity can be changed during playback
- Frame-accurate seeking
- Layers are always in sync with each other, and audio with video

### Non-goals (not in v1)
- Network streaming, DRM
- Safari support (target platform: desktop Chrome / Edge)
- Variable frame rate
- Editing features
- Home-grown compression algorithms (**compression always reuses existing encoders**)

---

## 1. Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Video codec | **VP9** (libvpx-vp9), 8-bit 4:2:0 | Natively supported by WebCodecs in browsers; with alt-ref disabled there is no frame reordering and every packet corresponds to exactly one displayed frame — the simplest implementation |
| Transparency | A separate VP9 grayscale stream (luma = alpha) | Same encoder and same GOP as the color stream; symmetric structure |
| Audio | Opus, 48 kHz | Natively supported by WebCodecs |
| Static layers | Stored as PNG in the resource region, shown over a frame range | Does not change over time, so it has no synchronization problem and wastes no bitrate |
| Time model | One frame rate for the whole file; time = frame index | Layers have no timestamps of their own, so "a layer's time" fundamentally does not exist |
| Byte order | Little-endian throughout | — |
| Metadata | UTF-8 JSON | Personal use, easy to debug |
| Packing tool | Python 3.10+, PyAV (`pip install av`), FFmpeg command line | — |
| Player | TypeScript + Vite, no framework, WebCodecs + WebGL2 | — |

---

## 2. File layout overview

```
┌──────────────────────┐ offset 0
│ File header (64 B)   │
├──────────────────────┤ meta_offset
│ Metadata JSON        │
├──────────────────────┤ resources_offset
│ Resources (PNG, ...) │
├──────────────────────┤ cau_offset
│ Composite frame 0    │
│ Composite frame 1    │
│ ...                  │
│ Composite frame N-1  │
├──────────────────────┤ index_offset
│ Index table          │
└──────────────────────┘
```

All offsets are **absolute offsets from the start of the file**. The packing tool first writes a placeholder header and back-fills it after everything else is written.

---

## 3. File header (64 bytes)

| Offset | Size | Type | Field | Description |
|---|---|---|---|---|
| 0 | 4 | bytes | magic | ASCII `LVF1` |
| 4 | 2 | u16 | version | Always `1` |
| 6 | 2 | u16 | flags | Reserved, write 0 |
| 8 | 8 | u64 | meta_offset | Start of the metadata (usually 64) |
| 16 | 4 | u32 | meta_length | Length of the metadata JSON in bytes |
| 20 | 4 | u32 | reserved | Write 0 |
| 24 | 8 | u64 | resources_offset | Start of the resource region |
| 32 | 8 | u64 | cau_offset | Start of the first composite frame |
| 40 | 8 | u64 | index_offset | Start of the index table |
| 48 | 16 | — | reserved | Write 0 |

---

## 4. Metadata JSON

```json
{
  "format": "LVF",
  "version": 1,
  "canvas": { "width": 1920, "height": 1080, "background": "#000000" },
  "fps": { "num": 30, "den": 1 },
  "frame_count": 3600,
  "max_rap_interval": 60,
  "layers": [
    {
      "id": "bg",
      "name": "Background",
      "kind": "video",
      "z": 0,
      "rect": { "x": 0, "y": 0, "w": 1920, "h": 1080 },
      "start_frame": 0,
      "end_frame": 3600,
      "codec": "vp09.00.40.08",
      "coded_width": 1920,
      "coded_height": 1080,
      "has_alpha": false,
      "alpha_codec": null,
      "blend": "normal",
      "opacity": 1.0,
      "visible": true
    },
    {
      "id": "person",
      "name": "Person",
      "kind": "video",
      "z": 1,
      "rect": { "x": 400, "y": 100, "w": 1280, "h": 960 },
      "start_frame": 0,
      "end_frame": 3600,
      "codec": "vp09.00.40.08",
      "coded_width": 1280,
      "coded_height": 960,
      "has_alpha": true,
      "alpha_codec": "vp09.00.40.08",
      "blend": "normal",
      "opacity": 1.0,
      "visible": true
    },
    {
      "id": "logo",
      "name": "Logo",
      "kind": "still",
      "z": 2,
      "rect": { "x": 1700, "y": 40, "w": 180, "h": 80 },
      "start_frame": 150,
      "end_frame": 1800,
      "resource": { "offset": 0, "length": 12345, "mime": "image/png" },
      "blend": "normal",
      "opacity": 1.0,
      "visible": true
    }
  ],
  "audio": {
    "codec": "opus",
    "sample_rate": 48000,
    "channels": 2,
    "description_b64": "<base64 of the OpusHead extradata, may be null>"
  }
}
```

### Field rules
- **Layer index** = the position of the layer in the `layers` array (from 0). Composite frames refer to layers by it.
- `z`: drawing order; smaller values are drawn first (underneath). May differ from the array order.
- `start_frame` / `end_frame`: the layer's active range, half-open `[start, end)`.
- `kind`: `"video"` or `"still"`.
- `codec`: the WebCodecs codec string, written by the packing tool with a level suited to the resolution (the implementer may decide how to compute it, as long as Chrome's `isConfigSupported` accepts it).
- `blend`: `"normal" | "add" | "multiply" | "screen"`.
- `opacity`, `visible`: defaults; the player UI can override them.
- `resource.offset`: offset **relative to the start of the resource region**.
- `audio`: `null` when there is no audio.
- `max_rap_interval`: the maximum number of frames between two random access points (equal to the GOP length used for encoding).

### Time conversion
- Timestamp of frame `f` in microseconds: `pts_us(f) = round(f * 1_000_000 * fps.den / fps.num)`
- This value is also used as the WebCodecs `EncodedVideoChunk.timestamp`, to map decoder output back to its composite frame.

---

## 5. Composite frames (CAU, Composite Access Unit) — the core

**Every video frame instant corresponds to exactly one composite frame**, stored contiguously in frame order 0 … frame_count-1.

### 5.1 Composite frame header (20 bytes)

| Offset | Size | Type | Field | Description |
|---|---|---|---|---|
| 0 | 4 | bytes | magic | ASCII `CAUF` |
| 4 | 4 | u32 | payload_size | Number of bytes from offset 8 to the end of this composite frame |
| 8 | 4 | u32 | frame_index | Frame index |
| 12 | 1 | u8 | flags | bit0 = random access point (RAP); other bits 0 |
| 13 | 1 | u8 | reserved | 0 |
| 14 | 2 | u16 | video_entry_count | Must equal the number of layers with kind=video |
| 16 | 2 | u16 | audio_packet_count | Number of audio packets carried by this frame |
| 18 | 2 | u16 | reserved | 0 |

### 5.2 Video entries (directly after the header, in ascending layer index)

| Size | Type | Field | Description |
|---|---|---|---|
| 2 | u16 | layer_index | Layer index |
| 1 | u8 | type | `0` = EMPTY (not active in this frame), `1` = FRAME; `2` is reserved for a future HOLD |
| 1 | u8 | frame_flags | bit0 = key frame |
| 4 | u32 | color_len | Length of the color stream data (0 for EMPTY) |
| 4 | u32 | alpha_len | Length of the alpha stream data (0 without alpha or for EMPTY) |
| color_len | bytes | color_data | One VP9 frame |
| alpha_len | bytes | alpha_data | One VP9 frame |

### 5.3 Audio packets (directly after all video entries)

| Size | Type | Field |
|---|---|---|
| 8 | i64 | pts_us (absolute time, microseconds) |
| 4 | u32 | duration_us |
| 4 | u32 | length |
| length | bytes | Opus packet |

An audio packet goes into the composite frame `f` for which `pts_us(f) <= packet pts < pts_us(f+1)`.

### 5.4 Format invariants (the packing tool must guarantee them; the validator must check them)

- **I1** Composite frame indices start at 0 and increase by one; there are exactly frame_count of them, one per frame.
- **I2** Every composite frame has exactly one entry for **every** video layer, in ascending layer_index.
- **I3** An entry is FRAME ⇔ `start_frame <= f < end_frame`, otherwise EMPTY.
- **I4** At `f == start_frame`, both the color frame and the alpha frame of the layer must be key frames.
- **I5** Within an entry, the key-frame flags of the color frame and the alpha frame must agree.
- **I6** The RAP flag is set ⇔ every color and alpha frame of every FRAME entry in this composite frame is a key frame (a frame whose entries are all EMPTY also counts as a RAP).
- **I7** Frame 0 must be a RAP.
- **I8** The distance between any two adjacent RAPs is ≤ `max_rap_interval`.
- **I9** Every audio packet's pts lies inside the time window of the composite frame that holds it (see 5.3).
- **I10** The index table corresponds one-to-one to the actual composite frames (see section 7).

---

## 6. Resource region

Stores the image data (PNG) of still layers back to back; each layer refers to its image by `resource.offset/length`. The player decodes them once at load time with `createImageBitmap` and uploads them as textures.

---

## 7. Index table

| Size | Type | Field |
|---|---|---|
| 4 | bytes | magic `IDX1` |
| 4 | u32 | count (= frame_count) |
| count × 16 | — | entries |

Each entry is 16 bytes:

| Size | Type | Field |
|---|---|---|
| 4 | u32 | frame_index |
| 1 | u8 | flags (bit0 = RAP, same as in the composite frame header) |
| 3 | — | reserved |
| 8 | u64 | cau_offset (absolute offset of that composite frame) |

Every frame is indexed, which makes seeking and validation easy (about 1.7 MB for one hour at 30 fps — acceptable).

---

## 8. Packing tool `lvf_pack.py`

### 8.1 Input: project file (JSON)

```json
{
  "output": "out.lvf",
  "canvas": { "width": 1920, "height": 1080, "background": "#000000" },
  "fps": "30/1",
  "duration": 120.0,
  "gop": 60,
  "quality": { "crf": 32 },
  "layers": [
    { "id": "bg", "name": "Background", "kind": "video", "src": "bg.mp4",
      "z": 0, "rect": [0, 0, 1920, 1080], "start": 0, "end": 120.0,
      "alpha": false, "blend": "normal", "opacity": 1.0, "visible": true },
    { "id": "person", "name": "Person", "kind": "video", "src": "person.mov",
      "z": 1, "rect": [400, 100, 1280, 960], "start": 0, "end": 120.0,
      "alpha": true },
    { "id": "logo", "name": "Logo", "kind": "still", "src": "logo.png",
      "z": 2, "rect": [1700, 40, 180, 80], "start": 5.0, "end": 60.0 }
  ],
  "audio": { "src": "music.wav" }
}
```

- Times may be given in seconds; the packing tool converts them to frames: `frame = round(seconds * fps)`.
- When `start` / `end` are omitted, the layer covers the whole duration.

### 8.2 Processing steps

1. **Parse the project**, compute `frame_count = round(duration * fps)` and each layer's `[start_frame, end_frame)`.
2. **Transcode every video layer** (one FFmpeg call per layer; they may run in parallel):
   - Normalize the frame rate (`fps` filter), scale to the `rect` size, and cut or extend to exactly `end_frame - start_frame` frames (extend with `tpad=stop_mode=clone`, which repeats the last frame).
   - Reference command for the color stream:
     ```
     ffmpeg -i SRC -vf "fps=30,scale=W:H,format=yuv420p" -frames:v N \
       -c:v libvpx-vp9 -crf 32 -b:v 0 -row-mt 1 \
       -g 60 -keyint_min 60 -auto-alt-ref 0 -lag-in-frames 0 \
       -f ivf color.ivf
     ```
   - Reference command for the alpha stream (the source must have alpha):
     ```
     ffmpeg -i SRC -vf "fps=30,scale=W:H,format=rgba,alphaextract,format=yuv420p" -frames:v N \
       (same encoding parameters as the color stream) -f ivf alpha.ivf
     ```
   - **Crucial**: fixed GOP, no automatic key frames on scene changes, alt-ref off — so the key frames of all layers fall on the same positions and every packet corresponds to exactly one displayed frame.
   - When a layer does not start at 0, its key frames should be aligned to the global grid: preferably `start_frame` is a multiple of `gop`; when that is not possible the packing tool must still handle it correctly (there will just be fewer RAPs), but I8 must hold — otherwise report an error and suggest an adjustment.
   - The implementer may choose the exact parameters, but **must not trust the encoder**: every frame must be verified in step 3.
3. **Read the packets**: demux the IVF with PyAV, take the packets in order and read their key-frame flag (PyAV's `packet.is_keyframe`; if in doubt, parse the frame_type in the VP9 uncompressed header as a second check).
   - Check that the number of packets equals the layer's frame count exactly; otherwise report an error.
   - Check that the key-frame positions of the color and alpha streams are identical (I5); otherwise report an error.
4. **Audio**: transcode to Opus, reference command `ffmpeg -i SRC -c:a libopus -b:a 128k -ar 48000 -t DURATION audio.ogg`; take the packets (pts, duration, data) and the extradata with PyAV.
5. **Interleave and write**: for every frame `f`, assemble the composite frame as in section 5, compute its RAP flag, and record the index entry.
6. **Write the resource region and the index table, back-fill the header**.
7. **Run the validator automatically after writing**; exit with a non-zero code if it fails.

### 8.3 Helper tools
- **`lvf_inspect.py FILE`**: prints the file header, the metadata and statistics (per-layer bitrate, number and positions of RAPs), and checks every invariant of section 5.4. `--frame N` prints the structure of composite frame N.
- **`make_test_assets.py`**: generates the test material and the test project with FFmpeg's lavfi (see section 11).

---

## 9. Player

### 9.1 Form
- A static web page built with TypeScript + Vite. Local `.lvf` files are opened with a file picker or by drag and drop.
- Read on demand with `File.slice()`; **never read the whole file into memory** (files may be several GB).

### 9.2 Architecture

```
File ──► Reader (reads composite frames sequentially, read-ahead buffer)
            │
            ▼
        Dispatcher ──► per layer: color VideoDecoder, alpha VideoDecoder
            │               │
            │               ▼ decoder output (mapped back to its composite frame by timestamp)
            ▼
        Composite-frame assembler: waits until every plane of the frame is decoded
            │
            ▼
        Ready queue (only "complete" composite frames, bounded)
            │
            ▼
        Render loop (requestAnimationFrame): take the frame for the master clock → WebGL2 compositing
```

### 9.3 Synchronization guarantees — player invariants that must hold

- **P1 Atomicity**: a composite frame enters the ready queue only after **every plane of every FRAME entry** has been decoded.
- **P2 Single composition point**: whatever is on screen at any moment comes from **one and the same** composite frame (still layers are shown or hidden according to that composite frame's frame_index). Mixing layers from different frames is forbidden.
- **P3 Wait as a whole**: when a composite frame is due but the next one is not ready yet, keep showing the current composite frame (all layers stop together). If the wait exceeds a threshold (e.g. 100 ms), pause the master clock and enter a buffering state; resume once ready.
- **P4 Drop as a whole**: when playback falls behind, drop whole composite frames (close all VideoFrames of that frame).
- **P5 Hidden layers keep decoding**: because frames depend on each other, hidden layers must keep decoding, so that a layer can be shown again immediately and in sync.
- **P6 Debug assertion**: in development mode, before rendering a composite frame, assert that the `timestamp` of every one of its VideoFrames equals `pts_us(frame_index)`; on mismatch, log an error to the console and show an obvious warning on screen.

### 9.4 Decoding details
- Configure each `VideoDecoder` with the codec string and `coded_width/height` from the metadata; check with `isConfigSupported` at startup.
- The `EncodedVideoChunk` sent to the decoder takes its `type` from the key-frame flag and uses `timestamp = pts_us(frame_index)`.
- **Back-pressure**: cap the ready queue plus the frames being decoded (e.g. 8 frames); stop reading and dispatching when the cap is reached. `decoder.decodeQueueSize` can be consulted as well.
- **Memory**: `close()` every VideoFrame as soon as it is no longer used (replaced or dropped), otherwise hardware decoders stall.

### 9.5 Master clock
- **With audio**: the playback position of the AudioContext is the master clock. Opus is decoded with `AudioDecoder`, and the decoded AudioData is scheduled on the AudioContext by pts.
- **Without audio**: the clock is `performance.now()` plus play/pause offsets.
- On every render-loop tick: `t = clock.now()`, take the last composite frame with `pts <= t` from the ready queue and show it; drop all earlier ones as a whole.

### 9.6 Compositing (WebGL2)
- Draw every layer that is "visible and active in this frame" in ascending `z`: a rectangle at the layer's `rect` in canvas coordinates.
- Color texture: upload the VideoFrame directly with `texImage2D` (the browser does the YUV→RGB conversion).
- Alpha texture: upload the alpha stream's VideoFrame the same way; the shader reads `.r` as alpha.
  - Note that video may be limited range (16–235). It must be verified with the test material that alpha=0 and alpha=255 map to 0.0 and 1.0; if they do not, apply a range mapping in the shader (`clamp((v - 16/255) * 255/219, 0, 1)`) — the implementer decides from measurements whether it is needed.
- Final alpha = alpha texture value × layer opacity. Colors are treated as straight (non-premultiplied) alpha.
- Blend modes: normal / add / multiply / screen, with blendFunc or in the shader.
- Clear the canvas with `canvas.background` first.

### 9.7 Seeking
1. Pause the clock, empty the ready queue and close all its frames, `reset()` and then `configure()` every decoder again, clear the audio.
2. Find the nearest RAP at or before the target frame `T` in the index table.
3. Read and decode from that RAP, dropping complete composite frames with frame_index < T.
4. Show frame T and resume playback (if it was playing). Audio resumes from the time of frame T.

### 9.8 UI
- Play / pause, a progress bar (drag to seek), current time and frame number.
- Layer panel: one row per layer showing its name, with a visibility toggle and an opacity slider.
- Frame step forward / backward (backward = seek to T-1).
- Shortcuts: Space for play/pause, ← → for frame steps.
- Debug panel (collapsible): current frame, ready-queue length, dropped frames, buffering events, number of P6 assertion failures.

---

## 10. Directory layout (suggested)

```
lvf/
  LVF_SPEC.md            this document
  tools/
    lvf_pack.py
    lvf_inspect.py
    make_test_assets.py
    lvf/                 shared reading/writing module (constants, struct packing/unpacking)
  player/                Vite + TypeScript
    src/
      format/            parsing of header, metadata, composite frames, index
      decode/            decoder management, composite-frame assembly
      render/            WebGL2 compositing
      clock/             master clock
      ui/
  test_assets/           generated test material (not committed)
```

---

## 11. Testing and acceptance

### 11.1 Test material (generated by `make_test_assets.py`)
- **Background layer**: the `testsrc2` test pattern, with the frame number drawn in the top-left corner by `drawtext`.
- **Sync test layer A** (with alpha): on a transparent background, a white vertical line at x = `(frame number × 16) mod width`, plus the frame number at a fixed position.
- **Sync test layer B** (with alpha): a red vertical line at exactly the same position as in A, narrower, plus the frame number at another position.
  → When in sync, the red line always sits exactly in the middle of the white line; being off by even one frame is plainly visible.
- **A layer that appears midway**: a moving square with alpha starting at frame 45 (not a multiple of the GOP), to test unaligned starts.
- **Still layer**: a PNG shown only within a range of frames.
- **Audio**: a short "beep" every second (at whole seconds), with a white frame flashed on screen at the same moment, to check audio/video sync.
- **Alpha calibration layer**: a horizontal alpha gradient from 0 to 255 left to right, to verify the range mapping of section 9.6.

### 11.2 Acceptance criteria
1. `lvf_pack.py` packs the test project, and `lvf_inspect.py` reports every invariant as passing for the output file.
2. Deliberately broken files (missing entry, misaligned key frames, non-contiguous frame numbers) are reported precisely by the validator.
3. The player plays the test file continuously; all frame numbers shown agree, and the red line is always in the middle of the white line.
4. While quickly and repeatedly toggling layers, dragging the progress bar and stepping frames forward and backward during playback, all frame numbers on screen are identical whenever playback is paused.
5. With the CPU slowed down 6× in Chrome DevTools: stutter and dropped frames are allowed, but **no layer may ever be out of sync**; the number of P6 assertion failures must be 0.
6. The "beep" and the white flash are aligned to the ear.
7. The two ends of the alpha calibration layer are fully transparent and fully opaque respectively.
8. Playing for more than 10 minutes shows no continuous memory growth (all VideoFrames are closed correctly).

### 11.3 Implementation order (milestones)
1. **M1**: shared reading/writing module + packing tool + validation tool + test-material generation. (Accepted first against items 1 and 2 of section 11.2.)
2. **M2**: player reading, decoding, compositing, playback (no audio, `performance.now()` clock).
3. **M3**: audio and the audio master clock.
4. **M4**: seeking, frame stepping, layer panel, debug panel.
5. **M5**: full acceptance (every item of section 11.2).

---

## 12. Possible future extensions (not implemented in v1, but the design must not rule them out)
- Entry type `2 = HOLD`: a video layer refers to its previous frame in frames whose content does not change, saving bitrate.
- A different codec per layer (AV1, H.264).
- Key-frame animation of layer transforms (position, scale, opacity over time).
- Layer groups.

---

## Appendix A: Implementation notes (clarifications, additions and measured facts from the v1 implementation)

Nothing here changes the format defined in sections 1–12; it records the concrete decisions made during implementation and why.

**A.1 Rounding.** The `round` of section 4 rounds half up and is implemented in integer arithmetic: `pts_us(f) = (2·f·10⁶·den + num) div (2·num)`. Python (`fflv/format/timing.py`) and the player (`player/src/format/timing.ts`, BigInt) agree bit for bit; both are tested.

**A.2 Key-frame alignment.** Measured: libvpx's fixed GOP does not restart its count after a forced key frame (after forcing a key frame at frame 15 it still inserts one at frame 60 by itself). The packing tool therefore pushes the encoder's own key-frame interval past the length of the clip (`-g N+G -keyint_min N+G`) and places key frames exactly with `-force_key_frames expr:eq(n,0)+eq(mod(n+S,G),0)`: the layer's first frame plus every GOP boundary of the global grid. Every layer — including those with unaligned starts — is then a key frame at every global frame `k·gop`, RAPs fall exactly on the grid, and I8 holds by construction. Every packet is still verified after encoding: PyAV's `is_keyframe` is cross-checked against our own parse of the VP9 uncompressed header, and we check that every packet shows exactly one frame, that key frames have the coded size, the packet count, and that color and alpha key frames agree.

**A.3 Alpha range and the alpha path in the player.** Alpha 0..255 is coded as limited-range luma Y = 16..235, and the VP9 header says BT.709, color_range = limited. Measured (Chrome for Testing 153, Edge 153): calling `texImage2D` on the alpha VideoFrame directly, the browser's YUV→RGB conversion maps Y = 235 to 253/255 (an offset of about −2 levels), and Edge's software-decoding path also corrupts the last column. So the player does not use the browser's conversion for alpha planes. Instead:

- It copies the raw luma out with `VideoFrame.copyTo()`; the plane only counts as decoded when the copy is done (P1 still holds). The alpha VideoFrame is closed right after the copy.
- The luma is uploaded as an R8 texture and the shader applies the range mapping from section 9.6, `clamp((v − 16/255)·255/219, 0, 1)`.

Measured: alpha = 0 → 0.0 and alpha = 255 → 1.0, both exact; the largest deviation from the ideal ramp in between is 1.9 levels, which is lossy-coding noise. If a pixel format cannot be copied (rare), the player falls back to uploading the frame directly. Color planes are still converted by the browser, as section 9.6 says.

**A.4 Opus pre-skip.** Audio packet `pts_us` values are on the "decoder timeline": the first packet is at 0, as in WebM. The metadata `audio` object gains a `pre_skip` field (in 48 kHz samples, the same value as in the OpusHead); presentation time = decoder time − pre_skip. Measured: Chromium's AudioDecoder, given the OpusHead description, drops the pre-skip itself and outputs presentation timestamps — after every configure, including after a seek. The player detects from the size of the first output whether the decoder did this, and subtracts the pre-skip itself if not. The end-to-end tests confirm that the onset of every decoded beep lies within 0.1 ms of a whole second.

**A.5 Metadata additions.** Only `audio.pre_skip` (see A.4). Readers must ignore unknown fields. (Appendix B adds more.)

**A.6 Coded size.** VP9 4:2:0 needs even dimensions; when the width or height of `rect` is odd, the coded size is rounded up to the next even number. Superseded by B.4: the padding is now recorded in `content_size` and cropped by players.

**A.7 Display-latency compensation.** What is drawn now reaches the screen at the next refresh, so the render loop picks frames for `t = clock.now() + refresh interval` (the interval is estimated from rAF timestamps, capped at 50 ms). Measured at the display, the white flash comes about 9 ms after the beep — far below what is perceptible.

**A.8 Jump when too far behind.** When the video is more than 2 s behind the master clock (e.g. audio kept playing while the tab was hidden and rAF was suspended), the player seeks straight to the clock's frame instead of catching up frame by frame. Ordinary stalls are handled by P3: the clock stops after 100 ms and buffering starts, so the lag never builds up to 2 s.

**A.9 Known browser issue.** In Edge 153, software-decoded VP9 frames uploaded to WebGL have the wrong chroma in the last pixel column (white becomes 198,255,187). Hardware decoding does not have this problem, nor does Chrome. Color planes are converted by the browser as section 9.6 says, so this is left as is; `?hw=prefer-hardware` avoids it. Alpha is not affected (see A.3).

**A.10 Test material.** Homebrew's FFmpeg 8.1 is built without `drawtext`, so frame numbers are drawn with a built-in bitmap font. In addition, every video layer carries a barcode of its frame number (1 start bit + 16 data bits + 1 parity bit, 8×8-pixel cells). During playback, seeking, layer toggling and CPU throttling, the end-to-end tests read the barcodes of all visible layers back from the canvas and check, frame by frame, that they all equal the frame_index of the current composite frame.

**A.11 Additional validator checks (beyond I1–I10).** The validator also checks: that VP9 frame headers agree with frame_flags; that every packet shows exactly one frame; key-frame sizes and profile / bit depth / chroma subsampling; that EMPTY entries carry no data; the types and values of metadata fields; resource bounds and the PNG signature; that audio pts values increase. "The last RAP is further than max_rap_interval from the end of the file" is only a warning.

**A.12 Two refinements of seeking (section 9.7).**

- During a seek, the current composite frame stays on screen until the target frame is ready and then replaces it as a whole, avoiding a black flash. P2 still holds: the screen always shows exactly one complete composite frame.
- A short forward seek (e.g. stepping forward) does not reset the decoders when there is no RAP before the target or the distance is ≤ 8 frames: the pipeline produces the following composite frames in order anyway, and the whole frames in between are simply closed.

Everything else follows section 9.7 exactly: `reset()` + `configure()` every decoder, decode from the nearest RAP at or before the target, drop complete composite frames with frame_index < T. Consecutive seek requests (dragging the progress bar, holding an arrow key) are coalesced; only the last one is carried out.

---

## Appendix B: the fflv toolchain and v1 format extensions

The v1 binary structures (file header, composite frames, index) are unchanged. The extensions below are all optional, backward-compatible metadata fields, or relax an earlier restriction.

**B.1 File extension**: `.lvd`. The header magic is still `LVF1` and the metadata `format` is still `"LVF"`.

**B.2 Lossless layers** (`layers[].lossless: true`).
- The color plane is coded losslessly as VP9 profile 1, 8-bit 4:4:4 RGB (`gbrp`, identity matrix, full range); codec string `vp09.01.LL.08.03.01.13.00.01`.
- The alpha plane is still a profile 0 luma stream, but coded losslessly in full range (Y = alpha).
- Measured: Chrome 153 / Edge 153 decode profile 1 RGB into I444 frames, and after `texImage2D` the pixels are bit-identical; the Python side (PyAV) is bit-identical as well.
- The validator checks the bitstream profile against the codec string: profile 0 must be 4:2:0; profile 1 must be 4:4:4 and RGB.

**B.3 `layers[].alpha_range`**: `"limited"` (default, Y = 16..235) or `"full"` (Y = 0..255). States the value range of the alpha plane's luma; the validator checks that it matches the color_range in the VP9 frame header. It is written explicitly into the metadata because, as measured, the `VideoFrame.colorSpace.fullRange` reported by browsers is unreliable: full-range streams are reported as limited too.

**B.4 `layers[].content_size: [w, h]`**. 4:2:0 needs even dimensions; layers with an odd width or height are padded to even size by repeating their last row/column. This field records the region of valid pixels (default: the coded size). Players and decoders crop the padding before scaling to `rect`, so changing `rect` later does not distort the image. The padding must repeat real pixels: `fflv pack` pads in 4:4:4 RGB, because FFmpeg's `pad` rounds the size of an odd, chroma-subsampled frame down to even and fills the dropped row/column with black.

**B.5 Reserved metadata space**. Writers fill the space after the metadata JSON with spaces up to `resources_offset` (by default they reserve max(4 KiB, 2 × metadata size), rounded up to 4 KiB). `meta_length` counts only the JSON itself. This lets renaming and changes to z / rect / blend / opacity / visible rewrite the metadata in place — instantly, whatever the file size (`fflv set`).

**B.6 `generator`** (optional): name of the writing tool, e.g. `"fflv"`.

**B.7 Editing semantics** (`fflv add / rm / set`).
- Adding or removing a layer or the audio track rewrites the whole file; the packets of existing layers are copied byte for byte and never re-encoded. The rewrite goes to a temporary file, which atomically replaces the original after it passes validation.
- The key frames of an added layer are its own first frame plus every RAP of the original file within its range, so all original RAPs are preserved.
- After a layer is removed, the layer indices of the layers after it shift down by one; RAP flags are recomputed from the entries.

**B.8 Python writer semantics** (`fflv.Writer`).
- All layers must be declared before the first `write()`, because every composite frame has to list every video layer.
- A layer becomes active at the first frame it is given an image. If a later frame gives it no image, it keeps showing the previous one (the same image is encoded again, at almost zero bitrate), until `end_layer()` or the end of the file.
- Key frames are at each layer's first frame and on the global `gop` grid.
- The file is written to `<path>.part`; on close the metadata (into the reserved space) and the index are written, then it is atomically renamed and validated.

**B.9 `fflv view`**. A local HTTP server (listening on 127.0.0.1 only) serves the player and the file, with Range and ETag support.
- The player sends `If-Match` with every range read; when the file has been replaced, the server answers 412 and the player reloads instead.
- The player polls the ETag once a second and reloads automatically when the file changes. A reload keeps the current frame, the play state, and the layer settings the user changed in the UI; everything else takes the new file's defaults.

**B.10 Decoding in Python** (`fflv.open` / `fflv render` / `fflv extract`) decodes only the selected layers, starting at the nearest RAP at or before the target frame. YUV→RGB uses the BT.601/709/2020 formulas in numpy, because swscale's result depends on the frame width (when the width is not a multiple of 16, Y = 235 becomes 253) and on the CPU. Compositing uses the same formulas as the WebGL player: straight alpha; add is min(1, d + a·c); multiply / screen follow the W3C separable blend modes.

**B.11 Publishing files.** Every writer — `fflv pack`, the edit commands and `fflv.Writer` — writes to a hidden temporary file next to the destination (`.<name>.fflv-tmp`), validates it, and only then atomically renames it over the destination. A reader of the destination (for example `fflv view`) therefore only ever sees a complete, valid file, and a failed or interrupted write leaves the previous file untouched. This replaces the in-place writing of section 8.2 step 6.

**B.12 Metadata is standard JSON.** The metadata must be RFC 8259 JSON in UTF-8 without a byte-order mark: no `NaN`, `Infinity` or other non-finite numbers (browsers' `JSON.parse` rejects them, and they would make a file unreadable in the player). Writers refuse such values and the validator reports them. Layers with equal `z` are drawn in ascending layer-index order.
