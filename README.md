# fflv — layered video (.lvd) for debugging

`fflv` is a small ffmpeg-like tool. It reads and writes **LVF layered video** (extension `.lvd`): one file holds several layers that are **always strictly frame-synchronized**, and during playback any layer can be switched on or off and have its opacity changed.

It is meant as a debugging aid. For example, you can store the input frames, predictions, masks and heat maps as separate layers, then compare them frame by frame.

- Format specification: [`LVF_SPEC.md`](LVF_SPEC.md). Implementation notes and measured facts are in Appendix A; fflv's format extensions are in Appendix B.
- **Compression:** every video layer is encoded with VP9, the same class of modern video codec as WebM/YouTube and as H.264/H.265 in MP4. Audio is encoded with Opus. Any layer can instead use lossless VP9. The 20-second test file holds 3.3 GB of raw layer data in 12 MB.

Synchronization does not depend on the player "doing its best". The format guarantees it structurally: the basic unit of a file is the **composite frame**, one frame of every layer at the same instant. The player only ever shows complete composite frames, and when it has to drop frames it drops whole composite frames.

## Installation

Requirements:
- Python ≥ 3.10.
- The FFmpeg command line, built with libvpx-vp9 and libopus. It is only needed to import from video or audio files.
- Desktop Chrome or Edge, to watch files.

```bash
conda activate lvf            # or any environment with Python >= 3.10
pip install -e .              # installs the fflv command and Python package (depends on av, numpy)
```

The bundled player (`fflv/viewer/`) is already built. Node is only needed after changing the player sources in `player/`: `cd player && npm install && npm run build`.

## Command line

```bash
fflv info    debug.lvd [--frame N] [--json]    # file, layers, bitrates, RAPs; check the format invariants
fflv check   a.lvd b.lvd                       # validate (exit code 1 if any file is invalid)
fflv view    debug.lvd                         # open the interactive player in Chrome/Edge

fflv add     debug.lvd --src pred.mp4 --id pred --alpha --rect 0,0,640,360 --start 2s
fflv add     debug.lvd --src mask.mov --id mask --lossless      # lossless: pixel values kept exactly
fflv add     debug.lvd --still legend.png --id legend --rect 1100,20,160,80
fflv add     debug.lvd --audio voice.wav                        # replace the audio track
fflv rm      debug.lvd pred legend [--audio]
fflv set     debug.lvd mask opacity=0.5 blend=screen z=10 visible=false name=Mask

fflv render  debug.lvd -o out.mp4 -l frame,mask -f 2s:5s       # any layer combination -> video / images / npy
fflv render  debug.lvd -o frame120.png -f 120 --hide pred
fflv extract debug.lvd mask -o mask/%05d.png                    # one layer's own RGBA pixels

fflv pack    project.json -o out.lvd                            # build a file from a project (spec §8.1)
```

**Arguments:**
- **Layers** are given by id or by index.
- **Frame ranges** are written `N`, `A:B` (B excluded), `A:` or `:B`. Values are frames, or seconds with an `s` suffix (e.g. `1.5s:3s`).

**How edits work:**
- **Edits modify the file in place by default.** They write a temporary file and atomically replace the original once it passes validation. `-o` writes to another file instead.
- **Existing layers are never re-encoded.** Adding or removing a layer only encodes the new layer; all other packets are copied byte for byte, and the new layer's key frames are aligned to the file's existing random-access points.
- **`set` only touches the metadata.** It rewrites it in place in the reserved space, which is instant regardless of file size.

**`render` / `extract` pick the output type from the file name:**
- `.png` / `.jpg`: a single frame.
- A name with `%d`, or a directory: one image per frame.
- `.mp4` / `.mov`: H.264.
- `.webm`: VP9.
- `.mkv`: lossless FFV1.
- `.npy`: a numpy array.
- `--transparent`: output with alpha (PNG, `.mov`, `.webm`, `.mkv`).

**`render` decodes only the selected layers**; the data of every other layer is skipped, so switching between layer combinations costs nothing extra.

## Interactive player (`fflv view`)

`fflv view debug.lvd` starts a small local server (127.0.0.1) and opens the player in Chrome, or in Edge when Chrome is not installed. The player reads the file on demand through HTTP range requests and never loads it whole.

**Keys:**
- <kbd>Space</kbd>: play / pause.
- <kbd>←</kbd> / <kbd>→</kbd>: frame step.
- <kbd>1</kbd>–<kbd>9</kbd>: toggle the n-th layer of the layer panel.
- <kbd>⇧</kbd>+<kbd>1</kbd>–<kbd>9</kbd>: show only that layer (solo); press again to restore.
- <kbd>0</kbd>: show all layers.

**Following changes to the file:** when the file is regenerated during debugging, or changed with `fflv add/rm/set`, the page reloads it within about a second. The current frame, the play state and the layer settings you changed in the UI are kept; new defaults stored in the file (e.g. an opacity changed with `fflv set`) take effect.

You can also open the player page directly and drag an `.lvd` file onto it.

## Python API

```python
import numpy as np
import fflv

# Write: numpy frames become layers directly
with fflv.Writer("debug.lvd", size=(1280, 720), fps=30) as w:
    w.add_layer("frame")                                  # opaque, whole canvas, lossy
    w.add_layer("mask", alpha=True, lossless=True)        # with alpha, lossless
    w.add_layer("boxes", alpha=True, rect=(0, 0, 640, 360), blend="screen")
    w.add_still("legend", "legend.png", rect=(1100, 20, 160, 80))
    for img, mask, boxes in data:                          # uint8 / bool / float (0..1)
        w.write(frame=img, mask=mask, boxes=boxes)         # H×W, H×W×3 or H×W×4

# Read: only the layers you ask for are decoded
with fflv.open("debug.lvd") as f:
    rgb = f.frame(120, layers=["frame", "mask"])           # composited RGB
    for i, rgba in f.layer_frames("mask", 100, 200):       # a layer's own RGBA pixels
        ...

# Edit: existing layers are never re-encoded
fflv.add_layer("debug.lvd", "pred", pred_frames, alpha=True, start=30)
fflv.remove_layers("debug.lvd", ["boxes"])
fflv.set_layer("debug.lvd", "mask", opacity=0.5)
fflv.render("debug.lvd", "clip.mp4", layers=["frame", "pred"], start=0, end=300)
```

**`Writer` rules:**
- All layers must be declared before the first `write()`.
- A layer appears from the first frame it is given an image. If a later frame gives it no image, it keeps showing the previous one; `end_layer()` ends it early.
- Layers are encoded in parallel and every frame is written to disk immediately, so memory use does not grow with the length of the video.
- The file is first written as `.part`, renamed on close, and validated automatically.

**Quality and size:**
- `crf` (default 32; lower means better quality and bigger files).
- `speed` (`fast` / `balanced` / `best`).
- `lossless=True`: pixel values are kept bit for bit. Meant for masks and numeric debug images; files get noticeably larger.

## Repository layout

```
fflv/                Python package (command line + API)
  format/            binary structures, reading/writing, VP9 header parsing, validator (invariants I1–I10)
  encode/            VP9 encoding (numpy in process / FFmpeg), Opus audio, Writer
  decode.py          Reader: decoding selected layers, compositing
  render.py          render / extract outputs
  edit.py            add / rm / set (remux, no re-encoding)
  view.py            local server for fflv view
  project.py         pack (from a project file)
  devtools/          testsrc (test material), corrupt (broken files)
  viewer/            the built web player
player/              web player sources (TypeScript + Vite, WebCodecs + WebGL2, no framework)
tests/               pytest
```

## Tests

```bash
conda activate lvf
python -m pytest                          # Python: format, Writer, editing, decoding, render, view server, CLI
fflv testsrc                              # generate test_assets/ and test_assets/test.lvd
fflv corrupt test_assets/test.lvd --check # all 13 kinds of broken files must be reported precisely
cd player && npm test && npm run build
npm run e2e                               # end-to-end in Chrome for Testing + Edge (run where fflv is installed)
LVF_SOAK_SECONDS=600 npx playwright test -g "long playback" --project=chromium   # 10-minute playback
```

**How the end-to-end tests check synchronization:** every video layer of the test material carries a barcode of its own frame number. In continuous playback, seeking, scrubbing, frame stepping, layer toggling, CPU throttling, slow storage and other scenarios, the tests read the barcodes of all visible layers back from the canvas **for every composite frame drawn**, and check that all of them equal the current frame number.

**How the lossless layer is checked:** the tests compare every pixel on the canvas against the generated data. Both RGB and alpha must match exactly.

| Acceptance item (spec §11.2) | Test |
|---|---|
| 1 Pack + all invariants pass | `tests/test_pack.py`; `fflv pack` validates automatically |
| 2 Broken files are reported precisely | `tests/test_bad_files.py`, `fflv corrupt --check` |
| 3 Continuous playback, red line centred in the white line | `player/e2e/acceptance.spec.ts` |
| 4 Frame numbers agree after toggling / scrubbing / stepping | same (45 random operations) |
| 5 No desync at 6× CPU slowdown, P6 failures = 0 | same (plus slow-storage and busy-main-thread stress tests) |
| 6 Beep aligned with the white flash | same (audio timeline + offset measured at the display) |
| 7 Alpha calibration ends fully transparent / fully opaque | same |
| 8 No memory growth in long playback | same (60 s by default, 600 s configurable) |
