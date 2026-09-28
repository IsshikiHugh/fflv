# fflv

**LVF** is a layered video format, and **fflv** is its core implementation: a command-line tool, a
Python package and a minimal web player.

## What it is

A *layered video* (`.lvd`) holds several video layers, still images and an audio track in one
file. Layers stay strictly frame-synchronized: the file stores **composite frames**, and each one
carries one frame of every layer for the same instant. A player therefore always shows layers from
the same moment. When it has to skip frames, it skips whole composite frames. During playback any
layer can be shown, hidden, or given another opacity.

The format was built for debugging. For example, you can put input frames, predictions, masks and
heat maps in separate layers and compare them frame by frame.

- **Format:** [`LVF_SPEC.md`](LVF_SPEC.md).
- **Codecs:** video layers are VP9 (with optional alpha, and an optional lossless mode that keeps
  pixel values exactly); audio is Opus.
- **Core implementation:** written in Rust (`crates/`). It covers the container, a validator,
  encoding and selective decoding (only the layers you ask for), compositing, and editing without
  re-encoding existing layers.

## Web player

`fflv view` starts a small local server and opens a minimal web player in Chrome or Edge. The
player uses WebCodecs and WebGL2. It reads the file on demand through HTTP range requests, plays
all layers in sync with the audio, and reloads automatically when the file changes on disk.

- <kbd>Space</kbd>: play / pause.
- <kbd>←</kbd> / <kbd>→</kbd> (or <kbd>,</kbd> / <kbd>.</kbd>): step one frame; with <kbd>⇧</kbd>: ten frames.
- <kbd>Home</kbd> / <kbd>End</kbd>: first / last frame. <kbd>G</kbd> (or click the frame number): type a frame number.
- <kbd>1</kbd>–<kbd>9</kbd> (or the layer's eye icon): show / hide a layer. <kbd>0</kbd> (or the eye above the list): show all.
- <kbd>⇧</kbd>+<kbd>1</kbd>–<kbd>9</kbd> (or <kbd>⌥</kbd>+click the eye): show only that layer; again shows all.
- Clicking a layer row opens its details: format, active frames, and the opacity slider.

The player sources are in `player/`.

## Usage

### Install

Building needs Rust ≥ 1.83, libvpx and pkg-config:

- macOS: `brew install libvpx pkg-config`
- Debian/Ubuntu: `apt install libvpx-dev pkg-config libclang-dev`

The FFmpeg command line is needed only for importing media files, audio and non-PNG images, and
for video or JPEG output.

```bash
cargo install --path crates/fflv   # the fflv command
pip install .                      # the Python package (needs numpy)
```

### Python

```python
import fflv

with fflv.Writer("debug.lvd", size=(1280, 720), fps=30) as w:
    w.add_layer("frame")                              # opaque, full canvas
    w.add_layer("mask", alpha=True, lossless=True)    # exact pixel values
    for img, mask in data:                            # numpy arrays: H×W, H×W×3 or H×W×4
        w.write(frame=img, mask=mask)

with fflv.open("debug.lvd") as f:
    rgb = f.frame(120, layers=["frame", "mask"])      # composited; only these layers are decoded

fflv.set_layer("debug.lvd", "mask", opacity=0.5)      # edits never re-encode existing layers
```

### Command line

```bash
fflv view    debug.lvd                                  # open the web player
fflv info    debug.lvd                                  # layers, statistics, format checks
fflv add     debug.lvd --src pred.mp4 --id pred --alpha # add a layer from any video file
fflv rm      debug.lvd pred                             # remove a layer
fflv set     debug.lvd mask opacity=0.5 visible=false   # change layer properties in place
fflv render  debug.lvd -o out.mp4 -l frame,mask         # composite chosen layers to video / images / .npy
fflv pack    project.json -o out.lvd                    # build a file from a project description
```

Run `fflv --help` or `fflv <command> --help` for all options.

## Development

```bash
cargo test                                   # Rust
pip install maturin && maturin develop --release && python -m pytest   # Python
cd player && npm install && npm test && npm run build   # player (then rebuild fflv: it embeds the player)
```

## License

MIT
