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

`fflv view` starts a small local server for a minimal web player and prints its URL; open it in
Chrome or Edge, or pass `--open` to have fflv open it. The player uses WebCodecs and WebGL2. It
reads the file on demand through HTTP range requests, plays all layers in sync with the audio, and
reloads automatically when the file changes on disk.

- <kbd>Space</kbd>: play / pause.
- <kbd>←</kbd> / <kbd>→</kbd> (or <kbd>,</kbd> / <kbd>.</kbd>): step one frame; with <kbd>⇧</kbd>: ten frames.
- <kbd>Home</kbd> / <kbd>End</kbd>: first / last frame. <kbd>G</kbd> (or click the frame number): type a frame number.
- <kbd>1</kbd>–<kbd>9</kbd> (or the layer's eye icon): show / hide a layer. <kbd>0</kbd> (or the eye above the list): show all.
- <kbd>⇧</kbd>+<kbd>1</kbd>–<kbd>9</kbd> (or <kbd>⌥</kbd>+click the eye): show only that layer; again shows all.
- Clicking a layer row opens its details: format, active frames, the opacity slider, and
  Move up / Move down.
- Drag a layer row to change the draw order; "reset order" goes back to the file's z order. Like
  visibility and opacity, this changes what the player shows and exports, not the file.
- **Export video** (top right): fflv renders the layers shown, at their opacities and in the panel's order, to an MP4,
  WebM, MOV or MKV file (as `fflv render` does), and the browser downloads it. Not available for a
  file opened with "Open .lvd…" or dropped on the page. Exports have no sound.

The player sources are in `player/`.

## Usage

### Install

Activate the Python environment to install into (venv, conda, …), then:

```bash
curl -fsSL https://raw.githubusercontent.com/IsshikiHugh/fflv/main/scripts/install.sh | sh
```

This installs the latest [release](https://github.com/IsshikiHugh/fflv/releases): the Python
package and the `fflv` command, as a prebuilt wheel with libvpx built in — no compiler or system
libraries needed. Add `-s -- --bin` after `sh` for the standalone `fflv` binary only (into
`~/.local/bin`); set `FFLV_VERSION=<tag>` for another release. See
[`scripts/install.sh`](scripts/install.sh) for what it runs and the other settings.

Wheels exist for:

- Linux x86_64 and aarch64 with glibc ≥ 2.28 (RHEL/Rocky/Alma 8+, Debian 10+, Ubuntu 18.10+);
  not musl (Alpine) or older systems such as CentOS 7;
- macOS arm64 and x86_64 (11+);
- Python ≥ 3.9 (the script upgrades pip to ≥ 20.3 if needed; older pip cannot see the wheels).

Elsewhere, build from source (below).

The FFmpeg command line is needed only for importing media files, audio and non-PNG images, and
for video or JPEG output; everything else (writing from numpy, reading, editing, PNG output, the
player) works without it.

#### On a remote Linux server

`fflv view` is a local web app: it serves the file on 127.0.0.1 and needs a desktop Chrome or Edge.
On a headless server start it there and forward the port:

```bash
fflv view debug.lvd --port 8765                      # on the server
ssh -L 8765:127.0.0.1:8765 user@server               # on your machine, then open the URL it printed
```

#### From source

Needs Rust ≥ 1.89, libvpx and pkg-config (`brew install libvpx pkg-config` on macOS;
`apt install libvpx-dev pkg-config libclang-dev` on Debian/Ubuntu):

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
fflv view    debug.lvd [--open]                         # serve the web player, print its URL
fflv info    debug.lvd                                  # layers, statistics, format checks
fflv add     debug.lvd --src pred.mp4 --id pred --alpha # add a layer from any video file
fflv rm      debug.lvd pred                             # remove a layer
fflv set     debug.lvd mask opacity=0.5 visible=false   # change layer properties in place
fflv render  debug.lvd -o out.mp4 -l frame,mask         # composite chosen layers to video / images / .npy
fflv pack    project.json -o out.lvd                    # build a file from a project description
```

Run `fflv --help` or `fflv <command> --help` for all options.

### Claude Code skill

[`skills/fflv`](skills/fflv/SKILL.md) teaches Claude Code when and how to use fflv, and to install it
from this README when it is missing. Link it into your skills:
`ln -s "$PWD/skills/fflv" ~/.claude/skills/fflv` (from a clone of this repository).

## Development

Prerequisites: the build dependencies above, FFmpeg (the test material is generated with it),
Node ≥ 20 for the player, and Python ≥ 3.9 for the package.

```bash
cargo test --release                                     # Rust
python -m venv .venv && . .venv/bin/activate             # any environment works; the package is
pip install maturin numpy pytest                         # built into the active one
maturin develop --release && python -m pytest            # Python
cd player && npm ci && npm test && npm run build         # player: unit tests, then build it into
                                                         # crates/fflv/viewer (compiled into fflv)
cargo build --release && ./target/release/fflv testsrc   # rebuild fflv, generate test_assets/
cd player && npx playwright install chromium && npm run e2e -- --project chromium   # end-to-end tests
                                                         # (without --project they also run in Edge)
```

The built player in `crates/fflv/viewer/` is committed, so building fflv does not need Node;
CI checks that it matches the player sources. All of the above runs in CI
(`.github/workflows/ci.yml`) on Linux and macOS.

## License

MIT
