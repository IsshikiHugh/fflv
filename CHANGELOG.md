# Changelog

## Unreleased

- Player: the export panel ("Export…") takes a range of frames (fields, or I / O
  on the frame on screen; shown under the seek bar) and a path: fflv writes the video there
  directly (next to the .lvd by default), instead of the browser downloading it. An existing file
  is replaced only after asking; a "download" link still fetches a copy. On a non-loopback address
  exports stay inside the .lvd's folder.
- `POST /export` takes `start`, `end`, `path` and `overwrite`; `GET /export` also returns the
  .lvd's folder and name.
- Player: exports can also render in the browser (WebCodecs, MP4: H.264 or VP9), with no fflv:
  for files opened in the page, and as a choice ("Render") under `fflv view`. BT.709 limited
  range and even-size padding as in `fflv render`; saved with the browser's save dialog.
- Player: color planes are converted to RGB by the player, with fflv's formulas, instead of by the
  browser, whose BT.709 conversion got the blue of saturated colors wrong by up to ~10 levels. The
  player (and exports in the browser) now match `fflv render` (LVF_SPEC.md A.13).
- The player is published as a static page on GitHub Pages (`.github/workflows/pages.yml`).
- `fflv view` without `--port` tries 8765 and then at most 10 ports after it (it tried 100, then
  any port), and skips a port another process holds on 127.0.0.1 also when listening on another
  address (such as an ssh tunnel's: binding 0.0.0.0 there succeeded, but requests reached the
  tunnel).

## 0.4.0 (2026-10-01)

- The built web player is no longer tracked: building fflv (`cargo build`, `maturin develop`,
  `pip install .`) builds it from `player/` when its sources change, so it never goes stale and
  pull requests no longer carry a bundle. This needs Node.js with npm (version in the README);
  without it, or when the player fails to build, fflv builds without the player (with a warning)
  and `fflv view` says so (`FFLV_PLAYER=skip` does that on purpose). Release builds build it
  once; the wheels, binaries and sdist include it, so installing needs no Node.
- `fflv view` (and `fflv.view`) no longer opens a browser by default: it prints the URL. `--open`
  (`open_page=True`) or `--browser` opens it as before; `--no-open` is still accepted.
- `scripts/install.sh`: installs the latest (or a chosen) release into the active Python
  environment, or the standalone binary with `--bin`; README's install instructions use it, so they
  name no version. Each release is installed this way on clean systems after it is published.
- `skills/fflv`: a Claude Code skill that points to the README and `--help` for installation and
  usage instead of repeating them.
- Player: the layer panel can be dragged wider (its left edge; double-click resets), so long layer
  names show in full. Layer thumbnails can be shown over a checkerboard (white and light gray, as in
  Photoshop), black or white (switch at the top of the panel). Both choices are kept in the browser.
- Player: "Export video" (with `fflv view`) renders the layers shown, at their opacities and in
  the panel's order, to an MP4 / WebM / MOV / MKV file on the server and downloads it.
  `RenderOptions` gains `opacity` and `order` (per-layer opacities and a draw order to use instead
  of the file's).
- Player: the layer order can be changed by dragging rows (or Move up / down in a row's details);
  it is kept when the file reloads, and "reset order" restores the file's z order.
- Robustness: malformed files can no longer stall or overflow the reader or validator; edits lock
  the source until the result is published (unlocked where the file system cannot lock), keep its
  permissions and follow symbolic links; `rm FILE --audio -o OUT` reads FILE (it edited OUT);
  `fflv view` bounds request bodies and writes exports to a private directory.
- Player: no silence after a backward seek or loop, no stuck end, sound keeps playing in a hidden
  tab, overlapping loads and WebGL context loss are handled; fewer requests and allocations.
- Faster: the compositing canvas is reused, encoder and decoder threads are shared, and
  `Reader.frame(i)` reuses its decoder for sequential access.
- Behaviour changes: Rust ≥ 1.89 to build; canvas sides are limited to 16384; chroma of lossy
  alpha layers is alpha-weighted (transparent pixels no longer bleed into edges); cancelling an
  export needs a JSON request.

## 0.3.0 (2026-09-27)

- Player: Photoshop-style layer panel (eye toggles, live thumbnails with real alpha, details
  drawer), Alt+click / Shift+number to show one layer, first/last-frame buttons, 10-frame steps,
  Home/End, a frame-number field, keyboard and screen-reader fixes, an error state with retry,
  and a layout that keeps the transport on screen in short windows.
- Distribution: wheels for Linux (x86_64, aarch64, manylinux_2_28) and macOS with libvpx built
  in and standalone binaries, on the GitHub release; each build is installed and smoke-tested on
  Rocky Linux 8, Debian 11 and current Debian before it is published. CI runs the Rust, Python
  and player test suites on Linux and macOS.
- Docs: the spec names the implemented tools; README covers installation on servers.

## 0.2.0

- Rust implementation of the container, codecs, editing, rendering and `fflv view`; Python
  bindings replace the Python prototype; the player is compiled into the `fflv` binary.
