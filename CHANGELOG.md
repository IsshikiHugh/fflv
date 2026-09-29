# Changelog

## Unreleased

- `fflv view` (and `fflv.view`) no longer opens a browser by default: it prints the URL. `--open`
  (`open_page=True`) or `--browser` opens it as before; `--no-open` is still accepted.
- `scripts/install.sh`: installs the latest (or a chosen) release into the active Python
  environment, or the standalone binary with `--bin`; README's install instructions use it, so they
  name no version. Each release is installed this way on clean systems after it is published.
- `skills/fflv`: a Claude Code skill that points to the README and `--help` for installation and
  usage instead of repeating them.
- Player: the layer panel can be dragged wider (its left edge; double-click resets), so long layer
  names show in full. Layer thumbnails can be shown over a checkerboard, black or white (switch at
  the top of the panel). Both choices are kept in the browser.

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
