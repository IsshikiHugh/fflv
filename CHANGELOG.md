# Changelog

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
