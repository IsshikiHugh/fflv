#!/bin/sh
# Install fflv from a GitHub release: the wheel (Python package + the `fflv` command) into the
# active Python environment, or with --bin the standalone `fflv` binary.
#
#   curl -fsSL https://raw.githubusercontent.com/IsshikiHugh/fflv/main/scripts/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/IsshikiHugh/fflv/main/scripts/install.sh | sh -s -- --bin
#
# Environment: FFLV_VERSION (a release tag, default: the latest release), PYTHON (default: python3,
# i.e. the active venv / conda environment), FFLV_BIN_DIR (for --bin, default: ~/.local/bin).
# This script is the one place that knows how a release is installed; README.md points here.
set -eu

REPO=IsshikiHugh/fflv
PYTHON=${PYTHON:-python3}
MODE=wheel
case "${1:-}" in
  --bin) MODE=bin ;;
  "") ;;
  *) echo "usage: install.sh [--bin]" >&2; exit 2 ;;
esac

die() { echo "fflv install: $*" >&2; exit 1; }
command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found (set PYTHON=/path/to/python)"

# The latest release's tag: github.com/<repo>/releases/latest redirects to .../releases/tag/<tag>.
TAG=${FFLV_VERSION:-$("$PYTHON" -c "
import urllib.request
print(urllib.request.urlopen('https://github.com/$REPO/releases/latest').geturl().rsplit('/', 1)[1])
")} || die "cannot reach github.com to find the latest release"
VERSION=${TAG#v}
echo "fflv $TAG"

if [ "$MODE" = bin ]; then
  case "$(uname -s)" in Linux) OS=linux ;; Darwin) OS=macos ;; *) die "no binary for $(uname -s)" ;; esac
  case "$(uname -m)" in x86_64|amd64) ARCH=x86_64 ;; aarch64|arm64) ARCH=$([ $OS = macos ] && echo arm64 || echo aarch64) ;; *) die "no binary for $(uname -m)" ;; esac
  DIR=${FFLV_BIN_DIR:-$HOME/.local/bin}
  mkdir -p "$DIR"
  URL="https://github.com/$REPO/releases/download/$TAG/fflv-$TAG-$OS-$ARCH.tar.gz"
  "$PYTHON" -c "
import io, sys, tarfile, urllib.request
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(sys.argv[1]).read())).extract('fflv', sys.argv[2])
" "$URL" "$DIR" 2>/dev/null || die "download failed: $URL"
  chmod +x "$DIR/fflv"
  "$DIR/fflv" --version
  case ":$PATH:" in *":$DIR:"*) ;; *) echo "note: $DIR is not on PATH" ;; esac
  exit 0
fi

# Wheels need pip >= 20.3 (manylinux_2_28 tags); upgrade an older pip in this environment only.
if ! "$PYTHON" -c "
import sys, pip
sys.exit(0 if tuple(map(int, pip.__version__.split('.')[:2])) >= (20, 3) else 1)
" 2>/dev/null; then
  "$PYTHON" -m pip install --upgrade "pip>=20.3"
fi
# Pinned to the release's version, and found on that release's assets page.
"$PYTHON" -m pip install "fflv==$VERSION" --find-links "https://github.com/$REPO/releases/expanded_assets/$TAG" ||
  die "no wheel for this system; see https://github.com/$REPO#install (supported systems, building from source)"
"$PYTHON" -m fflv --version
