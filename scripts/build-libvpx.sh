#!/bin/sh
# Build a static, position-independent libvpx and install it under PREFIX, for release builds
# that must not depend on a system libvpx (wheels, the binaries on the Releases page).
#
#   scripts/build-libvpx.sh PREFIX
#
# Then build with PKG_CONFIG_PATH=PREFIX/lib/pkgconfig and PKG_CONFIG_ALL_STATIC=1 so that
# vpx-sys links the .a. Needs git, make, a C compiler, and nasm on x86-64.
set -eu
VERSION=v1.16.0
PREFIX=$1
SRC=$(mktemp -d)
trap 'rm -rf "$SRC"' EXIT
git clone --quiet --depth 1 --branch "$VERSION" https://chromium.googlesource.com/webm/libvpx "$SRC/libvpx"
cd "$SRC/libvpx"
./configure --prefix="$PREFIX" \
  --enable-static --disable-shared --enable-pic \
  --enable-vp9 --enable-vp8 \
  --disable-examples --disable-tools --disable-docs --disable-unit-tests \
  --disable-install-bins --disable-install-docs
make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
make install
echo "libvpx $VERSION installed under $PREFIX"
