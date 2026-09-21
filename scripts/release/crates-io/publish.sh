#!/usr/bin/env bash
# Publish a crates.io name-reservation stub.
#
#   ./scripts/release/crates-io/publish.sh vxc 0.0.2             # dry run
#   ./scripts/release/crates-io/publish.sh vxc 0.0.2 --publish   # for real
#
# Neither `vxc` nor `vxlang` is the compiler. The compiler links against LLVM 22 with MLIR, and
# optionally Enzyme and z3, so `cargo install` cannot build it -- a real publish would hand every
# user a build that fails at build.rs. What is published is a few lines that hold the name and
# point at the toolchain, which is why these stubs exist as their own sources rather than as
# something derived from the workspace manifest.
#
# The stub is copied out of the repository before building: a package inside the Vx tree belongs
# to the Vx workspace, and `cargo publish` would try to resolve it against that.
#
# A publish cannot be undone. A version number can never be reused, even after `cargo yank`, so
# the default is a dry run and the real thing needs --publish.
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

CRATE="${1:-}"
VERSION="${2:-}"
MODE="${3:---dry-run}"
WORKDIR="${VX_RELEASE_DIR:-$HOME/go/vx-releases}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "usage: $0 <vxc|vxlang> <version> [--publish]" >&2
    echo "       version is a bare semver, e.g. 0.0.2, with no leading v" >&2
    exit 1
}

[ -n "$CRATE" ] && [ -n "$VERSION" ] || usage
[ -d "$HERE/$CRATE" ] || { echo "error: no stub for '$CRATE' in $HERE" >&2; usage; }

case "$VERSION" in
    v*) echo "error: drop the leading v -- cargo wants ${VERSION#v}, not $VERSION" >&2; exit 1 ;;
    *[!0-9.]*) echo "error: '$VERSION' is not a bare semver" >&2; exit 1 ;;
esac

DEST="$WORKDIR/$CRATE"
echo "==> Staging $CRATE $VERSION in $DEST"
mkdir -p "$WORKDIR"
rm -rf "$DEST"
cp -R "$HERE/$CRATE" "$DEST"

# The checked-in manifest carries a version so it is readable on its own; the one that ships is
# whatever was asked for here.
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$DEST/Cargo.toml"
rm -f "$DEST/Cargo.toml.bak"
grep -E '^(name|version) = ' "$DEST/Cargo.toml"

cd "$DEST"

# config.local points CARGO_HOME at the repository so the toolchain stays self-contained, and a
# shell that has sourced it looks for the crates.io token there and finds none -- "no token found,
# please run `cargo login`" while ~/.cargo/credentials.toml holds a perfectly good one. These
# stubs have no LLVM dependency, so they want the ordinary cargo home.
cargo() { env -u CARGO_HOME command cargo "$@"; }

if [ "$MODE" = "--publish" ]; then
    echo "==> Publishing $CRATE $VERSION to crates.io. This cannot be undone."
    cargo publish --allow-dirty
else
    echo "==> Dry run. Nothing is uploaded. Pass --publish to do it for real."
    cargo publish --dry-run --allow-dirty
    echo
    echo "    Files that would ship:"
    cargo package --list --allow-dirty | sed 's/^/      /'
fi
