#!/usr/bin/env bash
# Build a redistributable Vx toolchain tarball for the host platform.
#
#   ./scripts/release/package.sh v0.1.0
#
# Produces  dist/vx-<version>-<target>.tar.gz  and its .sha256.
#
# The tarball is "slim": it carries the Vx compiler, its runtime library, the standard library
# and the machine files, but NOT LLVM. The user supplies LLVM 22 through their own package
# manager, and the installer records where it found it. That keeps the download near 300 MB
# instead of over a gigabyte, at the cost of one prerequisite.
#
# Layout produced:
#
#   vx-<version>-<target>/
#     bin/            wrapper scripts -- what the user runs
#     libexec/        the real compiler binaries
#     lib/            libvx_std_core, plus any non-system libraries the compiler links
#     stdlib/         the standard library, as Vx source
#     fleet/          machine files for real parts
#     examples/       runnable programs
#     etc/            written at install time, not here
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
    echo "usage: $0 <version>        e.g. $0 v0.1.0" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ------------------------------------------------------------------ target --

os=$(uname -s)
arch=$(uname -m)
case "$os/$arch" in
    Darwin/arm64)  TARGET="aarch64-apple-darwin"      ; DLL=dylib ;;
    Linux/x86_64)  TARGET="x86_64-unknown-linux-gnu"  ; DLL=so    ;;
    *) echo "error: no release target defined for $os/$arch" >&2; exit 1 ;;
esac

STAGE="$REPO_ROOT/dist/vx-${VERSION}-${TARGET}"
echo "==> Packaging Vx ${VERSION} for ${TARGET}"

# ------------------------------------------------------------------- build --

if [ -z "${VX_SKIP_BUILD:-}" ]; then
    echo "==> Building (release)"
    # shellcheck disable=SC1091
    [ -f config.local ] && source config.local
    cargo build --release --workspace
fi

TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release"

for required in vxc vx-format vx-opt; do
    [ -x "$TARGET_DIR/$required" ] || {
        echo "error: $TARGET_DIR/$required is missing. Build first, or unset VX_SKIP_BUILD." >&2
        exit 1
    }
done

# ------------------------------------------------------------------- stage --

rm -rf "$STAGE"
mkdir -p "$STAGE"/{bin,libexec,lib,etc}

echo "==> Staging binaries"
for tool in vxc vx-format vx-opt vx-analyzer; do
    if [ -x "$TARGET_DIR/$tool" ]; then
        cp "$TARGET_DIR/$tool" "$STAGE/libexec/$tool"
    else
        echo "    note: $tool not built, skipping"
    fi
done

cp "$TARGET_DIR/libvx_std_core.${DLL}" "$STAGE/lib/"

# The dispatch backend and the MLIR print shims are written by build.rs into its OUT_DIR, and the
# compiler records their ABSOLUTE build-time paths. On the machine that built it those paths exist,
# which is exactly why this stayed hidden: a tarball unpacked on the build host works, and the same
# tarball on any other machine hands clang a path to a file that was never shipped. Copy them in,
# and let the wrappers point the compiler at these copies.
for out_dir in "$TARGET_DIR"/build/Vx-*/out; do
    [ -d "$out_dir" ] || continue
    for lib in "$out_dir"/*dispatch."${DLL}" "$out_dir"/*dispatch.a "$out_dir"/libvx_mlir_shims."${DLL}"; do
        [ -f "$lib" ] && cp "$lib" "$STAGE/lib/"
    done
done
ls "$STAGE/lib/"

echo "==> Staging the standard library, machine files and examples"
# Source only. The .vxlib interface format carries a version tag that the compiler rejects when
# it does not match, so shipping prebuilt interfaces would break on the next format bump.
mkdir -p "$STAGE/stdlib"
for dir in std graph; do
    [ -d "stdlib/$dir" ] && cp -R "stdlib/$dir" "$STAGE/stdlib/"
done
find "$STAGE/stdlib" -name '*.vxlib' -delete

cp -R fleet "$STAGE/fleet"

mkdir -p "$STAGE/examples"
cp examples/*.vx "$STAGE/examples/" 2>/dev/null || true

cp LICENSE "$STAGE/LICENSE"
cp README.md "$STAGE/README.md"
echo "$VERSION" > "$STAGE/VERSION"

# ---------------------------------------------------- relocate the libraries --

# The compiler links a couple of libraries from the build machine's package manager. Left alone,
# they are absolute paths into /opt/homebrew or /usr/lib that may not exist on the user's box.
# Copy them in beside the binary and rewrite the references to be relative to it.
echo "==> Relocating linked libraries"

if [ "$os" = "Darwin" ]; then
    for bin in "$STAGE"/libexec/*; do
        [ -f "$bin" ] || continue
        # Non-system absolute paths: Homebrew and friends. /usr/lib and the frameworks ship
        # with macOS and are left as they are.
        # Collected first rather than piped, so a binary with no such dependencies leaves an
        # empty list instead of failing the pipeline on grep's exit status.
        deps=$(otool -L "$bin" | tail -n +2 | awk '{print $1}' | grep -E '^/(opt|usr/local)/' || true)
        printf '%s\n' "$deps" | while read -r dep; do
            [ -n "$dep" ] || continue
            base=$(basename "$dep")
            [ -f "$STAGE/lib/$base" ] || cp "$dep" "$STAGE/lib/$base"
            chmod u+w "$STAGE/lib/$base"
            install_name_tool -change "$dep" "@executable_path/../lib/$base" "$bin"
        done
        install_name_tool -add_rpath "@executable_path/../lib" "$bin" 2>/dev/null || true
    done
    # Re-sign: mutating a Mach-O invalidates the ad-hoc signature it was built with, and macOS
    # refuses to run a binary whose signature no longer matches its contents.
    for bin in "$STAGE"/libexec/* "$STAGE"/lib/*; do
        [ -f "$bin" ] && codesign --force --sign - "$bin" 2>/dev/null || true
    done
elif [ "$os" = "Linux" ]; then
    if command -v patchelf >/dev/null 2>&1; then
        for bin in "$STAGE"/libexec/*; do
            [ -f "$bin" ] || continue
            patchelf --set-rpath '$ORIGIN/../lib' "$bin" || true
        done
    else
        echo "    warning: patchelf not installed; skipping RPATH rewrite."
        echo "             Install it so the toolchain finds its own libraries."
    fi
fi

# -------------------------------------------------------------- wrappers ----

# What the user actually runs. Each one finds the toolchain prefix from its own location, points
# the compiler at the LLVM the installer found, and hands off to the real binary.
#
# Without this the compiler is unusable when installed: it shells out to mlir-translate, opt, llc
# and clang by bare name, so with LLVM absent from PATH the whole --run path dies with
# "Failed to run mlir-translate: No such file or directory" -- naming a binary the user has never
# heard of rather than the toolchain they are missing.
echo "==> Writing wrappers"
for tool in vxc vx-format vx-opt vx-analyzer; do
    [ -f "$STAGE/libexec/$tool" ] || continue
    cat > "$STAGE/bin/$tool" <<WRAPPER
#!/bin/sh
# Vx toolchain wrapper for ${tool}. Generated by scripts/release/package.sh.
set -eu

# Resolve this script to its real location, so the prefix is right whether the user runs it
# through a symlink in ~/.vx/bin or directly.
self="\$0"
while [ -L "\$self" ]; do
    link=\$(readlink "\$self")
    case "\$link" in
        /*) self="\$link" ;;
        *)  self="\$(dirname "\$self")/\$link" ;;
    esac
done
PREFIX=\$(CDPATH= cd -- "\$(dirname -- "\$self")/.." && pwd)

# Where the installer found LLVM on this machine.
if [ -f "\$PREFIX/etc/llvm-env.sh" ]; then
    . "\$PREFIX/etc/llvm-env.sh"
fi

if [ -n "\${VX_LLVM_BIN:-}" ] && [ -d "\$VX_LLVM_BIN" ]; then
    # Both mechanisms: the env vars for the call sites that read them, and PATH for the ones
    # that still resolve by bare name.
    PATH="\$VX_LLVM_BIN:\$PATH"
    export PATH
    export LLVM_CONFIG_PATH="\${LLVM_CONFIG_PATH:-\$VX_LLVM_BIN/llvm-config}"
    export MLIR_TRANSLATE_PATH="\${MLIR_TRANSLATE_PATH:-\$VX_LLVM_BIN/mlir-translate}"
    export OPT_PATH="\${OPT_PATH:-\$VX_LLVM_BIN/opt}"
    export LLC_PATH="\${LLC_PATH:-\$VX_LLVM_BIN/llc}"
    export CLANG_PATH="\${CLANG_PATH:-\$VX_LLVM_BIN/clang}"
fi

export VX_RUNTIME_LIB_DIR="\${VX_RUNTIME_LIB_DIR:-\$PREFIX/lib}"
export VX_STD_PATH="\${VX_STD_PATH:-\$PREFIX/stdlib/std:\$PREFIX/stdlib}"

# The dispatch backend and the print shims, as shipped in this toolchain rather than wherever they
# sat on the machine that built it. Each is set only if the file is actually present, so a
# toolchain built without one falls back to the compiler's own handling instead of naming a file
# that is not there.
for candidate in "\$PREFIX"/lib/*dispatch.*; do
    if [ -f "\$candidate" ]; then
        export VX_DISPATCH_LIB="\${VX_DISPATCH_LIB:-\$candidate}"
        break
    fi
done
if [ -f "\$PREFIX/lib/libvx_mlir_shims.${DLL}" ]; then
    export VX_MLIR_SHIMS="\${VX_MLIR_SHIMS:-\$PREFIX/lib/libvx_mlir_shims.${DLL}}"
fi

exec "\$PREFIX/libexec/${tool}" "\$@"
WRAPPER
    chmod +x "$STAGE/bin/$tool"
done

# --------------------------------------------------------------- tarball ----

echo "==> Creating the tarball"
cd "$REPO_ROOT/dist"
ARCHIVE="vx-${VERSION}-${TARGET}.tar.gz"
tar -czf "$ARCHIVE" "vx-${VERSION}-${TARGET}"

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
else
    shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"
fi

size=$(du -h "$ARCHIVE" | cut -f1)
echo ""
echo "    dist/$ARCHIVE  ($size)"
echo "    dist/$ARCHIVE.sha256"
echo ""
echo "Smoke-test it somewhere that is not this checkout:"
echo "    tar -xzf dist/$ARCHIVE -C /tmp"
echo "    printf 'VX_LLVM_BIN=\"%s\"\\n' \"\$(dirname \"\$(command -v llvm-config)\")\" \\"
echo "        > /tmp/vx-${VERSION}-${TARGET}/etc/llvm-env.sh"
echo "    cd /tmp && /tmp/vx-${VERSION}-${TARGET}/bin/vxc --run <a .vx file>"
