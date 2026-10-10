#!/usr/bin/env bash
# Build a redistributable Vx toolchain tarball for the host platform.
#
#   ./scripts/release/package.sh v0.0.3
#
# Produces  dist/vx-<version>-<target>.tar.gz  and its .sha256.
#
# The tarball carries the Vx compiler, its runtime library, the standard library and the machine
# files. With VX_BUNDLE_LLVM it also carries the LLVM tools vxc runs, built by
# scripts/provision/build_llvm.sh. Without it, the user supplies LLVM 22 through their own package
# manager, and the installer records where it found it.
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
#     llvm/           the LLVM tools and libraries, only with VX_BUNDLE_LLVM
#     etc/            written at install time, not here
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
    echo "usage: $0 <version>        e.g. $0 v0.0.3" >&2
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

# A CUDA build is a separate toolchain, not a variant of the portable one, and its name has to say
# so. The dispatch backend links -lcudart, -lcublas and -lcuda and carries an rpath into the build
# host's toolkit, so this tarball cannot even be loaded on a machine with no CUDA -- it is for
# NVIDIA hosts and nothing else. The portable tarball keeps the plain triple and runs anywhere.
#
# Set by the release workflow on the job that installs the toolkit. Detecting CUDA here instead
# would make the artifact's name depend on what happened to be installed on the builder, which is
# how a toolchain ends up named for a machine it cannot run on.
if [ -n "${VX_TARGET_SUFFIX:-}" ]; then
    TARGET="${TARGET}${VX_TARGET_SUFFIX}"
fi

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
# Any build directory, not this package's by name. Cargo derives that directory from the package
# name, so `build/Vx-*` stopped matching the moment the package was renamed to vxc -- and nothing
# noticed, because a machine that had built under the old name still had stale Vx-* directories
# lying around for the glob to find. A clean runner had only vxc-*, shipped no dispatch library,
# and every program containing `spawn on` failed to link with undefined vx_plugin_* symbols.
# The filenames below are specific enough to select on their own.
for out_dir in "$TARGET_DIR"/build/*/out; do
    [ -d "$out_dir" ] || continue
    for lib in "$out_dir"/*dispatch."${DLL}" "$out_dir"/*dispatch.a "$out_dir"/libvx_mlir_shims."${DLL}"; do
        [ -f "$lib" ] && cp "$lib" "$STAGE/lib/"
    done
done
ls "$STAGE/lib/"

# A toolchain without a dispatch backend links no program containing `spawn on`, which is the
# language's headline feature. Better to fail the build than to publish that.
# Tested by glob rather than by counting: `wc -l` pads its output, so a `case` against "0" never
# matches and the guard silently passes. An unmatched glob leaves the pattern itself in $1, which
# -e then rejects.
# The shared library is the one `vxc --run` links; a static archive alone used to pass this check.
set -- "$STAGE"/lib/*dispatch."${DLL}"
if [ ! -e "$1" ]; then
    echo "error: no shared dispatch library staged into lib/." >&2
    echo "  Looked in $TARGET_DIR/build/*/out for *dispatch.$DLL." >&2
    echo "  Without it every program using 'spawn on' fails to link." >&2
    exit 1
fi

echo "==> Staging the standard library, machine files and examples"
# Source only. The .vxlib interface format carries a version tag that the compiler rejects when
# it does not match, so shipping prebuilt interfaces would break on the next format bump.
#
# Every Vx module under stdlib/, rather than a list kept by hand. The list said `std graph`, and
# when `core` was added nothing updated it, so the toolchain shipped without it and `import
# std::vec` -- which reaches core::ops -- failed to resolve in a released build while working in
# every checkout. rust_core is the Rust crate behind libvx_std_core, not Vx source, so it is the
# one directory skipped.
mkdir -p "$STAGE/stdlib"
for dir in stdlib/*/; do
    name=$(basename "$dir")
    [ "$name" = "rust_core" ] && continue
    # No trailing slash: `cp -R dir/ dest` copies the *contents* of dir, which spills every
    # module's sources flat into stdlib/ and leaves no std/ or core/ to import.
    cp -R "${dir%/}" "$STAGE/stdlib/"
done
find "$STAGE/stdlib" -name '*.vxlib' -delete

# A toolchain whose standard library is missing a module fails only when a program imports it,
# which is how the gap above reached a release. Name the ones the smoke test depends on.
for required in std core; do
    [ -d "$STAGE/stdlib/$required" ] || {
        echo "error: stdlib/$required was not staged; a released toolchain needs it." >&2
        exit 1
    }
done

cp -R fleet "$STAGE/fleet"

mkdir -p "$STAGE/examples"
# llama.vx imports a module from tests/ and reads model files from it, so it runs only in a checkout.
for example in examples/*.vx; do
    [ "$(basename "$example")" = "llama.vx" ] && continue
    cp "$example" "$STAGE/examples/"
done

cp LICENSE "$STAGE/LICENSE"
cp README.md "$STAGE/README.md"
echo "$VERSION" > "$STAGE/VERSION"

# ------------------------------------------------------------------- LLVM --

# With VX_BUNDLE_LLVM set to the install from scripts/provision/build_llvm.sh, the toolchain
# carries the part of LLVM that vxc runs, in llvm/, and the user needs no LLVM of their own. The
# compiler itself already has MLIR and LLVM linked in; these are the tools it shells out to, the
# libraries a compiled program loads, and Enzyme.
if [ -n "${VX_BUNDLE_LLVM:-}" ]; then
    echo "==> Staging LLVM from $VX_BUNDLE_LLVM"
    mkdir -p "$STAGE/llvm/bin" "$STAGE/llvm/lib"
    for tool in llvm-config mlir-translate opt llc clang-22 clang; do
        [ -e "$VX_BUNDLE_LLVM/bin/$tool" ] || {
            echo "error: $VX_BUNDLE_LLVM/bin/$tool is missing." >&2
            exit 1
        }
        # -P keeps clang a link to clang-22 rather than a second copy.
        cp -P "$VX_BUNDLE_LLVM/bin/$tool" "$STAGE/llvm/bin/"
    done
    for lib in mlir_c_runner_utils mlir_runner_utils mlir_float16_utils; do
        set -- "$VX_BUNDLE_LLVM/lib/lib$lib.$DLL"*
        [ -e "$1" ] || {
            echo "error: $VX_BUNDLE_LLVM/lib/lib$lib.$DLL is missing." >&2
            exit 1
        }
        cp -P "$@" "$STAGE/llvm/lib/"
    done
    # clang's own headers and its runtime library directory.
    cp -R "$VX_BUNDLE_LLVM/lib/clang" "$STAGE/llvm/lib/"
    cp -R "$VX_BUNDLE_LLVM/lib/enzyme" "$STAGE/llvm/lib/"
    du -sh "$STAGE/llvm"
fi

# With VX_BUNDLE_ENZYME set to an Enzyme plugin, a toolchain that carries no LLVM ships it in
# lib/enzyme/, to load into the user's LLVM. scripts/provision/build_enzyme.sh builds one.
if [ -n "${VX_BUNDLE_ENZYME:-}" ]; then
    echo "==> Staging Enzyme from $VX_BUNDLE_ENZYME"
    mkdir -p "$STAGE/lib/enzyme"
    cp "$VX_BUNDLE_ENZYME" "$STAGE/lib/enzyme/"
fi

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
    # Enzyme loads into the user's opt and must use that opt's libLLVM. The path to the build
    # machine's libLLVM, and its absolute rpaths, could name a different LLVM on the user's Mac.
    for plugin in "$STAGE"/lib/enzyme/*.dylib; do
        [ -f "$plugin" ] || continue
        llvm=$(otool -L "$plugin" | tail -n +2 | awk '{print $1}' | grep '/libLLVM[^/]*\.dylib$' || true)
        [ -z "$llvm" ] || install_name_tool -change "$llvm" "@rpath/libLLVM.dylib" "$plugin"
        rpaths=$(otool -l "$plugin" | awk '/cmd LC_RPATH/ {getline; getline; print $2}' | grep '^/' || true)
        for rpath in $rpaths; do
            install_name_tool -delete_rpath "$rpath" "$plugin"
        done
        codesign --force --sign - "$plugin"
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

# The LLVM this toolchain carries, if it has one; otherwise the one the installer found on this
# machine.
if [ -x "\$PREFIX/llvm/bin/llvm-config" ]; then
    VX_LLVM_BIN="\$PREFIX/llvm/bin"
elif [ -f "\$PREFIX/etc/llvm-env.sh" ]; then
    . "\$PREFIX/etc/llvm-env.sh"
fi

# Enzyme, for grad, vjp and jvp: inside the LLVM this toolchain carries, or on its own in lib/.
for candidate in "\$PREFIX"/llvm/lib/enzyme/LLVMEnzyme-* "\$PREFIX"/lib/enzyme/LLVMEnzyme-*; do
    if [ -f "\$candidate" ]; then
        export ENZYME_LIB="\${ENZYME_LIB:-\$candidate}"
        break
    fi
done

# Upgrading LLVM can delete the folder the installer saved.
if [ -n "\${VX_LLVM_BIN:-}" ] && [ ! -d "\$VX_LLVM_BIN" ]; then
    echo "error: Vx was set up to use LLVM in \$VX_LLVM_BIN, which no longer exists." >&2
    echo "  Run the installer again to find LLVM:" >&2
    echo "    curl -fsSL https://vxlang.org/install.sh | VX_VERSION=${VERSION} sh" >&2
    exit 1
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
#
# Shared libraries only. lib/ also holds libnpu_dispatch.a, which the AOT linker needs, and an
# archive is not something dlopen can ever load -- naming it here made the first emit-obj of a
# fresh install print a dyld failure about a "slice is not valid mach-o file".
for candidate in "\$PREFIX"/lib/*dispatch."${DLL}"; do
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
if [ -z "${VX_BUNDLE_LLVM:-}" ]; then
    echo "    printf 'VX_LLVM_BIN=\"%s\"\\n' \"\$(dirname \"\$(command -v llvm-config)\")\" \\"
    echo "        > /tmp/vx-${VERSION}-${TARGET}/etc/llvm-env.sh"
fi
echo "    cd /tmp && /tmp/vx-${VERSION}-${TARGET}/bin/vxc --run <a .vx file>"
