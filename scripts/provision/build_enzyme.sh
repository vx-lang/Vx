#!/usr/bin/env bash
# Builds Enzyme against an LLVM that is already installed, such as Homebrew's LLVM 22 on macOS, at
# the commit scripts/provision/build_llvm.sh pins. The macOS release ships the result, and on a Mac
# it is also how a checkout gets grad, vjp and jvp working.
#
#   scripts/provision/build_enzyme.sh <llvm bin directory> <output directory>
#
# Writes <output directory>/LLVMEnzyme-<LLVM major version>.dylib (.so on Linux). Run it from the
# repository root; it works in toolchain/enzyme/, which is safe to delete afterwards.
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

LLVM_BIN="${1:?usage: $0 <llvm bin directory> <output directory>}"
OUT="${2:?usage: $0 <llvm bin directory> <output directory>}"

# The pin is read from build_llvm.sh, so the two cannot drift apart. Editing that file instead
# would change the key CI caches its LLVM build under.
ENZYME_COMMIT=$(sed -n 's/^ENZYME_COMMIT="\([0-9a-f]*\)"$/\1/p' scripts/provision/build_llvm.sh)
if [ -z "$ENZYME_COMMIT" ]; then
    echo "Could not read ENZYME_COMMIT from scripts/provision/build_llvm.sh." >&2
    exit 1
fi

if [ -z "${JOBS:-}" ]; then
    CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
    JOBS=$((CORES > 3 ? CORES - 2 : 1))
fi

MAJOR=$("$LLVM_BIN/llvm-config" --version | cut -d. -f1)
SRC=toolchain/enzyme/src
BUILD=toolchain/enzyme/build

echo "Fetching Enzyme $ENZYME_COMMIT..."
if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/EnzymeAD/Enzyme.git
fi
if [ "$(git -C "$SRC" rev-parse -q --verify HEAD || true)" != "$ENZYME_COMMIT" ]; then
    git -C "$SRC" fetch -q --depth 1 origin "$ENZYME_COMMIT"
    git -C "$SRC" checkout -q --detach "$ENZYME_COMMIT"
fi

echo "Building Enzyme against LLVM $("$LLVM_BIN/llvm-config" --version)..."
cmake -S "$SRC/enzyme" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="$("$LLVM_BIN/llvm-config" --cmakedir)" \
    -DENZYME_BUILD_TESTS=OFF \
    -DENZYME_CLANG=OFF
ninja -C "$BUILD" -j "$JOBS" "LLVMEnzyme-$MAJOR"

mkdir -p "$OUT"
cp "$BUILD"/Enzyme/LLVMEnzyme-"$MAJOR".* "$OUT/"
ls -l "$OUT"/LLVMEnzyme-"$MAJOR".*
