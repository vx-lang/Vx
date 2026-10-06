#!/usr/bin/env bash
# Builds the LLVM that Vx uses, from source, into toolchain/ in this checkout: LLVM, MLIR, clang
# (vxc runs it to link programs), FileCheck for the tests, and Enzyme built against that LLVM.
# Run it from the repository root, then run ./setup.sh, which points config.local at the result.
#
#   toolchain/src/llvm-project   a shallow clone of the pinned commit, with only the parts we build
#   toolchain/build/...          the build directories, safe to delete after the install
#   toolchain/install            what vxc, the build and the tests use
#
# Running it again reuses the clone and the build directories, so only what changed is rebuilt.
# JOBS sets the number of parallel jobs; it defaults to all cores but two.

set -euo pipefail

LLVM_TAG="llvmorg-22.1.8"
LLVM_COMMIT="ca7933e47d3a3451d81e72ac174dcb5aa28b59d1"
ENZYME_TAG="v0.0.302"
ENZYME_COMMIT="b1d93d0bb23a8cb01123e0b66f1b50d2c505c6d6"

PROJECT_DIR=$(pwd)
if [ ! -f "$PROJECT_DIR/setup.sh" ]; then
    echo "Run this script from the root of the Vx repository." >&2
    exit 1
fi

TOOLCHAIN="$PROJECT_DIR/toolchain"
SRC="$TOOLCHAIN/src"
BUILD="$TOOLCHAIN/build"
INSTALL="$TOOLCHAIN/install"

if [ -z "${JOBS:-}" ]; then
    if command -v nproc >/dev/null 2>&1; then
        CORES=$(nproc)
    else
        CORES=$(sysctl -n hw.ncpu)
    fi
    JOBS=$((CORES > 3 ? CORES - 2 : 1))
fi

for tool in git cmake ninja; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "$tool is needed to build LLVM and was not found." >&2
        exit 1
    fi
done

# Clone one commit of a repository into $1, fetching only the directories listed after the URL
# and commit. With no directories, the whole tree is checked out.
fetch() {
    local dir=$1 url=$2 commit=$3
    shift 3
    if [ ! -d "$dir/.git" ]; then
        git init -q "$dir"
        git -C "$dir" remote add origin "$url"
    fi
    if [ "$#" -gt 0 ]; then
        git -C "$dir" sparse-checkout set "$@"
    fi
    if [ "$(git -C "$dir" rev-parse -q --verify HEAD || true)" != "$commit" ]; then
        git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$commit"
        git -C "$dir" checkout -q --detach "$commit"
    fi
}

echo "Fetching LLVM $LLVM_TAG..."
fetch "$SRC/llvm-project" https://github.com/llvm/llvm-project.git "$LLVM_COMMIT" \
    llvm mlir clang cmake third-party

echo "Fetching Enzyme $ENZYME_TAG..."
fetch "$SRC/enzyme" https://github.com/EnzymeAD/Enzyme.git "$ENZYME_COMMIT"

case "$(uname -m)" in
    x86_64) HOST_TARGET=X86 ;;
    arm64 | aarch64) HOST_TARGET=AArch64 ;;
    *) echo "Unsupported CPU: $(uname -m)" >&2; exit 1 ;;
esac

# Only what Vx runs is built: no tests, examples or benchmarks, code generation for the host and
# for NVIDIA GPUs, and no optional system libraries that a user's machine would then need too.
echo "Building LLVM, MLIR and clang with $JOBS jobs (log: $BUILD/llvm.log)..."
mkdir -p "$BUILD/llvm"
cmake -S "$SRC/llvm-project/llvm" -B "$BUILD/llvm" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL" \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="$HOST_TARGET;NVPTX" \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF \
    -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DCLANG_INCLUDE_TESTS=OFF \
    > "$BUILD/llvm.log" 2>&1
ninja -C "$BUILD/llvm" -j "$JOBS" install >> "$BUILD/llvm.log" 2>&1

# An LLVM plugin loads only into the LLVM it was built for, so Enzyme is built here too.
echo "Building Enzyme (log: $BUILD/enzyme.log)..."
mkdir -p "$BUILD/enzyme"
cmake -S "$SRC/enzyme/enzyme" -B "$BUILD/enzyme" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="$INSTALL/lib/cmake/llvm" \
    -DENZYME_BUILD_TESTS=OFF \
    -DENZYME_CLANG=OFF \
    > "$BUILD/enzyme.log" 2>&1
ninja -C "$BUILD/enzyme" -j "$JOBS" LLVMEnzyme-22 >> "$BUILD/enzyme.log" 2>&1
mkdir -p "$INSTALL/lib/enzyme"
cp "$BUILD"/enzyme/Enzyme/LLVMEnzyme-22.* "$INSTALL/lib/enzyme/"

echo "Installed in $INSTALL. Run ./setup.sh to point config.local at it."
