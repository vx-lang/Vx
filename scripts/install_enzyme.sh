#!/usr/bin/env bash
# scripts/install_enzyme.sh
# Downloads the appropriate Enzyme LLVM plugin for the current OS and LLVM version.

set -e

# Configuration
ENZYME_VERSION="v0.0.263"
DEST_DIR="$(pwd)/.cargo/enzyme"
mkdir -p "$DEST_DIR"

echo "Detecting system..."
OS="$(uname -s)"
ARCH="$(uname -m)"

# Determine OS suffix
if [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        SYS="aarch64-apple-darwin"
    else
        SYS="x86_64-apple-darwin"
    fi
elif [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "aarch64" ]; then
        SYS="aarch64-linux-gnu-cxx11"
    else
        SYS="x86_64-linux-gnu-cxx11"
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Detect LLVM version
LLVM_CONFIG=""
for cmd in llvm-config llvm-config-22 llvm-config-21 llvm-config-20 llvm-config-19 llvm-config-18 llvm-config-17 llvm-config-16 llvm-config-15; do
    if command -v $cmd >/dev/null 2>&1; then
        LLVM_CONFIG=$cmd
        break
    fi
done

# Check if LLVM is explicitly passed via PATH or Homebrew
if [ -z "$LLVM_CONFIG" ] && [ -x "/opt/homebrew/opt/llvm/bin/llvm-config" ]; then
    LLVM_CONFIG="/opt/homebrew/opt/llvm/bin/llvm-config"
fi

if [ -z "$LLVM_CONFIG" ]; then
    echo "Error: llvm-config not found. Please install LLVM."
    exit 1
fi

LLVM_VERSION_FULL=$($LLVM_CONFIG --version)
LLVM_MAJOR=$(echo $LLVM_VERSION_FULL | cut -d. -f1)

echo "Detected LLVM version: $LLVM_MAJOR (from $LLVM_VERSION_FULL)"

if [ "$LLVM_MAJOR" -gt 20 ]; then
    echo "Enzyme pre-built binaries for LLVM > 20 are not available yet."
    echo "Building Enzyme from source for LLVM $LLVM_MAJOR..."
    
    ENZYME_SRC="/tmp/enzyme_src_$$"
    rm -rf "$ENZYME_SRC"
    git clone --depth 1 https://github.com/EnzymeAD/Enzyme.git "$ENZYME_SRC"
    
    mkdir -p "$ENZYME_SRC/build_dir"
    cd "$ENZYME_SRC/build_dir"
    
    # Enzyme requires matching LLVMDIR
    LLVM_DIR=$($LLVM_CONFIG --prefix)/lib/cmake/llvm
    cmake -G Ninja ../enzyme -DENZYME_LLVM_DIR="$LLVM_DIR" -DCMAKE_BUILD_TYPE=Release
    ninja
    
    if [ "$OS" = "Darwin" ]; then
        cp Enzyme/LLVMEnzyme-*.dylib "$DEST_DIR/"
    else
        cp Enzyme/LLVMEnzyme-*.so "$DEST_DIR/"
    fi
    
    cd "$DEST_DIR"
    rm -rf "$ENZYME_SRC"
else
    # Not all LLVM versions have binaries (e.g. 17, 19 might be missing). We fallback if needed.
    LLVM_BIN_VER=$LLVM_MAJOR
    if [ "$LLVM_BIN_VER" = "17" ]; then LLVM_BIN_VER=16; fi
    if [ "$LLVM_BIN_VER" = "19" ]; then LLVM_BIN_VER=18; fi

    FILENAME="Enzyme.${ENZYME_VERSION}.${SYS}-llvm_version+${LLVM_BIN_VER}.tar.gz"
    URL="https://github.com/JuliaBinaryWrappers/Enzyme_jll.jl/releases/download/Enzyme-${ENZYME_VERSION}%2B0/${FILENAME}"

    echo "Downloading Enzyme from $URL"
    curl -sL -o "/tmp/enzyme.tar.gz" "$URL"

    echo "Extracting..."
    tar -xzf "/tmp/enzyme.tar.gz" -C "$DEST_DIR"
    rm -f "/tmp/enzyme.tar.gz"
fi

# Find the library
if [ "$OS" = "Darwin" ]; then
    LIB_EXT="dylib"
else
    LIB_EXT="so"
fi

ENZYME_LIB=$(find "$DEST_DIR" -name "LLVMEnzyme*.$LIB_EXT" | head -n 1)

if [ -z "$ENZYME_LIB" ]; then
    echo "Error: Could not find Enzyme library in extracted files."
    exit 1
fi

echo "Successfully installed Enzyme to: $ENZYME_LIB"
echo "Set ENZYME_LIB environment variable to use it:"
echo "export ENZYME_LIB=\"$ENZYME_LIB\""
