#!/usr/bin/env bash
set -e

PROJECT_DIR=$(pwd)

echo "Locating LLVM installation..."
if command -v brew >/dev/null 2>&1 && brew --prefix llvm >/dev/null 2>&1; then
    LLVM_PREFIX=$(brew --prefix llvm)
    LLVM_PATH="$LLVM_PREFIX/bin"
elif command -v llvm-config >/dev/null 2>&1; then
    LLVM_PATH=$(dirname $(command -v llvm-config))
elif [ -d "/opt/homebrew/opt/llvm/bin" ]; then
    LLVM_PATH="/opt/homebrew/opt/llvm/bin"
else
    echo "Could not automatically locate LLVM. Please install LLVM or manually create config.local based on config.template."
    exit 1
fi

echo "LLVM found at: $LLVM_PATH"

echo "Generating config.local from config.template..."
sed -e "s|{{PROJECT_DIR}}|$PROJECT_DIR|g" \
    -e "s|{{LLVM_PATH}}|$LLVM_PATH|g" \
    config.template > config.local

echo "config.local successfully generated! You can now source config.local to load the toolchain environment."
