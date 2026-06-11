#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/config.local"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "Error: config.local not found. Please run $PROJECT_ROOT/setup.sh first."
    exit 1
fi
cargo install cargo-fuzz cargo-llvm-cov
