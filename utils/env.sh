#!/bin/bash

# Vx Local Toolchain Environment Setup
# Source this file before working on the project to use the locally isolated rustup and cargo paths.
# Usage: source env.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/config.local"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "Error: config.local not found. Please run $PROJECT_ROOT/setup.sh first."
    return 1 2>/dev/null || exit 1
fi

# Ensure stable is selected
rustup default stable
