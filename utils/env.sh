#!/bin/bash

# Vx Local Toolchain Environment Setup
# Source this file before working on the project to use the locally isolated rustup and cargo paths.
# Usage: source env.sh

export CARGO_HOME=/Users/adityak/go/Vx/.cargo
export RUSTUP_HOME=/Users/adityak/go/Vx/.rustup
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

# Ensure stable is selected
rustup default stable
