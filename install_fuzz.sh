#!/bin/bash
export CARGO_HOME=/Users/adityak/go/Vx/.cargo
export RUSTUP_HOME=/Users/adityak/go/Vx/.rustup
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
cargo install cargo-fuzz cargo-llvm-cov
