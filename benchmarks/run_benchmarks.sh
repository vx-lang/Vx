#!/bin/bash
#===- run_benchmarks.sh - Vx Compiler --------------------------*- Shell -*-===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#

# Ensure we're in the project root
cd "$(dirname "$0")/.."

echo "Building Vx Compiler in release mode..."
cargo build --release

echo ""
echo "====================================="
echo "       Running Vx Benchmarks         "
echo "====================================="
echo ""

# Loop through all benchmark files
for file in benchmarks/*.vx; do
  echo "▶ Benchmarking $file..."
  # Run and time the execution
  # Suppress cargo output
  cargo run --release --bin vxc -- "$file" -O3 --run > /dev/null
  echo "-------------------------------------"
done

echo "Benchmarks completed!"
