#!/bin/bash

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
    time ./target/release/vxc --run "$file" > /dev/null
    
    echo "-------------------------------------"
done

echo "Benchmarks completed!"
