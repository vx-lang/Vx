#!/usr/bin/env bash
# Akar Compiler - Pre-commit validation script
# This script enforces our project rules before allowing a commit.

set -e

echo "======================================"
echo "Running Pre-commit Checks for Akar..."
echo "======================================"

# Rule 1: Formatting Check
echo "[1/3] Checking Code Formatting (cargo fmt)..."
cargo fmt --all -- --check
echo "✅ Formatting is perfect!"

# Rule 2: Linting Check
echo "[2/3] Checking Lints (cargo clippy)..."
cargo clippy --all-targets --all-features -- -D warnings
echo "✅ No clippy warnings found!"

# Rule 3: Test Suite
echo "[3/3] Running Test Suite (cargo test)..."
cargo test
echo "✅ All tests passed!"

echo "======================================"
echo "🎉 All checks passed! Ready to commit."
echo "======================================"
