#!/usr/bin/env bash
# Akar Compiler - Pre-commit validation script
# This script enforces our project rules before allowing a commit.

set -e

echo "======================================"
echo "Running Pre-commit Checks for Akar..."
echo "======================================"

# Rule 1: Markdown Formatting
echo "[1/4] Checking Markdown Formatting..."
find docs/ -name "*.md" -type f -exec perl -pi -e 's/[ \t]+$//' {} +
echo "✅ Markdown files stripped of trailing whitespaces!"

# Rule 2: Rust Formatting Check
echo "[2/4] Checking Code Formatting (cargo fmt)..."
cargo fmt --all -- --check
echo "✅ Formatting is perfect!"

# Rule 3: Linting Check
echo "[3/4] Checking Lints (cargo clippy)..."
cargo clippy --all-targets --all-features -- -D warnings
echo "✅ No clippy warnings found!"

# Rule 4: Test Suite
echo "[4/4] Running Test Suite (cargo test)..."
cargo test
echo "✅ All tests passed!"

echo "======================================"
echo "🎉 All checks passed! Ready to commit."
echo "======================================"
