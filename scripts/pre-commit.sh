#!/usr/bin/env bash
# Vx Compiler - Pre-commit validation script
# This script enforces our project rules before allowing a commit.

set -e

echo "======================================"
echo "Running Pre-commit Checks for Vx..."
echo "======================================"

# Rule 1: Markdown Formatting
echo "[1/4] Checking Markdown Formatting (mdformat)..."
if ! command -v mdformat &> /dev/null; then
    echo "❌ mdformat could not be found. Please install it using: pipx install mdformat"
    exit 1
fi
mdformat --check docs/ README.md || (echo "❌ Markdown files are not formatted properly. Run 'mdformat docs/ README.md' locally to fix." && exit 1)
echo "✅ Markdown files formatted perfectly!"

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
