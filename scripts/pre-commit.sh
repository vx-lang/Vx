#!/usr/bin/env bash
#===- pre-commit.sh - Vx Compiler ------------------------------*- Shell -*-===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
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
mdformat docs/ README.md
git add docs/ README.md
echo "✅ Markdown files formatted perfectly!"

# Rule 2: Rust Formatting Check
echo "[2/5] Checking Code Formatting (cargo fmt)..."
cargo fmt --all -- --check
echo "✅ Rust Formatting is perfect!"

# Rule 3: Vx Formatting Check
echo "[3/5] Checking Vx Formatting (vx-format)..."
cargo build --bin vx-format
find tests stdlib -name "*.vx" -exec ./target/debug/vx-format {} +
git add tests/ stdlib/
echo "✅ Vx Formatting is perfect!"

# Rule 4: Linting Check
echo "[4/5] Checking Lints (cargo clippy)..."
cargo clippy --all-targets --all-features -- -D warnings
echo "✅ No clippy warnings found!"

# Rule 5: Test Suite
echo "[5/5] Running Test Suite (cargo test)..."
cargo test
echo "✅ All tests passed!"

echo "======================================"
echo "🎉 All checks passed! Ready to commit."
echo "======================================"
