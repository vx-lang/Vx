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

# Export paths required for llvm-config and Cargo during pre-commit compilation
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/config.local"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "Error: config.local not found. Paths may be missing. Please run $PROJECT_ROOT/setup.sh."
    exit 1
fi
echo "======================================"
echo "Running Pre-commit Checks for Vx..."
echo "======================================"

# Rule 1: Markdown Formatting
echo "[1/4] Checking Markdown Formatting (mdformat)..."
STAGED_MD=$(git diff --cached --name-only --diff-filter=ACM | grep '\.md$' || true)
if [ -n "$STAGED_MD" ]; then
    if git diff --name-only | grep -q '\.md$'; then
        echo "⚠️  Unstaged Markdown changes detected. Skipping mdformat to prevent accidental data loss."
    else
        if ! command -v mdformat &> /dev/null; then
            echo "❌ mdformat could not be found. Please install it using: pipx install mdformat"
            exit 1
        fi
        mdformat $STAGED_MD
        git add $STAGED_MD
        echo "✅ Markdown files formatted perfectly!"
    fi
else
    echo "✅ No markdown files to format."
fi

# Rule 2: Rust Formatting Check
echo "[2/5] Checking Code Formatting (cargo fmt)..."
cargo fmt --all -- --check
echo "✅ Rust Formatting is perfect!"

# Rule 3: Vx Formatting Check
echo "[3/5] Checking Vx Formatting (vx-format)..."
STAGED_VX=$(git diff --cached --name-only --diff-filter=ACM | grep '\.vx$' || true)
if [ -n "$STAGED_VX" ]; then
    if git diff --name-only | grep -q '\.vx$'; then
        echo "⚠️  Unstaged Vx changes detected. Skipping vx-format to prevent accidental data loss."
    else
        cargo build --bin vx-format
        for file in $STAGED_VX; do
            ./target/debug/vx-format "$file"
        done
        git add $STAGED_VX
        echo "✅ Vx Formatting is perfect!"
    fi
else
    echo "✅ No Vx files to format."
fi

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
