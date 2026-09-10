#!/usr/bin/env bash
#===- run_shakedown.sh - Vx Compiler --------------------------*- bash -*-===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Tier-0 protocol shakedown: predict -> measure -> compare, end to end, on the
# machine we already own.
#
# The output is an error table for one memory link. Its purpose is NOT data about Apple silicon --
# it is to find the bugs in our own harness (a wrong formula, a mislabeled column, a unit mismatch)
# before any GPU time is paid for. The MLSys admission matrix was dry-run the same way and caught
# an 8x arithmetic error before any money was spent; this one caught a timer-granularity bug that
# would have produced infinite bandwidths on the rented box too.
#
#   ./run_shakedown.sh [--machine fleet/m4-uma.vx]
#
#===----------------------------------------------------------------------===#
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=$(pwd)
# shellcheck disable=SC1091
[ -f config.local ] && source config.local

MACHINE=${1:-fleet/m4-uma.vx}
[ "${1:-}" = "--machine" ] && MACHINE=${2:-fleet/m4-uma.vx}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$ROOT/utils/memalg/results/shakedown-$STAMP"
mkdir -p "$OUT"

# Provenance. A calibration result is a claim about one compiler on one machine; a log that cannot
# say which is not evidence.
{
    echo "stamp=$STAMP"
    echo "host=$(hostname)"
    echo "uname=$(uname -a)"
    echo "machine_file=$MACHINE"
    echo "commit=$(git rev-parse HEAD)"
    echo "branch=$(git rev-parse --abbrev-ref HEAD)"
    echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
    echo "cc=$(cc --version 2>/dev/null | head -1)"
    if command -v sysctl >/dev/null; then
        echo "cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
        echo "memsize=$(sysctl -n hw.memsize 2>/dev/null || true)"
        echo "l2cache=$(sysctl -n hw.perflevel0.l2cachesize 2>/dev/null || true)"
        echo "ncpu=$(sysctl -n hw.ncpu 2>/dev/null || true)"
    fi
    # The declared figures the predictions come from, copied in so the error table is readable
    # without going back to the machine file at the commit it was produced from.
    echo "--- declared ---"
    grep -E "capacity:|bandwidth:|transfer " "$MACHINE" | sed 's/^/  /'
} > "$OUT/env.txt"

echo "=== building ==="
cargo build --release --bin vxc
cc -O2 utils/memalg/measure_link.c -o utils/memalg/measure_link

echo "=== raw measurement ==="
./utils/memalg/measure_link > "$OUT/measured.csv" 2> "$OUT/measured.log"

echo "=== predict / measure / compare ==="
python3 utils/memalg/shakedown.py \
    --machine "$MACHINE" \
    --out "$OUT/error_table.csv" 2> "$OUT/shakedown.log" | tee "$OUT/error_table.txt"

echo
echo "results -> $OUT"
