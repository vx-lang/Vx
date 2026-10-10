#!/usr/bin/env bash
#===- xpu_bandwidth_spike.sh - the Arc A770's achieved bandwidth ----------===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Measure the memory bandwidth this machine really gets: a device-to-device
# copy through Level Zero, and a triad kernel (x = a*y + b*z) launched the same
# way the other spike launches a kernel. Both are timed best-of-several after a
# warm-up, and both are checked against the CPU.
#
#   ./scripts/tools/xpu_bandwidth_spike.sh [size in MiB, default 512]
#
# Memory: at the default size the copy phase holds 1 GiB of VRAM and the triad
# phase 1.5 GiB. The card is shared and this machine is unstable above 12 GiB in
# use, so the program prints its footprint and refuses to go past
# `VX_VRAM_CEILING` (12 GiB by default).
#
#===----------------------------------------------------------------------===#

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${TMPDIR:-/tmp}/vx-xpu-bandwidth"
SIZE_MIB="${1:-512}"
mkdir -p "$OUT"

if [ -f "$ROOT/config.local" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/config.local"
fi

for tool in clang spirv-val; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool not on PATH (source config.local)" >&2
    exit 1
  }
done

echo "==> building the triad kernel"
clang -target spirv64-unknown-unknown -x cl -cl-std=CL2.0 \
  -c "$HERE/xpu_bandwidth_spike.cl" -o "$OUT/triad.spv"
spirv-val "$OUT/triad.spv"
echo "    $OUT/triad.spv validates"

echo "==> building the measurement"
"${CXX:-g++}" -std=c++17 -O2 -o "$OUT/bandwidth" \
  "$HERE/xpu_bandwidth_spike.cpp" -lze_loader

echo "==> measuring ($SIZE_MIB MiB per buffer)"
cd "$OUT"
./bandwidth triad.spv triad "$SIZE_MIB" 2>&1 | tee "$OUT/summary.txt"
NUMBERS="$(grep -E 'copy:|triad:' "$OUT/summary.txt" || true)"
[ -n "$NUMBERS" ] || {
  echo "error: no bandwidth numbers came out" >&2
  exit 1
}

echo "==> sabotage: tell the triad only half the length"
if ./bandwidth triad.spv triad "$SIZE_MIB" short-n >"$OUT/sabotage.txt" 2>&1; then
  echo "error: the triad check passed with a short length; it cannot fail" >&2
  exit 1
else
  echo "    the check failed as it should: $(grep -E 'triad:' "$OUT/sabotage.txt")"
fi

echo "==> sabotage: skip the device copy"
if ./bandwidth triad.spv triad "$SIZE_MIB" skip-copy >"$OUT/copy_sabotage.txt" 2>&1; then
  echo "error: the copy check passed with the copy skipped; it cannot fail" >&2
  exit 1
else
  echo "    the check failed as it should: $(grep -E 'copy:' "$OUT/copy_sabotage.txt")"
fi

echo "==> sabotage: skip the readback"
if ./bandwidth triad.spv triad "$SIZE_MIB" skip-readback >"$OUT/readback_sabotage.txt" 2>&1; then
  echo "error: the copy check passed with the readback skipped; it cannot fail" >&2
  exit 1
else
  echo "    the check failed as it should: $(grep -E 'copy:' "$OUT/readback_sabotage.txt")"
fi

echo "==> done"
