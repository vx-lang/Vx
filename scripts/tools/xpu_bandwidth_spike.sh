#!/usr/bin/env bash
#===- xpu_bandwidth_spike.sh - the Arc A770's achieved bandwidth ----------===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Measure the memory bandwidth this machine really gets from the Arc A770: a
# device-to-device copy through Level Zero, a triad kernel (x = a*y + b*z), and
# the two host-transfer directions through pinned host memory. The triad is
# compiled to SPIR-V and launched with the call sequence a device image uses
# (`zeModuleCreate`, `zeKernelCreate`, `zeCommandListAppendLaunchKernel`), so it
# takes the path a Vx-compiled kernel takes. Every measurement is timed
# best-of-several after a warm-up, and every one is checked against the CPU.
#
#   ./scripts/tools/xpu_bandwidth_spike.sh [size in MiB, default 512]
#
# Memory: the program prints its footprint and its ceiling and refuses to
# allocate past `VX_VRAM_CEILING` -- 2 GiB by default (vram_ceiling.h). At the
# default size the triad phase holds 1.5 GiB, the copy phase 1 GiB and the
# transfer phase 256 MiB.
#
# Only one of these runs may touch the card at a time, so the whole script holds
# a lock that every GPU helper in this directory takes.
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

for tool in clang spirv-val flock; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool not on PATH (source config.local)" >&2
    exit 1
  }
done

# Serialise GPU access, for the whole run including the sabotage steps below.
# The lock is held on file descriptor 9 until the script exits, so a second
# helper waits rather than allocating beside this one.
exec 9>"${TMPDIR:-/tmp}/vx-gpu.lock"
flock 9

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
NUMBERS="$(grep -E 'copy:|triad:|host to device:|device to host:' "$OUT/summary.txt" || true)"
[ -n "$NUMBERS" ] || {
  echo "error: no bandwidth numbers came out" >&2
  exit 1
}
if grep -q WRONG "$OUT/summary.txt"; then
  echo "error: the measured run reported WRONG" >&2
  exit 1
fi

# Each sabotage turns on one deliberate fault and has to come back with the
# WRONG line of the phase it broke. A non-zero exit alone is not proof: a run
# that dies for an unrelated reason also exits non-zero, and then the check the
# step exists to prove could not fail stays unproven.
sabotage() {
  local what="$1" arg="$2" phase="$3" out="$4"
  echo "==> sabotage: $what"
  if ./bandwidth triad.spv triad "$SIZE_MIB" "$arg" >"$out" 2>&1; then
    echo "error: $what passed; the check cannot fail" >&2
    exit 1
  fi
  if ! grep -q "$phase.*WRONG" "$out"; then
    echo "error: $what exited non-zero but printed no $phase WRONG line" >&2
    cat "$out" >&2
    exit 1
  fi
  echo "    the check failed as it should: $(grep "$phase" "$out")"
}

sabotage "tell the triad only half the length" short-n "triad:" \
  "$OUT/sabotage.txt"
sabotage "skip the device copy" skip-copy "copy:" "$OUT/copy_sabotage.txt"
sabotage "skip the readback" skip-readback "copy:" "$OUT/readback_sabotage.txt"
sabotage "skip the upload" skip-upload "host to device:" \
  "$OUT/upload_sabotage.txt"
sabotage "skip the download" skip-download "device to host:" \
  "$OUT/download_sabotage.txt"

echo "==> done"
