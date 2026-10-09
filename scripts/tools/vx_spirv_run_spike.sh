#!/usr/bin/env bash
#===- vx_spirv_run_spike.sh - run a Vx-compiled kernel on the Intel GPU ---===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Compile a Vx program whose topology declares `arch: spirv64`, take the SPIR-V
# image out of its dispatch payload, load that image through Level Zero, launch
# it with the flattened argument list in ABI order, and check the numbers
# against the same computation on the CPU.
#
#   ./scripts/tools/vx_spirv_run_spike.sh
#
# The program is tests/backend/pass/spirv_device_image_loop_kernel.vx. Its body
# is four rank-2 operands (`o[i][j] = a[i][j] * b[i][j] + c[i][j]`, 8 rows of
# 4) and a loop nest, which is the shape a real placed kernel has. Its payload
# carries the image in the length-prefixed section, marked `format=spirv`.
#
# What a passing run establishes, on real hardware:
#
#   - a compiler-emitted SPIR-V image loads through `zeModuleCreate`;
#   - the image's 28 arguments (four tensors of two pointers, an offset, two
#     sizes and two strides) are exactly what a launch with the flattened ABI
#     supplies;
#   - the values the kernel computes agree with the host's;
#   - and the check can fail: the same launch with one argument changed, and
#     with one byte of the image flipped, both come back wrong.
#
# Memory: four tensors of 128 bytes. The card is shared and this machine goes
# unstable above 12 GiB of VRAM in use; the launcher prints its footprint.
#
#===----------------------------------------------------------------------===#

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${TMPDIR:-/tmp}/vx-spirv-run-spike"
FIXTURE="$ROOT/tests/backend/pass/spirv_device_image_loop_kernel.vx"
KERNEL=vx_npu_kernel_0
EXPECTED_ARGS=28
mkdir -p "$OUT"

# config.local puts the LLVM tools this branch is pinned to on PATH.
if [ -f "$ROOT/config.local" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/config.local"
fi

for tool in clang spirv-val spirv-dis; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool not on PATH (source config.local)" >&2
    exit 1
  }
done

VXC="${VXC:-$ROOT/target/release/vxc}"
if [ ! -x "$VXC" ]; then
  VXC="$(command -v vxc || true)"
fi
[ -n "$VXC" ] && [ -x "$VXC" ] || {
  echo "error: no vxc; build it with 'cargo build --release -j 6'" >&2
  exit 1
}
[ -d /usr/include/level_zero ] || {
  echo "error: no /usr/include/level_zero (install level-zero-headers)" >&2
  exit 1
}

echo "==> compiling the Vx program"
"$VXC" "$FIXTURE" --emit-llvm >"$OUT/loop.ll"

echo "==> taking the image out of the dispatch payload"
python3 "$HERE/vx_spirv_extract.py" "$OUT/loop.ll" "$KERNEL" "$OUT/loop.spv"

echo "==> validating the image"
spirv-val "$OUT/loop.spv"
echo "    spirv-val accepts $OUT/loop.spv"

echo "==> checking the launch's argument count"
spirv-dis "$OUT/loop.spv" -o "$OUT/loop.spvasm"
PARAMS="$(awk '/OpFunctionParameter/{n++} END{print n+0}' "$OUT/loop.spvasm")"
echo "    spirv-dis reports $PARAMS OpFunctionParameter (vx_launch_param_width: $EXPECTED_ARGS)"
if [ "$PARAMS" != "$EXPECTED_ARGS" ]; then
  echo "error: the image takes $PARAMS arguments, the ABI packs $EXPECTED_ARGS" >&2
  exit 1
fi

echo "==> building the launcher"
"${CXX:-g++}" -std=c++17 -O2 -o "$OUT/spike" "$HERE/vx_spirv_run_spike.cpp" \
  -lze_loader

echo "==> launching the Vx kernel"
cd "$OUT"
./spike loop.spv "$KERNEL"

echo "==> sabotage 1: change one argument (the first tensor's offset)"
if ./spike loop.spv "$KERNEL" shift-offset; then
  echo "error: the check passed with a changed argument; it cannot fail" >&2
  exit 1
else
  echo "    the check failed as it should: the offset really reaches the kernel"
fi

echo "==> sabotage 2: corrupt one byte of the image (the row-stride constant)"
if ./spike loop.spv "$KERNEL" corrupt-image; then
  echo "error: the check passed with a corrupted image; it cannot fail" >&2
  exit 1
else
  echo "    the check failed as it should: the image's bytes really run"
fi

echo "==> done"
