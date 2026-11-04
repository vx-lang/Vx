#!/usr/bin/env bash
#===- vx_spirv_run_image.sh - run a Vx-compiled kernel on the Intel GPU ---===#
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
#   ./scripts/tools/vx_spirv_run_image.sh
#
# The program is tests/backend/pass/spirv_device_image_loop_kernel.vx. Its body
# is four rank-2 operands (`o[i][j] = a[i][j] * b[i][j] + c[i][j]`, 8 rows of
# 4) and a loop nest, which is the shape a real placed kernel has. Its payload
# carries the image in the length-prefixed section, marked `format=spirv`.
#
# What a passing run establishes, on real hardware:
#
#   - a compiler-emitted SPIR-V image loads through `zeModuleCreate`
#   - the image's 28 arguments (four tensors of two pointers, an offset, two
#     sizes and two strides) are exactly what a launch with the flattened ABI
#     supplies.
#   - the values the kernel computes agree with the host's.
#   - and the check can fail: the same launch with one argument changed, and
#     with one byte of the image flipped, both come back wrong.
#
# Memory: four tensors of 128 bytes. The launcher prints its footprint and its
# ceiling and refuses to allocate past `VX_VRAM_CEILING` -- 2 GiB by default --
# the same rule the other helper in this directory uses.
#
# Only one of these runs may touch the card at a time, so the whole script holds
# a lock that every GPU helper in this directory takes.
#
#===----------------------------------------------------------------------===#

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${TMPDIR:-/tmp}/vx-spirv-run-image"
FIXTURE="$ROOT/tests/backend/pass/spirv_device_image_loop_kernel.vx"
KERNEL=vx_npu_kernel_0
EXPECTED_ARGS=28
mkdir -p "$OUT"

# config.local puts the LLVM tools this branch builds against on PATH.
if [ -f "$ROOT/config.local" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/config.local"
fi

for tool in clang++ spirv-val spirv-dis flock; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool not on PATH (source config.local)" >&2
    exit 1
  }
done

# Serialise GPU access, for the whole run including the checks below.
# The lock is held on file descriptor 9 until the script exits, so a second
# helper waits rather than allocating beside this one.
exec 9>"${TMPDIR:-/tmp}/vx-gpu.lock"
flock 9

VXC="${VXC:-$ROOT/target/release/vxc}"
if [ ! -x "$VXC" ]; then
  VXC="$(command -v vxc || true)"
fi
[ -n "$VXC" ] && [ -x "$VXC" ] || {
  echo "error: there is no vxc, so build it with 'cargo build --release -j 6'" >&2
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
# The parameters of the entry point, not of the module: a kernel that calls a
# helper has a second function, and counting its parameters too reports an
# argument count the ABI never packs. The entry point's id comes from
# `OpEntryPoint`, and the count stops at that function's `OpFunctionEnd`.
PARAMS="$(awk -v want="\"$KERNEL\"" '
  { op = ($2 == "=") ? $3 : $1 }
  op == "OpEntryPoint" {
    for (i = 3; i < NF; i++)
      if ($i == want) { id = $(i - 1); break }
    next
  }
  op == "OpFunction" && $1 == id { inside = 1; next }
  inside && op == "OpFunctionParameter" { n++ }
  inside && op == "OpFunctionEnd" { inside = 0 }
  END { print n + 0 }
' "$OUT/loop.spvasm")"
echo "    spirv-dis reports $PARAMS OpFunctionParameter (vx_launch_param_width: $EXPECTED_ARGS)"
if [ "$PARAMS" != "$EXPECTED_ARGS" ]; then
  echo "error: the image takes $PARAMS arguments, the ABI packs $EXPECTED_ARGS" >&2
  exit 1
fi

echo "==> building the launcher"
"${CXX:-clang++}" -std=c++17 -O2 -o "$OUT/run_image" "$HERE/vx_spirv_run_image.cpp" \
  -lze_loader

echo "==> launching the Vx kernel"
cd "$OUT"
./run_image loop.spv "$KERNEL"

echo "==> the check can fail: change one argument (the first tensor's offset)"
if ./run_image loop.spv "$KERNEL" shift-offset; then
  echo "error: the check passed with a changed argument, so it cannot fail" >&2
  exit 1
else
  echo "    the check failed as it should: the offset really reaches the kernel"
fi

echo "==> the check can fail: corrupt one byte of the image (the row-stride constant)"
if ./run_image loop.spv "$KERNEL" corrupt-image; then
  echo "error: the check passed with a corrupted image, so it cannot fail" >&2
  exit 1
else
  echo "    the check failed as it should: the image's bytes really run"
fi

echo "==> done"
