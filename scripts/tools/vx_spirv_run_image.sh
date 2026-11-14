#!/usr/bin/env bash
#===- vx_spirv_run_image.sh - run a Vx kernel through the dispatch library ---===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Compile a Vx program whose topology declares `arch: spirv64`, take the
# dispatch payload out of the compiled program, and run the kernel through the
# dispatch library's own entry points -- `vx_plugin_alloc_and_transfer`,
# `vx_plugin_dispatch_async` and `vx_plugin_transfer_device_to_host` -- which is
# the ABI a compiled Vx program calls. The numbers are then checked against the
# same computation on the CPU.
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
#   - the library picks the Intel GPU and accepts the `format=spirv` image;
#   - a compiler-emitted SPIR-V image loads through the library's own ABI, and
#     the image's 28 arguments are what the library's flattened ABI supplies;
#   - the values the kernel computes agree with the host's;
#   - the same launch through Level Zero alone (the control) agrees too;
#   - and the checks can fail: a corrupted image byte, a wrong parameter count,
#     a changed `format=` string and a changed argument each come back refused
#     or wrong.
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

# The sabotage steps below make the library refuse, and it refuses by aborting.
# No core file is wanted for a planned refusal.
ulimit -c 0 || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${TMPDIR:-/tmp}/vx-spirv-run-image"
FIXTURE="$ROOT/tests/backend/pass/spirv_device_image_loop_kernel.vx"
KERNEL=vx_npu_kernel_0
EXPECTED_ARGS=28
mkdir -p "$OUT"

# config.local puts the LLVM tools this branch is pinned to on PATH.
if [ -f "$ROOT/config.local" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/config.local"
fi

for tool in clang spirv-val spirv-dis flock; do
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

# The dispatch library cargo built when oneAPI was present. It is the same
# artifact the compiler links a placed program against, so the run exercises
# the library vxc uses and not a copy. The newest one is taken because cargo
# keeps a directory per build fingerprint.
DISPATCH_LIB="$(find "$ROOT/target/release/build" -name 'libvx_xpu_dispatch.so' \
  -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [ -z "$DISPATCH_LIB" ] || [ ! -f "$DISPATCH_LIB" ]; then
  echo "error: no libvx_xpu_dispatch.so in target/release; build it with" >&2
  echo "       'cargo build --release -j 6' on a machine with oneAPI installed" >&2
  exit 1
fi
if [ "$ROOT/runtime/xpu_dispatch.cpp" -nt "$DISPATCH_LIB" ]; then
  echo "error: $DISPATCH_LIB is older than runtime/xpu_dispatch.cpp; rebuild" >&2
  echo "       with 'cargo build --release -j 6' so the run tests the source" >&2
  exit 1
fi
DISPATCH_DIR="$(dirname "$DISPATCH_LIB")"

echo "==> compiling the Vx program"
"$VXC" "$FIXTURE" --emit-llvm >"$OUT/loop.ll"

echo "==> taking the payload out of the compiled program"
python3 "$HERE/vx_spirv_extract.py" "$OUT/loop.ll" "$KERNEL" "$OUT/loop.spv" \
  "$OUT/loop.payload"

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

echo "==> building the launcher against $DISPATCH_LIB"
"${CXX:-g++}" -std=c++17 -O2 -I"$ROOT/include" -o "$OUT/run_image" \
  "$HERE/vx_spirv_run_image.cpp" \
  -L"$DISPATCH_DIR" -lvx_xpu_dispatch -Wl,-rpath,"$DISPATCH_DIR" -lze_loader

cd "$OUT"

echo "==> control: launching through Level Zero alone"
./run_image loop.payload "$KERNEL" direct

echo "==> launching through the dispatch library"
VX_DISPATCH_VERBOSE=1 ./run_image loop.payload "$KERNEL"

echo "==> sabotage 1: one byte of the image corrupted"
if VX_DISPATCH_VERBOSE=1 ./run_image loop.payload "$KERNEL" corrupt-image; then
  echo "error: the check passed with a corrupted image; it cannot fail" >&2
  exit 1
else
  echo "    the check failed as it should: the image's bytes really run"
fi

echo "==> sabotage 2: a wrong parameter count"
if VX_DISPATCH_VERBOSE=1 ./run_image loop.payload "$KERNEL" wrong-param-count; then
  echo "error: the check passed with too few arguments; it cannot fail" >&2
  exit 1
else
  echo "    the library refused as it should: see the FATAL line above"
fi

echo "==> sabotage 3: a wrong format= string"
if VX_DISPATCH_VERBOSE=1 ./run_image loop.payload "$KERNEL" bad-format; then
  echo "error: the check passed with a format the backend cannot load" >&2
  exit 1
else
  echo "    the library refused as it should: see the line above"
fi

echo "==> sabotage 4: a changed argument (the first tensor's base pointer)"
if VX_DISPATCH_VERBOSE=1 ./run_image loop.payload "$KERNEL" shift-base; then
  echo "error: the check passed with a changed argument; it cannot fail" >&2
  exit 1
else
  echo "    the check failed as it should: the pointer really reaches the kernel"
fi

echo "==> done"
