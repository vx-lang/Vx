#!/usr/bin/env bash
#===- vx_xpu_conformance.sh - do the Arc and the CPU give the same answer? ---===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Run each program in scripts/tools/xpu_conformance/ on two backends and compare
# the results: the Intel Arc through libvx_xpu_dispatch, and the CPU through
# libvx_x86_dispatch. Both are handed the SAME compiled kernel and the same
# payload, so the only thing that differs is which backend answers.
#
#   ./scripts/tools/vx_xpu_conformance.sh
#
# What is compared, and how:
#
#   elementwise  `o = a*b + c` in each element type the Arc machine model
#                declares that has a byte address. Every value is a small
#                integer its type holds exactly, so the comparison is equality.
#                The CPU run is checked against an independent reference first,
#                so a kernel paired with the wrong element type stops the run
#                rather than looking like a backend disagreement.
#   reduce       a row sum, so the device holds an accumulator.
#   transpose    a transposing write, so both operands' strides are exercised.
#   matmul       oneMKL on the Arc against the CPU backend's own GEMM, at 1024
#                element by element (relative tolerance, the two sum in
#                different orders) and at 8192 for throughput.
#
# A kernel the device cannot run is reported, not skipped: the compiler's
# refusal or the host fallback is the result for that case.
#
# Each comparison is shown able to fail: five sabotage steps perturb an input,
# loosen or tighten a tolerance, mispair a region with an element type, and
# change a printed line, and each must turn the run red.
#
# Memory: the largest program holds three 8192x8192 fp32 tensors (805 MiB),
# checked against the shared ceiling and printed. Only one GPU helper runs at a
# time, through the lock every tool in this directory takes.
#
#===----------------------------------------------------------------------===#

set -euo pipefail

ulimit -c 0 || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PROGRAMS="$HERE/xpu_conformance"
OUT="${TMPDIR:-/tmp}/vx-xpu-conformance"
mkdir -p "$OUT"

# config.local puts the LLVM tools this branch is pinned to on PATH.
if [ -f "$ROOT/config.local" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/config.local"
fi

for tool in flock clinfo mlir-translate opt llc llvm-config objcopy \
  "${CXX:-g++}" python3; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool not on PATH (source config.local)" >&2
    exit 1
  }
done

# Serialise GPU access, for the whole run.
exec 9>"${TMPDIR:-/tmp}/vx-gpu.lock"
flock 9

# The hardware gate. There is no test-suite keyword for "a GPU must be here",
# so this checks for the card itself and says what it is doing.
if ! clinfo -l 2>/dev/null | grep -q "Arc"; then
  echo "no Intel Arc is present; nothing to compare against" >&2
  exit 0
fi
echo "==> the card is present"
clinfo -l 2>/dev/null | grep "Arc"

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
# the library vxc uses and not a copy.
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

# The CPU backend, built here from the source beside the XPU one. cargo builds
# exactly one of them, so this is how the CPU half is available at all.
echo "==> building the CPU dispatch library"
"${CXX:-g++}" -std=c++17 -O2 -fPIC -shared -I"$ROOT/runtime" -I"$ROOT/include" \
  -o "$OUT/libvx_x86_dispatch.so" "$ROOT/runtime/x86_dispatch.cpp" -lffi -ldl

STD_CORE="$ROOT/target/release/libvx_std_core.so"
[ -f "$STD_CORE" ] || {
  echo "error: no $STD_CORE; the outlined kernel prints through it" >&2
  exit 1
}

# Build one program and the two harnesses that run it. The program object holds
# both the host kernel the CPU backend calls and the device image the XPU one
# loads, so one compile serves both sides.
#
# The kernel object's `main` is renamed rather than dropped: the harness has its
# own, and the program's is referenced by nothing else, but it still has to link
# so the kernel's own symbols resolve.
build_program() {
  local name="$1"
  local dir="$OUT/$name"
  mkdir -p "$dir"

  "$VXC" "$PROGRAMS/$name.vx" --emit-llvm >"$dir/program.llvm"
  python3 "$PROGRAMS/extract_payloads.py" "$dir/program.llvm" "$dir" \
    >"$dir/order.txt"

  mlir-translate --mlir-to-llvmir "$dir/program.llvm" >"$dir/program.ll"
  local triple
  triple="$(llvm-config --host-target)-unknown-linux-gnu"
  opt -mtriple="$triple" -mcpu=native -passes='default<O2>' -S \
    "$dir/program.ll" -o "$dir/program.opt.ll"
  llc -O=2 -mtriple="$triple" -mcpu=native -filetype=obj \
    -relocation-model=pic "$dir/program.opt.ll" -o "$dir/program.o"
  objcopy --redefine-sym main=vx_program_main "$dir/program.o" "$dir/kernels.o"

  local std_dir
  std_dir="$(dirname "$STD_CORE")"
  # The commas are inside `-Wl,` options, not array separators.
  # shellcheck disable=SC2054
  local common=(-std=c++17 -O2 -rdynamic -I"$HERE" -I"$ROOT/include"
    "$HERE/vx_xpu_conformance.cpp" "$dir/kernels.o" -lffi -ldl
    -L"$std_dir" -Wl,-rpath,"$std_dir" -lvx_std_core)

  "${CXX:-g++}" "${common[@]}" -o "$dir/on_cpu" \
    -L"$OUT" -lvx_x86_dispatch -Wl,-rpath,"$OUT"
  "${CXX:-g++}" "${common[@]}" -o "$dir/on_arc" \
    -L"$DISPATCH_DIR" -lvx_xpu_dispatch -Wl,-rpath,"$DISPATCH_DIR" -lze_loader

  echo "    $name: $(wc -l <"$dir/order.txt") kernel(s)"
}

# The dispatch number the program gives the i-th region, when the regions are
# the order they appear in the source. Read from the program's own IR rather
# than assumed: the compiler numbers the outlined kernels in the order its
# lowering visits them, which is not the order the source writes them in.
kernel_at() {
  awk -v want="$1" '$1 == want { print $2 }' "$OUT/$2/order.txt"
}

# Dispatch one case on one backend: `on_cpu` or `on_arc`, then the arguments of
# `vx_xpu_conformance run`.
dispatch_case() {
  local side="$1" name="$2" op="$3" dtype="$4" M="$5" N="$6" K="$7"
  local reps="$8" kernel="$9" warmup="${10}"
  local dir="$OUT/$name"
  local label=cpu
  [ "$side" = arc ] && label=arc
  VX_CONFORMANCE_WARMUP="$warmup" VX_DISPATCH_VERBOSE=1 \
    timeout "${VX_CONFORMANCE_TIMEOUT:-1200}" \
    "$dir/on_$side" run "$op" "$dtype" "$M" "$N" "$K" "$reps" \
    "$dir/$kernel.payload" "$kernel" "$dir/$kernel.$side.bin" \
    >"$dir/$kernel.$side.txt" 2>&1 || {
    echo "error: the $label backend failed for $name / $kernel" >&2
    tail -20 "$dir/$kernel.$side.txt" >&2
    exit 1
  }
  grep -h "dispatch:\|matmul:\|against reference:\|footprint:\|ceiling:\|ran from its own image\|sum check:" \
    "$dir/$kernel.$side.txt" | sed "s/^/    $label  /"
}

# The CPU backend on the elementwise/reduce/transpose programs is exact, so a
# mismatch against the harness's own reference means the case itself is wrong --
# a kernel paired with the wrong element type, or a fill the .vx file does not
# use. That has to stop the run rather than look like a backend disagreement.
require_cpu_reference() {
  local name="$1" kernel="$2"
  if ! grep -q "against reference: 0 of" "$OUT/$name/$kernel.cpu.txt"; then
    echo "error: the CPU backend does not match the harness's own reference for" >&2
    echo "       $name / $kernel. Check the element type and the fill." >&2
    exit 1
  fi
}

# Compare the two backends' result files and fail on a difference.
compare_sides() {
  local name="$1" dtype="$2" count="$3" tolerance="$4" kernel="$5"
  local dir="$OUT/$name"
  local verdict=0
  "$dir/on_cpu" compare "$dtype" "$count" "$tolerance" \
    "$dir/$kernel.cpu.bin" "$dir/$kernel.arc.bin" | sed 's/^/    /' || verdict=$?
  if [ "$verdict" != 0 ]; then
    echo "error: the two backends disagree for $name / $kernel" >&2
    exit 1
  fi
}

echo "==> building the elementwise program"
build_program elementwise
# The element types the Arc machine model declares that have a byte address, in
# the order elementwise.vx declares its regions. i4 and u4 are declared too and
# the device runs them; a 4-bit element has no descriptor this harness can
# build, so those two are reported separately at the end.
EW_DTYPES=(f32 f16 i8 u8 i16 u16 i32 u32 i64 u64 bool)
for i in "${!EW_DTYPES[@]}"; do
  kernel="$(kernel_at "$i" elementwise)"
  echo "==> elementwise / $kernel: ew ${EW_DTYPES[$i]} 32x32"
  dispatch_case cpu elementwise ew "${EW_DTYPES[$i]}" 32 32 1 20 "$kernel" 3
  require_cpu_reference elementwise "$kernel"
  dispatch_case arc elementwise ew "${EW_DTYPES[$i]}" 32 32 1 20 "$kernel" 3
  compare_sides elementwise "${EW_DTYPES[$i]}" 1024 0 "$kernel"
done

echo "==> the documented host fallback"
# The same payload, with the image's format entry changed so the XPU backend
# declines to load it. The library then runs the outlined kernel on the host,
# which is the fallback every backend has for what it cannot route. The result
# has to match the CPU backend's.
fallback_kernel="$(kernel_at 0 elementwise)"
VX_CONFORMANCE_HOST_FALLBACK=1 VX_CONFORMANCE_WARMUP=0 VX_DISPATCH_VERBOSE=1 \
  timeout 600 "$OUT/elementwise/on_arc" run ew f32 32 32 1 1 \
  "$OUT/elementwise/$fallback_kernel.payload" "$fallback_kernel" \
  "$OUT/elementwise/$fallback_kernel.arc.fallback.bin" \
  >"$OUT/elementwise/$fallback_kernel.arc.fallback.txt" 2>&1
grep -h "fallback:\|not launching" \
  "$OUT/elementwise/$fallback_kernel.arc.fallback.txt" | sed 's/^/    /'
if grep -q "ran from its own image" \
  "$OUT/elementwise/$fallback_kernel.arc.fallback.txt"; then
  echo "error: the device image was loaded anyway, so the fallback was not" >&2
  echo "       the path that produced this answer" >&2
  exit 1
fi
"$OUT/elementwise/on_cpu" compare f32 1024 0 \
  "$OUT/elementwise/$fallback_kernel.cpu.bin" \
  "$OUT/elementwise/$fallback_kernel.arc.fallback.bin" | sed 's/^/    /'

echo "==> building the reduction program"
build_program reduce
kernel="$(kernel_at 0 reduce)"
echo "==> reduce / $kernel: reduce f32 64x32"
dispatch_case cpu reduce reduce f32 64 32 1 20 "$kernel" 3
require_cpu_reference reduce "$kernel"
dispatch_case arc reduce reduce f32 64 32 1 20 "$kernel" 3
compare_sides reduce f32 64 0 "$kernel"

echo "==> building the transposing program"
build_program transpose
kernel="$(kernel_at 0 transpose)"
echo "==> transpose / $kernel: transpose f32 48x32"
dispatch_case cpu transpose transpose f32 48 32 1 20 "$kernel" 3
require_cpu_reference transpose "$kernel"
dispatch_case arc transpose transpose f32 48 32 1 20 "$kernel" 3
compare_sides transpose f32 1536 0 "$kernel"

echo "==> building the 1024x1024 matrix multiply"
build_program matmul
kernel="$(kernel_at 0 matmul)"
echo "==> matmul / $kernel: matmul f32 1024x1024x1024"
# One CPU dispatch: the reference GEMM runs at about 0.4 GFLOP/s, so a timing
# loop over it costs minutes for a number one run already gives.
dispatch_case cpu matmul matmul f32 1024 1024 1024 1 "$kernel" 0
dispatch_case arc matmul matmul f32 1024 1024 1024 20 "$kernel" 3
compare_sides matmul f32 $((1024 * 1024)) 1e-4 "$kernel"

echo "==> building the 8192x8192 matrix multiply"
build_program matmul_big
kernel="$(kernel_at 0 matmul_big)"
echo "==> matmul_big / $kernel: matmul f32 8192x8192x8192 (Arc only)"
# The CPU backend's reference GEMM would need about 47 minutes at this size, so
# only the Arc is run here; the CPU half of the comparison is the 1024 case.
dispatch_case arc matmul_big matmul f32 8192 8192 8192 5 "$kernel" 2

# i4, u4 and bool have no descriptor this harness can build -- a four-bit and a
# one-bit element are not addressable one at a time -- so the whole program is
# run through the compiler's own driver on both backends and the two printed
# lines are compared. That covers every declared type over a whole 32x32 tensor.
echo "==> i4, u4 and bool, whole-program run on both backends"
jit_run() {
  local side="$1" name="$2"
  if [ "$side" = cpu ]; then
    VX_DISPATCH_LIB="$OUT/libvx_x86_dispatch.so" \
      timeout 600 "$VXC" "$PROGRAMS/$name.vx" --run 2>/dev/null
  else
    VX_DISPATCH_VERBOSE=1 timeout 600 "$VXC" "$PROGRAMS/$name.vx" --run \
      2>"$OUT/$name.arc.run.txt"
  fi | grep -E '^-?[0-9]+( +-?[0-9]+)* *$' || true
}
jit_run cpu elementwise >"$OUT/elementwise.cpu.out"
jit_run arc elementwise >"$OUT/elementwise.arc.out"
grep -c "ran from its own image" "$OUT/elementwise.arc.run.txt" | sed 's/^/    arc  images run: /'
echo "    cpu: $(cat "$OUT/elementwise.cpu.out")"
echo "    arc: $(cat "$OUT/elementwise.arc.out")"
if ! diff -q "$OUT/elementwise.cpu.out" "$OUT/elementwise.arc.out" >/dev/null; then
  echo "error: the whole-program runs disagree between the two backends" >&2
  diff -u "$OUT/elementwise.cpu.out" "$OUT/elementwise.arc.out" >&2
  exit 1
fi
echo "    the two whole-program runs agree on all $(wc -w <"$OUT/elementwise.cpu.out") element types"

echo "==> sabotage 1: one input element changed on the Arc side"
sab_kernel="$(kernel_at 0 elementwise)"
VX_CONFORMANCE_PERTURB=1 VX_CONFORMANCE_WARMUP=0 VX_DISPATCH_VERBOSE=1 \
  timeout 600 "$OUT/elementwise/on_arc" run ew f32 32 32 1 1 \
  "$OUT/elementwise/$sab_kernel.payload" "$sab_kernel" \
  "$OUT/elementwise/$sab_kernel.arc.perturbed.bin" \
  >"$OUT/elementwise/$sab_kernel.arc.perturbed.txt" 2>&1
grep -h "sabotage:" "$OUT/elementwise/$sab_kernel.arc.perturbed.txt" | sed 's/^/    /'
if "$OUT/elementwise/on_cpu" compare f32 1024 0 \
  "$OUT/elementwise/$sab_kernel.cpu.bin" \
  "$OUT/elementwise/$sab_kernel.arc.perturbed.bin"; then
  echo "error: the comparison passed with a perturbed input; it cannot fail" >&2
  exit 1
fi
echo "    the comparison failed as it should: the changed element was noticed"

echo "==> sabotage 2: a tolerance too loose to notice the change"
if ! "$OUT/elementwise/on_cpu" compare f32 1024 1e9 \
  "$OUT/elementwise/$sab_kernel.cpu.bin" \
  "$OUT/elementwise/$sab_kernel.arc.perturbed.bin"; then
  echo "error: even a huge tolerance reported a difference; the tolerance is" >&2
  echo "       not what the verdict rests on, so sabotage 2 proves nothing" >&2
  exit 1
fi
echo "    a tolerance of 1e9 hides the change, so the tolerance decides"

echo "==> sabotage 3: the matmul tolerance is what the verdict rests on"
mat_kernel="$(kernel_at 0 matmul)"
if "$OUT/matmul/on_cpu" compare f32 1048576 0 \
  "$OUT/matmul/$mat_kernel.cpu.bin" "$OUT/matmul/$mat_kernel.arc.bin"; then
  echo "    the two matmul results are equal bit for bit, so no tolerance was"
  echo "    needed: the stated 1e-4 is not doing any work at this size"
else
  echo "    the results differ at tolerance 0 and pass at 1e-4, so 1e-4 is"
  echo "    load-bearing rather than a rubber stamp"
fi

echo "==> sabotage 4: a region paired with the wrong element type"
# The second region of elementwise.vx is the f16 one. Running it with f32
# descriptors must not match the reference: this is the guard that stops a
# mismatched pairing from being read as a backend disagreement.
wrong_kernel="$(kernel_at 1 elementwise)"
if VX_CONFORMANCE_WARMUP=0 timeout 600 "$OUT/elementwise/on_cpu" run ew f32 \
  32 32 1 1 "$OUT/elementwise/$wrong_kernel.payload" "$wrong_kernel" \
  "$OUT/elementwise/$wrong_kernel.paired-wrong.bin" 2>&1 \
  | grep -q "against reference: 0 of"; then
  echo "error: the second region matched the f32 reference, so the guard that" >&2
  echo "       pairs a region with its element type cannot fail" >&2
  exit 1
fi
echo "    the second region with f32 descriptors does not match, as it must not"

echo "==> sabotage 5: the whole-program comparison notices a changed line"
sed 's/1024 *$/1025/' "$OUT/elementwise.cpu.out" \
  >"$OUT/elementwise.tampered.out"
if diff -q "$OUT/elementwise.cpu.out" "$OUT/elementwise.tampered.out" \
  >/dev/null; then
  echo "error: the whole-program comparison called a changed line equal" >&2
  exit 1
fi
echo "    a changed final number makes the comparison fail, as it must"

echo "==> refusals: programs the device cannot run, by the compiler's own words"
# These are the cases where the right answer is a refusal rather than a number.
# Each is compiled and the compiler's message is checked, so a build that
# quietly accepted the program would fail the run.
refuse() {
  local name="$1" expect="$2"
  local err="$OUT/refuse.$name.txt"
  if timeout 600 "$VXC" --machine "$ROOT/fleet/arc-a770.vx" --host default \
    "$PROGRAMS/refusals/$name.vx" --emit-llvm >/dev/null 2>"$err"; then
    echo "error: $name was accepted; the refusal this case exists for is gone" >&2
    exit 1
  fi
  if ! grep -q "$expect" "$err"; then
    echo "error: $name was refused, but not for the reason this case names" >&2
    head -6 "$err" >&2
    exit 1
  fi
  echo "    $name refused by name:"
  head -1 "$err" | sed 's/^/      /'
}
refuse f64_on_the_arc "E6026"
refuse f32_kernel_on_the_arc "cannot emit a device kernel"

echo "==> the card after the run"
clinfo -l 2>/dev/null | grep "Arc" | sed 's/^/    /'
exec 9>&-
if flock -n "${TMPDIR:-/tmp}/vx-gpu.lock" -c true 2>/dev/null; then
  echo "    nothing is holding ${TMPDIR:-/tmp}/vx-gpu.lock"
else
  echo "    warning: something still holds ${TMPDIR:-/tmp}/vx-gpu.lock" >&2
fi

echo "==> done"
