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
# Two more cases check the library's own state rather than one kernel's
# arithmetic, because neither fault shows up as a number that looks wrong:
#
#   same name    reduce and transpose each call their one kernel
#                `vx_npu_kernel_0`, and the two images are different kernels
#                with the same parameter count. Both are dispatched in one
#                process, A then B then A, and each answer has to be its own.
#                A cache keyed by the name alone serves the first library's
#                kernel to the second and reports 1520 of 1536 wrong.
#   host memory  every operand is left in this program's own buffer, which the
#                device cannot address. The backend has to refuse the launch
#                ("argument 0 is at an address the device cannot reach") and run
#                the kernel's host copy instead, with the CPU backend's answer.
#   machine      a program that declares nothing and uses the machine file's
#   file         own `Memory::HBM` and `Topology::Device`. The hash must be
#                aliased to the GPU band and the region must run from its image;
#   declared     a tensor transferred into a declared space that is not the
#   space        topology's `memory:`. The compiler sends no alias for it and
#                the space says the host cannot read it, so host memory is not
#                an answer. The backend has to refuse and name the space id;
#                before it did, the region quietly ran on the host and the run
#                printed numbers.
#
# A kernel the device cannot run is reported, not skipped: the compiler's
# refusal or the host fallback is the result for that case.
#
# Each comparison is shown able to fail: seven sabotage steps perturb an input,
# loosen or tighten a tolerance, mispair a region with an element type, key the
# kernel cache by the name alone, drop the address check before a launch, and
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

# Wrap a payload in a shared object that exports it.
#
# A program's payload lives in the library it was compiled into, and the XPU
# backend names that library in its kernel cache key because two libraries can
# use one kernel name. A payload in a heap buffer has no library, so this is how
# the same-name check hands the backend two payloads that differ in that one
# respect.
payload_library() {
  local payload="$1" out="$2"
  python3 - "$payload" "$out.c" <<'PY'
import sys

data = open(sys.argv[1], "rb").read()
with open(sys.argv[2], "w") as f:
    f.write("#include <stddef.h>\n")
    f.write("const unsigned char vx_test_payload[] = {")
    f.write(", ".join(str(b) for b in data) or "0")
    f.write("};\n")
    f.write("const size_t vx_test_payload_size = %d;\n" % len(data))
PY
  "${CC:-cc}" -shared -fPIC -o "$out" "$out.c"
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

echo "==> two libraries that both call their kernel vx_npu_kernel_0"
# reduce.vx and transpose.vx each hold one placed region, so each program names
# its kernel `vx_npu_kernel_0` and the two images are different kernels with the
# same parameter count. Dispatched in one process, a cache keyed by the name
# alone runs the first library's kernel for the second one's call, and the count
# check before the launch agrees because the counts are equal.
#
# The cache key names the library that holds a payload, so each payload has to
# be in a library for this to mean anything.
reduce_kernel="$(kernel_at 0 reduce)"
transpose_kernel="$(kernel_at 0 transpose)"
if [ "$reduce_kernel" != "$transpose_kernel" ]; then
  echo "error: reduce is $reduce_kernel and transpose is $transpose_kernel," >&2
  echo "       and this check needs them to share a name." >&2
  exit 1
fi
payload_library "$OUT/reduce/$reduce_kernel.payload" "$OUT/reduce/libpayload.so"
payload_library "$OUT/transpose/$transpose_kernel.payload" \
  "$OUT/transpose/libpayload.so"
if ! VX_DISPATCH_VERBOSE=1 timeout 600 "$OUT/reduce/on_arc" same_name f32 reduce 64 32 1 \
  "$OUT/reduce/libpayload.so" transpose 48 32 1 \
  "$OUT/transpose/libpayload.so" >"$OUT/same_name.txt" 2>&1; then
  grep -h "same name:\|  A\|  B\|result:" "$OUT/same_name.txt" | sed 's/^/    /' >&2
  echo "error: a kernel was served from the other library's cache" >&2
  exit 1
fi
grep -h "same name:\|  A:\|  B:\|result:\|footprint:" "$OUT/same_name.txt" |
  sed 's/^/    /'
if ! grep -q "result: equal (0 of 3 dispatches differ)" "$OUT/same_name.txt"; then
  echo "error: the second library's call did not run its own kernel" >&2
  exit 1
fi
# Both payloads ran from their own image, so a host copy did not stand in for
# one of them and make the comparison say nothing.
if [ "$(grep -c "ran from its own image" "$OUT/same_name.txt")" != 2 ]; then
  echo "error: one of the two payloads did not launch its own image" >&2
  grep -h "ran from its own image\|not launching" "$OUT/same_name.txt" >&2
  exit 1
fi
echo "    each library ran its own kernel, and the first one still does"

echo "==> a launch whose operands are the program's own memory"
# A memref argument can point anywhere, and the driver does not refuse a host
# address: the kernel reads memory the GPU does not know, the process dies of a
# page fault, and nothing says which launch did it. The backend checks every
# memref's data pointer before it launches, so this case has to be refused and
# has to run its host copy instead -- with the same answer as the CPU backend.
hp_kernel="$(kernel_at 0 elementwise)"
VX_CONFORMANCE_HOST_POINTER=1 VX_CONFORMANCE_WARMUP=0 VX_DISPATCH_VERBOSE=1 \
  timeout 600 "$OUT/elementwise/on_arc" run ew f32 32 32 1 1 \
  "$OUT/elementwise/$hp_kernel.payload" "$hp_kernel" \
  "$OUT/elementwise/$hp_kernel.arc.host_pointer.bin" \
  >"$OUT/elementwise/$hp_kernel.arc.host_pointer.txt" 2>&1
grep -h "host pointer:\|cannot reach\|against reference:" \
  "$OUT/elementwise/$hp_kernel.arc.host_pointer.txt" | sed 's/^/    /'
if ! grep -q "argument 0 is at an address the device cannot reach; not launching" \
  "$OUT/elementwise/$hp_kernel.arc.host_pointer.txt"; then
  echo "error: the launch at a host address was not refused. Either the check" >&2
  echo "       before a launch is gone, or it no longer names the argument." >&2
  exit 1
fi
if grep -q "ran from its own image" \
  "$OUT/elementwise/$hp_kernel.arc.host_pointer.txt"; then
  echo "error: the device image was launched at a host address anyway" >&2
  exit 1
fi
"$OUT/elementwise/on_cpu" compare f32 1024 0 \
  "$OUT/elementwise/$hp_kernel.cpu.bin" \
  "$OUT/elementwise/$hp_kernel.arc.host_pointer.bin" | sed 's/^/    /'

echo "==> eight threads dispatching at once, under ThreadSanitizer"
# The backend keeps one kernel handle and one command list for the whole
# process, so two threads that dispatch at once write the same arguments and
# append to the same list. Level Zero allows neither, and without the lock the
# result is a wrong answer or a crash rather than an error, so a plain
# assertion on it is a coin flip. ThreadSanitizer is what turns the missing lock
# into a report that names the map, so the library and this harness are built
# again with it.
#
# The build mirrors build.rs's XPU command line, which is the only other place
# these flags are written down.
tsan_dir="$OUT/tsan"
mkdir -p "$tsan_dir"
tsan_cxx="${XPU_CXX:-/opt/intel/oneapi/compiler/latest/bin/icpx}"
if [ ! -x "$tsan_cxx" ]; then
  echo "error: no $tsan_cxx, so the lock cannot be checked" >&2
  exit 1
fi
tsan_compiler_dir="$(dirname "$(dirname "$tsan_cxx")")"
tsan_mkl_dir="/opt/intel/oneapi/mkl/latest"
"$tsan_cxx" -shared -fPIC -O2 -g -fsanitize=thread \
  "$ROOT/runtime/xpu_dispatch.cpp" -lffi \
  -o "$tsan_dir/libvx_xpu_dispatch_tsan.so" -fsycl -lze_loader \
  -I"$tsan_compiler_dir/include" -I"$tsan_mkl_dir/include" -I"$ROOT/include" \
  -I"$ROOT/runtime" -L"$tsan_compiler_dir/lib" -L"$tsan_mkl_dir/lib" \
  -Wl,-rpath,"$tsan_compiler_dir/lib" -Wl,-rpath,"$tsan_mkl_dir/lib" \
  -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl \
  -lOpenCL -lpthread -lm -ldl
"${CXX:-g++}" -std=c++17 -O1 -g -fsanitize=thread -rdynamic \
  -I"$HERE" -I"$ROOT/include" "$HERE/vx_xpu_conformance.cpp" \
  "$OUT/elementwise/kernels.o" -lffi -ldl \
  -L"$(dirname "$STD_CORE")" -Wl,-rpath,"$(dirname "$STD_CORE")" \
  -lvx_std_core -L"$tsan_dir" -lvx_xpu_dispatch_tsan \
  -Wl,-rpath,"$tsan_dir" -lze_loader -o "$OUT/elementwise/on_arc_tsan"
# Two of ThreadSanitizer's reports are not about this library, and the run is
# not allowed to fail on them:
#
#   detect_deadlocks=0        the Level Zero driver takes its own locks, and
#                             one of them is read as this library's, so the
#                             reported lock order cannot happen: the second
#                             stack is inside the driver and names no frame of
#                             ours.
#   exitcode=0                the driver delivers signals, and a `malloc` this
#                             library makes afterwards is reported as
#                             "signal-unsafe call inside of a signal". That is
#                             a warning about the state a signal left behind,
#                             not a race.
#
# What is asserted is a data race or a lock order that involves this library,
# which is the shape the missing lock takes.
tsan_kernel="$(kernel_at 0 elementwise)"
tsan_log="$OUT/elementwise.threads.tsan.txt"
tsan_status=0
VX_CONFORMANCE_THREADS=8 VX_CONFORMANCE_WARMUP=0 VX_DISPATCH_VERBOSE=1 \
  TSAN_OPTIONS="detect_deadlocks=0:exitcode=0" timeout 900 \
  "$OUT/elementwise/on_arc_tsan" run ew f32 32 32 1 20 \
  "$OUT/elementwise/$tsan_kernel.payload" "$tsan_kernel" \
  "$OUT/elementwise/$tsan_kernel.threads.bin" >"$tsan_log" 2>&1 || tsan_status=$?
grep -h "threads:\|against reference:\|result:\|ran from its own image" \
  "$tsan_log" | sed 's/^/    /'
if grep -qE "WARNING: ThreadSanitizer: (data race|lock-order-inversion)" \
  "$tsan_log"; then
  grep -A 6 "WARNING: ThreadSanitizer" "$tsan_log" | head -20 >&2
  echo "error: ThreadSanitizer reported a data race, so the launch is not" >&2
  echo "       serialised. The report above names where." >&2
  exit 1
fi
if [ "$tsan_status" != 0 ]; then
  tail -20 "$tsan_log" >&2
  echo "error: the threaded run failed with status $tsan_status" >&2
  exit 1
fi
if ! grep -q "threads: 8 threads, 20 dispatches each, all returned" "$tsan_log"; then
  echo "error: the threaded run did not dispatch from eight threads" >&2
  exit 1
fi
if ! grep -q "result: equal (0 of 8 threads differ)" "$tsan_log"; then
  echo "error: a thread's answer was not the reference's" >&2
  exit 1
fi
if ! grep -q "ran from its own image" "$tsan_log"; then
  echo "error: no thread launched the image, so this says nothing about the" >&2
  echo "       lock around a launch" >&2
  exit 1
fi
echo "    no data race, and all eight threads' answers are the reference's"

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

# The other half of the declared-space case below. The space the machine file
# itself names has to reach the card: the program declares nothing and reads
# `Memory::HBM` and `Topology::Device` from fleet/arc-a770.vx through
# `--machine`. The ids are hashes, so the transfer is only device memory if
# `Topology Device { memory: Memory::HBM }` sent the pair (fnv("HBM"), 500) and
# the library maps the hash to the GPU band. The staged topology id, the
# `vx.local_devices` pair and the image launch are all named, so a spelling that
# silently lands on the host cannot pass.
echo "==> the machine file's own space reaches the card"
machine_spelling="$PROGRAMS/the_machine_files_hbm_is_the_cards_memory.vx"
machine_spelling_mlir="$OUT/machine_spelling.mlir"
if ! "$VXC" --machine "$ROOT/fleet/arc-a770.vx" --host default \
  "$machine_spelling" --emit-mlir 2>/dev/null >"$machine_spelling_mlir"; then
  echo "error: the machine-file conformance program did not compile" >&2
  exit 1
fi
machine_spelling_status=0
VX_DISPATCH_VERBOSE=1 timeout 600 "$VXC" \
  --machine "$ROOT/fleet/arc-a770.vx" --host default "$machine_spelling" --run \
  >"$OUT/machine_spelling.out" 2>"$OUT/machine_spelling.err" ||
  machine_spelling_status=$?
if [ "$machine_spelling_status" != 0 ]; then
  echo "error: the machine-file run failed" >&2
  tail -8 "$OUT/machine_spelling.err" >&2
  exit 1
fi
machine_spelling_id="$(awk '
  match($0, /staged [0-9]+ bytes on topology [0-9]+/) {
    id = substr($0, RSTART, RLENGTH)
    sub(/.*topology /, "", id)
    print id
    exit
  }' "$OUT/machine_spelling.err")"
if [ -z "$machine_spelling_id" ] || [ "$machine_spelling_id" = 500 ]; then
  echo "error: the transfer was not staged on a declared topology id;" >&2
  echo "       got '${machine_spelling_id:-none}'" >&2
  head -8 "$OUT/machine_spelling.err" >&2
  exit 1
fi
if ! grep -q "vx.local_devices = array<i32: .*$machine_spelling_id, 500" \
  "$machine_spelling_mlir"; then
  echo "error: the staged topology $machine_spelling_id has no alias to the GPU" >&2
  echo "       band 500 in vx.local_devices, so this says nothing about the" >&2
  echo "       machine file's spelling" >&2
  grep -h "vx.local_devices" "$machine_spelling_mlir" >&2
  exit 1
fi
if ! grep -q "ran from its own image" "$OUT/machine_spelling.err"; then
  echo "error: the region on the machine file's topology did not run from its" >&2
  echo "       own image" >&2
  tail -8 "$OUT/machine_spelling.err" >&2
  exit 1
fi
machine_spelling_answer="$(tail -n 1 "$OUT/machine_spelling.out")"
if [ "$machine_spelling_answer" != 4 ]; then
  echo "error: the region printed '$machine_spelling_answer', not 4" >&2
  exit 1
fi
grep -h "staged .* on topology $machine_spelling_id\|ran from its own image" \
  "$OUT/machine_spelling.err" | sed 's/^/    /'
echo "    topology $machine_spelling_id is aliased to 500 and the region ran on the card"

# A tensor transferred into a declared space that is not the topology's
# `memory:`. The compiler sends no alias for it, the runtime has only a hash,
# and the space is `managed: explicit`, so host memory cannot stand in for it.
# The backend must refuse and name the space: before it did, the run printed the
# host fallback's numbers and looked successful. 1855308478 is the id
# src/arch.rs gives `fnv("HBM")`, which is the number the message has to name.
echo "==> a transfer into a declared space this backend cannot place"
declared_space_id=1855308478
declared_stdout="$OUT/declared_space.out"
declared_stderr="$OUT/declared_space.err"
declared_status=0
VX_DISPATCH_VERBOSE=1 timeout 600 "$VXC" \
  "$PROGRAMS/a_declared_space_the_backend_cannot_place.vx" --run \
  >"$declared_stdout" 2>"$declared_stderr" || declared_status=$?
if [ "$declared_status" = 0 ]; then
  echo "error: a transfer into a declared space the backend cannot place" >&2
  echo "       answered with memory instead of refusing. The run printed:" >&2
  sed 's/^/         /' "$declared_stdout" >&2
  echo "       those are the host fallback's numbers." >&2
  exit 1
fi
grep -h "asks for memory space\|Make the space the topology's" "$declared_stderr" |
  sed 's/^/    /'
if ! grep -q "asks for memory space $declared_space_id" "$declared_stderr"; then
  echo "error: the refusal does not name the space id $declared_space_id," >&2
  echo "       so a reader cannot tell which memory was refused" >&2
  head -8 "$declared_stderr" >&2
  exit 1
fi
if ! grep -q "managed: explicit" "$declared_stderr" ||
  ! grep -q "Make the space the topology's \`memory:\`" "$declared_stderr"; then
  echo "error: the refusal does not say why host memory is wrong, or does not" >&2
  echo "       say to make the space the topology's memory" >&2
  head -8 "$declared_stderr" >&2
  exit 1
fi
echo "    refused by name, and the program printed no numbers"

# An elementwise loop on the machine file's own topology, whose memory the host
# cannot read. The compiler used to refuse it -- only a matmul got a device
# kernel -- and now emits an image for it, so the check is that it runs on the
# card and prints the same number the CPU backend prints.
echo "==> a loop region on the Arc's own topology, through the machine file"
machine_run() {
  local side="$1"
  if [ "$side" = cpu ]; then
    VX_DISPATCH_LIB="$OUT/libvx_x86_dispatch.so" \
      timeout 600 "$VXC" --machine "$ROOT/fleet/arc-a770.vx" --host default \
      "$PROGRAMS/elementwise_on_the_arc.vx" --run 2>/dev/null
  else
    VX_DISPATCH_VERBOSE=1 timeout 600 "$VXC" --machine "$ROOT/fleet/arc-a770.vx" \
      --host default "$PROGRAMS/elementwise_on_the_arc.vx" --run \
      2>"$OUT/elementwise_on_the_arc.arc.run.txt"
  fi | grep -E '^-?[0-9]+( +-?[0-9]+)* *$' || true
}
machine_run cpu >"$OUT/elementwise_on_the_arc.cpu.out"
machine_run arc >"$OUT/elementwise_on_the_arc.arc.out"
grep -h "ran from its own image" "$OUT/elementwise_on_the_arc.arc.run.txt" |
  sed 's/^/    arc  /'
if ! grep -q "ran from its own image" "$OUT/elementwise_on_the_arc.arc.run.txt"; then
  echo "error: the Arc run did not load a device image, so this says nothing" >&2
  echo "       about the device path" >&2
  exit 1
fi
echo "    cpu: $(cat "$OUT/elementwise_on_the_arc.cpu.out")"
echo "    arc: $(cat "$OUT/elementwise_on_the_arc.arc.out")"
if ! diff -q "$OUT/elementwise_on_the_arc.cpu.out" \
  "$OUT/elementwise_on_the_arc.arc.out" >/dev/null; then
  echo "error: the two backends disagree on the machine-file program" >&2
  exit 1
fi
echo "    both backends print the same number"

echo "==> the card after the run"
clinfo -l 2>/dev/null | grep "Arc" | sed 's/^/    /'
exec 9>&-
if flock -n "${TMPDIR:-/tmp}/vx-gpu.lock" -c true 2>/dev/null; then
  echo "    nothing is holding ${TMPDIR:-/tmp}/vx-gpu.lock"
else
  echo "    warning: something still holds ${TMPDIR:-/tmp}/vx-gpu.lock" >&2
fi

echo "==> done"
