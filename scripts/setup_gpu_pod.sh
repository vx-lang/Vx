#!/usr/bin/env bash
#===- setup_gpu_pod.sh - Prepare a rented GPU box to run Vx programs ------===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Brings a rented GPU pod to the point where it can compile and run a Vx
# program, without a checkout of this repository on it.
#
# What gets copied there is an archive of build outputs and a handful of runtime
# sources -- no git history, no compiler source -- and the directory is removed
# when the run is done. See "Build & deployment discipline" in
# docs/discussions/implementation_plans/gpu_disaggregated_inference.md.
#
# The bundle is:
#
#   vxc                      the compiler, built on the x86 build box. It links
#                            LLVM statically, so it needs only libc, libstdc++,
#                            libz and libzstd, all of which a pod image has.
#   runtime/, include/       the four files runtime/cuda_dispatch.cpp needs.
#   *.vx                     the programs to run.
#
# The pod still needs LLVM's command-line tools, because JIT execution shells
# out to mlir-translate and clang++ rather than translating in process. That is
# the one thing installed here. Emitting an object on the build box and copying
# only that would avoid it entirely, which is the better arrangement and is
# blocked on #332.
#
# Usage, from the bundle directory on the pod:
#   ./setup_gpu_pod.sh          # install tools, build the dispatch library
#   ./setup_gpu_pod.sh --check  # report what is present and exit
#
#===----------------------------------------------------------------------===#

set -euo pipefail

LLVM_VERSION=22
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

report() {
  echo "=== GPU pod status ==="
  echo -n "nvidia-smi:     "
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
    # Every device, not just the first. A disaggregated run needs two and aborts
    # in the plugin if it named one this box does not have (#347), which is
    # worth knowing before the run rather than during it.
    echo "GPUs:           $(nvidia-smi --query-gpu=index,name --format=csv,noheader | wc -l | tr -d ' ')"
    nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/                /'
    # What the link between them actually is. fleet/node-2gpu-a100.vx declares
    # the PCIe figure because a rented pod does not say, and this is what
    # settles it -- recorded next to the run so the declared number and the
    # observed one are archived together rather than one being edited to match
    # the other.
    echo "interconnect:"
    nvidia-smi topo -m 2>/dev/null | sed 's/^/                /' || echo "                unavailable"
  else
    echo "absent"
  fi
  echo -n "CUDA toolkit:   "
  if [ -f "$CUDA_HOME/include/cuda_runtime.h" ]; then
    echo "$CUDA_HOME ($(basename "$(readlink -f "$CUDA_HOME")"))"
  else
    echo "absent at $CUDA_HOME"
  fi
  echo -n "cuBLAS:         "
  ls "$CUDA_HOME"/lib64/libcublas.so* >/dev/null 2>&1 && echo "present" || echo "absent"
  echo -n "mlir-translate: "
  command -v mlir-translate >/dev/null 2>&1 && mlir-translate --version | head -1 || echo "absent"
  echo -n "clang++:        "
  command -v clang++ >/dev/null 2>&1 && clang++ --version | head -1 || echo "absent"
  echo -n "libffi:         "
  ls /usr/lib/x86_64-linux-gnu/libffi.so >/dev/null 2>&1 && echo "present" || echo "absent"
  echo -n "MLIR runner:    "
  ls /usr/lib/llvm-$LLVM_VERSION/lib/libmlir_c_runner_utils.so >/dev/null 2>&1 &&
    echo "present" || echo "absent"
  echo -n "vxc:            "
  [ -x "$BUNDLE_DIR/vxc" ] && echo "present" || echo "absent"
  echo -n "vx_std_core:    "
  [ -f "$BUNDLE_DIR/target/release/libvx_std_core.so" ] && echo "present" || echo "absent"
}

if [ "${1:-}" = "--check" ]; then
  report
  exit 0
fi

# A pod image has the CUDA runtime but rarely the toolkit headers. Without them
# the dispatch library cannot be built here, and there is no point continuing.
if [ ! -f "$CUDA_HOME/include/cuda_runtime.h" ]; then
  echo "error: no CUDA toolkit at $CUDA_HOME (set CUDA_HOME, or pick an image with the toolkit)" >&2
  exit 1
fi

echo "==> Installing LLVM $LLVM_VERSION tools and libffi"
export DEBIAN_FRONTEND=noninteractive
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

$SUDO apt-get update -qq
$SUDO apt-get install -y -qq wget gnupg lsb-release software-properties-common libffi-dev

if ! command -v mlir-translate >/dev/null 2>&1; then
  wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh
  chmod +x /tmp/llvm.sh
  $SUDO /tmp/llvm.sh $LLVM_VERSION
fi

# mlir-*-tools provides mlir-translate; libmlir-*-dev provides the runner-utils
# libraries that a JIT-linked program needs for printing memrefs. Missing the
# second one fails at link time, not at setup time, which is a worse place to
# find out.
$SUDO apt-get install -y -qq "mlir-$LLVM_VERSION-tools" "libmlir-$LLVM_VERSION-dev"

# vxc finds its tools through PATH, so the versioned directory has to lead.
echo "export PATH=/usr/lib/llvm-$LLVM_VERSION/bin:\$PATH" > "$BUNDLE_DIR/env.sh"
export PATH="/usr/lib/llvm-$LLVM_VERSION/bin:$PATH"

echo "==> Building the CUDA dispatch backend"
clang++ -shared -fPIC -O2 -Wall \
  "$BUNDLE_DIR/runtime/cuda_dispatch.cpp" \
  -I"$CUDA_HOME/include" \
  -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" \
  -lcudart -lcublas -lffi \
  -o "$BUNDLE_DIR/libvx_cuda_dispatch.so"

# vxc links whatever dispatch library was built alongside it, which is a path in
# the build box's OUT_DIR and does not exist here. Point it at the one just
# built, against this box's CUDA.
echo "export VX_DISPATCH_LIB=$BUNDLE_DIR/libvx_cuda_dispatch.so" >> "$BUNDLE_DIR/env.sh"

echo
report
echo
echo "Ready. Run:  source $BUNDLE_DIR/env.sh && $BUNDLE_DIR/vxc <program>.vx --run"
echo "When done:   rm -rf $BUNDLE_DIR"
