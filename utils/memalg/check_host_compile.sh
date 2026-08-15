#!/usr/bin/env bash
# Syntax- and format-check measure_device.cu's HOST code without a CUDA toolkit.
#
# Why this exists: two blocks in measure_device.cu were written on a machine with no nvcc and
# committed unverified -- `seam 2b` and the `M4: the declared numbers` block. An unverified
# instrument is a rented GPU session spent debugging a compiler error instead of measuring, so
# this catches the errors that are catchable here: typos, wrong printf formats, wrong types,
# missing variables.
#
# What it CANNOT catch, and do not read it as catching:
#   * wrong `cudaDeviceProp` field names -- the stub declares them from our own belief, so a
#     misremembered field compiles here and fails under nvcc;
#   * anything in device code. `__global__` bodies, `<<<grid,block>>>` launches and `clock64()`
#     are stripped or stubbed. The launch's ARGUMENT LIST is still type-checked, which is the
#     part that actually goes wrong.
#   * whether the numbers mean anything. That needs the hardware.
#
# Usage:  bash utils/memalg/check_host_compile.sh
# Exit 0 = the host code compiles. Exit non-zero = it would not have compiled on the pod either.

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

SRC=utils/memalg/measure_device.cu
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/cuda_stub.h" <<'EOF'
#pragma once
#include <cstddef>
typedef enum { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 } cudaError_t;
typedef enum { cudaDevAttrClockRate = 13, cudaDevAttrMemoryClockRate = 36 } cudaDeviceAttr;
struct cudaDeviceProp {
  char name[256]; size_t totalGlobalMem; size_t sharedMemPerBlock;
  size_t sharedMemPerMultiprocessor; size_t sharedMemPerBlockOptin;
  int l2CacheSize; int memoryBusWidth; int multiProcessorCount; int major, minor;
};
struct float4 { float x, y, z, w; };
struct cudaEvent_st; typedef cudaEvent_st *cudaEvent_t;
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int);
cudaError_t cudaDeviceGetAttribute(int *, cudaDeviceAttr, int);
cudaError_t cudaMemGetInfo(size_t *, size_t *);
// A template in the real headers, so the stub must be one too or every typed pointer is an error.
template <class T> cudaError_t cudaMalloc(T **, size_t);
cudaError_t cudaFree(void *);
cudaError_t cudaMemset(void *, int, size_t);
cudaError_t cudaGetLastError(void);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaEventCreate(cudaEvent_t *);
cudaError_t cudaEventDestroy(cudaEvent_t);
// The stream argument defaults to 0 in the real headers; without the default every
// single-argument cudaEventRecord call is an error here and not on the pod.
cudaError_t cudaEventRecord(cudaEvent_t, int = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
cudaError_t cudaHostAlloc(void **, size_t, unsigned);
cudaError_t cudaFreeHost(void *);
cudaError_t cudaMemcpy(void *, const void *, size_t, int);
const char *cudaGetErrorString(cudaError_t);
EOF

# Pull the shared preamble (CK macro, sweep sizes, helpers) and one block at a time, by the
# comment banners rather than by line number, so an edit above a block does not silently shift
# what gets checked.
python3 - "$SRC" "$WORK" <<'PY'
import re, sys
src, work = sys.argv[1], sys.argv[2]
text = open(src).read()
lines = text.split("\n")

def between(start_pat, end_pat):
    a = next(i for i, l in enumerate(lines) if re.search(start_pat, l))
    b = next(i for i, l in enumerate(lines) if i > a and re.search(end_pat, l))
    return "\n".join(lines[a:b])

preamble = "\n".join([
    between(r"^#define CK\(x\)", r"^// The frozen cell sizes"),
    between(r"^static const size_t SIZES", r"^// One declared-number row"),
    between(r"^static void fact_row", r"^static double median_of"),
    between(r"^static double median_of", r"^// ---- seam 2"),
])

blocks = {
    "m4_facts": between(r"---- M4: the declared numbers", r"---- seam 1: CPU_DRAM"),
    "seam2b":   between(r"---- seam 2b: HBM -> L2, the actual fill", r"---- seam 3: L2 -> SMEM"),
}

for name, body in blocks.items():
    body = re.sub(r"<<<[^>]*>>>", "", body)  # launch syntax; the arg list survives
    open(f"{work}/{name}.cpp", "w").write("\n".join([
        "#include <cstdio>", "#include <cstdlib>", "#include <cstring>",
        "#include <algorithm>", "#include <vector>",
        f'#include "{work}/cuda_stub.h"',
        "#define REPS 11",
        "static void l2_stream(const float4*, size_t, float*, int);",
        preamble,
        "int main() {",
        "  int dev = 0; CK(cudaSetDevice(dev));",
        "  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));",
        "  int mem_clock_khz = 0, sm_clock_khz = 0;",
        "  CK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev));",
        "  CK(cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, dev));",
        "  cudaEvent_t ev_a, ev_b; CK(cudaEventCreate(&ev_a)); CK(cudaEventCreate(&ev_b));",
        body,
        "  (void)ev_a; (void)ev_b; return 0; }"]))
    print(f"extracted {name}")
PY

rc=0
for f in "$WORK"/*.cpp; do
    name=$(basename "$f" .cpp)
    if clang++ -std=c++17 -Wall -Wformat -fsyntax-only "$f" 2>"$WORK/$name.err"; then
        echo "  OK    $name"
    else
        echo "  FAIL  $name"
        grep -E "error:" "$WORK/$name.err" | head -10
        rc=1
    fi
done

if [ $rc -eq 0 ]; then
    echo "host code compiles (device code and cudaDeviceProp field names still unverified)"
fi
exit $rc
