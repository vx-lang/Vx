#!/usr/bin/env bash
# The 2xA100 session, push-button: local prep, one archive, one pod-side run, results back.
#
# Usage:
#   bash utils/memalg/run_a100_session.sh prep                      # local: build the archive
#   bash utils/memalg/run_a100_session.sh run  <host> <port> <key>  # ship, run, fetch results
#
# What the session measures, and which issue each item belongs to:
#   1. measure_device (M1 sweep + M4 device facts + seam 2b + M6 walk rows)
#   2. launch_smem_kernel: the FIRST shared-memory kernel Vx emits, run and
#      checked exact against the host answer                                  Vx#352/#353
#   3. probe_peer: the 2-GPU peer edge vs the declared 31.5 GB/s bound
#   4. if `ncu` exists on the pod: DRAM/shared traffic counts for the SMEM
#      kernel vs its global-only twin -- A4's first data point (traffic is
#      valid even single-threaded; TIME is not, so no timing comparison)      Vx#353 A4
#
# Policy (from the campaign's standing rules): only an archive is shipped, never a checkout;
# the pod directory is removed at the end; every artifact lands under results/ with an env
# capture so the run is citable.

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
WORK=utils/memalg/session-prep

prep() {
    rm -rf "$WORK" && mkdir -p "$WORK"

    # -- PTX for the launch harness, extracted here where the compiler lives --
    echo "== extracting the shared-memory kernel's PTX =="
    ./target/release/vxc tests/backend/pass/custom_topology_device_image.vx \
        --emit-llvm -o /dev/null 2>/dev/null > "$WORK/smem.ll"
    python3 - "$WORK" <<'PY'
import re, sys
work = sys.argv[1]
s = open(f"{work}/smem.ll").read()
m = re.search(r'image=(.*?)\\00"?\)', s, re.S)
assert m, "no image= in the emitted LLVM -- did the arch gate regress?"
# LLVM string escapes are two-digit HEX (\\0A = newline), not C escapes -- unicode_escape
# read \\0A as NUL + 'A' and corrupted every line break; the .shared assertion passed
# because that substring carries no escapes. Found when the pod's entry lookup failed.
ptx = re.sub(r"\\([0-9A-Fa-f]{2})", lambda g: chr(int(g.group(1), 16)), m.group(1))
assert "\x00" not in ptx, "NUL inside decoded PTX -- extraction wrong"
# The storage DECLARATION, not the substring: a kernel can carry ld.shared/st.shared
# instructions against local storage (measured: ILLEGAL_ADDRESS on an A100) and the
# substring check waves it through -- the mislabelled-artifact failure, one layer deeper.
assert re.search(r"\.shared \.align", ptx), "no .shared STORAGE declared -- wrong artifact"
open(f"{work}/smem_kernel.ptx", "w").write(ptx)
print(f"  smem_kernel.ptx: {len(ptx)} bytes, .shared x{ptx.count('.shared')}")
PY

    # -- the global-only twin, derived from the corpus fixture so it cannot drift --
    echo "== deriving the global-only twin =="
    sed -e '/let tile = transfer(ad, Memory::SMEM);/d' \
        -e 's/tile\[i\]\[d\]/ad[i][d]/' \
        tests/backend/pass/custom_topology_device_image.vx > "$WORK/global_twin.vx"
    ./target/release/vxc "$WORK/global_twin.vx" --emit-llvm -o /dev/null 2>/dev/null \
        > "$WORK/twin.ll"
    python3 - "$WORK" <<'PY'
import re, sys
work = sys.argv[1]
s = open(f"{work}/twin.ll").read()
m = re.search(r'image=(.*?)\\00"?\)', s, re.S)
assert m, "the global twin lost its image"
# LLVM string escapes are two-digit HEX (\\0A = newline), not C escapes -- unicode_escape
# read \\0A as NUL + 'A' and corrupted every line break; the .shared assertion passed
# because that substring carries no escapes. Found when the pod's entry lookup failed.
ptx = re.sub(r"\\([0-9A-Fa-f]{2})", lambda g: chr(int(g.group(1), 16)), m.group(1))
assert "\x00" not in ptx, "NUL inside decoded PTX -- extraction wrong"
assert ".shared" not in ptx, "the twin still uses shared memory -- the sed drifted"
open(f"{work}/global_kernel.ptx", "w").write(ptx)
print(f"  global_kernel.ptx: {len(ptx)} bytes, no .shared")
PY

    # -- sources the pod compiles itself (host-checked here, nvcc there) --
    bash utils/memalg/check_host_compile.sh
    # The peer probe is plain CUDA runtime API; a syntax pass with a stub costs nothing and the
    # first session lost the probe to a macro-shadowing bug nothing had compiled. The empty
    # cuda_runtime.h on the include path shadows the real include; the stub supplies the decls.
    local stubdir
    stubdir=$(mktemp -d)
    : > "$stubdir/cuda_runtime.h"
    clang++ -fsyntax-only -std=c++17 -x c++ -I "$stubdir" \
        -include utils/memalg/.peer_stub.h utils/memalg/probe_peer.cu
    # The launch harness too -- its first shipping failed on a struct-name typo a syntax pass
    # would have caught. cuda.h gets the same empty-shadow treatment; a minimal driver-API stub
    # supplies the decls.
    cat > "$stubdir/cuda.h" <<'CUDASTUB'
#pragma once
#include <cstddef>
#include <cstdint>
typedef int CUresult; typedef int CUdevice; typedef unsigned long long CUdeviceptr;
struct CUctx_st; typedef CUctx_st *CUcontext;
struct CUmod_st; typedef CUmod_st *CUmodule;
struct CUfunc_st; typedef CUfunc_st *CUfunction;
static const int CUDA_SUCCESS = 0;
enum CUdevice_attribute { CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
                          CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76 };
CUresult cuInit(unsigned); CUresult cuDeviceGet(CUdevice *, int);
CUresult cuCtxCreate(CUcontext *, unsigned, CUdevice);
CUresult cuDeviceGetName(char *, int, CUdevice);
CUresult cuDeviceGetAttribute(int *, CUdevice_attribute, CUdevice);
CUresult cuModuleLoadData(CUmodule *, const void *);
CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
CUresult cuMemAlloc(CUdeviceptr *, size_t);
CUresult cuMemcpyHtoD(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH(void *, CUdeviceptr, size_t);
CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                        unsigned, unsigned, void *, void **, void **);
CUresult cuCtxSynchronize(void);
CUresult cuGetErrorName(CUresult, const char **);
CUresult cuGetErrorString(CUresult, const char **);
CUDASTUB
    clang++ -fsyntax-only -std=c++17 -I "$stubdir" utils/memalg/launch_smem_kernel.cpp
    rm -rf "$stubdir"
    echo "  probe_peer.cu and launch_smem_kernel.cpp syntax-check against the stubs" 
    cp utils/memalg/measure_device.cu "$WORK/"
    cp utils/memalg/probe_peer.cu "$WORK/"
    cp utils/memalg/launch_smem_kernel.cpp "$WORK/"
    cp runtime/vx_kernel_launch.h "$WORK/"
    cp include/vx_hardware_runtime.h "$WORK/"
    # Flatten the include paths for the pod's flat directory -- vx_kernel_launch.h itself pulls
    # ../include/vx_hardware_runtime.h, which is what broke the first session's build.
    python3 - "$WORK" <<'FLAT'
import sys
work = sys.argv[1]
for fname, old, new in [
    ("launch_smem_kernel.cpp", '#include "../../runtime/vx_kernel_launch.h"', '#include "vx_kernel_launch.h"'),
    ("vx_kernel_launch.h", '#include "../include/vx_hardware_runtime.h"', '#include "vx_hardware_runtime.h"'),
]:
    path = f"{work}/{fname}"
    s = open(path).read()
    assert old in s, f"include flatten missed in {fname}"
    open(path, "w").write(s.replace(old, new, 1))
print("  includes flattened")
FLAT

    # -- the pod-side runner --
    cat > "$WORK/on_pod.sh" <<'POD'
#!/usr/bin/env bash
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")"
mkdir -p out
{
    echo "stamp=$(date -u +%Y%m%dT%H%M%SZ)"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,clocks.max.sm --format=csv
    nvidia-smi topo -m
    nvcc --version | tail -1
} > out/env.txt 2>&1

echo "== 1/4 measure_device (M1 + M4 + seam2b + M6 walk) =="
nvcc -O3 -arch=sm_80 measure_device.cu -o measure_device 2> out/build_measure.log \
    && ./measure_device > out/measured.csv 2> out/measured.log \
    || echo "MEASURE_DEVICE FAILED" | tee -a out/measured.log

echo "== 2/4 the shared-memory kernel =="
g++ -std=c++17 -O2 -I/usr/local/cuda/include launch_smem_kernel.cpp \
    -L/usr/local/cuda/lib64/stubs -lcuda -o launch_smem 2> out/build_smem.log \
    && ./launch_smem smem_kernel.ptx vx_npu_kernel_0 > out/smem_launch.txt 2>&1 \
    || echo "SMEM LAUNCH FAILED" | tee -a out/smem_launch.txt

echo "== 3/4 the peer edge (needs 2 GPUs) =="
nvcc -O3 -arch=sm_80 probe_peer.cu -o probe_peer 2> out/build_peer.log \
    && ./probe_peer > out/peer.csv 2>&1 \
    || echo "PEER PROBE FAILED" | tee -a out/peer.csv

echo "== 4/4 traffic counts (optional, needs ncu) =="
if command -v ncu >/dev/null 2>&1; then
    # Traffic, not time: dram bytes and shared transactions per kernel. Valid
    # single-threaded; a timing comparison would not be, so none is taken.
    ncu --metrics dram__bytes.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum \
        --csv ./launch_smem smem_kernel.ptx vx_npu_kernel_0 > out/traffic_smem.csv 2>&1 || true
    ncu --metrics dram__bytes.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum \
        --csv ./launch_smem global_kernel.ptx vx_npu_kernel_0 --global > out/traffic_global.csv 2>&1 || true
else
    echo "ncu not present; traffic counts skipped" > out/traffic_skipped.txt
fi
echo "== done; results in $(pwd)/out =="
POD
    chmod +x "$WORK/on_pod.sh"
    tar czf utils/memalg/session-$STAMP.tgz -C "$WORK" .
    echo "== archive ready: utils/memalg/session-$STAMP.tgz =="
    ls -la "$WORK"
}

run() {
    local host=$1 port=$2 key=$3
    local tgz
    tgz=$(ls -t utils/memalg/session-*.tgz | head -1)
    echo "== shipping $tgz to $host:$port =="
    scp -P "$port" -i "$key" "$tgz" "root@$host:/root/session.tgz"
    ssh -p "$port" -i "$key" "root@$host" \
        'mkdir -p /root/vxsession && tar xzf /root/session.tgz -C /root/vxsession && rm /root/session.tgz && bash /root/vxsession/on_pod.sh'
    local dest="utils/memalg/results/a100-session-$STAMP"
    mkdir -p "$dest"
    scp -P "$port" -i "$key" -r "root@$host:/root/vxsession/out/*" "$dest/"
    # The standing rule: nothing of ours stays on the pod.
    ssh -p "$port" -i "$key" "root@$host" 'rm -rf /root/vxsession'
    echo "== results in $dest; pod directory removed =="
    echo "== next: =="
    echo "  python3 utils/memalg/compare_m1.py --predictions <frozen> --measured $dest/measured.csv --sku a100-80"
    echo "  python3 utils/memalg/walk.py --machine fleet/a100-80.vx --measured $dest/measured.csv"
    echo "  cat $dest/smem_launch.txt  $dest/peer.csv"
}

case "${1:-}" in
    prep) prep ;;
    run) shift; run "$@" ;;
    *) echo "usage: $0 prep | run <host> <port> <keyfile>"; exit 1 ;;
esac
