#!/usr/bin/env bash
# The 2xA100 session, push-button: local prep, one archive, one pod-side run, results back.
#
# Usage:
#   bash utils/memalg/run_a100_session.sh prep                      # local: build the archive
#   bash utils/memalg/run_a100_session.sh run  <host> <port> <key>  # ship, run, fetch results
#
# What the session measures, and which issue each item belongs to:
#   1. measure_device (M1 sweep + M4 device facts + seam 2b + M6 walk rows)   vx-review#15/#18/#20
#   2. launch_smem_kernel: the FIRST shared-memory kernel Vx emits, run and
#      checked exact against the host answer                                  Vx#352/#353
#   3. probe_peer: the 2-GPU peer edge vs the declared 31.5 GB/s bound        vx-review#19/#21
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
ptx = m.group(1).encode().decode("unicode_escape")
assert ".shared" in ptx, "image carries no .shared -- wrong artifact"
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
ptx = m.group(1).encode().decode("unicode_escape")
assert ".shared" not in ptx, "the twin still uses shared memory -- the sed drifted"
open(f"{work}/global_kernel.ptx", "w").write(ptx)
print(f"  global_kernel.ptx: {len(ptx)} bytes, no .shared")
PY

    # -- sources the pod compiles itself (host-checked here, nvcc there) --
    bash utils/memalg/check_host_compile.sh
    cp utils/memalg/measure_device.cu "$WORK/"
    cp utils/memalg/probe_peer.cu "$WORK/"
    cp utils/memalg/launch_smem_kernel.cpp "$WORK/"
    cp runtime/vx_kernel_launch.h "$WORK/"
    # launch_smem_kernel includes ../../runtime/vx_kernel_launch.h; flatten for the pod.
    sed -i '' 's|#include "../../runtime/vx_kernel_launch.h"|#include "vx_kernel_launch.h"|' \
        "$WORK/launch_smem_kernel.cpp" 2>/dev/null || \
    sed -i 's|#include "../../runtime/vx_kernel_launch.h"|#include "vx_kernel_launch.h"|' \
        "$WORK/launch_smem_kernel.cpp"

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
        --csv ./launch_smem global_kernel.ptx vx_npu_kernel_0 > out/traffic_global.csv 2>&1 || true
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
