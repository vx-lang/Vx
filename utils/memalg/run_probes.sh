#!/usr/bin/env bash
#===- run_probes.sh - Vx Compiler ------------------------------*- bash -*-===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Measure the full transfer-edge powerset (vx-review#15, #22).
#
#   ./run_probes.sh
#
# These are DIAGNOSTICS, not scored cells: they measure hardware properties in order to decide
# what the Phase-2 model extension should be. They do not join against the frozen predictions, so
# no freeze and no vx-review checkout is needed -- unlike run_m1.sh, which must have the tag.
#
# Build first, and stop on failure. probe_edges.cu has NOT been compiled anywhere: the CUDA box
# used to pre-verify the M1 instrument is gone, so this build is its first. That is deliberate --
# a compile error costs seconds here and nothing at all in wrong conclusions.
#
#===----------------------------------------------------------------------===#
set -euo pipefail

cd "$(dirname "$0")"
export PATH=/usr/local/cuda/bin:$PATH

command -v nvcc >/dev/null || { echo "FATAL: nvcc not on PATH (try /usr/local/cuda/bin)" >&2; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || {
    echo "FATAL: nvidia-smi cannot talk to a GPU -- no device, or the driver is not loaded." >&2
    exit 1
}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')
OUT="results/probes-${NAME}-${STAMP}"
mkdir -p "$OUT"

ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
case "$ARCH" in
    [0-9][0-9]*) ;;
    *) echo "FATAL: could not derive compute capability (got '$ARCH')" >&2; exit 1 ;;
esac

{
    echo "stamp=$STAMP"
    echo "host=$(hostname)"
    echo "uname=$(uname -a)"
    echo "vx_commit=${VX_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN_NOT_A_GIT_CHECKOUT)}"
    echo "nvcc=$(nvcc --version | tail -1)"
    echo "arch=sm_${ARCH}"
    nvidia-smi --query-gpu=name,driver_version,memory.total,ecc.mode.current,clocks.max.sm,clocks.max.mem \
        --format=csv 2>/dev/null || true
} > "$OUT/env.txt"

# Clock pinning is best-effort and will fail in an unprivileged container. Recorded either way, so
# nobody can later claim a run was pinned when it was not. The per-SM columns are denominated in
# CYCLES, which are clock-invariant, so an unpinned run still yields usable numbers there.
if nvidia-smi -pm 1 >/dev/null 2>&1 && \
   nvidia-smi -lgc "$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)" >/dev/null 2>&1; then
    echo "clocks_pinned=yes" >> "$OUT/env.txt"
else
    echo "clocks_pinned=no" >> "$OUT/env.txt"
    echo "note: clocks NOT pinned (needs privileged access); per-SM columns are in cycles and unaffected"
fi

echo "=== building probe_edges.cu for sm_${ARCH} ==="
nvcc -O3 -arch="sm_${ARCH}" probe_edges.cu -o probe_edges
echo "  built"

echo "=== measuring ==="
./probe_edges 2>&1 | tee "$OUT/probes.txt"

echo
echo "results -> $OUT"
