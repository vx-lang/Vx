#!/usr/bin/env bash
#===- run_m1.sh - Vx Compiler ---------------------------------*- bash -*-===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# M1: measure every declared seam and compare against the FROZEN predictions (vx-review#15).
#
#   ./run_m1.sh [--sku h100-sxm] [--predictions <dir>]
#
# Machine discipline, per PREDICTIONS.md and EXPERIMENTS.md -- all of it recorded, none of it
# assumed:
#   * GPU clocks pinned (`nvidia-smi -lgc`), so a thermal excursion cannot masquerade as a result
#   * ECC state, driver, topology, NUMA layout captured
#   * the vendor's own bandwidth tool runs FIRST as the machine ceiling
#   * predictions are READ from the freeze, never regenerated -- regenerating them after seeing
#     the data is precisely what the tag exists to prevent
#
#===----------------------------------------------------------------------===#
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=$(pwd)

SKU=h100-sxm
PRED="$ROOT/../vx-review/memory-algebra-paper/predictions"
while [ $# -gt 0 ]; do
    case "$1" in
        --sku) SKU="$2"; shift 2 ;;
        --predictions) PRED="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ ! -d "$PRED" ]; then
    echo "FATAL: frozen predictions not found at $PRED" >&2
    echo "Pass --predictions <dir>. M1 scores against the freeze, and must not regenerate it." >&2
    exit 1
fi

command -v nvcc >/dev/null || { echo "FATAL: nvcc not found. Run utils/memalg/setup_h100.sh" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "FATAL: nvidia-smi not found" >&2; exit 1; }

# nvidia-smi being INSTALLED is not the same as a GPU being present. A GPU-AMI box booted on a
# CPU-only instance type has the binary, the toolkit and the DKMS module and still has no device --
# in which case compute_cap comes back empty and the build below would silently become `-arch=sm_`.
# Caught exactly that way on a c4.8xlarge running a GPU AMI.
nvidia-smi -L >/dev/null 2>&1 || {
    echo "FATAL: nvidia-smi is installed but cannot talk to a GPU." >&2
    echo "  Either the driver module is not loaded, or this instance type has no GPU attached." >&2
    echo "  Check: lspci | grep -i nvidia   (empty => wrong instance type, not a driver problem)" >&2
    exit 1
}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$ROOT/utils/memalg/results/m1-$SKU-$STAMP"
mkdir -p "$OUT"

# ---- provenance -------------------------------------------------------------------------------
{
    echo "stamp=$STAMP"
    echo "sku=$SKU"
    echo "host=$(hostname)"
    echo "uname=$(uname -a)"
    # Both repos are private, so a rented box cannot clone them without being handed a credential.
    # Shipping the harness as loose files instead means this may not be a git checkout -- in which
    # case the operator states the commit via VX_COMMIT rather than the provenance silently going
    # blank. An empty commit field in a results directory is indistinguishable from a lost one.
    echo "commit=${VX_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN_NOT_A_GIT_CHECKOUT)}"
    echo "dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    echo "predictions_dir=$PRED"
    echo "predictions_commit=$(git -C "$PRED" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "predictions_tag=$(git -C "$PRED" describe --tags --exact-match 2>/dev/null || echo UNTAGGED)"
    echo "nvcc=$(nvcc --version | tail -1)"
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=name,driver_version,memory.total,ecc.mode.current,clocks.max.sm,clocks.max.mem \
        --format=csv 2>/dev/null || true
    echo "--- topology ---"
    nvidia-smi topo -m 2>/dev/null || true
    echo "--- numa ---"
    (command -v numactl >/dev/null && numactl --hardware) 2>/dev/null || echo "numactl absent"
} > "$OUT/env.txt"

# The freeze must be tagged. An untagged prediction set is not pre-registered -- it is a directory
# someone could have edited this morning, and no reader can tell from the outside.
if grep -q "predictions_tag=UNTAGGED" "$OUT/env.txt"; then
    echo "WARNING: the predictions directory is not at a tag." >&2
    echo "  M1 results scored against an untagged freeze are not pre-registered evidence." >&2
fi

# ---- pin clocks -------------------------------------------------------------------------------
# Best-effort: needs root on most boxes. Recorded either way, because an unpinned run is still
# usable data as long as nobody later claims it was pinned.
echo "=== pinning clocks ==="
if nvidia-smi -pm 1 >/dev/null 2>&1 && \
   nvidia-smi -lgc "$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)" >/dev/null 2>&1; then
    echo "clocks_pinned=yes" >> "$OUT/env.txt"
    echo "  pinned"
else
    echo "clocks_pinned=no" >> "$OUT/env.txt"
    echo "  could not pin (needs root?) -- recorded as unpinned"
fi

# ---- machine ceiling FIRST --------------------------------------------------------------------
# Model error is charged against achievable peak separately from declared peak: the first gap is
# the model's, the second is the declaration's, and conflating them would attribute a wrong
# spec-sheet number to the thesis.
echo "=== machine ceiling (vendor tool) ==="
if command -v bandwidthTest >/dev/null; then
    bandwidthTest --htod --dtoh --dtod --mode=shmoo --csv > "$OUT/ceiling_bandwidthTest.csv" 2>&1 \
        || echo "bandwidthTest failed (non-fatal)" | tee -a "$OUT/env.txt"
    tail -5 "$OUT/ceiling_bandwidthTest.csv" 2>/dev/null || true
else
    echo "bandwidthTest not on PATH -- the ceiling comes from measure_device's own header" \
        | tee -a "$OUT/env.txt"
fi

# ---- build + measure --------------------------------------------------------------------------
echo "=== building instrument ==="
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
case "$ARCH" in
    [0-9][0-9]*) ;;
    *) echo "FATAL: could not derive compute capability (got '$ARCH')" >&2; exit 1 ;;
esac
nvcc -O3 -arch="sm_${ARCH}" utils/memalg/measure_device.cu -o utils/memalg/measure_device
echo "  built for sm_${ARCH}"

echo "=== measuring ==="
./utils/memalg/measure_device > "$OUT/measured.csv" 2> "$OUT/measured.log"
tail -8 "$OUT/measured.log"

# ---- compare ----------------------------------------------------------------------------------
echo "=== predicted vs measured ==="
python3 utils/memalg/compare_m1.py \
    --predictions "$PRED" \
    --measured "$OUT/measured.csv" \
    --sku "$SKU" \
    --out "$OUT/error_table.csv" | tee "$OUT/error_table.txt"

echo
echo "results -> $OUT"
