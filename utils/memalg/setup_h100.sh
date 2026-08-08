#!/usr/bin/env bash
#===- setup_h100.sh - Vx Compiler -----------------------------*- bash -*-===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Provision a rented NVIDIA box for the M-series (vx-review#15..#21).
#
#   curl -sSL <this file> | bash     # or: ./setup_h100.sh
#
# Deliberately minimal. M1 needs a CUDA toolchain, python3, and the repo -- it does NOT need the
# Vx compiler built, because predictions are read from the frozen directory rather than
# regenerated. That keeps the metered box off the critical path of a Rust+MLIR build, which on a
# fresh machine is tens of minutes of rental time spent on nothing.
#
#===----------------------------------------------------------------------===#
set -euo pipefail

echo "=== what is already here ==="
uname -a
command -v nvidia-smi >/dev/null && nvidia-smi || { echo "FATAL: no nvidia-smi -- not a GPU box"; exit 1; }
echo

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COUNT=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
echo "GPU: $NAME  compute_cap=$CC  count=$COUNT"

# Which fleet machine file this box corresponds to. Getting this wrong means scoring measurements
# against another SKU's predictions, which would look like a spectacular model failure.
case "$NAME" in
    *H100*)   SKU=h100-sxm ;;
    *H200*)   SKU=h200 ;;
    *B200*)   SKU=b200 ;;
    *A100*80*) SKU=a100-80 ;;
    *A100*)   SKU=a100-40 ;;
    *MI300*)  SKU=mi300x ;;
    *)        SKU=UNKNOWN ;;
esac
echo "matching fleet file: fleet/$SKU.vx"
if [ "$SKU" = "UNKNOWN" ]; then
    echo "  !! no fleet file matches this GPU. Write one and freeze its predictions BEFORE"
    echo "     measuring, or the comparison is retrofitted by construction."
fi
echo

# --- THE QUESTION THAT DECIDES WHETHER THE HOST-SEAM PREDICTIONS ARE EVEN THE RIGHT KIND --------
# PREDICTIONS.md records the host link as a PCIe placeholder. A Grace/C2C-attached part reaches
# the host ~7x faster, in which case the M1 host-seam cells are measuring a wrong DECLARATION
# rather than a model residual, and must be re-frozen under an amendment instead of scored.
echo "=== host link: PCIe or C2C? ==="
if nvidia-smi -q 2>/dev/null | grep -qi "C2C\|Grace"; then
    echo "  C2C / Grace detected -- the declared PCIe figure is WRONG for this box."
    echo "  Amend PREDICTIONS.md and re-freeze the host-seam cells BEFORE running M1."
else
    echo "  no C2C indication; treating the host link as PCIe"
    nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv 2>/dev/null || true
    echo "  ^ compare against fleet/$SKU.vx's declared CPU_DRAM->HBM figure"
fi
echo

echo "=== installing ==="
if command -v apt-get >/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential python3 git numactl >/dev/null
fi
command -v nvcc >/dev/null || {
    echo "nvcc missing. On most GPU images the toolkit is present but not on PATH:"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    ls -d /usr/local/cuda*/bin 2>/dev/null || echo "  (no /usr/local/cuda* found -- install the CUDA toolkit)"
}

echo
echo "=== ready ==="
cat <<EOF
Next, from the Vx repo root, with the frozen predictions available:

  ./utils/memalg/run_m1.sh --sku $SKU --predictions <path-to>/memory-algebra-paper/predictions

The predictions directory must be at the tag memalg-freeze-2026-08-07. run_m1.sh warns if it
is not: results scored against an untagged freeze are not pre-registered evidence.
EOF
