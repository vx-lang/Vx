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
command -v nvidia-smi >/dev/null || { echo "FATAL: nvidia-smi not installed -- not a GPU image"; exit 1; }
if ! nvidia-smi; then
    # A GPU AMI on a CPU-only instance type has the toolkit, the driver package and the DKMS
    # module, and no device. Distinguishing that from a broken driver is the difference between
    # "relaunch on the right instance type" and "rebuild the kernel module", so check the bus.
    echo
    echo "FATAL: nvidia-smi is installed but cannot talk to a driver."
    if ! command -v lspci >/dev/null; then
        # Containers frequently ship without pciutils. Saying "no GPU on the bus" here would be a
        # confident wrong answer -- the bus was never checked.
        echo "  (lspci not installed, so the PCI bus was NOT checked -- cannot tell you which"
        echo "   of the two cases below applies. Install pciutils to find out.)"
    elif lspci 2>/dev/null | grep -qi nvidia; then
        echo "  A GPU IS on the PCI bus -- this is a driver problem."
        echo "  Try: modprobe nvidia   (or rebuild DKMS for $(uname -r))"
    else
        echo "  NO GPU on the PCI bus -- this machine has no GPU attached."
        echo "  This is not a driver problem. On a cloud VM, relaunch on a GPU instance type"
        echo "  (AWS p5.* for H100 SXM); on RunPod, the pod was created without a GPU."
    fi
    exit 1
fi
echo

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COUNT=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
echo "GPU: $NAME  compute_cap=$CC  count=$COUNT"

# Which fleet machine file this box corresponds to. Getting this wrong means scoring measurements
# against another SKU's predictions, which does not look like an error -- it looks like a large,
# clean model residual. That is the most dangerous failure mode in the whole experiment, so match
# on the SPECIFIC part and refuse to guess.
#
# "H100" is not one part. SXM is HBM3 at 3.35 TB/s, PCIe is HBM2e at ~2 TB/s, NVL is a 94 GB part.
# Only h100-sxm is in the freeze, and the PCIe card is the cheaper rental -- so the natural
# cost-saving choice is exactly the one that would silently invalidate the run.
SKU=UNKNOWN
SKU_NOTE=""
case "$NAME" in
    *H100*NVL*)              SKU=UNSUPPORTED; SKU_NOTE="H100 NVL: 94 GB part, not the 80 GB SXM in the freeze" ;;
    *H100*PCIe*|*H100*PCIE*) SKU=UNSUPPORTED; SKU_NOTE="H100 PCIe: HBM2e ~2 TB/s, not the 3.35 TB/s SXM in the freeze" ;;
    *H100*HBM3*|*H100*SXM*)  SKU=h100-sxm ;;
    *H100*)                  SKU=UNSUPPORTED; SKU_NOTE="H100 variant not identifiable from '$NAME'" ;;
    *H200*)                  SKU=h200 ;;
    *B200*)                  SKU=b200 ;;
    *A100*80*)               SKU=a100-80 ;;
    *A100*40*)               SKU=a100-40 ;;
    *MI300*)                 SKU=mi300x ;;
esac

if [ "$SKU" = "UNSUPPORTED" ]; then
    echo "  !! WRONG PART: $SKU_NOTE"
    echo "     Scoring this card against the frozen h100-sxm cells would produce a large residual"
    echo "     that is an artefact of the SKU mismatch, not a model error. Either rent an H100 SXM"
    echo "     (80GB HBM3), or write a fleet file for this part and freeze its predictions BEFORE"
    echo "     measuring -- as a dated amendment in PREDICTIONS.md."
    exit 1
fi
if [ "$SKU" = "UNKNOWN" ]; then
    echo "  !! no fleet file matches '$NAME'. Write one and freeze its predictions BEFORE"
    echo "     measuring, or the comparison is retrofitted by construction."
    exit 1
fi
echo "matching fleet file: fleet/$SKU.vx"

# Cross-check the capacity the device reports against what the fleet file declares. Name matching
# is a heuristic; capacity is a fact the card states about itself, and it catches a rebadged or
# MIG-partitioned device that the name alone would wave through.
MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  device reports ${MEM_MIB} MiB of memory"
case "$SKU" in
    h100-sxm|a100-80|h200) [ "$MEM_MIB" -ge 79000 ] || echo "  !! expected >=80 GiB for $SKU -- MIG partition or wrong part?" ;;
esac
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
# Containers (RunPod, Lambda, Docker images generally) run as root with no sudo installed, while
# cloud VMs run as an unprivileged user with sudo. Handle both rather than assuming either.
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null; then
    SUDO="sudo"
else
    SUDO=""
    echo "  !! not root and no sudo -- skipping package installation."
    echo "     If anything below is missing, install it by hand."
fi
if command -v apt-get >/dev/null && { [ "$(id -u)" -eq 0 ] || command -v sudo >/dev/null; }; then
    # pciutils so the GPU-vs-driver diagnostic above can actually check the bus next time.
    $SUDO apt-get update -qq || true
    $SUDO apt-get install -y -qq build-essential python3 git numactl pciutils >/dev/null || \
        echo "  !! package install failed -- continuing, the checks below will say what is missing"
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
