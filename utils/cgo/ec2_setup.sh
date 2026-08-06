#!/usr/bin/env bash
#===- ec2_setup.sh - Vx Compiler -----------------------------*- bash -*-===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Provision a fresh Ubuntu x86-64 instance for the CGO 2027 E1 measurement
# (#295/#296). Mirrors .github/workflows/ci.yml's dependency set, which is the
# only Linux build recipe this project has that is known to work.
#
#   sudo ./ec2_setup.sh                 # deps only
#   sudo ./ec2_setup.sh --clone <url>   # deps + clone + build
#
# Assumes Ubuntu 24.04. On a different release, the apt.llvm.org script still
# works but the package names may drift.
#
#===----------------------------------------------------------------------===#
set -euo pipefail

BRANCH="${BRANCH:-parallel-frontend-eval}"
CLONE_URL=""
WORKDIR="${WORKDIR:-/opt/vx}"

while [ $# -gt 0 ]; do
    case "$1" in
        --clone) CLONE_URL="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "run as root (apt + perf setup)" >&2
    exit 1
fi

echo "== base packages =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    build-essential cmake ninja-build git curl wget z3 lld pkg-config \
    linux-tools-common "linux-tools-$(uname -r)" linux-tools-generic \
    python3-pip numactl hwloc

echo "== LLVM/MLIR 22 =="
# Same source and version as CI. MLIR's C API is not stable across major
# versions, so 22 is not a floor -- melior is built against it specifically.
wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh
chmod +x /tmp/llvm.sh
/tmp/llvm.sh 22
apt-get install -y -qq \
    llvm-22-dev libmlir-22-dev mlir-22-tools libpolly-22-dev libclang-22-dev

echo 'export PATH=/usr/lib/llvm-22/bin:$PATH' > /etc/profile.d/llvm22.sh
echo 'export LLVM_CONFIG_PATH=llvm-config-22' >> /etc/profile.d/llvm22.sh

echo "== rust =="
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
    echo 'export PATH=$HOME/.cargo/bin:$PATH' > /etc/profile.d/rust.sh
fi

echo "== perf permissions =="
# `perf lock contention -b` is BPF-based (kernel 5.19+) and needs to read kernel
# symbols and attach programs. This is the measurement #295 actually turns on --
# wall clock alone cannot distinguish the two interning designs (see README.md),
# so a box where perf does not work is a box that cannot produce the result.
cat > /etc/sysctl.d/99-vx-perf.conf <<'EOF'
kernel.perf_event_paranoid = -1
kernel.kptr_restrict = 0
EOF
sysctl --system >/dev/null

echo "== corpus tmpfs =="
# The corpus is regenerated per cell and read once per rep. On instance storage
# that is filesystem-cache behaviour inside the timed region; on tmpfs it is not.
mkdir -p /dev/shm/vxbench
chmod 1777 /dev/shm/vxbench

if [ -n "$CLONE_URL" ]; then
    echo "== clone + build ($BRANCH) =="
    mkdir -p "$(dirname "$WORKDIR")"
    if [ ! -d "$WORKDIR/.git" ]; then
        git clone --branch "$BRANCH" "$CLONE_URL" "$WORKDIR"
    else
        git -C "$WORKDIR" fetch --all && git -C "$WORKDIR" checkout "$BRANCH"
    fi
    cd "$WORKDIR"
    # shellcheck disable=SC1091
    . /etc/profile.d/llvm22.sh
    # shellcheck disable=SC1091
    [ -f /etc/profile.d/rust.sh ] && . /etc/profile.d/rust.sh
    export RUSTFLAGS="-C link-arg=-fuse-ld=lld"
    export LLVM_CONFIG_PATH=llvm-config-22
    cargo build --release --bin intern_bench
fi

echo
echo "== machine =="
lscpu | grep -E '^(Model name|Socket|Core\(s\)|Thread\(s\)|CPU\(s\)|NUMA node\(s\))'
echo
echo "setup done. next: utils/cgo/run_e1.sh"
