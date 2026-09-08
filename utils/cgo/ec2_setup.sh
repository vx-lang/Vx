#!/usr/bin/env bash
#===- ec2_setup.sh - Vx Compiler -----------------------------*- bash -*-===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Provision a fresh Ubuntu x86-64 instance for the CGO 2027 E1 measurement
# (#295/#296). Mirrors .github/workflows/ci.yml's dependency set, which is the
# only Linux build recipe this project has that is known to work.
#
#   sudo ./ec2_setup.sh                 # deps only -- the usual case
#   sudo ./ec2_setup.sh --clone <url>   # deps + clone + build
#
# Prefer deps-only, then `utils/cgo/push.sh` from your own machine. It carries
# the working tree over the EC2 keypair, so the instance never holds a git
# credential -- and this is rented, shared-tenant hardware that gets terminated,
# so a key that has been on it should be considered disclosed. `--clone` is for a
# *public HTTPS* URL only; never hand this an `ssh://`/`git@` remote, because
# that means putting a key on the box.
#
# Tested on Ubuntu 24.04 and 26.04. LLVM comes from the distribution when it
# carries version 22, else from apt.llvm.org.
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
    python3-pip numactl hwloc \
    libzstd-dev libxml2-dev libz3-dev zlib1g-dev libtinfo-dev libedit-dev libffi-dev
# Those seven `-dev` packages are what a static LLVM links *against*. Nothing
# names them: `llvm-config --libs` emits `-lzstd -lxml2 -lz3 -lz -ltinfo`, so the
# failure surfaces as `rust-lld: error: unable to find library -lzstd` from
# inside a dependency crate's link step, several minutes into a build, with
# nothing pointing at the missing package. Installed up front instead.
# `perf` ships under a kernel-versioned package that does not always exist for
# the running kernel on a fresh AMI. Non-fatal: the wall-clock ladder is the
# primary measurement and does not need perf, and `run_e1.sh` already reports a
# missing lock profile rather than failing.
apt-get install -y -qq linux-tools-common "linux-tools-$(uname -r)" linux-tools-generic \
    || echo "WARNING: perf tools unavailable for kernel $(uname -r); lock profiling will be skipped"

echo "== LLVM/MLIR 22 =="
# MLIR's C API is not stable across major versions, so 22 is not a floor --
# `mlir-sys = 220.x` is built against it specifically.
#
# Prefer the distribution's own packages when it has them (Ubuntu 26.04 does),
# and fall back to apt.llvm.org otherwise. Fewer moving parts: a third-party apt
# source is one more thing that can lag a new release, and on a machine rented by
# the hour "the repo does not publish for this codename yet" is an expensive way
# to find out.
# `clang-22` is the compiler *driver*, and it is separate from `libclang-22-dev`,
# which is only the library. `build.rs` shells out to `clang++` to compile
# plugin_loader.cpp, so without the driver the build dies in the build script
# with "Failed to execute clang++" -- after cargo has already compiled the
# dependency tree, which is a slow way to learn about a missing package.
LLVM_PKGS="llvm-22-dev libmlir-22-dev mlir-22-tools libpolly-22-dev libclang-22-dev clang-22"
# shellcheck disable=SC2086
if apt-get install -y -qq $LLVM_PKGS 2>/dev/null; then
    echo "   from the distribution's own repositories"
else
    echo "   not in the distro; falling back to apt.llvm.org"
    wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh
    chmod +x /tmp/llvm.sh
    /tmp/llvm.sh 22
    # shellcheck disable=SC2086
    apt-get install -y -qq $LLVM_PKGS
fi

echo 'export PATH=/usr/lib/llvm-22/bin:$PATH' > /etc/profile.d/llvm22.sh
echo 'export LLVM_CONFIG_PATH=llvm-config-22' >> /etc/profile.d/llvm22.sh

echo "== rust =="
# Installed for the *invoking* user, not for root. This script needs root for apt
# and sysctl, but rustup follows $HOME, so installing it here as root would put
# cargo in /root and leave the login user without it -- and building as root then
# leaves a root-owned `target/` that the ordinary user cannot rebuild into and
# `push.sh` cannot manage. The build is not a privileged operation and should not
# run as one.
RUST_USER="${SUDO_USER:-root}"
RUST_HOME="$(getent passwd "$RUST_USER" | cut -d: -f6)"
if ! sudo -u "$RUST_USER" env HOME="$RUST_HOME" sh -c 'command -v cargo >/dev/null 2>&1 || [ -x "$HOME/.cargo/bin/cargo" ]'; then
    echo "   installing rustup for $RUST_USER ($RUST_HOME)"
    sudo -u "$RUST_USER" env HOME="$RUST_HOME" sh -c \
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path"
fi
echo 'export PATH=$HOME/.cargo/bin:$PATH' > /etc/profile.d/rust.sh

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
