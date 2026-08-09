#!/usr/bin/env bash
#===- setup_linux.sh - Vx Compiler ----------------------------*- bash -*-===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Provision an x86_64 Ubuntu box to build and test Vx. Written for the build
# machine the GPU campaign uses (#321), where artifacts are built and then
# copied to a rented GPU host -- see docs/discussions/implementation_plans/
# gpu_disaggregated_inference.md for why sources do not travel.
#
# Verified on Ubuntu 24.04 (noble). After this, run ./setup.sh to generate
# config.local, then `source config.local`.
#
#===----------------------------------------------------------------------===#
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# LLVM/MLIR version, pinned by mlir-sys in Cargo.toml (220.x means LLVM 22).
LLVM_VERSION="${LLVM_VERSION:-22}"

echo "=== [1/3] build dependencies ==="
# z3 is the *binary*, not just libz3-dev. Seam verification shells out to it
# (Command::new("z3") in src/hir/seam.rs) to discharge `relaxed` transfer-edge
# obligations in QF_BV. Without it four test suites fail in ways that read as
# compiler regressions -- a missing W1027, an expected-failure that passes --
# rather than as a missing tool.
#
# libffi is needed by the dispatch runtime, which calls outlined kernels through
# their MLIR C-interface (runtime/host_dispatch.cpp).
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends \
    build-essential ninja-build cmake pkg-config git curl ca-certificates gnupg \
    libffi-dev libz3-dev z3 zlib1g-dev libzstd-dev libedit-dev libxml2-dev

echo "=== [2/3] LLVM/MLIR ${LLVM_VERSION} ==="
# Ubuntu's own repositories lag the version mlir-sys pins, so take it from
# apt.llvm.org. libmlir-*-dev ships the MLIR C API that melior links against.
if ! command -v "llvm-config-${LLVM_VERSION}" >/dev/null 2>&1; then
    codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | sudo gpg --dearmor -o /usr/share/keyrings/llvm-snapshot.gpg
    echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/${codename}/ llvm-toolchain-${codename}-${LLVM_VERSION} main" \
        | sudo tee "/etc/apt/sources.list.d/llvm-${LLVM_VERSION}.list" >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq --no-install-recommends \
        "llvm-${LLVM_VERSION}" "llvm-${LLVM_VERSION}-dev" "llvm-${LLVM_VERSION}-tools" \
        "clang-${LLVM_VERSION}" "lld-${LLVM_VERSION}" \
        "libmlir-${LLVM_VERSION}-dev" "mlir-${LLVM_VERSION}-tools" \
        "libpolly-${LLVM_VERSION}-dev"
fi

echo "=== [3/3] Rust ==="
if ! command -v rustc >/dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/rustc" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path
fi

echo
echo "Provisioned:"
echo "  llvm-config : $("llvm-config-${LLVM_VERSION}" --version)"
echo "  z3          : $(z3 --version)"
echo "  rustc       : $("$HOME/.cargo/bin/rustc" --version 2>/dev/null || rustc --version)"
echo
echo "The LLVM tools are under /usr/lib/llvm-${LLVM_VERSION}/bin, which must come"
echo "first on PATH so the unsuffixed names resolve there. ./setup.sh writes that"
echo "into config.local; source it before building:"
echo
echo "  ./setup.sh && source config.local && cargo build --release"
