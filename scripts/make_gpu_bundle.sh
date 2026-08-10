#!/usr/bin/env bash
#===- make_gpu_bundle.sh - Assemble what a GPU pod needs -----------------===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Packs a compiler build and the few runtime sources a GPU box needs into one
# archive, to be copied to a rented pod and deleted after the run.
#
# Run this on the build box, from a checkout, after `cargo build --release`.
# Nothing here reads the git history and none of it goes into the archive: the
# pod gets build outputs, four runtime files and whichever programs are named.
#
# Usage:
#   scripts/make_gpu_bundle.sh [-o out.tar.gz] [--with <path>] program.vx ...
#
# Programs and `--with` paths keep their repository-relative location inside the
# bundle, because imports resolve by path.
#
# Then, on the pod:
#   tar -xzmf vx-gpu-bundle.tar.gz && cd vx-gpu-bundle
#   ./setup_gpu_pod.sh
#   source env.sh && ./vxc gpu_matmul_roles.vx --run
#   cd .. && rm -rf vx-gpu-bundle vx-gpu-bundle.tar.gz
#
#===----------------------------------------------------------------------===#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="vx-gpu-bundle.tar.gz"
PROGRAMS=()

EXTRAS=()

while [ $# -gt 0 ]; do
  case "$1" in
    -o) OUT="$2"; shift 2 ;;
    --with) EXTRAS+=("$2"); shift 2 ;;
    *)  PROGRAMS+=("$1"); shift ;;
  esac
done

if [ ${#PROGRAMS[@]} -eq 0 ]; then
  echo "usage: $0 [-o out.tar.gz] program.vx [program.vx ...]" >&2
  exit 1
fi

VXC="$REPO_ROOT/target/release/vxc"
if [ ! -x "$VXC" ]; then
  echo "error: no compiler at $VXC -- run 'cargo build --release' first" >&2
  exit 1
fi

# A compiler built on macOS will not run on a Linux pod. Say so here rather
# than letting it fail there, where the failure costs rental time.
case "$(uname -s)" in
  Linux) ;;
  *) echo "warning: building the bundle on $(uname -s); a pod runs Linux/x86_64" >&2 ;;
esac

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
BUNDLE="$STAGE/vx-gpu-bundle"
mkdir -p "$BUNDLE/runtime" "$BUNDLE/include" "$BUNDLE/target/release"

cp "$VXC" "$BUNDLE/vxc"

# The JIT links every program against the Vx standard library, by a path
# relative to the compiler's working directory. It is a build output rather
# than a source file, so it has to travel with the compiler: without it the
# link fails on the pod, after setup has appeared to succeed.
STD_CORE="$REPO_ROOT/target/release/libvx_std_core.so"
if [ ! -f "$STD_CORE" ]; then
  echo "error: no $STD_CORE -- run 'cargo build --release --workspace' first" >&2
  exit 1
fi
cp "$STD_CORE" "$BUNDLE/target/release/"
cp "$REPO_ROOT/runtime/cuda_dispatch.cpp" \
   "$REPO_ROOT/runtime/vx_dispatch_plan.h" \
   "$REPO_ROOT/runtime/vx_host_call.h" "$BUNDLE/runtime/"
cp "$REPO_ROOT/include/vx_hardware_runtime.h" "$BUNDLE/include/"
cp "$REPO_ROOT/scripts/setup_gpu_pod.sh" "$BUNDLE/"

# Programs keep their repository-relative path, and the standard library comes
# along. Module imports resolve against `stdlib/std`, `stdlib` and the working
# directory (src/module_loader.rs), so `import tests::modules::llama_rt` only
# finds its module if the layout is preserved -- flattening everything into one
# directory compiles here and fails there.
cp -R "$REPO_ROOT/stdlib" "$BUNDLE/stdlib"

for prog in "${PROGRAMS[@]}"; do
  rel="${prog#"$REPO_ROOT"/}"
  mkdir -p "$BUNDLE/$(dirname "$rel")"
  cp "$prog" "$BUNDLE/$rel"
done

# Anything else the run needs at its own path: imported modules, machine models,
# model weights. Data rather than source -- a checkpoint is not code, and the
# pod cannot run a model it does not have.
for extra in "${EXTRAS[@]}"; do
  rel="${extra#"$REPO_ROOT"/}"
  mkdir -p "$BUNDLE/$(dirname "$rel")"
  cp -R "$extra" "$BUNDLE/$rel"
done

# COPYFILE_DISABLE keeps macOS from writing ._* resource-fork members, which
# arrive on Linux as stray files that look like sources to anything scanning a
# directory.
COPYFILE_DISABLE=1 tar -czf "$OUT" -C "$STAGE" vx-gpu-bundle

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "Contents:"
tar -tzf "$OUT" | sed 's/^/  /'
