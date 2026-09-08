#!/usr/bin/env bash
#===- emit_gpu_kernel.sh - a Vx-shaped kernel through MLIR to PTX --------===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# The pass pipeline that turns an outlined `vx.kernel` body into PTX, verified
# on a kernel shaped like the one #251 exists for: FlashAttention, whose outer
# loop over queries is parallel and whose inner online-softmax loop is not.
#
# This is a spike, not the compiler path. It exists so that wiring it into
# src/dialect/VxLowering.cpp starts from a recipe known to work rather than from
# the several that do not, and so the failures below stay recorded. It needs no
# GPU: `format=isa` stops at PTX text, which is inspectable anywhere.
#
#   ./scripts/emit_gpu_kernel.sh [sm_80]
#
# Three things had to be right, and each was a dead end first:
#
#   1. The loops need `gpu.parallel_loop_mapping` before they can become a
#      launch. `--convert-parallel-loops-to-gpu` on its own is silent -- it does
#      not fail, it just leaves the `scf.parallel` alone, which reads like the
#      pass not existing. `--gpu-map-parallel-loops` supplies the attribute.
#
#   2. A hand-assembled nested pipeline leaves `i64`/`index` casts behind.
#      `convert-cf-to-llvm` is restricted to `builtin.module` and cannot be
#      nested under `gpu.module`, so the block arguments the loop lowering
#      creates never convert, `reconcile-unrealized-casts` has nothing to pair
#      them with, and serialization fails on a `builtin.unrealized_conversion_
#      cast` with no indication of which pass was missing. The composite
#      `--gpu-lower-to-nvvm-pipeline` sequences it correctly.
#
#   3. The default PTX version is 6.0, which does not admit sm_80 -- and that
#      one at least says so.
#
# Also settled: the bare-pointer calling convention
# (`use-bare-ptr-memref-call-conv=true`) cannot be used here. It needs static
# shapes and Vx emits `memref<?x?xf32>` throughout, so `gpu.func` fails to
# legalize. Descriptors work, at the cost of a wide parameter list -- the kernel
# below takes 18 `.param`s for three memrefs and two indices.
#
#===----------------------------------------------------------------------===#

set -euo pipefail

CHIP="${1:-sm_80}"
OUT="${TMPDIR:-/tmp}/vx-gpu-spike"
mkdir -p "$OUT"

command -v mlir-opt >/dev/null 2>&1 || {
  echo "error: mlir-opt not on PATH (source config.local)" >&2
  exit 1
}

# A kernel with FlashAttention's dependence structure: parallel over queries,
# sequential within one. Dynamic memrefs, because that is what Vx emits.
cat > "$OUT/kernel.mlir" <<'EOF'
func.func @vx_npu_kernel_0(%q: memref<?x?xf32>, %k: memref<?x?xf32>,
                           %o: memref<?x?xf32>, %n: index, %d: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    %init = arith.constant 0.0 : f32
    %acc = scf.for %j = %c0 to %d step %c1 iter_args(%a = %init) -> (f32) {
      %qv = memref.load %q[%i, %j] : memref<?x?xf32>
      %kv = memref.load %k[%i, %j] : memref<?x?xf32>
      %m = arith.mulf %qv, %kv : f32
      %s = arith.addf %a, %m : f32
      scf.yield %s : f32
    }
    memref.store %acc, %o[%i, %c0] : memref<?x?xf32>
    scf.reduce
  }
  return
}
EOF

echo "==> lowering for $CHIP"
mlir-opt "$OUT/kernel.mlir" \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=$CHIP cubin-features=+ptx76 cubin-format=isa" \
  -o "$OUT/lowered.mlir"

# The PTX is embedded in a gpu.binary as an escaped string, so it is unescaped
# here rather than read directly.
python3 - "$OUT/lowered.mlir" "$OUT/kernel.ptx" <<'PY'
import sys
# The PTX sits inside a `gpu.binary` as one escaped MLIR string. Both ends
# matter: taking from the opening quote to end-of-file leaves the closing quote
# and the attribute's `>]` in the file, and ptxas rejects that -- "Parsing error
# near '\"'" three lines past the last `}`, which reads like a compiler defect
# and is not one.
def unescape(s, i):
    start = s.rfind('"', 0, i) + 1
    end = s.find('"', start)
    if end < 0:
        sys.exit("unterminated PTX string in the gpu.binary")
    return s[start:end].replace("\\0A", "\n").replace("\\09", "\t")
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
i = s.find(".visible .entry")
if i < 0:
    sys.exit("no .visible .entry in the lowered module -- serialization produced no PTX")
open(dst, "w").write(unescape(s, i))
PY

echo "==> $OUT/kernel.ptx"
grep -E '^\.version|^\.target|\.visible \.entry' "$OUT/kernel.ptx" | sed 's/^/    /'

# What makes it a real kernel rather than a stub: the parallel index became a
# block index, both operands are loaded, the sequential loop survived as a
# branch, and the result is stored.
echo "==> the body"
for want in '%ctaid' 'ld.global' 'bra' 'st.global'; do
  n=$(grep -c -- "$want" "$OUT/kernel.ptx" || true)
  printf '    %-12s %s\n' "$want" "$n"
  [ "$n" -eq 0 ] && { echo "error: no '$want' in the PTX; this is not the loop" >&2; exit 1; }
done
echo "    instructions $(grep -c ';$' "$OUT/kernel.ptx" || true)"
