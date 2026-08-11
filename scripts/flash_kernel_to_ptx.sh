#!/usr/bin/env bash
#===- flash_kernel_to_ptx.sh - the real FlashAttention kernel, to PTX ----===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Takes the kernel `convert-vx-to-standard` outlines for
# tests/backend/pass/flash_attention_placed_verified.vx and drives it to sm_80
# PTX, so what #251 has left to do is measured against what the compiler emits
# rather than against a kernel written to be easy.
#
# scripts/emit_gpu_kernel.sh established the pass pipeline on such a kernel:
# `scf.parallel`, dynamic memrefs, no calls. What Vx outlines is none of those.
# This starts from `vx.kernel`, after the pass, so every compiler-side fix shows
# up here on the next run -- which is the point, and why it reads the lowered
# module rather than `--action emit-mlir`.
#
# The only edit it makes is structural: `vx.kernel` becomes `gpu.func`, because
# nothing in the compiler does that yet. That step *is* the remaining work.
#
# What this has settled so far:
#
#   The region is raw CFG (`cf.br`/`cf.cond_br`) with every induction variable
#   and accumulator in a `memref.alloca`. `--mem2reg` promotes all 21 and
#   `--lift-cf-to-scf` removes every `cf` op, but the loops come back as
#   `scf.while`, not `scf.for` -- a lifter cannot recover a trip count that was
#   in memory, and this build has no while-to-for pass. So raising is a dead end
#   for reaching `scf.parallel`. It costs nothing for correctness: NVVM lowers a
#   CFG fine and the kernel below keeps its branches. It costs parallelism,
#   which is a separate step.
#
#   `.exp()` lowered to `func.call @expf` -- host libm, which a kernel cannot
#   call. The pass now rewrites it to `math.exp`, which lowers to libm on the
#   host and to a device intrinsic here.
#
#   The in-region `Tensor<f32>([1,16])` -- `ts`, the score tile -- lowered to
#   `memref.alloc`, a device-side `malloc` call per launch for 64 bytes of
#   scratch. The pass now puts entry-block scratch on the stack.
#
# What is left, and what the numbers below are for:
#
#   __nv_expf   `math.exp` reaches libdevice. The module will not load unless
#               the pipeline links it, or `math.exp` becomes `ex2.approx.f32`
#               (exp x = ex2 (x * log2 e)). This is the last external.
#   the wrapper  nothing in the compiler emits `gpu.func` or runs this pipeline.
#   the launch   `cuLaunchKernel`, and 28 `.param`s to marshal for 4 memrefs.
#   parallelism  one thread runs all 32 queries. Deliberately not smuggled in
#               here: this script answers "does our kernel reach PTX", and it
#               does.
#
#   ./scripts/flash_kernel_to_ptx.sh [sm_80]
#
# Needs no GPU: `cubin-format=isa` stops at PTX text.
#
#===----------------------------------------------------------------------===#

set -euo pipefail

CHIP="${1:-sm_80}"
SRC="tests/backend/pass/flash_attention_placed_verified.vx"
OUT="${TMPDIR:-/tmp}/vx-flash-ptx"
mkdir -p "$OUT"

command -v mlir-opt >/dev/null 2>&1 || {
  echo "error: mlir-opt not on PATH (source config.local)" >&2
  exit 1
}
VXC="${VXC:-./target/release/vxc}"
[ -x "$VXC" ] || { echo "error: no $VXC (cargo build --release)" >&2; exit 1; }
[ -f "$SRC" ] || { echo "error: run from the repo root; no $SRC" >&2; exit 1; }

echo "==> outlining the kernel from $SRC"
"$VXC" "$SRC" --emit-mlir \
  -X mlir=--pass-pipeline="builtin.module(convert-vx-to-standard)" 2>&1 \
  | grep -vE '^Warning|^  help|^Expanding|^\[' > "$OUT/lowered.mlir"

grep -q 'vx.kernel' "$OUT/lowered.mlir" || {
  echo "error: no vx.kernel in the lowered module -- did the placement change?" >&2
  exit 1
}

echo "==> vx.kernel -> gpu.func"
python3 - "$OUT" <<'PY'
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])
lines = (out / "lowered.mlir").read_text().splitlines()

start = next((i for i, l in enumerate(lines) if l.lstrip().startswith("vx.kernel")), None)
if start is None:
    sys.exit("no vx.kernel")
depth, end = 0, None
for i in range(start, len(lines)):
    depth += lines[i].count("{") - lines[i].count("}")
    if depth == 0:
        end = i
        break
if end is None:
    sys.exit("unterminated vx.kernel region")

name = re.search(r'vx\.kernel\s+@([\w$.]+)', lines[start]).group(1)
# The captures are already the entry block's arguments, so the signature is a
# transcription rather than an analysis.
sig_line = lines[start + 1].strip()
m = re.match(r'\^bb0\((.*)\):$', sig_line)
if not m:
    sys.exit(f"expected an entry block signature, got: {sig_line}")
sig = m.group(1)
body = "\n".join(lines[start + 2:end])
body = re.sub(r'\bvx\.return\b', 'gpu.return', body)

(out / "kernel.mlir").write_text(
    "gpu.module @vx_kernels {\n"
    f"  gpu.func @{name}({sig}) kernel {{\n{body}\n  }}\n}}\n")
nargs = len([a for a in sig.split(",") if a.strip()])
print(f"    @{name}: {nargs} arguments, {end - start - 2} lines")
PY

mlir-opt "$OUT/kernel.mlir" -o /dev/null

echo "==> lowering for $CHIP"
mlir-opt "$OUT/kernel.mlir" \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=$CHIP cubin-features=+ptx76 cubin-format=isa" \
  -o "$OUT/nvvm.mlir"

python3 - "$OUT" <<'PY'
import sys, pathlib
out = pathlib.Path(sys.argv[1])
s = (out / "nvvm.mlir").read_text()
i = s.find(".visible .entry")
if i < 0:
    sys.exit("no .visible .entry -- serialization produced no PTX")
(out / "kernel.ptx").write_text(
    s[max(0, s.rfind('"', 0, i)) + 1:].replace("\\0A", "\n").replace("\\09", "\t"))
PY

PTX="$OUT/kernel.ptx"
echo "==> $PTX"
grep -E '^\.version|^\.target|\.visible \.entry' "$PTX" | sed 's/^/    /'

# Parameters of the entry itself. Counting every line that says `.param` counts
# each `ld.param` use and each extern's return slot as well; here that turned 28
# into 36. The number is the point of the measurement -- 28 is 4 memref
# descriptors at 7 fields each -- so it has to be the signature and nothing else.
ENTRY_PARAMS=$(awk '/\.visible \.entry/,/^\)/' "$PTX" | grep -c '\.param')
printf '    %-14s %s\n' \
  entry.params "$ENTRY_PARAMS" \
  ld.global    "$(grep -c 'ld.global' "$PTX")" \
  st.global    "$(grep -c 'st.global' "$PTX")" \
  branches     "$(grep -c 'bra' "$PTX")" \
  blocks       "$(grep -cE '^\$L__BB' "$PTX")" \
  instructions "$(grep -c ';$' "$PTX")"

# It must be the loop, not a stub: operands loaded, result stored, branches kept.
for want in 'ld.global' 'st.global' 'bra'; do
  [ "$(grep -c -- "$want" "$PTX")" -gt 0 ] || {
    echo "error: no '$want' in the PTX; this is not the kernel" >&2; exit 1; }
done

# Externals are the gap list. A new one appearing is a new gap, and silence
# about it is how a kernel that cannot load gets reported as working.
echo "==> device-side externals"
EXTERNS=$(grep -oE '^\.extern \.func .*\) [a-zA-Z_][a-zA-Z_0-9]*' "$PTX" \
          | awk '{print $NF}' | sort -u)
if [ -z "$EXTERNS" ]; then
  echo "    none -- the kernel is self-contained"
else
  echo "$EXTERNS" | sed 's/^/    /'
  UNEXPECTED=$(echo "$EXTERNS" | grep -vxE '__nv_expf' || true)
  if [ -n "$UNEXPECTED" ]; then
    echo "  FAILURE: an external this script does not account for:" >&2
    echo "$UNEXPECTED" | sed 's/^/    /' >&2
    echo "  Either the kernel gained a dependency or the emission changed; the" >&2
    echo "  header's gap list is now wrong and needs updating with it." >&2
    exit 1
  fi
  echo "    (libdevice; see the header)"
fi
