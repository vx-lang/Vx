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
# It makes no edits. The compiler emits the `gpu.module` now (#251); this lifts
# it out and drives it, so what is measured here is what the compiler produced.
# That was the remaining work as of this script's first version, and the header
# said so; it is done.
#
# The compiler now runs this pipeline itself, too, and carries the PTX in the
# dispatch payload -- so this script is no longer the only way to get one. It is
# still worth keeping, and worth running: it is an independent transcription of
# the same passes, and the two agreeing byte for byte is what says the in-tree
# one is right. They did, at 11253 bytes. tests/integration_test/
# device_image_test.rs is the CI-runnable half (it needs no mlir-opt).
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
#   The in-region `Tensor<f32, [1,16]>::uninit()` -- `ts`, the score tile -- lowered to
#   `memref.alloc`, a device-side `malloc` call per launch for 64 bytes of
#   scratch. The pass now puts entry-block scratch on the stack.
#
#   `math` ops lowered to libdevice calls -- `__nv_expf`, and `__nv_sqrtf` and
#   `__nv_fabsf` too, which have native PTX instructions. Linking
#   libdevice.10.bc resolves and inlines them: `math.exp` becomes a single
#   `ex2.approx.f32`, and the module has no externals at all. Set VX_LIBDEVICE
#   or have a CUDA toolkit installed; the file is architecture-independent
#   bitcode, so it does not have to come from this machine.
#
# What is left:
#
#   the shipping  a fifth wire message. A worker cannot be sent a kernel today,
#                which is why a non-matmul region is refused rather than run.
#   the launch   `cuLaunchKernel`, and 28 `.param`s to marshal for 4 memrefs.
#   the result   this program's output is a buffer the caller passed in, which
#                crosses back the way any operand does. A region that returns a
#                slot instead has its result named by a descriptor minted on the
#                far side, naming memory there -- that one is a design question,
#                not a mechanism.
#   parallelism  one thread runs all 32 queries. Deliberately not smuggled in
#               here: this script answers "does our kernel reach PTX", and it
#               does.
#
#   ./scripts/flash_kernel_to_ptx.sh [sm_80]
#
# Needs no GPU: `format=isa` stops at PTX text, and libdevice is bitcode.
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

echo "==> taking the compiler's gpu.func"
# This used to build the `gpu.func` here, from the `vx.kernel` above it, with a
# text transform -- and the header of this file said that step *was* the
# remaining work. The compiler emits it now (#251), so this only lifts it out.
# Which makes the script an actual test of what the compiler produces rather
# than of what this script could produce from it: if the emission regresses,
# there is no `gpu.module` to find and this stops.
python3 - "$OUT" <<'PY'
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])
lines = (out / "lowered.mlir").read_text().splitlines()

start = next((i for i, l in enumerate(lines)
              if l.lstrip().startswith("gpu.module")), None)
if start is None:
    sys.exit("no gpu.module -- the compiler did not emit a device kernel "
             "(convert-vx-to-standard, materializeGpuKernels)")
depth, end = 0, None
for i in range(start, len(lines)):
    depth += lines[i].count("{") - lines[i].count("}")
    if depth == 0:
        end = i
        break
if end is None:
    sys.exit("unterminated gpu.module")

# Dedented, so it parses as a top-level op rather than needing its enclosing
# builtin.module and the host code inside it -- which mlir-opt cannot parse,
# the vx dialect not being registered there.
block = lines[start:end + 1]
pad = len(block[0]) - len(block[0].lstrip())
(out / "kernel.mlir").write_text(
    "\n".join(l[pad:] if l.startswith(" " * pad) else l for l in block) + "\n")

names = re.findall(r'gpu\.func\s+@([\w$.]+)\((.*?)\)', "\n".join(block))
for name, sig in names:
    nargs = len([a for a in sig.split(",") if a.strip()])
    print(f"    @{name}: {nargs} arguments")
print(f"    {end - start + 1} lines of device module")
PY

mlir-opt "$OUT/kernel.mlir" -o /dev/null

# libdevice, if this machine has it.
#
# Every `math` op lowers to a libdevice call under the NVVM conversion --
# `__nv_expf`, and `__nv_sqrtf` and `__nv_fabsf` too, which have native PTX
# instructions. Decomposing `exp` into `exp2` does not avoid it. Linking the
# bitcode does, and it inlines: `math.exp` becomes one `ex2.approx.f32`.
#
# The composite `--gpu-lower-to-nvvm-pipeline` has no option for it, so linking
# means running its steps separately with `nvvm-attach-target l=...`. Without the
# file the kernel still compiles and the external is reported, because a run on a
# machine with no CUDA toolkit should say what is missing rather than fail.
if [ -z "${VX_LIBDEVICE:-}" ]; then
  for c in "${CUDA_HOME:-/usr/local/cuda}" /usr/local/cuda-*; do
    [ -f "$c/nvvm/libdevice/libdevice.10.bc" ] && {
      VX_LIBDEVICE="$c/nvvm/libdevice/libdevice.10.bc"; break; }
  done
fi

echo "==> lowering for $CHIP"
if [ -n "${VX_LIBDEVICE:-}" ] && [ -f "$VX_LIBDEVICE" ]; then
  echo "    linking $VX_LIBDEVICE"
  # --convert-scf-to-cf first: the attention fallback nest (Vx#378) is the one
  # kernel body emitted as structured loops rather than raw CFG, and the NVVM
  # conversion has no scf patterns. Mirrors deviceImageOf in VxLowering.cpp.
  mlir-opt "$OUT/kernel.mlir" \
    --nvvm-attach-target="chip=$CHIP features=+ptx76 l=$VX_LIBDEVICE" \
    --convert-scf-to-cf \
    --convert-gpu-to-nvvm --convert-arith-to-llvm --convert-math-to-llvm \
    --convert-vector-to-llvm \
    --gpu-to-llvm --reconcile-unrealized-casts \
    --gpu-module-to-binary="format=isa" \
    -o "$OUT/nvvm.mlir"
  LINKED=1
else
  echo "    no libdevice (set VX_LIBDEVICE to link it); math stays external"
  mlir-opt "$OUT/kernel.mlir" \
    --gpu-lower-to-nvvm-pipeline="cubin-chip=$CHIP cubin-features=+ptx76 cubin-format=isa" \
    -o "$OUT/nvvm.mlir"
  LINKED=0
fi

python3 - "$OUT" <<'PY'
import sys, pathlib
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
out = pathlib.Path(sys.argv[1])
s = (out / "nvvm.mlir").read_text()
i = s.find(".visible .entry")
if i < 0:
    sys.exit("no .visible .entry -- serialization produced no PTX")
(out / "kernel.ptx").write_text(unescape(s, i))
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
# `|| true`: no externals is the *good* outcome, and grep exits 1 on no match,
# which under `set -e` killed the script exactly when it had succeeded.
EXTERNS=$(grep -oE '^\.extern \.func .*\) [a-zA-Z_][a-zA-Z_0-9]*' "$PTX" \
          | awk '{print $NF}' | sort -u || true)
if [ -z "$EXTERNS" ]; then
  echo "    none -- the module is self-contained and could be loaded as it is"
  [ "$LINKED" = 1 ] && printf '    %-14s %s\n' ex2.approx "$(grep -c 'ex2.approx' "$PTX")"
else
  echo "$EXTERNS" | sed 's/^/    /'
  if [ "$LINKED" = 1 ]; then
    echo "  FAILURE: libdevice was linked and these are still unresolved." >&2
    exit 1
  fi
  UNEXPECTED=$(echo "$EXTERNS" | grep -vxE '__nv_expf' || true)
  if [ -n "$UNEXPECTED" ]; then
    echo "  FAILURE: an external this script does not account for:" >&2
    echo "$UNEXPECTED" | sed 's/^/    /' >&2
    echo "  Either the kernel gained a dependency or the emission changed; the" >&2
    echo "  header's gap list is now wrong and needs updating with it." >&2
    exit 1
  fi
  echo "    (libdevice; link it and this list goes empty)"
fi

# If a CUDA toolkit is here, assemble it. `mlir-opt` emitting PTX text says the
# pipeline ran; `ptxas` accepting it says the text is a program, and its report
# is the first real measurement of the kernel -- registers, spills, and the
# stack frame, which should be exactly the region's own scratch.
#
# This is also the check that caught the extractor above: the PTX lives inside
# the `gpu.binary` as an escaped string, and reading from its opening quote to
# end-of-file left the closing quote and the attribute's `>]` in the file.
# ptxas said "Parsing error near '\"'" three lines past the last `}`, which
# reads like a compiler defect and was a defect in this script.
# The trailing `|| true` is not decoration: not finding ptxas is the ordinary
# case on a machine with no toolkit, and without it `set -e` aborts the script
# on the lookup itself.
PTXAS="${VX_PTXAS:-$(command -v ptxas 2>/dev/null || ls /usr/local/cuda-*/bin/ptxas 2>/dev/null | head -1 || true)}"
if [ -n "$PTXAS" ] && [ -x "$PTXAS" ]; then
  echo "==> assembling with $PTXAS"
  if "$PTXAS" -arch="$CHIP" -O3 -v "$PTX" -o "$OUT/kernel.cubin" 2>&1 \
       | grep -E 'registers|stack frame|spill' | sed 's/^ptxas info[[:space:]]*:[[:space:]]*/    /;s/^[[:space:]]*/    /'; then
    echo "    -> $OUT/kernel.cubin"
  else
    echo "  FAILURE: ptxas rejected the PTX" >&2
    exit 1
  fi
else
  echo "==> no ptxas here; set VX_PTXAS to assemble (needs a toolkit, not a GPU)"
fi
