#!/usr/bin/env bash
#===- flash_kernel_to_ptx.sh - the real FlashAttention region, to PTX ----===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Takes the device region the compiler actually emits for
# tests/backend/pass/flash_attention_placed_verified.vx and drives it to sm_80
# PTX, so what #251 has left to do is measured against real output rather than
# against a kernel written to be easy.
#
# scripts/emit_gpu_kernel.sh established the pass pipeline on a hand-written
# kernel: `scf.parallel`, dynamic memrefs, no calls. This one starts from what
# Vx emits, which is none of those things, and the difference is the work:
#
#   1. The region is raw CFG (`cf.br`/`cf.cond_br`) with every induction
#      variable and accumulator in a `memref.alloca`. `--mem2reg` promotes all
#      21 allocas and `--lift-cf-to-scf` removes every `cf` op -- but the loops
#      come back as `scf.while`, not `scf.for`, because a lifter cannot see a
#      trip count that was in memory. This build has no while-to-for pass, so
#      raising is a dead end for reaching `scf.parallel`. It does not matter for
#      *correctness*: NVVM lowers a CFG perfectly well, and this script keeps
#      the CFG. It matters for parallelism, which is a later step.
#
#   2. `.exp()` lowers to `func.call @expf` -- host libm, which a kernel cannot
#      call. Substituted here for `math.exp`; the compiler must emit that
#      directly for a device region.
#
#   3. What survives to PTX names two device-side externals, and both are
#      gaps rather than details:
#
#        __nv_expf   libdevice. The module will not load without it linked, so
#                    either the pipeline links libdevice or `math.exp` has to
#                    become `ex2.approx.f32` (exp x = ex2 (x * log2 e)).
#        malloc      device-side heap. This is `Tensor<f32>([1,16])` declared
#                    *inside* the region -- `ts`, the score tile -- lowering to
#                    `memref.alloc`. Per-thread scratch belongs in `alloca` or
#                    in shared memory; a heap call per launch is not viable.
#
# The kernel is sequential: one thread runs all 32 queries. Mapping the outer
# loop across threads is a separate step, and deliberately not smuggled in here
# -- this script answers "does our IR reach PTX", and it does.
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

echo "==> compiling $SRC"
"$VXC" "$SRC" --action emit-mlir 2>&1 \
  | grep -vE '^Warning|^  help|^Expanding|^\[' > "$OUT/module.mlir"

# Lift the region out of `vx.spawn` into a `gpu.func`. Its free values are the
# captures and become the kernel's parameters -- computed rather than assumed,
# since a capture missed here shows up as an unrelated parse error later.
echo "==> lifting the device region"
python3 - "$OUT" <<'PY'
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])
lines = (out / "module.mlir").read_text().splitlines()

start = next((i for i, l in enumerate(lines) if l.strip().startswith("vx.spawn")), None)
if start is None:
    sys.exit("no vx.spawn in the emitted module -- did the placement change?")
depth, end = 0, None
for i in range(start, len(lines)):
    depth += lines[i].count("{") - lines[i].count("}")
    if depth == 0:
        end = i
        break
if end is None:
    sys.exit("unterminated vx.spawn region")
body = lines[start + 1:end]

defined = set()
for ln in body:
    m = re.match(r'\s*(%[\w$.]+(?:\s*,\s*%[\w$.]+)*)\s*=', ln)
    if m:
        defined.update(re.findall(r'%[\w$.]+', m.group(1)))
free = []
for ln in body:
    for u in re.findall(r'%[\w$.]+', ln):
        if u not in defined and u not in free:
            free.append(u)

# Each free value's type, read from its defining op above the region.
def type_of(name):
    pat = re.compile(re.escape(name) + r'\s*=.*: .*-> (\S+)$')
    for ln in lines[:start]:
        if ln.strip().startswith(name + " ="):
            m = re.search(r'->\s*(\S+)\s*$', ln) or re.search(r':\s*(\S+)\s*$', ln)
            if m:
                return m.group(1)
    return None

params, consts = [], []
for name in free:
    ty = type_of(name)
    if ty is None:
        sys.exit(f"cannot type the captured value {name}")
    if ty.startswith("memref"):
        params.append((name, ty))
    else:
        # A captured scalar constant (the softmax scale); rematerialize it in
        # the kernel rather than passing it, which is what a real emission
        # would do too.
        defn = next(l for l in lines[:start] if l.strip().startswith(name + " ="))
        consts.append(defn.strip())

sig = ", ".join(f"{n}: {t}" for n, t in params)
text = "\n".join(body)
# Host libm is not callable from a kernel.
text = re.sub(r'%(\d+) = func\.call @f32\$exp\((%[\w$.]+)\) : \(f32\) -> f32',
              r'%\1 = math.exp \2 : f32', text)
text = re.sub(r'\bvx\.yield\b', 'gpu.return', text)

(out / "kernel.mlir").write_text(
    "gpu.module @vx_kernels {\n"
    f"  gpu.func @vx_flash_kernel({sig}) kernel {{\n"
    + "".join("    " + c + "\n" for c in consts)
    + text + "\n  }\n}\n")
print(f"    captures: {len(params)} memref(s), {len(consts)} rematerialized constant(s)")
print(f"    kernel body: {len(body)} lines")
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
# about it is how a kernel that cannot load gets called working.
echo "==> device-side externals"
EXTERNS=$(grep -oE '^\.extern \.func .*\) [a-zA-Z_][a-zA-Z_0-9]*' "$PTX" \
          | awk '{print $NF}' | sort -u)
echo "$EXTERNS" | sed 's/^/    /'
UNEXPECTED=$(echo "$EXTERNS" | grep -vxE '__nv_expf|malloc' || true)
if [ -n "$UNEXPECTED" ]; then
  echo "  FAILURE: an external this script does not account for:" >&2
  echo "$UNEXPECTED" | sed 's/^/    /' >&2
  echo "  Either the kernel gained a dependency or the emission changed; the" >&2
  echo "  header's gap list is now wrong and needs updating with it." >&2
  exit 1
fi
echo "    (both known: libdevice exp, and the in-region Tensor's heap alloc)"
