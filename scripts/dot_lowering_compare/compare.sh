#!/usr/bin/env bash
# Compile both dot forms to PTX through the same pipeline deviceImageOf uses,
# then report the numbers that actually differ. Needs: mlir-opt (source config.local).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CHIP="${1:-sm_80}"

command -v mlir-opt >/dev/null || { echo "mlir-opt not on PATH (source config.local)" >&2; exit 1; }

mlir-opt "$HERE/two_dots.mlir" \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=$CHIP cubin-features=+ptx76 cubin-format=isa" \
  -o "$HERE/nvvm.mlir"

# The PTX rides inside gpu.binary as an escaped string; \0A is newline, \09 is tab.
python3 - "$HERE/nvvm.mlir" "$HERE/both.ptx" <<'PY'
import sys, re
s = open(sys.argv[1]).read()
blob = max(re.findall(r'"((?:[^"\\]|\\.)*)"', s, re.S), key=len)
b = blob.encode().decode('unicode_escape').encode('latin-1', 'ignore')
b = b.replace(b'\x00A', b'\n').replace(b'\x009', b'\t').replace(b'\x00', b'')
open(sys.argv[2], 'wb').write(b)
PY

# Split the PTX into one file per .visible .entry so each kernel is counted alone.
python3 - "$HERE/both.ptx" "$HERE" <<'PY'
import sys, re, os
txt = open(sys.argv[1]).read(); out = sys.argv[2]
parts = re.split(r'(?=\.visible \.entry )', txt)
for p in parts:
    m = re.match(r'\.visible \.entry (\w+)', p)
    if m:
        open(os.path.join(out, m.group(1) + '.ptx'), 'w').write(p)
        print("wrote", m.group(1) + ".ptx")
PY

# Longest chain of dependent scalar FP adds/fmas: the latency the kernel cannot hide.
chain() {
  python3 - "$1" <<'PY'
import sys, re
dep, best = {}, 0
for line in open(sys.argv[1]):
    m = re.match(r'\s*(?:add|fma|mul)\.rn\.f32\s+(%\w+),\s*(.+);', line)
    if not m: continue
    dst, src = m.group(1), re.findall(r'%\w+', m.group(2))
    dep[dst] = 1 + max([dep.get(s, 0) for s in src] or [0])
    best = max(best, dep[dst])
print(best)
PY
}

printf "\n%-22s %10s %10s\n" "" "dot_wide" "dot_chunked"
printf "%-22s %10s %10s\n" "----------------------" "----------" "-----------"
for metric in "ld.global.v4:ld\.global\.v4" "mul.rn.f32:mul\.rn\.f32" \
              "add.rn.f32:add\.rn\.f32" "fma.rn.f32:fma\.rn\.f32"; do
  name="${metric%%:*}"; pat="${metric#*:}"
  w=$(grep -c "$pat" "$HERE/dot_wide.ptx" || true)
  c=$(grep -c "$pat" "$HERE/dot_chunked.ptx" || true)
  printf "%-22s %10s %10s\n" "$name" "$w" "$c"
done
# NB: match inside %r<...> -- a naive [0-9]+ grabs the "32" out of ".b32" first.
wr=$(grep -oE '%r<[0-9]+>' "$HERE/dot_wide.ptx" | grep -oE '[0-9]+' | head -1)
cr=$(grep -oE '%r<[0-9]+>' "$HERE/dot_chunked.ptx" | grep -oE '[0-9]+' | head -1)
printf "%-22s %10s %10s\n" "b32 registers" "$wr" "$cr"
printf "%-22s %10s %10s\n" "dependent FP chain" "$(chain "$HERE/dot_wide.ptx")" "$(chain "$HERE/dot_chunked.ptx")"
echo
echo "PTX per kernel: $HERE/dot_wide.ptx  $HERE/dot_chunked.ptx"
