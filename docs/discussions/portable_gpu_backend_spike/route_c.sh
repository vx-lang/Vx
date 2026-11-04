#!/bin/bash
# Route C by hand: Vx source -> gpu.module -> structured MLIR -> MSL.
# Run from the repository root. Usage: route_c.sh <prog.vx> <out.metal> [<keep-dir>]
# VXC and VX_OPT name the compiler and the optimizer (default: target/debug/...).
set -euo pipefail
source "${VX_CONFIG:-config.local}"
here=$(cd "$(dirname "$0")" && pwd)
vxc=${VXC:-target/debug/vxc}
vxopt=${VX_OPT:-target/debug/vx-opt}
base=${3:-$(mktemp -d)}
mkdir -p "$base"
# The NVIDIA machine file only gives the kernel its `levels:`. The gpu.module is vendor-free.
"$vxc" "$1" --host default --machine fleet/h100-sxm.vx --emit-mlir > "$base/a.mlir" 2>/dev/null
"$vxopt" --convert-vx-to-standard "$base/a.mlir" -o "$base/b.mlir" 2>/dev/null
awk '/^  gpu.module @vx_kernels/{p=1} p{print} p && /^  }$/{exit}' "$base/b.mlir" > "$base/body"
{ echo 'module attributes {gpu.container_module} {'; cat "$base/body"; echo '}'; } > "$base/k.mlir"
# lift-cf-to-scf walks only func.func, so each gpu.func is a func.func while it runs.
sed -E 's/gpu\.func (@[a-z_0-9]+)\((.*)\) kernel attributes \{/func.func \1(\2) attributes {vx.was_kernel, /; s/gpu\.return/return/' "$base/k.mlir" > "$base/f.mlir"
mlir-opt "$base/f.mlir" --pass-pipeline='builtin.module(gpu.module(func.func(mem2reg,canonicalize),lift-cf-to-scf,func.func(canonicalize,cse)))' -o "$base/s.mlir"
sed -E 's/func\.func (@[a-z_0-9]+)\((.*)\) attributes \{(.*), vx\.was_kernel\}/gpu.func \1(\2) kernel attributes {\3}/; s/^      return$/      gpu.return/' "$base/s.mlir" > "$base/g.mlir"
python3 "$here/prep_kernel.py" "$base/g.mlir" "$base/p.mlir"
mlir-opt "$base/p.mlir" --canonicalize --cse -o "$base/final.mlir"
python3 "$here/emit_msl.py" "$base/final.mlir" "$2"
