#!/bin/bash
# Route A by hand, as far as this machine goes: the structured, flattened kernel from route_c.sh
# -> shader SPIR-V. Naga would take it from there. Usage: route_a.sh <final.mlir> <out.spv>
set -euo pipefail
source "${VX_CONFIG:-config.local}"
d=$(mktemp -d)
python3 - "$1" "$d/t.mlir" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
s = re.sub(r'gpu\.module @vx_kernels attributes \{[^}]*\}', 'gpu.module @vx_kernels attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader, Int64], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>}', s)
s = re.sub(r'(gpu\.func @\w+\(.*?\) kernel attributes \{)', r'\1spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [256, 1, 1]>, ', s)
open(sys.argv[2], 'w').write(s)
PY
mlir-opt "$d/t.mlir" --convert-gpu-to-spirv --pass-pipeline='builtin.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce,canonicalize))' -o "$d/spv.mlir" 2>/dev/null || mlir-opt "$d/t.mlir" --convert-gpu-to-spirv -o "$d/spv0.mlir"
[ -f "$d/spv0.mlir" ] && mlir-opt "$d/spv0.mlir" --pass-pipeline='builtin.module(spirv.module(spirv-lower-abi-attrs,spirv-update-vce,canonicalize))' -o "$d/spv.mlir"
awk '/^  spirv.module/{p=1} p{print} p && /^  }$/{exit}' "$d/spv.mlir" > "$d/m.mlir"
mlir-translate --no-implicit-module --serialize-spirv "$d/m.mlir" -o "$2"
