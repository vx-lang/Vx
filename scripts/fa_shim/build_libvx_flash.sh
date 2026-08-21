#!/usr/bin/env bash
#===- build_libvx_flash.sh - the vendor half of kind=attention -*- bash -*-===//
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===//
#
# Builds `libvx_flash.so`, the FlashAttention-2 forward behind Vx's
# `kind=attention` routing (Vx#378). runtime/cuda_dispatch.cpp dlopens it --
# `VX_FLASH_LIB=/path/to/libvx_flash.so`, or plain `libvx_flash.so` on the
# loader path -- and calls `vx_flash_fwd_f16_hd64`. No library found means no
# route: the outlined kernel computes the same values, slower.
#
# Needs (neither ships in this repo; both are public):
#   FLASH_ATTN  a checkout of github.com/Dao-AILab/flash-attention
#               (the csrc/flash_attn tree; tested at the tag shipping sm_80)
#   CUTLASS     a checkout of github.com/NVIDIA/cutlass (tested at v3.6.0)
#
# The shim is torch-free: Flash_fwd_params is plain data, and the only torch
# tentacles in the fwd path are the C10_CUDA_CHECK macros (stubbed to plain
# CUDA in fake_inc/) and a dropout-only philox include (overwritten with a
# bare `#pragma once`; dropout is compiled out below). Measured on an A100
# (SQ=8192 SK=2048 HD=64 f16): 0.062 ms standalone, 0.066 ms through the
# full Vx dispatch -- the number the campaign's own best fused kernel
# (1.125 ms FP32) exists to be compared against.
#
#   FLASH_ATTN=~/flash-attention/csrc/flash_attn CUTLASS=~/cutlass \
#     scripts/fa_shim/build_libvx_flash.sh
#
#===----------------------------------------------------------------------===//
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FLASH_ATTN="${FLASH_ATTN:?set FLASH_ATTN to flash-attention/csrc/flash_attn}"
CUTLASS="${CUTLASS:?set CUTLASS to a cutlass checkout}"
OUT="${OUT:-$HERE}"
ARCH="${ARCH:-sm_80}"

# The two torch stubs, generated rather than committed: they are workarounds
# for another project's headers, not code of ours.
FAKE="$OUT/fake_inc"
mkdir -p "$FAKE/c10/cuda"
cat > "$FAKE/c10/cuda/CUDAException.h" <<'EOF'
// Torch's error macros, as plain CUDA: all the fwd path uses of c10.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define C10_CUDA_CHECK(expr)                                                   \
  do {                                                                         \
    cudaError_t e_ = (expr);                                                   \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),      \
              __FILE__, __LINE__);                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
EOF
# Dropout-only include; dropout is disabled at compile time.
echo '#pragma once' > "$FAKE/philox_unpack.cuh"
cp "$FAKE/philox_unpack.cuh" "$FLASH_ATTN/src/philox_unpack.cuh"

nvcc -O3 -std=c++17 -arch="$ARCH" -Xcompiler -fPIC --shared \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_ALIBI \
  -DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_UNEVEN_K \
  -DFLASHATTENTION_DISABLE_LOCAL \
  -I "$FAKE" -I "$FLASH_ATTN" -I "$FLASH_ATTN/src" -I "$CUTLASS/include" \
  "$HERE/vx_flash_shim.cu" \
  "$FLASH_ATTN/src/flash_fwd_hdim64_fp16_sm80.cu" \
  -o "$OUT/libvx_flash.so"

echo "built $OUT/libvx_flash.so:"
nm -D "$OUT/libvx_flash.so" | grep vx_flash
