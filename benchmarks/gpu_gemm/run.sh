#!/bin/sh
#===- run.sh - Vx Compiler -------------------------------------*- Shell -*-===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#

# Times each GEMM in this folder and cuBLAS on one NVIDIA GPU, at two sizes. Each line gives the
# median of 50 launches and how many of 64 sampled entries of C are wrong.
#
# Run it from the repository root, after `cargo build --release`:
#   sh benchmarks/gpu_gemm/run.sh [folder for the results, default gemm-results]
# VERSIONS="v5 v8" times only those versions. MACHINE names the machine file.
set -u
out=${1:-gemm-results}
here=benchmarks/gpu_gemm
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p "$out"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv > "$out/gpu.txt"
nvcc --version | tail -1 >> "$out/gpu.txt"
nvcc -O3 -arch=sm_80 "$here/cublas_gemm.cu" -lcublas -o "$out/cublas_gemm"
. ./config.local
for size in "8192 3072 768" "4096 4096 4096"; do
  set -- $size
  M=$1 N=$2 K=$3
  "$out/cublas_gemm" $M $N $K > "$out/cublas-$M-$N-$K.log" 2>&1
  cat "$out/cublas-$M-$N-$K.log"
  for f in "$here"/v*.vx; do
    v=$(basename "$f" .vx | cut -d_ -f1)
    case " ${VERSIONS:-v1 v2 v3 v4 v5 v6 v7 v8} " in *" $v "*) ;; *) continue ;; esac
    p="$out/$v-$M-$N-$K"
    sed -e "s/^const M : i32 = .*;/const M : i32 = $M;/" \
        -e "s/^const N : i32 = .*;/const N : i32 = $N;/" \
        -e "s/^const K : i32 = .*;/const K : i32 = $K;/" "$f" > "$p.vx"
    VX_TIME_KERNEL=1 ./target/release/vxc "$p.vx" --machine "${MACHINE:-fleet/a100-80.vx}" \
      --host default > "$p.log" 2>&1
    # The first 3 launches warm up the GPU, so the median leaves them out.
    med=$(grep "device time" "$p.log" | awk '{print $6}' | tail -n +4 | sort -n |
      awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    wrong=$(grep "samples wrong" "$p.log" | tail -1)
    tflops=$(awk -v m=$M -v n=$N -v k=$K -v t="$med" 'BEGIN{printf "%.1f", 2*m*n*k/(t*1e9)}')
    echo "vx $v $M x $N x $K: median $med ms, $tflops TFLOP/s; $wrong"
  done
done
