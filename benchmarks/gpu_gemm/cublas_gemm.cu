//===- cublas_gemm.cu - Vx Compiler -----------------------------*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// C = A @ B with cuBLAS: A [m, k] and B [k, n] in bf16, row-major, C [m, n] in
// f32, sums in f32. The same inputs as the Vx programs in this folder. Prints
// the median of 50 single launches, timed with events, and the average of 1000
// launches in a row, repeated 7 times.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

int main(int argc, char **argv) {
  int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
  std::vector<__nv_bfloat16> ha((size_t)m * k), hb((size_t)k * n);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      ha[(size_t)i * k + j] =
          __float2bfloat16((float)((i * 3 + j * 5) % 7 - 3));
  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++)
      hb[(size_t)i * n + j] = __float2bfloat16((float)((i * 2 + j) % 5 - 2));
  __nv_bfloat16 *a, *b;
  float *c;
  cudaMalloc(&a, ha.size() * 2);
  cudaMalloc(&b, hb.size() * 2);
  cudaMalloc(&c, (size_t)m * n * 4);
  cudaMemcpy(a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(b, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice);
  cublasHandle_t h;
  cublasCreate(&h);
  float alpha = 1.0f, beta = 0.0f;
  // Row-major C = A B is column-major C^T = B^T A^T.
  auto run = [&] {
    cublasStatus_t s =
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b,
                     CUDA_R_16BF, n, a, CUDA_R_16BF, k, &beta, c, CUDA_R_32F, n,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) {
      printf("cublas error %d\n", (int)s);
      exit(1);
    }
  };
  for (int i = 0; i < 10; i++)
    run();
  cudaDeviceSynchronize();
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  std::vector<float> one;
  for (int i = 0; i < 50; i++) {
    cudaEventRecord(e0);
    run();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    one.push_back(ms);
  }
  std::sort(one.begin(), one.end());
  std::vector<float> steady;
  for (int r = 0; r < 7; r++) {
    cudaEventRecord(e0);
    for (int i = 0; i < 1000; i++)
      run();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    steady.push_back(ms / 1000);
  }
  std::sort(steady.begin(), steady.end());
  double flop = 2.0 * m * n * k;
  std::vector<float> hc((size_t)m * n);
  cudaMemcpy(hc.data(), c, hc.size() * 4, cudaMemcpyDeviceToHost);
  int wrong = 0;
  for (int s = 0; s < 64; s++) {
    int i = (s * 7919) % m, j = (s * 104729) % n;
    float want = 0;
    for (int q = 0; q < k; q++)
      want += __bfloat162float(ha[(size_t)i * k + q]) *
              __bfloat162float(hb[(size_t)q * n + j]);
    if (hc[(size_t)i * n + j] != want)
      wrong++;
  }
  printf("cublas %dx%dx%d: %d of 64 samples wrong\n", m, n, k, wrong);
  printf("cublas single launch median %.4f ms (%.1f TFLOP/s)\n", one[25],
         flop / one[25] / 1e9);
  printf("cublas 1000 in a row median %.4f ms (%.1f TFLOP/s), range %.4f .. "
         "%.4f\n",
         steady[3], flop / steady[3] / 1e9, steady[0], steady[6]);
  return 0;
}
