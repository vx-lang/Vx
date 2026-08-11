//===- gemm_anatomy.cu - where a dispatched GEMM's time actually goes ------===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Vx's GEMM path reaches about 8% of an A100's SGEMM peak, and "we just call
// cuBLAS" is true, so the gap is in what surrounds the call rather than in the
// call. This attributes it, by running the *same* cublasSgemm under layers
// added one at a time (#348, #321).
//
//   resident    operands already on the device, nothing copied or allocated
//   +sync       a full device synchronise per GEMM
//   +copies     operands staged in and the result staged out, pageable, per GEMM
//   +malloc     three cudaMalloc and three cudaFree per GEMM
//   pinned      the same as +malloc, but from pinned host memory
//   pinned+pool the same again, with the device buffers allocated once
//
// `+malloc` is what runtime/cuda_dispatch.cpp does today: stage() allocates and
// copies per operand, run_gemm allocates the result, DeviceBuffer frees all
// three, and every dispatch ends in cudaDeviceSynchronize. The last two rows
// are the two fixes, priced separately.
//
// Build on the pod:
//   nvcc -O2 -arch=native gemm_anatomy.cu -lcublas -o gemm_anatomy
//   ./gemm_anatomy 2048 200
//
//===----------------------------------------------------------------------===//

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                               \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,                \
              cudaGetErrorString(e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static cublasHandle_t g_handle;
static int N = 2048, ITERS = 200;

// Column-major cuBLAS computing what a row-major caller means by C = A*B, which
// is the same swap runtime/cuda_dispatch.cpp performs.
static void sgemm(const float *da, const float *db, float *dc) {
  const float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, db, N, da, N,
              &beta, dc, N);
}

static double seconds(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  return ms / 1000.0;
}

int main(int argc, char **argv) {
  if (argc > 1)
    N = atoi(argv[1]);
  if (argc > 2)
    ITERS = atoi(argv[2]);

  const size_t bytes = (size_t)N * N * sizeof(float);
  const double gflop = 2.0 * (double)N * N * N / 1e9;

  cublasCreate(&g_handle);

  // Host operands, both pageable and pinned, so the copy rows differ only in
  // which the driver is handed.
  float *ha = (float *)malloc(bytes), *hb = (float *)malloc(bytes),
        *hc = (float *)malloc(bytes);
  float *pa, *pb, *pc;
  CHECK(cudaHostAlloc(&pa, bytes, cudaHostAllocDefault));
  CHECK(cudaHostAlloc(&pb, bytes, cudaHostAllocDefault));
  CHECK(cudaHostAlloc(&pc, bytes, cudaHostAllocDefault));
  for (size_t i = 0; i < (size_t)N * N; ++i) {
    ha[i] = pa[i] = (float)((i % 7) + 1) * 0.001f;
    hb[i] = pb[i] = (float)((i % 5) + 1) * 0.001f;
  }

  float *da, *db, *dc;
  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMalloc(&db, bytes));
  CHECK(cudaMalloc(&dc, bytes));
  CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  // Warm the context, the handle and cuBLAS's own first-call setup.
  for (int i = 0; i < 5; ++i)
    sgemm(da, db, dc);
  CHECK(cudaDeviceSynchronize());

  printf("N=%d, %d iterations, %.2f GFLOP per GEMM\n\n", N, ITERS, gflop);
  printf("  %-14s %10s %10s %12s\n", "layer", "ms/GEMM", "GFLOP/s", "vs resident");

  double base = 0;
  auto report = [&](const char *name, double total) {
    double per = total / ITERS;
    if (base == 0)
      base = per;
    printf("  %-14s %10.3f %10.1f %11.2fx\n", name, per * 1000, gflop / per,
           per / base);
    return per;
  };

  // 1. resident: the device ceiling for this shape, queued back to back.
  cudaEventRecord(t0);
  for (int i = 0; i < ITERS; ++i)
    sgemm(da, db, dc);
  cudaEventRecord(t1);
  CHECK(cudaEventSynchronize(t1));
  report("resident", seconds(t0, t1));

  // 2. + a full synchronise per GEMM, as the dispatch path does.
  cudaEventRecord(t0);
  for (int i = 0; i < ITERS; ++i) {
    sgemm(da, db, dc);
    CHECK(cudaDeviceSynchronize());
  }
  cudaEventRecord(t1);
  CHECK(cudaEventSynchronize(t1));
  report("+sync", seconds(t0, t1));

  // 3. + staging both operands in and the result out, from pageable memory,
  //    into buffers that already exist.
  cudaEventRecord(t0);
  for (int i = 0; i < ITERS; ++i) {
    CHECK(cudaMemcpy2D(da, N * sizeof(float), ha, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy2D(db, N * sizeof(float), hb, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyHostToDevice));
    sgemm(da, db, dc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy2D(hc, N * sizeof(float), dc, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyDeviceToHost));
  }
  cudaEventRecord(t1);
  CHECK(cudaEventSynchronize(t1));
  report("+copies", seconds(t0, t1));

  // 4. + three allocations and three frees per GEMM. This is the current path.
  cudaEventRecord(t0);
  for (int i = 0; i < ITERS; ++i) {
    float *xa, *xb, *xc;
    CHECK(cudaMalloc(&xa, bytes));
    CHECK(cudaMemcpy2D(xa, N * sizeof(float), ha, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&xb, bytes));
    CHECK(cudaMemcpy2D(xb, N * sizeof(float), hb, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&xc, bytes));
    sgemm(xa, xb, xc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy2D(hc, N * sizeof(float), xc, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(xa));
    CHECK(cudaFree(xb));
    CHECK(cudaFree(xc));
  }
  cudaEventRecord(t1);
  CHECK(cudaEventSynchronize(t1));
  report("+malloc (ours)", seconds(t0, t1));

  // 5. the same, from pinned host memory: one line of difference in the plugin.
  cudaEventRecord(t0);
  for (int i = 0; i < ITERS; ++i) {
    float *xa, *xb, *xc;
    CHECK(cudaMalloc(&xa, bytes));
    CHECK(cudaMemcpy2D(xa, N * sizeof(float), pa, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&xb, bytes));
    CHECK(cudaMemcpy2D(xb, N * sizeof(float), pb, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&xc, bytes));
    sgemm(xa, xb, xc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy2D(pc, N * sizeof(float), xc, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(xa));
    CHECK(cudaFree(xb));
    CHECK(cudaFree(xc));
  }
  cudaEventRecord(t1);
  CHECK(cudaEventSynchronize(t1));
  report("pinned", seconds(t0, t1));

  // 6. pinned, and the device buffers kept rather than reallocated: still a
  //    copy per dispatch, but no allocator traffic. This is what a staging
  //    cache would achieve without any residency analysis.
  cudaEventRecord(t0);
  for (int i = 0; i < ITERS; ++i) {
    CHECK(cudaMemcpy2D(da, N * sizeof(float), pa, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy2D(db, N * sizeof(float), pb, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyHostToDevice));
    sgemm(da, db, dc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy2D(pc, N * sizeof(float), dc, N * sizeof(float),
                       N * sizeof(float), N, cudaMemcpyDeviceToHost));
  }
  cudaEventRecord(t1);
  CHECK(cudaEventSynchronize(t1));
  report("pinned+pool", seconds(t0, t1));

  printf("\n  %.1f MiB crosses per GEMM (3 x N*N*4)\n",
         3.0 * bytes / 1048576.0);
  cublasDestroy(g_handle);
  return 0;
}
