// measure_device.cu -- M1 per-seam instrument for an NVIDIA part (vx-review#15).
//
// Measures exactly the three seams the frozen predictions price for a discrete SKU, in the units
// they are priced in:
//
//   CPU_DRAM -> HBM   ps   link_rate     cudaMemcpy H2D/D2H, pinned and pageable
//   HBM -> L2         ps   containment   L2-resident streaming read (L2's own bandwidth)
//   L2 -> SMEM        cyc  containment   in-kernel clock64() around a shared-memory load
//
// The last one is measured in CYCLES on purpose. `L2->SMEM` is declared `B/cyc` on every fleet
// SKU, so its prediction is a cycle count; converting it to nanoseconds needs a clock figure the
// machine files do not carry. Protocol decision 5 in PREDICTIONS.md forbids comparing it against
// wall-clock, so the instrument reports cycles and the comparison stays in cycles.
//
// Protocol (fixed before any hardware was rented -- see PREDICTIONS.md):
//   * log-spaced sizes 4 KiB .. 1 GiB, matching the frozen cells exactly so the join is on bytes;
//   * >= 11 reps, median with IQR, never a mean;
//   * warm-up rep discarded;
//   * cudaEvent timing for device work, CLOCK_MONOTONIC for host-side;
//   * the machine ceiling is measured FIRST and printed, so model error can be charged against
//     achievable peak separately from declared peak.
//
// Build:  nvcc -O3 -arch=sm_90 measure_device.cu -o measure_device
// Run:    ./measure_device > measured.csv
//
// NOTE: this file has not been compiled -- it was written on a machine with no CUDA toolchain.
// `run_m1.sh` builds it as its first step and stops on failure rather than proceeding.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#ifndef REPS
#define REPS 11
#endif

#define CK(x)                                                                       \
  do {                                                                              \
    cudaError_t e_ = (x);                                                           \
    if (e_ != cudaSuccess) {                                                        \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
              __LINE__);                                                            \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

// The frozen cell sizes: f32 tiles [d,d], bytes = 4*d*d, d in {32..16384}. Hard-coded rather than
// computed so a change to the sweep cannot silently desynchronise the join with the predictions.
static const size_t SIZES[] = {4096ul,      16384ul,     65536ul,     262144ul,    1048576ul,
                               4194304ul,   16777216ul,  67108864ul,  268435456ul, 1073741824ul};
static const int N_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

static double median_of(std::vector<double> &v, double *q1, double *q3) {
  std::sort(v.begin(), v.end());
  if (q1) *q1 = v[v.size() / 4];
  if (q3) *q3 = v[(3 * v.size()) / 4];
  return v[v.size() / 2];
}

// ---- seam 2: L2-resident streaming read ------------------------------------------------------
// Grid-stride sum over a buffer sized to FIT in L2, so the traffic is L2->SM and not HBM->L2.
// `float4` because a 128-bit access is what saturates the path; a scalar loop measures issue rate.
__global__ void l2_stream(const float4 *__restrict__ src, size_t n4, float *__restrict__ sink) {
  float acc = 0.f;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  // Keep the loads live without a global reduction: only thread 0 of block 0 can ever store, and
  // the compiler cannot prove it does not.
  if (acc == 1234.5678f) sink[0] = acc;
}

// ---- seam 3: L2 -> SMEM, in cycles -----------------------------------------------------------
// One block, timed with clock64() around a bulk copy into shared memory. Reports the cycle count
// for `bytes`, which is the same quantity `bytes / (128 B/cyc)` predicts.
__global__ void l2_to_smem_cycles(const float4 *__restrict__ src, size_t n4,
                                  unsigned long long *out) {
  extern __shared__ float4 smem[];
  __syncthreads();
  unsigned long long t0 = clock64();
  for (size_t i = threadIdx.x; i < n4; i += blockDim.x) smem[i] = src[i];
  __syncthreads();
  unsigned long long t1 = clock64();
  // Keep the shared writes live.
  if (threadIdx.x == 0) {
    float4 v = smem[n4 - 1];
    if (v.x == 1234.5678f) out[1] = 1;
    out[0] = t1 - t0;
  }
}

int main() {
  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, dev));

  // ---- machine ceiling, FIRST -----------------------------------------------------------------
  // Printed to stderr so it lands in the log next to the data without polluting the CSV. Model
  // error is charged against this separately from the declared peak: the gap between achievable
  // and declared belongs to the declaration, not to the model.
  fprintf(stderr, "GPU: %s (sm_%d%d)\n", p.name, p.major, p.minor);
  fprintf(stderr, "  declared HBM peak    : %.1f GB/s (%d-bit @ %.0f MHz effective)\n",
          2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6, p.memoryBusWidth,
          p.memoryClockRate / 1000.0);
  fprintf(stderr, "  L2 cache             : %d MiB\n", p.l2CacheSize >> 20);
  fprintf(stderr, "  SMEM per block (opt) : %zu KiB\n", p.sharedMemPerBlockOptin >> 10);
  fprintf(stderr, "  SM clock             : %.0f MHz\n", p.clockRate / 1000.0);
  fprintf(stderr, "  SM count             : %d\n", p.multiProcessorCount);

  printf("seam,bytes,unit,median,q1,q3,derived_rate_GBps,reps,note\n");

  // ---- seam 1: CPU_DRAM -> HBM ----------------------------------------------------------------
  // Pinned AND pageable. Pinned is the fair comparison against a declared link rate; pageable is
  // what naive code does, and the difference between them is the harness's positive control --
  // if the two come out equal the instrument is not resolving the thing it exists to resolve.
  for (int pinned = 1; pinned >= 0; --pinned) {
    for (int i = 0; i < N_SIZES; ++i) {
      size_t bytes = SIZES[i];
      char *h = nullptr;
      void *d = nullptr;
      if (pinned) {
        if (cudaHostAlloc((void **)&h, bytes, cudaHostAllocDefault) != cudaSuccess) continue;
      } else {
        h = (char *)malloc(bytes);
        if (!h) continue;
      }
      if (cudaMalloc(&d, bytes) != cudaSuccess) {
        pinned ? (void)cudaFreeHost(h) : free(h);
        continue;
      }
      memset(h, 0x5a, bytes);

      cudaEvent_t a, b;
      CK(cudaEventCreate(&a));
      CK(cudaEventCreate(&b));
      CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));  // warm-up

      std::vector<double> s;
      for (int r = 0; r < REPS; ++r) {
        CK(cudaEventRecord(a));
        CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        s.push_back((double)ms * 1e9);  // ms -> ps
      }
      double q1, q3, med = median_of(s, &q1, &q3);
      printf("CPU_DRAM->HBM,%zu,ps,%.0f,%.0f,%.0f,%.2f,%d,%s\n", bytes, med, q1, q3,
             med > 0 ? (double)bytes / (med / 1000.0) : 0.0, REPS,
             pinned ? "pinned" : "pageable");
      fflush(stdout);

      CK(cudaEventDestroy(a));
      CK(cudaEventDestroy(b));
      CK(cudaFree(d));
      pinned ? CK(cudaFreeHost(h)) : (void)free(h);
    }
  }

  // ---- seam 2: HBM -> L2 ----------------------------------------------------------------------
  // Only sizes that FIT in L2 are physically L2-resident. Larger cells are still *predicted* --
  // the model prices them at L2's bandwidth regardless -- so they are emitted with a note rather
  // than silently measured as something else. A prediction for a placement that cannot be
  // L2-resident is itself a finding.
  {
    size_t l2 = (size_t)p.l2CacheSize;
    for (int i = 0; i < N_SIZES; ++i) {
      size_t bytes = SIZES[i];
      if (bytes % sizeof(float4)) continue;
      if (bytes > l2) {
        printf("HBM->L2,%zu,ps,,,,,%d,exceeds_l2_capacity_not_measured\n", bytes, REPS);
        continue;
      }
      float4 *d = nullptr;
      float *sink = nullptr;
      if (cudaMalloc(&d, bytes) != cudaSuccess) continue;
      CK(cudaMalloc(&sink, sizeof(float)));
      CK(cudaMemset(d, 1, bytes));
      size_t n4 = bytes / sizeof(float4);

      int block = 256;
      int grid = p.multiProcessorCount * 32;
      cudaEvent_t a, b;
      CK(cudaEventCreate(&a));
      CK(cudaEventCreate(&b));
      // Warm-up doubles as the residency step: it pulls the buffer into L2 so the timed reps
      // measure L2->SM and not the cold HBM fill.
      l2_stream<<<grid, block>>>(d, n4, sink);
      CK(cudaDeviceSynchronize());

      std::vector<double> s;
      for (int r = 0; r < REPS; ++r) {
        CK(cudaEventRecord(a));
        l2_stream<<<grid, block>>>(d, n4, sink);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        s.push_back((double)ms * 1e9);
      }
      double q1, q3, med = median_of(s, &q1, &q3);
      printf("HBM->L2,%zu,ps,%.0f,%.0f,%.0f,%.2f,%d,l2_resident\n", bytes, med, q1, q3,
             med > 0 ? (double)bytes / (med / 1000.0) : 0.0, REPS);
      fflush(stdout);

      CK(cudaEventDestroy(a));
      CK(cudaEventDestroy(b));
      CK(cudaFree(d));
      CK(cudaFree(sink));
    }
  }

  // ---- seam 3: L2 -> SMEM, in cycles ----------------------------------------------------------
  {
    size_t smem_max = p.sharedMemPerBlockOptin;
    for (int i = 0; i < N_SIZES; ++i) {
      size_t bytes = SIZES[i];
      if (bytes % sizeof(float4)) continue;
      if (bytes > smem_max) {
        printf("L2->SMEM,%zu,cyc,,,,,%d,exceeds_smem_capacity_not_measured\n", bytes, REPS);
        continue;
      }
      float4 *d = nullptr;
      unsigned long long *out = nullptr;
      if (cudaMalloc(&d, bytes) != cudaSuccess) continue;
      CK(cudaMalloc(&out, 2 * sizeof(unsigned long long)));
      CK(cudaMemset(d, 1, bytes));
      size_t n4 = bytes / sizeof(float4);

      CK(cudaFuncSetAttribute(l2_to_smem_cycles, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              (int)bytes));
      std::vector<double> s;
      l2_to_smem_cycles<<<1, 256, bytes>>>(d, n4, out);  // warm-up
      CK(cudaDeviceSynchronize());
      for (int r = 0; r < REPS; ++r) {
        l2_to_smem_cycles<<<1, 256, bytes>>>(d, n4, out);
        CK(cudaDeviceSynchronize());
        unsigned long long cyc = 0;
        CK(cudaMemcpy(&cyc, out, sizeof(cyc), cudaMemcpyDeviceToHost));
        s.push_back((double)cyc);
      }
      double q1, q3, med = median_of(s, &q1, &q3);
      // No GB/s column: this is a cycle count, and turning it into a rate needs the clock figure
      // the machine files do not declare. Left empty rather than invented.
      printf("L2->SMEM,%zu,cyc,%.0f,%.0f,%.0f,,%d,one_block\n", bytes, med, q1, q3, REPS);
      fflush(stdout);

      CK(cudaFree(d));
      CK(cudaFree(out));
    }
  }

  return 0;
}
