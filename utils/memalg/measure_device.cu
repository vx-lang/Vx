// measure_device.cu -- M1 per-seam instrument for an NVIDIA part (vx-review#15).
//
// Measures exactly the three seams the frozen predictions price for a discrete SKU, in the units
// they are priced in:
//
//   CPU_DRAM -> HBM   ps   link_rate     cudaMemcpy H2D/D2H, pinned and pageable
//   HBM -> L2         ps   containment   L2-resident streaming read (L2's own bandwidth)
//   L2 -> SMEM        cyc  containment   in-kernel clock64() around a shared-memory load
//
// It also emits `device/*` rows, which are not seams: they are the machine file's own declared
// numbers -- capacity, clock, SM count, allocation granule -- read back off the hardware, so that
// `spec:` lines can become `measured:` ones (vx-review#18). Those rows carry no distribution and
// join against no prediction.
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
// Compiles clean under nvcc 13.2 and 12.8 for sm_80/90/90a/100/120, and has been RUN against an
// H100 80GB HBM3 (2026-08-08) -- with two exceptions, both written after that box was released
// and with no toolkit reachable, so NEITHER HAS BEEN RUN:
//   * `seam 2b: HBM -> L2, the actual fill`
//   * the `M4: the declared numbers themselves` block
// Their HOST code is checked by `utils/memalg/check_host_compile.sh`, which compiles each block
// against a stub cuda_runtime.h. That catches typos, printf formats and type errors -- the things
// that would otherwise eat the first hour of a rented session. It does NOT check device code, and
// it cannot check `cudaDeviceProp` field names, because the stub declares those from our own
// belief. Run both blocks on real hardware before quoting anything from them.
//
// Two instrument defects that the H100 run exposed, both of which produced confident wrong numbers
// rather than errors, are fixed here and described at their sites: kernel-launch overhead swamping
// the on-die seams, and grid-stride re-reading serving out of L1.
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

// One declared-number row (vx-review#18). These are facts, not distributions: `median` carries the
// value, `unit` says what it is, q1/q3/rate are empty and reps is 1. compare_m1.py joins on
// (seam, bytes, note) and no frozen cell carries a `device/` seam name, so these are inert to the
// M1 error table -- they exist to be read by whoever annotates the machine files.
static void fact_row(const char *seam, size_t bytes, const char *unit, double value,
                     const char *note) {
  printf("%s,%zu,%s,%.3f,,,,1,%s\n", seam, bytes, unit, value, note);
}

static double median_of(std::vector<double> &v, double *q1, double *q3) {
  std::sort(v.begin(), v.end());
  if (q1) *q1 = v[v.size() / 4];
  if (q3) *q3 = v[(3 * v.size()) / 4];
  return v[v.size() / 2];
}

// ---- seam 2: L2-resident streaming read ------------------------------------------------------
// Grid-stride sum over a buffer sized to FIT in L2, so the traffic is L2->SM and not HBM->L2.
// `float4` because a 128-bit access is what saturates the path; a scalar loop measures issue rate.
// `iters` re-reads the buffer inside the kernel. Without it the measurement is pure launch
// overhead: a ~10 us floor swamps a 16 MiB L2 read that should take ~1.4 us, and the reported
// time is then flat across a 4096x size range -- which is exactly what the first run on an H100
// showed. Amortising is the correct comparison, not a flattering one: the model predicts the time
// to move bytes, and kernel launch is not part of what it claims.
//
// Float addition is not associative, so the compiler cannot fold the repeated passes into a
// multiply; the loads have to happen every iteration.
__global__ void l2_stream(const float4 *__restrict__ src, size_t n4, float *__restrict__ sink,
                          int iters) {
  float acc = 0.f;
  for (int it = 0; it < iters; ++it) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
      // __ldcg = cache-global: the line is cached in L2 but NOT in L1. Without this the
      // amortisation defeats itself. Under grid-stride re-reading each block's slice is only
      // total/num_blocks -- about 12 KB even for a 48 MiB buffer -- so a plain load re-reads out
      // of L1 no matter how large the buffer is. Measured on an H100 that reported a *rising*
      // 17.4 -> 29.2 TB/s with size (L1 bandwidth plus parallelism scaling) where the L1-bypassed
      // walk is flat at ~7.5 TB/s, which is what a bandwidth limit actually looks like.
      float4 v = __ldcg(&src[i]);
      acc += v.x + v.y + v.z + v.w;
    }
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
  // CUDA 13 removed cudaDeviceProp::clockRate and ::memoryClockRate. The attribute queries carry
  // the same values in the same units (kHz) and exist on 11.x/12.x too, so this builds against
  // whatever toolkit the rented box happens to ship.
  int mem_clock_khz = 0, sm_clock_khz = 0;
  CK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
  CK(cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, dev));

  fprintf(stderr, "GPU: %s (sm_%d%d)\n", p.name, p.major, p.minor);
  fprintf(stderr, "  declared HBM peak    : %.1f GB/s (%d-bit @ %.0f MHz effective)\n",
          2.0 * mem_clock_khz * (p.memoryBusWidth / 8) / 1.0e6, p.memoryBusWidth,
          mem_clock_khz / 1000.0);
  fprintf(stderr, "  L2 cache             : %d MiB\n", p.l2CacheSize >> 20);
  fprintf(stderr, "  SMEM per block (opt) : %zu KiB\n", p.sharedMemPerBlockOptin >> 10);
  fprintf(stderr, "  SM clock             : %.0f MHz\n", sm_clock_khz / 1000.0);
  fprintf(stderr, "  SM count             : %d\n", p.multiProcessorCount);

  printf("seam,bytes,unit,median,q1,q3,derived_rate_GBps,reps,note\n");

  // ---- M4: the declared numbers themselves (vx-review#18) -------------------------------------
  // Every figure in fleet/*.vx came from a vendor document. These rows are the same figures read
  // off the hardware, so each machine-file line can be marked `measured:` against a log reference
  // instead of `spec:`. They go to the CSV rather than only to stderr because "update the machine
  // files" is then a mechanical join; a human transcribing from a log is exactly the step where a
  // number quietly becomes the number that fits.
  //
  // NOT COMPILE-VERIFIED AND NOT RUN, like seam 2b -- written with no reachable toolkit. Compile
  // before quoting anything from it.
  //
  // Two things these rows are NOT:
  //   * they are not the vendor's marketing capacity. `totalGlobalMem` is already net of the ECC
  //     reservation and is reported in bytes, so an "80 GB" part reports ~79.6 GiB and the gap is
  //     units plus ECC, not a defect. The defect to look for is a gap that survives both.
  //   * they do not settle a +-4% bandwidth dispute by themselves. See `HBM_peak_from_clock`.
  size_t free_b = 0, total_b = 0;
  CK(cudaMemGetInfo(&free_b, &total_b));
  fact_row("device/HBM_total", 0, "B", (double)p.totalGlobalMem,
           "cudaDeviceProp::totalGlobalMem -- the HBM `capacity:` figure, net of ECC");
  fact_row("device/HBM_free", 0, "B", (double)free_b,
           "cudaMemGetInfo free with a context up and nothing allocated");
  fact_row("device/L2_capacity", 0, "B", (double)p.l2CacheSize,
           "cudaDeviceProp::l2CacheSize -- the L2 `capacity:` figure");
  // Both SMEM figures, because the machine files declare one number and the hardware has two: an
  // H100 reports 228 KiB per SM but caps a single block's opt-in at 227 KiB, the last KiB being
  // driver-reserved. `capacity:` is checked against a tile placement, which is a block, so if
  // these disagree the machine files are declaring the wrong one.
  fact_row("device/SMEM_per_SM", 0, "B", (double)p.sharedMemPerMultiprocessor,
           "cudaDeviceProp::sharedMemPerMultiprocessor");
  fact_row("device/SMEM_per_block_optin", 0, "B", (double)p.sharedMemPerBlockOptin,
           "cudaDeviceProp::sharedMemPerBlockOptin -- what a tile placement can actually get");
  fact_row("device/SM_count", 0, "count", (double)p.multiProcessorCount,
           "cudaDeviceProp::multiProcessorCount -- the SMEM `replicas:` figure");
  fact_row("device/SM_clock", 0, "Hz", sm_clock_khz * 1000.0,
           "cudaDevAttrClockRate -- the `clock:` figure; max, not current");
  fact_row("device/mem_bus_width", 0, "bit", (double)p.memoryBusWidth,
           "cudaDeviceProp::memoryBusWidth");

  // The B200 dispute (7.7 vs 8.0 TB/s across two vendor documents) is settled by the part's own
  // reported clock and bus width, not by an achieved-bandwidth number: a streaming read lands at
  // 80-90% of peak, and 85% of 7.7 TB/s is indistinguishable from 82% of 8.0. So this row is the
  // theoretical peak the hardware itself implies, and `HBM->L2_fill` is what it delivers -- the
  // two answer different questions and the dispute is about the first.
  //
  // Believe it only if the clock is non-zero: some driver/part combinations report 0 here, and a
  // 0 would otherwise publish a confident peak of 0.0 GB/s. Blackwell is also a multi-die package,
  // so check that the reported bus width is the whole part and not one die before quoting it.
  if (mem_clock_khz <= 0) {
    fprintf(stderr,
            "  WARNING: the driver reports memory clock %d kHz; device/HBM_peak_from_clock is not\n"
            "           usable on this box and the B200 dispute cannot be settled from it.\n",
            mem_clock_khz);
  }
  fact_row("device/HBM_peak_from_clock", 0, "GB/s",
           2.0 * mem_clock_khz * (p.memoryBusWidth / 8) / 1.0e6,
           "2 x cudaDevAttrMemoryClockRate x busWidth/8; 0 means the driver did not report it");

  // Usable vs quoted, the way a placement finds out. The frozen predictions record capacity
  // *rejections* as predictions, so where this boundary really sits is itself a scored quantity:
  // a placement the model refuses at 80 GiB that the hardware accepts -- or the reverse -- is a
  // miss that no error percentage would ever show.
  //
  // Binary search on cudaMalloc rather than trusting `free`, because free memory counts pages the
  // allocator cannot hand back as one contiguous block, and a tile placement needs one block.
  // cudaErrorMemoryAllocation is recoverable and not sticky, so the failed probes are cleared and
  // the run continues.
  {
    size_t lo = 0, hi = free_b + (1ull << 30);  // above `free`: it is a lower bound, not a ceiling
    const size_t resolution = 1ull << 20;
    while (lo + resolution < hi) {
      size_t mid = lo + (hi - lo) / 2;
      void *probe = nullptr;
      if (cudaMalloc(&probe, mid) == cudaSuccess) {
        CK(cudaFree(probe));
        lo = mid;
      } else {
        cudaGetLastError();
        hi = mid;
      }
    }
    fact_row("device/HBM_largest_single_alloc", 0, "B", (double)lo,
             "binary search on cudaMalloc at 1 MiB resolution -- what one tile can actually get");
  }

  // Real allocation block size. Every fleet file declares `granule: 1 KiB` on SMEM and none
  // declares one for HBM at all, so this is a number the model does not carry and a capacity
  // prediction is wrong by exactly this much per allocation.
  //
  // The signal is the free-memory delta, not pointer spacing: the allocator may return blocks
  // that are not adjacent, but it cannot hide what it took from the device. A warm-up allocation
  // comes first because the very first cudaMalloc in a process grows the allocator's heap, and
  // measuring that growth would report a granule of several MiB on every part.
  {
    void *warm = nullptr;
    CK(cudaMalloc(&warm, 1));
    CK(cudaFree(warm));
    static const size_t REQ[] = {1ul, 256ul, 1024ul, 4096ul, 65536ul, 1048576ul, 2097152ul};
    for (size_t i = 0; i < sizeof(REQ) / sizeof(REQ[0]); ++i) {
      size_t f0 = 0, f1 = 0, t = 0;
      CK(cudaMemGetInfo(&f0, &t));
      void *q = nullptr;
      if (cudaMalloc(&q, REQ[i]) != cudaSuccess) {
        cudaGetLastError();
        continue;
      }
      CK(cudaMemGetInfo(&f1, &t));
      CK(cudaFree(q));
      // Signed on purpose: another process on a shared box can free memory underneath this and
      // make the delta negative. Reporting the negative is honest; silently taking an unsigned
      // difference would publish 18 exabytes as an allocation granule.
      double consumed = (double)f0 - (double)f1;
      fact_row("device/alloc_block", REQ[i], "B", consumed,
               "cudaMemGetInfo delta across one cudaMalloc of `bytes`");
    }
  }

  // ---- seam 1: CPU_DRAM -> HBM ----------------------------------------------------------------
  // Pinned AND pageable. Pinned is the fair comparison against a declared link rate; pageable is
  // what naive code does, and the difference between them is the harness's positive control --
  // if the two come out equal the instrument is not resolving the thing it exists to resolve.
  for (int pinned = 1; pinned >= 0; --pinned) {
    for (int i = 0; i < N_SIZES; ++i) {
      size_t bytes = SIZES[i];
      char *h = nullptr;
      void *d = nullptr;
      // Report every refused allocation. A silent `continue` here would drop the PINNED rows --
      // the headline measurement -- and the run would still exit 0 with a plausible-looking table.
      // Containers routinely cap locked memory (`ulimit -l`), so this is a live failure mode, not
      // a theoretical one.
      if (pinned) {
        cudaError_t e = cudaHostAlloc((void **)&h, bytes, cudaHostAllocDefault);
        if (e != cudaSuccess) {
          fprintf(stderr, "  SKIP pinned %zu B: %s (locked-memory limit? check `ulimit -l`)\n",
                  bytes, cudaGetErrorString(e));
          continue;
        }
      } else {
        h = (char *)malloc(bytes);
        if (!h) {
          fprintf(stderr, "  SKIP pageable %zu B: host malloc failed\n", bytes);
          continue;
        }
      }
      cudaError_t de = cudaMalloc(&d, bytes);
      if (de != cudaSuccess) {
        fprintf(stderr, "  SKIP %s %zu B: cudaMalloc: %s\n", pinned ? "pinned" : "pageable", bytes,
                cudaGetErrorString(de));
        if (pinned) {
          CK(cudaFreeHost(h));
        } else {
          free(h);
        }
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
      // CK expands to a do{}while(0) statement and cannot appear in a ternary.
      if (pinned) {
        CK(cudaFreeHost(h));
      } else {
        free(h);
      }
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
      // Size the grid to the WORK, not to the device. The first H100 run launched 4224 blocks of
      // 256 threads to read a 4 KiB buffer of 256 float4s, so 99.98% of the threads existed only
      // to be scheduled. Cap at 32 waves for the buffers big enough to want them.
      size_t want_blocks = (n4 + block - 1) / block;
      int max_blocks = p.multiProcessorCount * 32;
      int grid = (int)(want_blocks < (size_t)max_blocks ? want_blocks : (size_t)max_blocks);
      if (grid < 1) grid = 1;

      cudaEvent_t a, b;
      CK(cudaEventCreate(&a));
      CK(cudaEventCreate(&b));
      // Warm-up doubles as the residency step: it pulls the buffer into L2 so the timed reps
      // measure L2->SM and not the cold HBM fill.
      l2_stream<<<grid, block>>>(d, n4, sink, 1);
      CK(cudaDeviceSynchronize());

      // Calibrate `iters` until one batch runs >= 5 ms, the same discipline the UMA shakedown
      // uses. Below that the launch/sync floor is a first-order term rather than a rounding error.
      int iters = 1;
      for (;;) {
        CK(cudaEventRecord(a));
        l2_stream<<<grid, block>>>(d, n4, sink, iters);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (ms >= 5.0f || iters >= (1 << 22)) break;
        int grow = (int)(6.0f / (ms > 0.01f ? ms : 0.01f)) + 1;
        iters = iters * (grow > 2 ? grow : 2);
      }

      std::vector<double> s;
      for (int r = 0; r < REPS; ++r) {
        CK(cudaEventRecord(a));
        l2_stream<<<grid, block>>>(d, n4, sink, iters);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        s.push_back((double)ms * 1e9 / (double)iters);  // ps for ONE pass over the buffer
      }
      double q1, q3, med = median_of(s, &q1, &q3);
      // L1 is bypassed, so this is an L2 read and is labelled as one. Small buffers still cannot
      // saturate -- the grid is sized to the work, so a 4 KiB cell runs one block -- and that is a
      // real property of a small transfer, not an instrument artefact.
      const char *note = "l2_read_l1_bypassed";
      fprintf(stderr, "  HBM->L2 %zu B: grid=%d iters=%d\n", bytes, grid, iters);
      printf("HBM->L2,%zu,ps,%.0f,%.0f,%.0f,%.2f,%d,%s\n", bytes, med, q1, q3,
             med > 0 ? (double)bytes / (med / 1000.0) : 0.0, REPS, note);
      fflush(stdout);

      CK(cudaEventDestroy(a));
      CK(cudaEventDestroy(b));
      CK(cudaFree(d));
      CK(cudaFree(sink));
    }
  }

  // ---- seam 2b: HBM -> L2, the actual fill -----------------------------------------------------
  // The cell above is an L2-resident read. Its own note says so (`l2_read_l1_bypassed`) and its
  // warm-up exists precisely to make it one -- but it is emitted under the seam name `HBM->L2`,
  // and that name is what the frozen predictions and every downstream table join on.
  //
  // It cannot be an HBM->L2 transfer, and the measurement says so without any modelling: it
  // reported 7579 GB/s on a part whose HBM is declared at 3350 GB/s. Nothing that leaves HBM can
  // exceed HBM's bandwidth. The two were never distinguishable while the model also priced this
  // seam at L2's rate alone; charging the source side (Vx 1f8591d9) is what made them disagree,
  // and this is the measurement that disagreement asked for. It is the third defect of this shape
  // in this file -- like the other two, it returned a confident number rather than an error.
  //
  // Cold by construction rather than by flushing. Timing a single cold pass would put the launch
  // floor back into every cell, which is defect #1 in this file's header. Instead the tile is laid
  // end to end until the working set is several times L2, and one pass over the whole span is
  // timed: every tile-sized stretch is then read from HBM because the span cannot be resident.
  // Cost for one tile is the span's cost divided by how many tiles it holds.
  //
  // Residual: the tail of a pass is still in L2 when the next begins, so up to `L2/span` of the
  // traffic can hit. At OVERSUB=8 that bounds it at 12.5%, and it biases the measured rate UP,
  // i.e. toward the old wrong answer -- so it cannot manufacture agreement with the HBM figure.
  {
    const size_t l2 = (size_t)p.l2CacheSize;
    const size_t OVERSUB = 8;
    const size_t target_span = l2 * OVERSUB;
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));

    for (int i = 0; i < N_SIZES; ++i) {
      size_t bytes = SIZES[i];
      if (bytes % sizeof(float4)) continue;

      // How many tiles it takes to overflow L2, and the span they occupy.
      size_t tiles = (target_span + bytes - 1) / bytes;
      if (tiles < 2) tiles = 2;
      size_t span = tiles * bytes;
      // Leave the device room to breathe; a span that does not fit is not measured rather than
      // silently shrunk to something L2-resident, which is the failure this cell exists to fix.
      if (span > free_b / 2) {
        printf("HBM->L2_fill,%zu,ps,,,,,%d,span_exceeds_free_memory_not_measured\n", bytes, REPS);
        continue;
      }

      float4 *d = nullptr;
      float *sink = nullptr;
      if (cudaMalloc(&d, span) != cudaSuccess) {
        printf("HBM->L2_fill,%zu,ps,,,,,%d,alloc_failed\n", bytes, REPS);
        continue;
      }
      CK(cudaMalloc(&sink, sizeof(float)));
      CK(cudaMemset(d, 1, span));
      size_t n4 = span / sizeof(float4);

      int block = 256;
      size_t want_blocks = (n4 + block - 1) / block;
      int max_blocks = p.multiProcessorCount * 32;
      int grid = (int)(want_blocks < (size_t)max_blocks ? want_blocks : (size_t)max_blocks);
      if (grid < 1) grid = 1;

      cudaEvent_t a, b;
      CK(cudaEventCreate(&a));
      CK(cudaEventCreate(&b));
      l2_stream<<<grid, block>>>(d, n4, sink, 1);
      CK(cudaDeviceSynchronize());

      int iters = 1;
      for (;;) {
        CK(cudaEventRecord(a));
        l2_stream<<<grid, block>>>(d, n4, sink, iters);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (ms >= 5.0f || iters >= (1 << 22)) break;
        int grow = (int)(6.0f / (ms > 0.01f ? ms : 0.01f)) + 1;
        iters = iters * (grow > 2 ? grow : 2);
      }

      std::vector<double> s;
      for (int r = 0; r < REPS; ++r) {
        CK(cudaEventRecord(a));
        l2_stream<<<grid, block>>>(d, n4, sink, iters);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        // ps for ONE tile: the batch covers `iters` passes over `tiles` tiles.
        s.push_back((double)ms * 1e9 / ((double)iters * (double)tiles));
      }
      double q1, q3, med = median_of(s, &q1, &q3);
      fprintf(stderr, "  HBM->L2_fill %zu B: span=%zu MiB tiles=%zu grid=%d iters=%d\n", bytes,
              span >> 20, tiles, grid, iters);
      printf("HBM->L2_fill,%zu,ps,%.0f,%.0f,%.0f,%.2f,%d,%s\n", bytes, med, q1, q3,
             med > 0 ? (double)bytes / (med / 1000.0) : 0.0, REPS, "hbm_read_l2_oversubscribed");
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
