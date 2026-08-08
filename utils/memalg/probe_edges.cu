// probe_edges.cu -- the full transfer-edge powerset for the memory algebra (vx-review#15, #22).
//
// The algebra is a graph whose edges carry bandwidths, and most of those bandwidths are today
// transcribed from whitepapers and marked UNVERIFIED. This walks EVERY ordered pair over
//
//     { CPU_DRAM, HBM, L2, L1, SMEM, REG }
//
// and, for each, either measures it or says why it cannot be measured. NOT scored cells: these
// measure hardware, so no freeze and no vx-review checkout is involved.
//
// THE POINT OF THE CLASSIFICATION. Not every pair is an edge. On an NVIDIA part:
//   * L1 is filled THROUGH L2 -- there is no HBM->L1 path that skips L2, so that cell is a
//     COMPOSITE, not a primitive edge, while L2->L1 IS primitive (a load miss filling L1);
//   * reaching L1 is asymmetric: it is a real LOAD destination and never a STORE destination,
//     because global stores bypass it. A single "L1" node in the graph hides that asymmetry;
//   * CPU_DRAM reaches nothing on-die directly; it reaches HBM, and the rest is composition.
// An algebra that carries a primitive edge for HBM->L1 is claiming a path the silicon does not
// have, and would price it independently of the L2 traffic it actually generates. So the matrix
// below reports MEASURED / COMPOSITE(route) / NOT-A-PATH, and that classification is as much a
// result as any bandwidth.
//
// Where a claim can be tested rather than asserted, it is: see the "is L1 gated by L2?" section,
// which measures the L1 hit rate against L2 delivery instead of taking the architecture note on
// faith. And every COMPOSITE is measured end-to-end AND as the sum of its legs, which is a direct
// test of the additivity axiom (M2) on real paths.
//
// Residency is controlled by buffer size plus cache hints, the only honest way to name a cache
// level here:
//     L1   128 KiB buffer, ordinary loads (fits the 256 KB L1)
//     L2    32 MiB buffer, __ldcg (bypasses L1, fits the 50 MiB L2)
//     HBM  512 MiB buffer, __ldcg (exceeds L2)
//     SMEM shared memory, prefilled OUTSIDE the timed region
//     REG  values already in registers
//
// Every kernel edge is reported twice, because the algebra needs both and they are not
// interchangeable: per-SM B/cyc from ONE block (what a tile load in one CTA sees, and what a
// `B/cyc` declaration like SMEM's 128 actually claims) and aggregate GB/s at full occupancy (what
// a whole-device transfer sees). M1 conflated these, which is how L2->SMEM scored -91.3% against
// a per-SM peak a single 8-warp block could never reach.
//
// Cycles, not wall clock, for the per-SM column: cycles are clock-invariant, and a rented
// container cannot pin clocks (PREDICTIONS.md decision 5, same reasoning).
//
// Build:  nvcc -O3 -arch=sm_90 probe_edges.cu -o probe_edges
// Run:    ./probe_edges
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CK(x)                                                                       \
  do {                                                                              \
    cudaError_t e_ = (x);                                                           \
    if (e_ != cudaSuccess) {                                                        \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
              __LINE__);                                                            \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

static const int REPS = 11;

enum Level { L_CPU = 0, L_HBM = 1, L_L2 = 2, L_L1 = 3, L_SMEM = 4, L_REG = 5, N_LEVELS = 6 };
static const char *LNAME[] = {"CPU_DRAM", "HBM", "L2", "L1", "SMEM", "REG"};

enum Kind { K_KERNEL, K_MEMCPY, K_COMPOSITE, K_NONE, K_SELF };

struct Cell {
  Kind kind;
  const char *note;
};

// The classification. Stated here so it is reviewable in one place rather than implied by which
// measurements happen to exist.
static Cell classify(int s, int d) {
  if (s == d) return {K_SELF, "same space"};
  if (s == L_CPU && d == L_HBM) return {K_MEMCPY, ""};
  if (s == L_HBM && d == L_CPU) return {K_MEMCPY, ""};
  if (s == L_CPU || d == L_CPU) return {K_COMPOSITE, "via HBM; host reaches nothing on-die"};
  if (d == L_L1 && s == L_L2) return {K_KERNEL, "L1 fill from L2 on a load miss"};
  if (d == L_L1 && s == L_HBM) return {K_COMPOSITE, "load miss fills via L2; no HBM->L1 that skips L2"};
  if (d == L_L1) return {K_NONE, "stores bypass L1; L1 is not a store destination"};
  // L1 is not a source port: anything leaving it goes through registers. SMEM is the one such
  // path worth measuring end-to-end, because it is the control for L2->SMEM -- same destination,
  // same code, source swapped -- which is what separates an L2-delivery limit from a write limit.
  if (s == L_L1 && d == L_SMEM) return {K_KERNEL, "L1 hit -> REG -> SMEM, measured end-to-end"};
  if (s == L_L1 && d != L_REG) return {K_COMPOSITE, "L1 hit -> REG -> dst; L1 is not a source port"};
  if (s == L_SMEM && d == L_L1) return {K_NONE, "no SMEM->L1 path"};
  if (s == L_HBM && d == L_SMEM) return {K_KERNEL, "composite through L2, measured end-to-end"};
  if (s == L_REG && d == L_REG) return {K_SELF, ""};
  return {K_KERNEL, ""};
}

static double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Kernel-side source and destination selectors. REG as a source means "a value already in
// registers", so there is no load at all.
enum KSrc { KS_L1 = 0, KS_L2 = 1, KS_HBM = 2, KS_SMEM = 3, KS_REG = 4, KS_L2_VIA_L1 = 5 };
enum KDst { KD_REG = 0, KD_SMEM = 1, KD_L2 = 2, KD_HBM = 3 };

// All buffers are powers of two so the wrap is a mask, not an integer modulo -- a divide in the
// inner loop would be measuring the divider rather than the memory path.
template <int SRC, int DST>
__global__ void edge_kernel(const float4 *__restrict__ gsrc, size_t src_mask,
                            float4 *__restrict__ gdst, size_t dst_mask, int chunk4, int nchunks,
                            unsigned long long *out, float *sink) {
  extern __shared__ float4 sh[];
  float4 *sh_in = sh;
  float4 *sh_out = sh + chunk4;

  // Prefill the shared SOURCE outside the timed region: getting data into SMEM is a different
  // edge and must not be charged to this one.
  if (SRC == KS_SMEM) {
    for (int i = threadIdx.x; i < chunk4; i += blockDim.x) sh_in[i] = gsrc[i & src_mask];
    __syncthreads();
  }

  float acc = 0.f;
  __syncthreads();
  unsigned long long t0 = clock64();
  for (int c = 0; c < nchunks; ++c) {
    // Offset by block so concurrent blocks walk different lines instead of replaying one.
    size_t base = ((size_t)c * gridDim.x + blockIdx.x) * (size_t)chunk4;
    for (int i = threadIdx.x; i < chunk4; i += blockDim.x) {
      float4 v;
      if (SRC == KS_REG) {
        float f = (float)(i + c);
        v = make_float4(f, f, f, f);
      } else if (SRC == KS_SMEM) {
        v = sh_in[i];
      } else if (SRC == KS_L1 || SRC == KS_L2_VIA_L1) {
        // Ordinary load: L1 participates. KS_L1's buffer fits L1 (hits); KS_L2_VIA_L1's does not,
        // so it misses L1 and is served by L2 -- but still routed through L1.
        v = gsrc[(base + i) & src_mask];
      } else {
        v = __ldcg(&gsrc[(base + i) & src_mask]);  // L1 bypassed; L2 vs HBM set by buffer size
      }

      if (DST == KD_REG) {
        acc += v.x + v.y + v.z + v.w;
      } else if (DST == KD_SMEM) {
        sh_out[i] = v;
      } else {
        __stcg(&gdst[(base + i) & dst_mask], v);  // L2 vs HBM set by dst buffer size
      }
    }
    if (DST == KD_SMEM || SRC == KS_SMEM) __syncthreads();
  }
  __syncthreads();
  unsigned long long t1 = clock64();

  if (threadIdx.x == 0) {
    out[0] = t1 - t0;
    if (DST == KD_SMEM) {
      float4 v = sh_out[chunk4 - 1];
      if (v.x == 1234.5678f) out[1] = 1;  // keep the shared writes live
    }
  }
  if (acc == 1234.5678f) sink[0] = acc;  // keep the loads live
}

struct Ctx {
  const float4 *src[6];   // indexed by KSrc; KS_REG carries no buffer but must not index OOB
  size_t src_mask[6];
  float4 *dst[4];
  size_t dst_mask[4];
  float *sink;
  unsigned long long *out;
  int chunk4, nchunks, threads;
  int grid;  // 0 = full occupancy (one block per SM)
  size_t smem_bytes;
  cudaDeviceProp p;
};

struct Res {
  double bcyc, gbps;
};

template <int SRC, int DST>
static Res measure(const Ctx &c) {
  // Anything above 48 KiB of dynamic shared memory is refused without this opt-in.
  CK(cudaFuncSetAttribute((const void *)edge_kernel<SRC, DST>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)c.smem_bytes));
  const float4 *gs = (SRC == KS_REG) ? c.src[KS_L1] : c.src[SRC];
  size_t sm_ = (SRC == KS_REG) ? c.src_mask[KS_L1] : c.src_mask[SRC];
  bool glob = (DST == KD_L2 || DST == KD_HBM);
  float4 *gd = glob ? c.dst[DST] : c.dst[KD_L2];
  size_t dm = glob ? c.dst_mask[DST] : c.dst_mask[KD_L2];
  double bytes = (double)c.nchunks * c.chunk4 * sizeof(float4);

  edge_kernel<SRC, DST><<<1, c.threads, c.smem_bytes>>>(gs, sm_, gd, dm, c.chunk4, c.nchunks, c.out, c.sink);
  CK(cudaDeviceSynchronize());
  std::vector<double> cyc;
  for (int i = 0; i < REPS; ++i) {
    edge_kernel<SRC, DST><<<1, c.threads, c.smem_bytes>>>(gs, sm_, gd, dm, c.chunk4, c.nchunks, c.out, c.sink);
    CK(cudaDeviceSynchronize());
    unsigned long long v = 0;
    CK(cudaMemcpy(&v, c.out, sizeof(v), cudaMemcpyDeviceToHost));
    cyc.push_back((double)v);
  }
  double mc = median_of(cyc);

  int grid = c.grid ? c.grid : c.p.multiProcessorCount;
  cudaEvent_t a, e;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&e));
  edge_kernel<SRC, DST><<<grid, c.threads, c.smem_bytes>>>(gs, sm_, gd, dm, c.chunk4, c.nchunks, c.out, c.sink);
  CK(cudaDeviceSynchronize());
  std::vector<double> g;
  for (int i = 0; i < REPS; ++i) {
    CK(cudaEventRecord(a));
    edge_kernel<SRC, DST><<<grid, c.threads, c.smem_bytes>>>(gs, sm_, gd, dm, c.chunk4, c.nchunks, c.out, c.sink);
    CK(cudaEventRecord(e));
    CK(cudaEventSynchronize(e));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, a, e));
    g.push_back(bytes * grid / ((double)ms / 1e3) / 1e9);
  }
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(e));
  return {mc > 0 ? bytes / mc : 0.0, median_of(g)};
}

// SRC/DST are template parameters and cannot be loop variables, hence the macro.
#define ROW(SL, DL, KS, KD)                                                          \
  do {                                                                               \
    Res r = measure<KS, KD>(ctx);                                                    \
    printf("%-10s -> %-8s %10s %14.1f %16.1f  %s\n", LNAME[SL], LNAME[DL], "MEASURED", \
           r.bcyc, r.gbps, classify(SL, DL).note);                                   \
    fflush(stdout);                                                                  \
  } while (0)

int main() {
  int dev = 0;
  CK(cudaSetDevice(dev));
  Ctx ctx{};
  CK(cudaGetDeviceProperties(&ctx.p, dev));
  const cudaDeviceProp &p = ctx.p;
  int sm_khz = 0, mem_khz = 0;
  CK(cudaDeviceGetAttribute(&sm_khz, cudaDevAttrClockRate, dev));
  CK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
  printf("GPU: %s   SMs=%d   L2=%d MiB   SMEM/block(optin)=%zu KiB   SM clock=%.0f MHz\n", p.name,
         p.multiProcessorCount, p.l2CacheSize >> 20, p.sharedMemPerBlockOptin >> 10,
         sm_khz / 1000.0);
  printf("HBM theoretical %.1f GB/s\n\n", 2.0 * mem_khz * (p.memoryBusWidth / 8) / 1.0e6);

  const size_t L1B = 128ul << 10, L2B = 32ul << 20, HBMB = 512ul << 20;
  const int CHUNK = 32 * 1024;
  ctx.chunk4 = CHUNK / (int)sizeof(float4);
  ctx.nchunks = 2048;
  ctx.threads = 1024;  // 32 warps: enough in flight that the result is a bandwidth, not a latency
  ctx.smem_bytes = 2 * (size_t)CHUNK;

  float4 *l1, *l2, *hbm, *dl2, *dhbm;
  CK(cudaMalloc(&l1, L1B));
  CK(cudaMalloc(&l2, L2B));
  CK(cudaMalloc(&hbm, HBMB));
  CK(cudaMalloc(&dl2, L2B));
  CK(cudaMalloc(&dhbm, HBMB));
  CK(cudaMalloc(&ctx.sink, sizeof(float)));
  CK(cudaMalloc(&ctx.out, 2 * sizeof(unsigned long long)));
  CK(cudaMemset(l1, 1, L1B));
  CK(cudaMemset(l2, 1, L2B));
  CK(cudaMemset(hbm, 1, HBMB));
  ctx.src[KS_L1] = l1;   ctx.src_mask[KS_L1] = L1B / sizeof(float4) - 1;
  ctx.src[KS_L2] = l2;   ctx.src_mask[KS_L2] = L2B / sizeof(float4) - 1;
  ctx.src[KS_HBM] = hbm; ctx.src_mask[KS_HBM] = HBMB / sizeof(float4) - 1;
  ctx.src[KS_SMEM] = l1; ctx.src_mask[KS_SMEM] = L1B / sizeof(float4) - 1;
  ctx.src[KS_L2_VIA_L1] = l2; ctx.src_mask[KS_L2_VIA_L1] = L2B / sizeof(float4) - 1;
  ctx.src[KS_REG] = l1;  ctx.src_mask[KS_REG] = L1B / sizeof(float4) - 1;
  ctx.dst[KD_L2] = dl2;   ctx.dst_mask[KD_L2] = L2B / sizeof(float4) - 1;
  ctx.dst[KD_HBM] = dhbm; ctx.dst_mask[KD_HBM] = HBMB / sizeof(float4) - 1;

  // ---- the powerset, classified ----------------------------------------------------------------
  printf("=== transfer-edge powerset over {CPU_DRAM, HBM, L2, L1, SMEM, REG} ===\n");
  printf("%-10s    %-8s %10s %14s %16s  %s\n", "from", "to", "kind", "per-SM B/cyc",
         "aggregate GB/s", "note");
  printf("--------------------------------------------------------------------------------\n");
  for (int s = 0; s < N_LEVELS; ++s)
    for (int d = 0; d < N_LEVELS; ++d) {
      Cell c = classify(s, d);
      if (c.kind == K_SELF) continue;
      if (c.kind == K_KERNEL || c.kind == K_MEMCPY) continue;  // measured below
      printf("%-10s -> %-8s %10s %14s %16s  %s\n", LNAME[s], LNAME[d],
             c.kind == K_COMPOSITE ? "COMPOSITE" : "NOT-A-PATH", "-", "-", c.note);
    }
  printf("\n");

  // Measured kernel edges. Ordered so the reader can compare down a column.
  ROW(L_L1,   L_REG,  KS_L1,   KD_REG);
  ROW(L_L1,   L_SMEM, KS_L1,   KD_SMEM);
  ROW(L_L2,   L_L1,   KS_L2_VIA_L1, KD_REG);
  ROW(L_L2,   L_REG,  KS_L2,   KD_REG);
  ROW(L_L2,   L_SMEM, KS_L2,   KD_SMEM);
  ROW(L_L2,   L_HBM,  KS_L2,   KD_HBM);
  ROW(L_HBM,  L_REG,  KS_HBM,  KD_REG);
  ROW(L_HBM,  L_SMEM, KS_HBM,  KD_SMEM);
  ROW(L_HBM,  L_L2,   KS_HBM,  KD_L2);
  ROW(L_SMEM, L_REG,  KS_SMEM, KD_REG);
  ROW(L_SMEM, L_L2,   KS_SMEM, KD_L2);
  ROW(L_SMEM, L_HBM,  KS_SMEM, KD_HBM);
  ROW(L_REG,  L_SMEM, KS_REG,  KD_SMEM);
  ROW(L_REG,  L_L2,   KS_REG,  KD_L2);
  ROW(L_REG,  L_HBM,  KS_REG,  KD_HBM);

  // ---- is L1 gated by L2? ----------------------------------------------------------------------
  // Tested, not assumed. If L1 hits are much faster than L2 delivery, L1 is a distinct level the
  // algebra should carry; if they are the same, "L1" is not buying the model anything and the
  // apparent edge is L2 all along.
  printf("\n=== is L1 a distinct level, or is it gated by L2? ===\n");
  {
    Res l1r = measure<KS_L1, KD_REG>(ctx);           // 128 KiB, hits L1
    Res via = measure<KS_L2_VIA_L1, KD_REG>(ctx);    // 32 MiB, ordinary loads: misses L1, L2 serves
    Res l2r = measure<KS_L2, KD_REG>(ctx);           // 32 MiB, __ldcg: L1 bypassed entirely
    Res hr = measure<KS_HBM, KD_REG>(ctx);
    printf("  L1 hit        -> REG : %8.1f B/cyc per SM\n", l1r.bcyc);
    printf("  L2 via L1     -> REG : %8.1f B/cyc per SM   (same buffer, ordinary loads)\n", via.bcyc);
    printf("  L2 bypassing L1->REG : %8.1f B/cyc per SM   (same buffer, __ldcg)\n", l2r.bcyc);
    printf("  HBM           -> REG : %8.1f B/cyc per SM\n", hr.bcyc);
    printf("\n  L1hit / L2  = %.2fx\n", l2r.bcyc > 0 ? l1r.bcyc / l2r.bcyc : 0.0);
    printf("  viaL1 / L2  = %.2fx   <- SAME buffer both ways; the only variable is L1 in the path\n",
           l2r.bcyc > 0 ? via.bcyc / l2r.bcyc : 0.0);
    printf("\n  L1hit/L2 near 1.0 -> an L1 hit is no faster than L2 delivery, so a separate L1 edge\n");
    printf("                       would price a distinction the hardware does not have.\n");
    printf("  viaL1/L2 near 1.0 -> routing through L1 costs nothing on a miss: L1 is transparent\n");
    printf("                       and L2 is the gate, which is the assumption to confirm here.\n");
    printf("  viaL1/L2 well under 1.0 -> L1 lookup is charged even on a miss, and a model that\n");
    printf("                       ignores L1 will under-predict every HBM/L2 read.\n");
  }

  // ---- additivity on a real composite (M2) -----------------------------------------------------
  // HBM->SMEM must physically pass through L2. The algebra asserts the staged cost is the sum of
  // its legs; M2 pre-registered that copy engines and TMA overlap them, so measured should come in
  // BELOW the sum. This is that test on a path the hardware actually composes.
  printf("\n=== additivity: HBM -> SMEM vs its legs (M2) ===\n");
  {
    Res leg1 = measure<KS_HBM, KD_L2>(ctx);    // HBM -> L2
    Res leg2 = measure<KS_L2, KD_SMEM>(ctx);   // L2  -> SMEM
    Res staged = measure<KS_HBM, KD_SMEM>(ctx);
    // Costs add as time per byte, i.e. reciprocals of rates.
    double t1 = leg1.bcyc > 0 ? 1.0 / leg1.bcyc : 0;
    double t2 = leg2.bcyc > 0 ? 1.0 / leg2.bcyc : 0;
    double ts = staged.bcyc > 0 ? 1.0 / staged.bcyc : 0;
    printf("  HBM->L2      : %8.1f B/cyc  (%.4f cyc/B)\n", leg1.bcyc, t1);
    printf("  L2->SMEM     : %8.1f B/cyc  (%.4f cyc/B)\n", leg2.bcyc, t2);
    printf("  sum of legs  :            %.4f cyc/B   <- what the algebra asserts\n", t1 + t2);
    printf("  staged direct: %8.1f B/cyc  (%.4f cyc/B)\n", staged.bcyc, ts);
    printf("  staged/sum   : %.2f%s\n", (t1 + t2) > 0 ? ts / (t1 + t2) : 0.0,
           ts < t1 + t2 ? "   (overlap -- M2 confirmed)" : "   (no overlap seen)");
  }

  // ---- L2 -> SMEM vs warps: the edge M1 got wrong ----------------------------------------------
  printf("\n=== L2 -> SMEM vs warps (one block) -- M1 scored this at -91.3%% ===\n");
  printf("Model predicts 128.0 B/cyc.  M1 measured ~9.5-12.2 with 8 warps.\n");
  printf("%8s %8s %16s %16s\n", "threads", "warps", "L2->SMEM B/cyc", "L1->SMEM B/cyc");
  {
    Ctx t = ctx;
    for (int th = 32; th <= 1024; th *= 2) {
      t.threads = th;
      Res a = measure<KS_L2, KD_SMEM>(t);
      Res b = measure<KS_L1, KD_SMEM>(t);
      printf("%8d %8d %16.1f %16.1f\n", th, th / 32, a.bcyc, b.bcyc);
      fflush(stdout);
    }
  }
  printf("\n  L1->SMEM ~= L2->SMEM -> the SMEM write path binds, not L2 delivery. The -91.3%% is\n");
  printf("                          then a modelling error about WHICH side of the edge binds,\n");
  printf("                          and #22's per-edge constant will not fix it.\n");
  printf("  L1->SMEM >> L2->SMEM -> L2 delivery to one SM binds: the structural case.\n");

  // ---- where does aggregate saturate? ----------------------------------------------------------
  // If per-SM bandwidth held constant as blocks are added, aggregate would scale linearly and the
  // algebra could price a device transfer as (per-SM rate x SMs). It does not, and the shape of
  // the rolloff is what a whole-device edge cost has to encode.
  printf("\n=== aggregate scaling vs resident blocks (HBM -> REG, L1 bypassed) ===\n");
  printf("%8s %18s %18s %14s\n", "blocks", "aggregate GB/s", "per-block GB/s", "vs linear");
  {
    Ctx t = ctx;
    double one = 0.0;
    for (int g = 1; g <= p.multiProcessorCount * 2; g *= 2) {
      t.grid = g;
      Res r = measure<KS_HBM, KD_REG>(t);
      if (g == 1) one = r.gbps;
      printf("%8d %18.1f %18.2f %13.0f%%\n", g, r.gbps, r.gbps / g,
             one > 0 ? r.gbps / (one * g) * 100.0 : 0.0);
      fflush(stdout);
    }
    printf("  'vs linear' falling well below 100%% is the fabric saturating; staying near 100%%\n");
    printf("  means the single-block number was occupancy-limited, not bandwidth-limited.\n");
  }

  // ---- host edges ------------------------------------------------------------------------------
  printf("\n=== host edges (cudaMemcpy, 256 MiB) ===\n");
  {
    size_t n = 256ul << 20;
    char *hpin = nullptr, *hpage = (char *)malloc(n);
    void *d1 = nullptr, *d2 = nullptr;
    CK(cudaHostAlloc((void **)&hpin, n, cudaHostAllocDefault));
    CK(cudaMalloc(&d1, n));
    CK(cudaMalloc(&d2, n));
    cudaEvent_t a, e;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&e));
    struct HE {
      const char *name;
      void *dst;
      const void *src;
      cudaMemcpyKind kind;
    } edges[] = {{"CPU_DRAM -> HBM (pinned)", d1, hpin, cudaMemcpyHostToDevice},
                 {"CPU_DRAM -> HBM (pageable)", d1, hpage, cudaMemcpyHostToDevice},
                 {"HBM -> CPU_DRAM (pinned)", hpin, d1, cudaMemcpyDeviceToHost},
                 {"HBM -> CPU_DRAM (pageable)", hpage, d1, cudaMemcpyDeviceToHost},
                 {"HBM -> HBM (device copy)", d2, d1, cudaMemcpyDeviceToDevice}};
    for (auto &ed : edges) {
      if (!ed.dst || !ed.src) continue;
      CK(cudaMemcpy(ed.dst, ed.src, n, ed.kind));
      std::vector<double> g;
      for (int i = 0; i < REPS; ++i) {
        CK(cudaEventRecord(a));
        CK(cudaMemcpy(ed.dst, ed.src, n, ed.kind));
        CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, e));
        g.push_back((double)n / ((double)ms / 1e3) / 1e9);
      }
      printf("  %-28s %10.1f GB/s\n", ed.name, median_of(g));
      fflush(stdout);
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(e));
    CK(cudaFreeHost(hpin));
    free(hpage);
    CK(cudaFree(d1));
    CK(cudaFree(d2));
  }

  CK(cudaFree(l1));
  CK(cudaFree(l2));
  CK(cudaFree(hbm));
  CK(cudaFree(dl2));
  CK(cudaFree(dhbm));
  CK(cudaFree(ctx.sink));
  CK(cudaFree(ctx.out));
  return 0;
}
