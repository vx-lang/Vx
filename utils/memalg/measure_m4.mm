// measure_m4.mm -- on-die seam instrument for an Apple GPU (vx-review#15, #16, #20).
//
// The Metal counterpart of measure_device.cu. It exists because fleet/m4-uma.vx is a HELD-OUT SKU
// -- 30 frozen cells, never scored -- that also happens to be the development machine, so the whole
// predict -> measure -> compare loop runs for free and settles a question the CUDA instrument
// cannot: whether the composition law found on an H100 survives a change of vendor and ISA.
//
// Read utils/memalg/M4_PREREGISTRATION.md first. It states P1..P4 with their falsifiers and was
// committed before this file existed.
//
// Build:  clang++ -std=c++17 -ObjC++ -fobjc-arc -O2 measure_m4.mm \
//             -framework Metal -framework Foundation -o measure_m4
// Run:    ./measure_m4 > m4.csv 2> m4.log
//
// No `xcrun metal` is needed and none is installed here: the shaders are compiled at runtime by
// `newLibraryWithSource:`, which is part of the Metal framework itself.
//
// PROTOCOL
//   * >= 11 reps, median with IQR, never a mean; one warm-up discarded.
//   * GPU time from the command buffer's own GPUStartTime/GPUEndTime, not host wall-clock, so
//     encoding and submission are excluded.
//   * Every rate is reported against the declared 120 GB/s peak. A ratio above 1.0x is impossible
//     for DRAM traffic and means the buffer was served from a cache -- printing the ratio makes
//     that visible instead of leaving it to be noticed. Six of ten points in the Tier-0 CPU
//     shakedown had exactly that defect and it went unreported for five days.
//   * Buffers for DRAM-scale seams are sized from the MEASURED cache knee, not from a guess.
//
// WHAT IS DELIBERATELY NOT MEASURED
//   `L2 -> SMEM` in cycles. It is declared `B/cyc` against an UNVERIFIED 1.4 GHz placeholder and
//   Metal exposes no cycle counter to a kernel, so it can only be timed in wall-clock. Protocol
//   decision 5 forbids converting with an invented clock, so that cell stays unscorable (P4).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The declared figure every rate is sanity-checked against. spec: Apple M4, 120 GB/s
// (LPDDR5X-7500 on a 128-bit bus), the same number fleet/m4-uma.vx declares.
static const double DECLARED_PEAK_GBPS = 120.0;

static const int REPS = 13;

// Threads per threadgroup. 256 keeps every configuration below maxTotalThreadsPerThreadgroup
// (1024 here) while giving eight SIMD groups of 32 to hide latency with.
static const int TG_THREADS = 256;

// Threadgroups dispatched for an "aggregate" (whole-GPU) measurement. The M4 has 10 GPU cores, so
// 64 oversubscribes them enough to hide scheduling.
static const int TG_COUNT = 64;

// Total bytes every seam moves, held CONSTANT across sizes and across seams.
//
// This is not a detail. The first run varied it -- `iters` was chosen to cap total work, so a
// 512 MiB buffer moved 512 MiB in one dispatch while a 16 MiB buffer moved only 64 MiB across four
// -- and the fixed dispatch floor is then amortised unequally. The sweep came out NON-MONOTONIC,
// with 256 MiB "faster" than 16 MiB, which is backwards for a working-set sweep and is an artifact
// of the harness rather than anything about the cache. Constant work per point is what makes the
// sizes comparable at all.
static const size_t MOVED_BYTES = 256ull * 1024 * 1024;

/// Re-read count that moves `MOVED_BYTES` through a buffer of `bytes`.
static uint32_t iters_for(size_t bytes) {
  return (uint32_t)std::max<size_t>(1, MOVED_BYTES / bytes);
}

// ---- shaders ---------------------------------------------------------------------------------
// Every kernel guards its result behind a comparison the compiler cannot fold away, so the loads
// and stores it exists to time cannot be eliminated. `1e30f` never occurs in the data.
//
// The threadgroup kernels keep their stores live with a single element read back after a barrier
// rather than a full sweep. Threadgroup memory is shared, so one thread reading one element after
// a barrier means every store is observable and none can be dropped -- and it adds one element of
// traffic instead of doubling the measurement with a read leg that is not part of the seam.
static NSString *const kShaders = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// A streaming read. `iters` re-reads the buffer so that dispatch overhead is amortised rather
// than measured: the model predicts the time to move bytes, and launch is not part of that claim.
// Float addition is not associative, so the passes cannot be folded into a multiply.
kernel void stream_read(device const float4* src   [[buffer(0)]],
                        constant uint&      n4     [[buffer(1)]],
                        constant uint&      iters  [[buffer(2)]],
                        device float*       sink   [[buffer(3)]],
                        uint gid  [[thread_position_in_grid]],
                        uint gsz  [[threads_per_grid]]) {
  float4 acc = float4(0.0f);
  for (uint it = 0; it < iters; ++it)
    for (uint i = gid; i < n4; i += gsz)
      acc += src[i];
  if (acc.x == 1e30f) sink[0] = acc.x + acc.y + acc.z + acc.w;
}

// Registers -> threadgroup. Touches no global memory at all: the values are made from the thread
// index, so what is timed is the store into threadgroup memory and nothing else.
kernel void reg_to_tg(constant uint&        iters [[buffer(0)]],
                      device float*         sink  [[buffer(1)]],
                      constant uint&        tgn4  [[buffer(2)]],
                      threadgroup float4*   tg    [[threadgroup(0)]],
                      uint tid  [[thread_position_in_threadgroup]],
                      uint tsz  [[threads_per_threadgroup]]) {
  float4 acc = float4(0.0f);
  for (uint it = 0; it < iters; ++it) {
    for (uint i = tid; i < tgn4; i += tsz)
      tg[i] = float4(float(i + it));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) acc += tg[it % tgn4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (acc.x == 1e30f) sink[0] = acc.x;
}

// Global -> threadgroup: a load into a register followed by a threadgroup store. This is the route
// kind P1 is about -- on Apple family 9 there is no asynchronous copy engine, so this is two
// instructions and not a hardware-streamed fill.
kernel void global_to_tg(device const float4* src   [[buffer(0)]],
                         constant uint&       n4    [[buffer(1)]],
                         constant uint&       iters [[buffer(2)]],
                         device float*        sink  [[buffer(3)]],
                         constant uint&       tgn4  [[buffer(4)]],
                         threadgroup float4*  tg    [[threadgroup(0)]],
                         uint tid  [[thread_position_in_threadgroup]],
                         uint tsz  [[threads_per_threadgroup]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint ntg  [[threadgroups_per_grid]]) {
  float4 acc = float4(0.0f);
  for (uint it = 0; it < iters; ++it) {
    for (uint base = tgid * tgn4; base < n4; base += ntg * tgn4) {
      uint lim = min(tgn4, n4 - base);
      for (uint i = tid; i < lim; i += tsz)
        tg[i] = src[base + i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0) acc += tg[0];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (acc.x == 1e30f) sink[0] = acc.x;
}
)MSL";

// ---- harness ---------------------------------------------------------------------------------

static double median_of(std::vector<double> &v, double *q1, double *q3) {
  std::sort(v.begin(), v.end());
  if (q1) *q1 = v[v.size() / 4];
  if (q3) *q3 = v[(3 * v.size()) / 4];
  return v[v.size() / 2];
}

/// One declared-number row (vx-review#18), same shape as measure_device.cu's.
static void fact_row(const char *seam, size_t bytes, const char *unit, double value,
                     const char *note) {
  printf("%s,%zu,%s,%.3f,,,,1,%s\n", seam, bytes, unit, value, note);
}

struct Timing {
  double median, q1, q3;
};

/// Median GPU seconds over REPS dispatches of an already-configured encoder callback.
static Timing time_dispatch(id<MTLCommandQueue> q, void (^encode)(id<MTLComputeCommandEncoder>),
                            MTLSize grid, MTLSize tg) {
  std::vector<double> samples;
  samples.reserve(REPS);
  for (int r = 0; r < REPS + 1; ++r) {
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      encode(enc);
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (r == 0) continue;  // warm-up discarded
      samples.push_back(cb.GPUEndTime - cb.GPUStartTime);
    }
  }
  Timing t{};
  t.median = median_of(samples, &t.q1, &t.q3);
  return t;
}

/// Emit one seam row, and flag a rate that exceeds the part's declared peak.
static void seam_row(const char *seam, size_t bytes, const Timing &t, int reps, const char *note) {
  const double gbps = bytes / t.median / 1.0e9;
  printf("%s,%zu,ps,%.1f,%.1f,%.1f,%.2f,%d,%s\n", seam, bytes, t.median * 1.0e12,
         t.q1 * 1.0e12, t.q3 * 1.0e12, gbps, reps, note);
  if (gbps > DECLARED_PEAK_GBPS) {
    fprintf(stderr,
            "  NOTE %s at %zu B: %.1f GB/s is %.2fx the declared %.0f GB/s peak. Nothing leaving\n"
            "       DRAM can exceed DRAM bandwidth, so this row is served from a cache and does\n"
            "       not measure the seam its name claims.\n",
            seam, bytes, gbps, gbps / DECLARED_PEAK_GBPS, DECLARED_PEAK_GBPS);
  }
}

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      fprintf(stderr, "no Metal device\n");
      return 1;
    }
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kShaders options:nil error:&err];
    if (!lib) {
      fprintf(stderr, "runtime MSL compile failed: %s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    id<MTLComputePipelineState> ps_stream =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream_read"]
                                           error:&err];
    id<MTLComputePipelineState> ps_reg_tg =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"reg_to_tg"] error:&err];
    id<MTLComputePipelineState> ps_g_tg =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"global_to_tg"]
                                           error:&err];
    if (!ps_stream || !ps_reg_tg || !ps_g_tg) {
      fprintf(stderr, "pipeline creation failed: %s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }

    const size_t tg_max = dev.maxThreadgroupMemoryLength;
    fprintf(stderr, "GPU: %s\n", [[dev name] UTF8String]);
    fprintf(stderr, "  unified memory       : %s\n", dev.hasUnifiedMemory ? "yes" : "no");
    fprintf(stderr, "  threadgroup memory   : %zu B\n", tg_max);
    fprintf(stderr, "  max working set      : %llu B\n",
            (unsigned long long)dev.recommendedMaxWorkingSetSize);
    fprintf(stderr, "  SIMD width           : %lu\n",
            (unsigned long)ps_stream.threadExecutionWidth);
    fprintf(stderr, "  declared peak        : %.0f GB/s\n", DECLARED_PEAK_GBPS);

    printf("seam,bytes,unit,median,q1,q3,derived_rate_GBps,reps,note\n");

    // ---- the declared numbers (vx-review#18) -------------------------------------------------
    fact_row("device/threadgroup_memory", 0, "B", (double)tg_max,
             "MTLDevice::maxThreadgroupMemoryLength -- the SMEM `capacity:` figure");
    fact_row("device/max_working_set", 0, "B", (double)dev.recommendedMaxWorkingSetSize,
             "MTLDevice::recommendedMaxWorkingSetSize -- usable against the declared HBM capacity");
    fact_row("device/unified_memory", 0, "bool", dev.hasUnifiedMemory ? 1.0 : 0.0,
             "MTLDevice::hasUnifiedMemory -- 1 means CPU_DRAM and HBM are one memory (P2)");
    fact_row("device/simd_width", 0, "count", (double)ps_stream.threadExecutionWidth,
             "MTLComputePipelineState::threadExecutionWidth");
    fact_row("device/max_threads_per_tg", 0, "count",
             (double)ps_stream.maxTotalThreadsPerThreadgroup,
             "MTLComputePipelineState::maxTotalThreadsPerThreadgroup");

    id<MTLBuffer> sink = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];

    // ---- P3: where is the GPU's cache knee? --------------------------------------------------
    // The machine file declares `L2` from hw.perflevel0.l2cachesize, which is the CPU P-core
    // cluster's cache. The topology that declares it is `arch: applegpu`. This sweep asks the GPU
    // where its own working set stops fitting, and the answer is what the composite test below
    // uses to pick a cache-resident source -- rather than a guess, which is the mistake that put
    // six impossible rows in the CPU shakedown.
    fprintf(stderr, "\n== working-set sweep (P3): where does the GPU's rate fall off? ==\n");
    size_t knee_bytes = 0;
    double best_rate = 0.0;
    const int tg_count = TG_COUNT;
    // Starts at 256 KiB, not lower, and that bound is derived rather than chosen: the grid is
    // TG_COUNT * TG_THREADS = 16384 threads, and a grid-stride walk over a buffer with fewer than
    // 16384 float4s leaves the surplus threads idle for every pass. Below this size the sweep
    // measures how many threads found work, not how fast the memory is -- which is what made the
    // first run report 64 KiB slower than 128 KiB.
    const size_t sweep_min = (size_t)tg_count * TG_THREADS * 16;
    for (size_t bytes = sweep_min; bytes <= (size_t)512 * 1024 * 1024; bytes *= 2) {
      id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      if (!buf) continue;
      memset(buf.contents, 1, bytes);
      uint32_t n4 = (uint32_t)(bytes / 16);
      uint32_t iters = iters_for(bytes);
      Timing t = time_dispatch(
          queue,
          ^(id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:ps_stream];
            [enc setBuffer:buf offset:0 atIndex:0];
            [enc setBytes:&n4 length:4 atIndex:1];
            [enc setBytes:&iters length:4 atIndex:2];
            [enc setBuffer:sink offset:0 atIndex:3];
          },
          MTLSizeMake(tg_count, 1, 1), MTLSizeMake(TG_THREADS, 1, 1));
      const double moved = (double)bytes * iters;
      const double gbps = moved / t.median / 1.0e9;
      fprintf(stderr, "  %9zu B  %8.1f GB/s  %5.2fx peak\n", bytes, gbps,
              gbps / DECLARED_PEAK_GBPS);
      printf("knee/stream_read,%zu,ps,%.1f,%.1f,%.1f,%.2f,%d,working-set sweep, %u passes\n", bytes,
             t.median * 1.0e12, t.q1 * 1.0e12, t.q3 * 1.0e12, gbps, REPS, iters);
      if (gbps > best_rate) {
        best_rate = gbps;
        knee_bytes = bytes;
      }
    }
    fprintf(stderr, "  fastest at %zu B (%.1f GB/s)\n", knee_bytes, best_rate);

    // A cache-resident source for the on-chip composite, and a DRAM-scale one for the far seam.
    const size_t cached_bytes = std::max<size_t>(knee_bytes, 256 * 1024);
    const size_t dram_bytes = 512 * 1024 * 1024;

    // ---- P1: the composition test ------------------------------------------------------------
    // Composite CACHE -> SMEM against its legs CACHE -> REG and REG -> SMEM. Both legs are on-chip
    // so their rates are comparable, which is what lets sum and bottleneck disagree: with a DRAM
    // source the read leg dominates so completely that the two laws predict nearly the same thing
    // and the experiment could not discriminate.
    fprintf(stderr, "\n== composition (P1): source %zu B, cache-resident ==\n", cached_bytes);

    struct Seam {
      const char *name;
      double per_tg_gbps;
      double aggregate_gbps;
    };
    std::vector<Seam> seams;

    for (int aggregate = 0; aggregate <= 1; ++aggregate) {
      const int groups = aggregate ? tg_count : 1;

      // leg 1: CACHE -> REG
      {
        id<MTLBuffer> buf =
            [dev newBufferWithLength:cached_bytes options:MTLResourceStorageModeShared];
        memset(buf.contents, 1, cached_bytes);
        uint32_t n4 = (uint32_t)(cached_bytes / 16);
        uint32_t iters = iters_for(cached_bytes);
        Timing t = time_dispatch(
            queue,
            ^(id<MTLComputeCommandEncoder> enc) {
              [enc setComputePipelineState:ps_stream];
              [enc setBuffer:buf offset:0 atIndex:0];
              [enc setBytes:&n4 length:4 atIndex:1];
              [enc setBytes:&iters length:4 atIndex:2];
              [enc setBuffer:sink offset:0 atIndex:3];
            },
            MTLSizeMake(groups, 1, 1), MTLSizeMake(TG_THREADS, 1, 1));
        double gbps = (double)cached_bytes * iters / t.median / 1.0e9;
        if (aggregate)
          seams[0].aggregate_gbps = gbps;
        else
          seams.push_back({"L2 -> REG", gbps, 0});
      }

      // leg 2: REG -> SMEM
      {
        uint32_t tgn4 = (uint32_t)(tg_max / 16);
        // Same MOVED_BYTES as every other seam, per threadgroup, so the legs and the composite
        // are amortised identically. A leg measured over a different amount of work is a leg
        // measured against a different dispatch floor.
        uint32_t iters = (uint32_t)(MOVED_BYTES / tg_max);
        Timing t = time_dispatch(
            queue,
            ^(id<MTLComputeCommandEncoder> enc) {
              [enc setComputePipelineState:ps_reg_tg];
              [enc setBytes:&iters length:4 atIndex:0];
              [enc setBuffer:sink offset:0 atIndex:1];
              [enc setBytes:&tgn4 length:4 atIndex:2];
              [enc setThreadgroupMemoryLength:tg_max atIndex:0];
            },
            MTLSizeMake(groups, 1, 1), MTLSizeMake(TG_THREADS, 1, 1));
        double moved = (double)tg_max * iters * groups;
        double gbps = moved / t.median / 1.0e9;
        if (aggregate)
          seams[1].aggregate_gbps = gbps;
        else
          seams.push_back({"REG -> SMEM", gbps, 0});
      }

      // composite: CACHE -> SMEM
      {
        id<MTLBuffer> buf =
            [dev newBufferWithLength:cached_bytes options:MTLResourceStorageModeShared];
        memset(buf.contents, 1, cached_bytes);
        uint32_t n4 = (uint32_t)(cached_bytes / 16);
        uint32_t tgn4 = (uint32_t)(tg_max / 16);
        uint32_t iters = iters_for(cached_bytes);
        Timing t = time_dispatch(
            queue,
            ^(id<MTLComputeCommandEncoder> enc) {
              [enc setComputePipelineState:ps_g_tg];
              [enc setBuffer:buf offset:0 atIndex:0];
              [enc setBytes:&n4 length:4 atIndex:1];
              [enc setBytes:&iters length:4 atIndex:2];
              [enc setBuffer:sink offset:0 atIndex:3];
              [enc setBytes:&tgn4 length:4 atIndex:4];
              [enc setThreadgroupMemoryLength:tg_max atIndex:0];
            },
            MTLSizeMake(groups, 1, 1), MTLSizeMake(TG_THREADS, 1, 1));
        double gbps = (double)cached_bytes * iters / t.median / 1.0e9;
        if (aggregate)
          seams[2].aggregate_gbps = gbps;
        else
          seams.push_back({"L2 -> SMEM", gbps, 0});
      }

      // the far seam, DRAM -> SMEM, at both scales
      {
        id<MTLBuffer> buf =
            [dev newBufferWithLength:dram_bytes options:MTLResourceStorageModeShared];
        if (buf) {
          memset(buf.contents, 1, dram_bytes);
          uint32_t n4 = (uint32_t)(dram_bytes / 16);
          uint32_t tgn4 = (uint32_t)(tg_max / 16);
          uint32_t iters = iters_for(dram_bytes);
          Timing t = time_dispatch(
              queue,
              ^(id<MTLComputeCommandEncoder> enc) {
                [enc setComputePipelineState:ps_g_tg];
                [enc setBuffer:buf offset:0 atIndex:0];
                [enc setBytes:&n4 length:4 atIndex:1];
                [enc setBytes:&iters length:4 atIndex:2];
                [enc setBuffer:sink offset:0 atIndex:3];
                [enc setBytes:&tgn4 length:4 atIndex:4];
                [enc setThreadgroupMemoryLength:tg_max atIndex:0];
              },
              MTLSizeMake(groups, 1, 1), MTLSizeMake(TG_THREADS, 1, 1));
          double gbps = (double)dram_bytes * iters / t.median / 1.0e9;
          if (aggregate) {
            seams[3].aggregate_gbps = gbps;
            seam_row("HBM->SMEM", dram_bytes, t, REPS, "global to threadgroup, DRAM-scale source");
          } else {
            seams.push_back({"HBM -> SMEM", gbps, 0});
          }
        }
      }

      // and the DRAM read leg on its own
      {
        id<MTLBuffer> buf =
            [dev newBufferWithLength:dram_bytes options:MTLResourceStorageModeShared];
        if (buf) {
          memset(buf.contents, 1, dram_bytes);
          uint32_t n4 = (uint32_t)(dram_bytes / 16);
          uint32_t iters = iters_for(dram_bytes);
          Timing t = time_dispatch(
              queue,
              ^(id<MTLComputeCommandEncoder> enc) {
                [enc setComputePipelineState:ps_stream];
                [enc setBuffer:buf offset:0 atIndex:0];
                [enc setBytes:&n4 length:4 atIndex:1];
                [enc setBytes:&iters length:4 atIndex:2];
                [enc setBuffer:sink offset:0 atIndex:3];
              },
              MTLSizeMake(groups, 1, 1), MTLSizeMake(TG_THREADS, 1, 1));
          double gbps = (double)dram_bytes * iters / t.median / 1.0e9;
          if (aggregate) {
            seams[4].aggregate_gbps = gbps;
            seam_row("HBM->REG", dram_bytes, t, REPS, "DRAM streaming read, L1/L2 defeated by size");
          } else {
            seams.push_back({"HBM -> REG", gbps, 0});
          }
        }
      }
    }

    // ---- the powerset table, in the format compose.py already parses -------------------------
    // Same shape as probes.txt from the H100 run so the composition laws can be scored on this
    // machine with the same script and no special case. Columns are GB/s per threadgroup and GB/s
    // device-wide -- NOT B/cyc, because Metal exposes no cycle counter (P4). compose.py's
    // arithmetic is unit-agnostic; pass --units to label it correctly.
    fprintf(stderr, "\n=== transfer-edge powerset over {HBM, L2, SMEM, REG} ===\n");
    fprintf(stderr, "from          to             kind   per-TG GB/s   aggregate GB/s\n");
    fprintf(stderr, "--------------------------------------------------------------------\n");
    for (const auto &s : seams) {
      // Split "A -> B" back into its endpoints for the powerset line.
      char from[32] = {0}, to[32] = {0};
      if (sscanf(s.name, "%31s -> %31s", from, to) == 2) {
        fprintf(stderr, "%-10s -> %-10s MEASURED %12.1f %16.1f\n", from, to, s.per_tg_gbps,
                s.aggregate_gbps);
      }
    }
    return 0;
  }
}
