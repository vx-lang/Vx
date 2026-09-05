// Time the two dot lowerings against each other on real hardware.
// Loads the PTX through the driver API (same way Vx's runtime loads its images),
// runs both kernels over identical inputs, checks they agree, and times them.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){ const char*s; cuGetErrorString(r,&s); \
  printf("ERR %s @%d: %s\n", #x, __LINE__, s); exit(1);} } while(0)

// MLIR's memref<?x64xf32> ABI: {allocated, aligned, offset, size[0], size[1], stride[0], stride[1]}
struct MemRef2D { void* a; void* b; long long off, s0, s1, st0, st1; };
struct MemRef1D { void* a; void* b; long long off, s0, st0; };

int main(int argc, char** argv) {
  long long N = (argc > 1) ? atoll(argv[1]) : (1LL << 20);
  int threads = (argc > 2) ? atoi(argv[2]) : 256;
  int blocks  = (argc > 3) ? atoi(argv[3]) : 432;   // 4 per SM on 108 SMs
  const int D = 64;

  CK(cuInit(0));
  CUdevice dev; CK(cuDeviceGet(&dev, 0));
  CUcontext ctx; CK(cuCtxCreate(&ctx, 0, dev));
  CUmodule mod; CK(cuModuleLoad(&mod, "bench.ptx"));
  CUfunction fw, fc;
  CK(cuModuleGetFunction(&fw, mod, "bench_wide"));
  CK(cuModuleGetFunction(&fc, mod, "bench_chunked"));

  size_t bytes = (size_t)N * D * sizeof(float);
  std::vector<float> hq((size_t)N * D), hk((size_t)N * D);
  for (size_t i = 0; i < hq.size(); ++i) { hq[i] = 0.01f; hk[i] = 0.01f; }
  // dot of a 64-wide row of 0.01 with itself = 64 * 1e-4 = 0.0064

  CUdeviceptr dq, dk, dow, doc;
  CK(cuMemAlloc(&dq, bytes)); CK(cuMemAlloc(&dk, bytes));
  CK(cuMemAlloc(&dow, (size_t)N * sizeof(float)));
  CK(cuMemAlloc(&doc, (size_t)N * sizeof(float)));
  CK(cuMemcpyHtoD(dq, hq.data(), bytes));
  CK(cuMemcpyHtoD(dk, hk.data(), bytes));

  auto run = [&](CUfunction f, CUdeviceptr out, int reps, float* ms) {
    MemRef2D q{(void*)dq,(void*)dq,0,N,D,D,1};
    MemRef2D k{(void*)dk,(void*)dk,0,N,D,D,1};
    MemRef1D o{(void*)out,(void*)out,0,N,1};
    long long n = N;
    void* args[] = { &q.a,&q.b,&q.off,&q.s0,&q.s1,&q.st0,&q.st1,
                     &k.a,&k.b,&k.off,&k.s0,&k.s1,&k.st0,&k.st1,
                     &o.a,&o.b,&o.off,&o.s0,&o.st0, &n };
    for (int i = 0; i < 3; ++i)   // warm
      CK(cuLaunchKernel(f, blocks,1,1, threads,1,1, 0,0, args, 0));
    CK(cuCtxSynchronize());
    CUevent t0,t1; CK(cuEventCreate(&t0,0)); CK(cuEventCreate(&t1,0));
    CK(cuEventRecord(t0,0));
    for (int i = 0; i < reps; ++i)
      CK(cuLaunchKernel(f, blocks,1,1, threads,1,1, 0,0, args, 0));
    CK(cuEventRecord(t1,0)); CK(cuEventSynchronize(t1));
    CK(cuEventElapsedTime(ms, t0, t1)); *ms /= reps;
  };

  float mw=0, mc=0;
  const int REPS = 20;
  run(fw, dow, REPS, &mw);
  run(fc, doc, REPS, &mc);

  std::vector<float> ow(N), oc(N);
  CK(cuMemcpyDtoH(ow.data(), dow, (size_t)N*sizeof(float)));
  CK(cuMemcpyDtoH(oc.data(), doc, (size_t)N*sizeof(float)));
  double maxdiff = 0;
  for (long long i = 0; i < N; ++i) maxdiff = fmax(maxdiff, fabs((double)ow[i]-oc[i]));

  double gb = (double)bytes * 2 / 1e9;            // q and k read once each
  printf("N=%lld  grid=%d x %d\n", N, blocks, threads);
  printf("  wide     %8.3f ms   %7.1f GB/s   o[0]=%.6f\n", mw, gb/(mw/1e3), ow[0]);
  printf("  chunked  %8.3f ms   %7.1f GB/s   o[0]=%.6f\n", mc, gb/(mc/1e3), oc[0]);
  printf("  speedup  %8.3fx    max|diff|=%.3e (expect o=0.0064)\n", mw/mc, maxdiff);
  return 0;
}
