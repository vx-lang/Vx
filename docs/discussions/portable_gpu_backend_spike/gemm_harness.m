// Spike harness: run the Vx tiled GEMM on the Apple GPU and check it against C.
// Usage: gemm_harness <file.metal>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { N = 1024 };

int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                              encoding:NSUTF8StringEncoding error:&err];
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { fprintf(stderr, "library failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> p =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"vx_npu_kernel_0"] error:&err];
    if (!p) { fprintf(stderr, "pipeline failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    size_t bytes = sizeof(float) * N * N;
    id<MTLBuffer> bC = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bA = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bB = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    float *a = bA.contents, *b = bB.contents, *c = bC.contents;
    for (int i = 0; i < N * N; i++) {
      a[i] = (float)((i * 7) % 13) / 13.0f - 0.5f;
      b[i] = (float)((i * 5) % 11) / 11.0f - 0.5f;
    }
    int n = N;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    void (^encode)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:p];
      [e setBuffer:bC offset:0 atIndex:0];
      [e setBuffer:bA offset:0 atIndex:1];
      [e setBuffer:bB offset:0 atIndex:2];
      for (int k = 3; k < 7; k++) [e setBytes:&n length:4 atIndex:k];
      [e dispatchThreadgroups:MTLSizeMake(N, N / 256, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e endEncoding];
    };
    double best = 1e9;
    for (int rep = 0; rep < 8; rep++) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      for (int i = 0; i < 10; i++) encode(cb);
      [cb commit];
      [cb waitUntilCompleted];
      double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / 10;
      if (rep > 0 && ms < best) best = ms;
    }
    // References: the same order of adds in float (no fused multiply-add), and a double result.
    double maxf = 0, maxd = 0;
    int unequal = 0;
    for (int r = 0; r < N; r++)
      for (int col = 0; col < N; col++) {
        float s = 0;
        double sd = 0;
        for (int k = 0; k < N; k++) {
          volatile float prod = a[r * N + k] * b[k * N + col];
          s = s + prod;
          sd += (double)a[r * N + k] * b[k * N + col];
        }
        float g = c[r * N + col];
        unequal += g != s;
        if (fabs(g - s) > maxf) maxf = fabs(g - s);
        if (fabs(g - sd) > maxd) maxd = fabs(g - sd);
      }
    printf("gemm %dx%d: %.3f ms, %.1f GFLOP/s\n", N, N, best, 2.0 * N * N * N / (best * 1e6));
    printf("vs float reference (no fma): max diff %.3g, %d of %d not bit-equal\n", maxf, unequal, N * N);
    printf("vs double reference: max diff %.3g\n", maxd);
  }
  return 0;
}
