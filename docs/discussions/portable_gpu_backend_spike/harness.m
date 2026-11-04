// Spike harness: run the two Vx blur kernels on the Apple GPU and check them.
// Usage: harness <file.metal | file.metallib> <rows-kernel> <cols-kernel>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { H = 720, W = 1280, LANES = 256 };

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

// The same algorithm as blur.vx, in C, for an element-by-element check.
static void reference(float *out, const float *src) {
  float *tmp = malloc(sizeof(float) * H * W);
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) {
      float acc = 0;
      for (int d = 0; d < 9; d++) {
        int xx = x + d - 4;
        xx = xx < 0 ? 0 : xx > W - 1 ? W - 1 : xx;
        acc += src[y * W + xx];
      }
      tmp[y * W + x] = acc / 9.0f;
    }
  for (int x = 0; x < W; x++)
    for (int y = 0; y < H; y++) {
      float acc = 0;
      for (int d = 0; d < 9; d++) {
        int yy = y + d - 4;
        yy = yy < 0 ? 0 : yy > H - 1 ? H - 1 : yy;
        acc += tmp[yy * W + x];
      }
      out[y * W + x] = acc / 9.0f;
    }
  free(tmp);
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 4) {
      fprintf(stderr, "usage: %s <file.metal|file.metallib> <rows-kernel> <cols-kernel>\n", argv[0]);
      return 2;
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    NSString *path = [NSString stringWithUTF8String:argv[1]];
    id<MTLLibrary> lib = nil;
    double t0 = now_ms();
    if ([path hasSuffix:@".metallib"]) {
      lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    } else {
      NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
      MTLCompileOptions *opts = [MTLCompileOptions new];
      if (getenv("VX_SAFE_MATH")) opts.mathMode = MTLMathModeSafe;
      lib = [dev newLibraryWithSource:src options:opts error:&err];
    }
    double tLib = now_ms() - t0;
    if (!lib) {
      fprintf(stderr, "library failed: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLFunction> fRows = [lib newFunctionWithName:[NSString stringWithUTF8String:argv[2]]];
    id<MTLFunction> fCols = [lib newFunctionWithName:[NSString stringWithUTF8String:argv[3]]];
    if (!fRows || !fCols) {
      fprintf(stderr, "kernel not found in library\n");
      return 1;
    }
    t0 = now_ms();
    id<MTLComputePipelineState> pRows = [dev newComputePipelineStateWithFunction:fRows error:&err];
    id<MTLComputePipelineState> pCols = [dev newComputePipelineStateWithFunction:fCols error:&err];
    double tPso = now_ms() - t0;
    if (!pRows || !pCols) {
      fprintf(stderr, "pipeline failed: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }

    // Shared storage: the CPU and the GPU see the same pages, so no copy is made.
    size_t bytes = sizeof(float) * H * W;
    id<MTLBuffer> bSrc;
    if (getenv("VX_NOCOPY")) {
      // A host allocation, as the host shell or a CPU stage would make it, wrapped without a copy.
      void *host = NULL;
      size_t page = 16384, len = (bytes + page - 1) / page * page;
      posix_memalign(&host, page, len);
      bSrc = [dev newBufferWithBytesNoCopy:host length:len options:MTLResourceStorageModeShared deallocator:nil];
      printf("src wraps host memory at %p, buffer contents %p (same pointer: %s)\n", host, bSrc.contents,
             host == bSrc.contents ? "yes" : "no");
    } else {
      bSrc = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> bTmp = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bOut = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    float *src = bSrc.contents;
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++)
        src[y * W + x] = (float)((x * 7 + y * 13) % 256) / 255.0f;

    int h = H, w = W, rowLen = W;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    void (^encode)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:pRows];
      [e setBuffer:bTmp offset:0 atIndex:0];
      [e setBuffer:bSrc offset:0 atIndex:1];
      [e setBytes:&h length:4 atIndex:2];
      [e setBytes:&w length:4 atIndex:3];
      [e setBytes:&rowLen length:4 atIndex:4];
      [e setBytes:&rowLen length:4 atIndex:5];
      [e dispatchThreadgroups:MTLSizeMake(H, 1, 1) threadsPerThreadgroup:MTLSizeMake(LANES, 1, 1)];
      [e setComputePipelineState:pCols];
      [e setBuffer:bOut offset:0 atIndex:0];
      [e setBuffer:bTmp offset:0 atIndex:1];
      [e dispatchThreadgroups:MTLSizeMake(W, 1, 1) threadsPerThreadgroup:MTLSizeMake(LANES, 1, 1)];
      [e endEncoding];
    };

    // Warm up, then time each frame on the GPU clock.
    enum { FRAMES = 200 };
    double gpu[FRAMES];
    for (int i = 0; i < 10; i++) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      encode(cb);
      [cb commit];
      [cb waitUntilCompleted];
    }
    double wall0 = now_ms();
    for (int i = 0; i < FRAMES; i++) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      encode(cb);
      [cb commit];
      [cb waitUntilCompleted];
      gpu[i] = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    }
    double wall = (now_ms() - wall0) / FRAMES;
    // Batched: 100 frames in one command buffer, as a steady 30 fps stream keeps the GPU awake.
    double batched = 1e9;
    for (int rep = 0; rep < 5; rep++) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      for (int i = 0; i < 100; i++) encode(cb);
      [cb commit];
      [cb waitUntilCompleted];
      double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / 100;
      if (ms < batched) batched = ms;
    }
    qsort(gpu, FRAMES, sizeof(double), cmp_double);

    float *out = bOut.contents;
    float *ref = malloc(bytes);
    reference(ref, src);
    double sum = 0, maxdiff = 0;
    for (int i = 0; i < H * W; i++) {
      sum += out[i];
      double d = fabs((double)out[i] - ref[i]);
      if (d > maxdiff) maxdiff = d;
    }
    printf("library %.1f ms, pipelines %.1f ms\n", tLib, tPso);
    printf("gpu ms/frame: median %.3f  p10 %.3f  p90 %.3f   wall ms/frame (incl. submit+wait) %.3f\n",
           gpu[FRAMES / 2], gpu[FRAMES / 10], gpu[FRAMES * 9 / 10], wall);
    int mism = 0;
    for (int i = 0; i < H * W; i++) mism += out[i] != ref[i];
    printf("batched gpu ms/frame (best of 5 x 100 frames): %.3f\n", batched);
    printf("max |gpu - C reference| = %.3g, elements not bit-equal: %d of %d\n", maxdiff, mism, H * W);
    printf("sum %.11f\n%.8g\n%.7g\n%.8g\n%.8g\n", sum, out[0], out[100 * W + 200], out[719 * W + 1279],
           out[360 * W + 5]);
    free(ref);
  }
  return 0;
}
