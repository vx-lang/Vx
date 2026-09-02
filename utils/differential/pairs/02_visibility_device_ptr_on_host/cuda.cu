// Host code reads device memory directly, with no copy back.
//
// The reverse of pair 01, and the mistake a program makes by forgetting the
// transfer *back*. The Vx counterpart is refused; here the pointer is just a
// pointer and the host dereferences it.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void fill(float *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    p[i] = 1.0f;
  }
}

int main() {
  const int n = 1024;
  float *dev = nullptr;
  if (cudaMalloc(&dev, n * sizeof(float)) != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed\n");
    return 3;
  }

  fill<<<(n + 255) / 256, 256>>>(dev, n);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "fill kernel failed\n");
    return 3;
  }

  // `dev` addresses device memory. Nothing in its type says so, and nothing
  // copied it back, so this is an ordinary-looking read of a pointer the host
  // cannot follow. Expected outcome is SIGSEGV -- which means the printf below
  // never runs, and the harness records the signal rather than an exit code.
  printf("about to read device memory from the host\n");
  fflush(stdout);
  float value = dev[0];
  printf("read succeeded: %f (expected 1.000000)\n", value);

  cudaFree(dev);
  return (value == 1.0f) ? 0 : 2;
}
