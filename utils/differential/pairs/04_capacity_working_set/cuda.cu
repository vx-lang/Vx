// Three buffers that each fit and together do not.
//
// 16 GiB each onto a 40 GiB A100. No single allocation is too big, so nothing
// about any one of them is wrong. The mistake is a property of the set, and CUDA
// has no notion of a set: it allocates until one fails.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main() {
  // Same figure as the Vx half: 65536 * 65536 * sizeof(float).
  const size_t bytes = 17179869184ULL;
  const int count = 3;

  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  printf("device total=%zu bytes free=%zu bytes\n", total_bytes, free_bytes);
  printf("requesting %d x %zu bytes = %zu bytes\n", count, bytes,
         (size_t)count * bytes);

  float *buf[count];
  for (int i = 0; i < count; ++i) {
    buf[i] = nullptr;
  }

  int allocated = 0;
  for (int i = 0; i < count; ++i) {
    cudaError_t rc = cudaMalloc(&buf[i], bytes);
    printf("cudaMalloc[%d]=%s\n", i, cudaGetErrorString(rc));
    if (rc != cudaSuccess) {
      break;
    }
    ++allocated;
  }

  printf("allocated %d of %d before failing\n", allocated, count);

  for (int i = 0; i < allocated; ++i) {
    cudaFree(buf[i]);
  }

  // Anything short of all three is the failure this pair is about.
  return (allocated == count) ? 0 : 1;
}
