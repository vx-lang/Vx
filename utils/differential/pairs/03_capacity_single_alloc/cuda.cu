// One allocation larger than the device has.
//
// 64 GiB onto a 40 GiB A100. Vx refuses this before the card is rented, in bytes,
// against the capacity its machine model declares. CUDA finds out on the card.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main() {
  // Same figure as the Vx half: 131072 * 131072 * sizeof(float).
  const size_t bytes = 68719476736ULL;

  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  printf("device total=%zu bytes free=%zu bytes\n", total_bytes, free_bytes);
  printf("requesting=%zu bytes\n", bytes);

  float *dev = nullptr;
  cudaError_t rc = cudaMalloc(&dev, bytes);
  printf("cudaMalloc=%s\n", cudaGetErrorString(rc));

  if (rc != cudaSuccess) {
    // The graceful path, and the one this program takes because it checks.
    // Worth stating plainly for anyone reading the table: a great deal of real
    // code does not check, and then uses the null pointer, which is pair 02's
    // failure mode arrived at by a different route.
    return 1;
  }

  cudaFree(dev);
  return 0;
}
