// A static __shared__ tile over the 48 KiB per-block ceiling.
//
// This is the pair CUDA catches. It is here on purpose: a suite where every case
// favours one toolchain is a suite whose cases were chosen.

#include <cstdio>
#include <cuda_runtime.h>

__global__ void tiled(float *out) {
  // 16384 floats = 65536 bytes. The static per-block limit is 48 KiB; going past
  // it requires the dynamic extern __shared__ form plus an opt-in call to
  // cudaFuncSetAttribute, and nvcc rejects the static spelling outright.
  __shared__ float tile[16384];

  int i = threadIdx.x;
  tile[i] = (float)i;
  __syncthreads();
  if (i == 0) {
    out[0] = tile[0];
  }
}

int main() {
  float *dev = nullptr;
  if (cudaMalloc(&dev, sizeof(float)) != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed\n");
    return 3;
  }
  tiled<<<1, 256>>>(dev);
  cudaError_t sync = cudaDeviceSynchronize();
  printf("sync=%s\n", cudaGetErrorString(sync));
  cudaFree(dev);
  return sync == cudaSuccess ? 0 : 1;
}
