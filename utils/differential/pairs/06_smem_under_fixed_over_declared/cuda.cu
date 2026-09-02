// The same 64 KiB tile pair 05 could not declare statically, allocated
// dynamically and used.
//
// This is the other half of pair 05's story. `nvcc` refuses a 64 KiB *static*
// __shared__ array against a fixed 48 KiB ceiling. The A100 has 164 KiB of shared
// memory per SM, so the tile fits the part comfortably; what the program has to do
// is ask for it dynamically and opt in with cudaFuncSetAttribute.
//
// If this runs, the static refusal in pair 05 was about a compiler constant rather
// than about the hardware.

#include <cstdio>
#include <cuda_runtime.h>

extern __shared__ float tile[];

__global__ void tiled(float *out, int n) {
  int i = threadIdx.x;
  for (int j = i; j < n; j += blockDim.x) {
    tile[j] = (float)j;
  }
  __syncthreads();
  if (i == 0) {
    // Sum the first 128 entries, so the answer depends on the tile really existing.
    float acc = 0.0f;
    for (int j = 0; j < 128; ++j) {
      acc += tile[j];
    }
    out[0] = acc;
  }
}

int main() {
  const int floats = 16384;              // 65536 bytes = 64 KiB
  const size_t smem = floats * sizeof(float);

  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("sharedMemPerMultiprocessor=%zu bytes\n", prop.sharedMemPerMultiprocessor);
  printf("sharedMemPerBlock(default)=%zu bytes\n", prop.sharedMemPerBlock);
  printf("requesting dynamic smem=%zu bytes\n", smem);

  // The opt-in. Without it the launch fails even though the hardware has the room.
  cudaError_t attr = cudaFuncSetAttribute(
      tiled, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  printf("cudaFuncSetAttribute=%s\n", cudaGetErrorString(attr));

  float *dout = nullptr;
  if (cudaMalloc(&dout, sizeof(float)) != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed\n");
    return 3;
  }

  tiled<<<1, 256, smem>>>(dout, floats);
  cudaError_t launch = cudaGetLastError();
  cudaError_t sync = cudaDeviceSynchronize();

  float out = -1.0f;
  cudaMemcpy(&out, dout, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dout);

  printf("launch=%s\n", cudaGetErrorString(launch));
  printf("sync=%s\n", cudaGetErrorString(sync));
  // 0 + 1 + ... + 127 = 8128
  printf("out=%f expected=8128.000000\n", out);

  if (sync != cudaSuccess || launch != cudaSuccess) {
    return 1;
  }
  return (out == 8128.0f) ? 0 : 2;
}
