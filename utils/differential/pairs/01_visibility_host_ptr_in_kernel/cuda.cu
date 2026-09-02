// A device kernel reads host memory that nothing brought across.
//
// The Vx counterpart is refused at compile time: the value lives in CPU_DRAM, the
// topology sees only its own memory, and the diagnostic names the transfer that
// would fix it. Here the same mistake compiles.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void read_first(const float *p, float *out) {
  // `p` is an ordinary host allocation. Nothing copied it to the device, and
  // nothing in the type of `p` records where it lives.
  *out = p[0];
}

int main() {
  const int n = 1024;
  float *host = (float *)malloc(n * sizeof(float));
  if (!host) {
    fprintf(stderr, "host allocation failed\n");
    return 3;
  }
  for (int i = 0; i < n; ++i) {
    host[i] = 1.0f;
  }

  float *dout = nullptr;
  if (cudaMalloc(&dout, sizeof(float)) != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed\n");
    return 3;
  }

  // The host pointer goes straight into the kernel. nvcc has no complaint:
  // `const float *` is a pointer, and which memory it points into is not part
  // of its type.
  read_first<<<1, 1>>>(host, dout);

  cudaError_t launch = cudaGetLastError();
  cudaError_t sync = cudaDeviceSynchronize();

  float out = -1.0f;
  cudaMemcpy(&out, dout, sizeof(float), cudaMemcpyDeviceToHost);

  printf("launch=%s\n", cudaGetErrorString(launch));
  printf("sync=%s\n", cudaGetErrorString(sync));
  printf("out=%f expected=1.000000\n", out);

  cudaFree(dout);
  free(host);

  if (sync != cudaSuccess) {
    return 1; // ran, faulted
  }
  if (out != 1.0f) {
    return 2; // ran, wrong answer
  }
  return 0;
}
