// Standalone driver: the family's closed form against FA-2's own kernel,
// timed with the same cuda-event clock every Vx bench uses.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" int vx_flash_fwd_f16_hd64(void*, void*, void*, void*, void*,
                                     int, int, int, int, float, cudaStream_t);

int main(int argc, char **argv) {
  const int SQ = 8192, SK = 2048, HD = 64;
  const float scale = 0.5f;
  std::vector<__half> qh(SQ * HD), kh(SK * HD), vh(SK * HD);
  for (int i = 0; i < SQ * HD; i++) qh[i] = __float2half(0.0f);
  for (int j = 0; j < SK; j++)
    for (int t = 0; t < HD; t++) {
      kh[j * HD + t] = __float2half(0.01f);
      vh[j * HD + t] = __float2half(j * 0.01f);
    }
  void *q, *k, *v, *o, *lse;
  cudaMalloc(&q, qh.size() * 2); cudaMalloc(&k, kh.size() * 2);
  cudaMalloc(&v, vh.size() * 2); cudaMalloc(&o, (size_t)SQ * HD * 2);
  cudaMalloc(&lse, (size_t)SQ * 4);
  cudaMemcpy(q, qh.data(), qh.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(k, kh.data(), kh.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(v, vh.data(), vh.size() * 2, cudaMemcpyHostToDevice);
  for (int w = 0; w < 3; w++)
    vx_flash_fwd_f16_hd64(q, k, v, o, lse, 1, 1, SQ, SK, scale, 0);
  cudaDeviceSynchronize();
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  const int reps = 20;
  for (int r = 0; r < reps; r++)
    vx_flash_fwd_f16_hd64(q, k, v, o, lse, 1, 1, SQ, SK, scale, 0);
  cudaEventRecord(t1); cudaEventSynchronize(t1);
  float ms = 0; cudaEventElapsedTime(&ms, t0, t1); ms /= reps;
  __half o0;
  cudaMemcpy(&o0, o, 2, cudaMemcpyDeviceToHost);
  double fl = 4.0 * SQ * SK * HD;
  printf("fa2-shim: %.3f ms  %.1f TF/s  o[0][0]=%.4f (expected 10.235)\n",
         ms, fl / ms / 1e9, __half2float(o0));
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}
