// Check vx_peer_buffer_create on real GPUs: each rank writes its own number
// into its part, and every rank then reads every part.
//
// NCCL refuses two ranks on one GPU, so this needs at least two GPUs. Each
// process sees every GPU and uses GPU <rank>. Build it against the CUDA
// dispatch library, then start one process per rank:
//
//   clang++ -O1 -std=c++17 scripts/tools/vx_peer_buffer_check.cpp \
//     -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" -lcudart \
//     -L<dir> -lvx_cuda_dispatch -Wl,-rpath,<dir> -o vx_peer_buffer_check
//   rm -f /tmp/vx_peer_id
//   for r in 0 1; do
//     VX_NCCL_ID_FILE=/tmp/vx_peer_id ./vx_peer_buffer_check $r 2 &
//   done; wait
//
// Each rank prints "rank R: ok" when it reads the right number from every
// part, and the rest of every part is 0. Memory: 64 bytes for the buffer and
// 4 bytes for the barrier on each GPU.

#include "vram_ceiling.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void vx_collective_init(int32_t rank, int32_t ranks);
void vx_collective_finish(void);
void vx_collective_all_reduce_sum_f32(const float *in, float *out,
                                      int64_t count);
void *vx_peer_buffer_create(int64_t bytes);
void vx_peer_buffer_pointers(void *buffer, uint64_t *out);
void vx_peer_buffer_destroy(void *buffer);
}

static void cuda(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(e));
    exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: vx_peer_buffer_check <rank> <ranks>\n");
    return 2;
  }
  const int rank = atoi(argv[1]);
  const int ranks = atoi(argv[2]);
  const int64_t bytes = 64;
  requireWithinCeiling("vx_peer_buffer_check", (uint64_t)bytes + sizeof(float),
                       vramCeiling());
  int gpus = 0;
  cuda(cudaGetDeviceCount(&gpus), "cudaGetDeviceCount");
  if (gpus < ranks) {
    fprintf(stderr, "%d ranks need %d GPUs, and this machine has %d\n", ranks,
            ranks, gpus);
    return 1;
  }
  cuda(cudaSetDevice(rank), "cudaSetDevice");
  vx_collective_init(rank, ranks);

  void *buffer = vx_peer_buffer_create(bytes);
  std::vector<uint64_t> parts((size_t)ranks);
  vx_peer_buffer_pointers(buffer, parts.data());

  // Each rank writes its number into its own part. An all-reduce ends only
  // after every rank starts it, so after it every rank's write is done.
  const int32_t mine = 1000 + rank;
  cuda(cudaMemcpy((void *)parts[(size_t)rank], &mine, sizeof mine,
                  cudaMemcpyHostToDevice),
       "the write into this rank's part");
  float *one = nullptr;
  cuda(cudaMalloc(&one, sizeof *one), "cudaMalloc");
  vx_collective_all_reduce_sum_f32(one, one, 1);
  cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  int bad = 0;
  for (int r = 0; r < ranks; ++r) {
    unsigned char seen[64];
    cuda(cudaMemcpy(seen, (const void *)parts[(size_t)r], sizeof seen,
                    cudaMemcpyDefault),
         "the read of a part");
    int32_t number = 0;
    memcpy(&number, seen, sizeof number);
    if (number != 1000 + r) {
      fprintf(stderr, "rank %d: part %d holds %d, not %d\n", rank, r, number,
              1000 + r);
      ++bad;
    }
    for (size_t i = sizeof number; i < sizeof seen; ++i) {
      if (seen[i] != 0) {
        fprintf(stderr, "rank %d: byte %zu of part %d is not 0\n", rank, i, r);
        ++bad;
        break;
      }
    }
  }
  cuda(cudaFree(one), "cudaFree");
  vx_peer_buffer_destroy(buffer);
  vx_collective_finish();
  if (bad) {
    return 1;
  }
  printf("rank %d: ok\n", rank);
  return 0;
}
