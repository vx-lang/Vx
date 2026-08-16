// Stub for syntax-checking probe_peer.cu without a CUDA toolkit (see check_host_compile.sh for
// the caveats: this catches typos and shadowing, not wrong API signatures).
#pragma once
#include <cstddef>
typedef enum { cudaSuccess = 0 } cudaError_t;
struct cudaEvent_st; typedef cudaEvent_st *cudaEvent_t;
typedef int cudaMemcpyKind;
cudaError_t cudaGetDeviceCount(int *);
cudaError_t cudaSetDevice(int);
cudaError_t cudaDeviceCanAccessPeer(int *, int, int);
cudaError_t cudaDeviceEnablePeerAccess(int, unsigned);
cudaError_t cudaGetLastError(void);
template <class T> cudaError_t cudaMalloc(T **, size_t);
cudaError_t cudaFree(void *);
cudaError_t cudaMemset(void *, int, size_t);
cudaError_t cudaMemcpyPeer(void *, int, const void *, int, size_t);
cudaError_t cudaEventCreate(cudaEvent_t *);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, int = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
const char *cudaGetErrorString(cudaError_t);
