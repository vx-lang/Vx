#ifndef NPU_DISPATCH_H
#define NPU_DISPATCH_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A memref descriptor used to be described here twice, once per rank, with
// `float *` fields -- so reading one meant having already decided its element
// type and rank. The dispatch tags now carry both, and vx_memref_sizes() and
// friends in include/vx_hardware_runtime.h read a descriptor of any rank
// without that decision being made in advance.

// Dispatch the matmul to the AMX (Accelerate) or ANE (CoreML)
int vx_dispatch_amx(float *xout, float *x, float *w, int n, int d);
int vx_dispatch_ane(float *xout, float *x, float *w, int n, int d);
int vx_dispatch_ane_affine(float *out, float *x, float alpha, float beta,
                           int length);

#ifdef __cplusplus
}
#endif

#endif // NPU_DISPATCH_H
