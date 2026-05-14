#ifndef NPU_DISPATCH_H
#define NPU_DISPATCH_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// A simple memref struct mapped to what MLIR generates for a 2D memref
typedef struct {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
} MemRef2D;

// Dispatch the matmul to the AMX (Accelerate) or ANE (CoreML)
int akar_dispatch_amx(float* xout, float* x, float* w, int n, int d);
int akar_dispatch_ane(float* xout, float* x, float* w, int n, int d);

#ifdef __cplusplus
}
#endif

#endif // NPU_DISPATCH_H
