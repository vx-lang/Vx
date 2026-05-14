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

// Dispatch the model to the ANE.
// model_path: path to the .mlmodelc or .mlpackage directory
// input_ref: pointer to the MLIR memref containing the input data
// output_ref: pointer to the MLIR memref where output should be written
bool _mlir_ciface_akar_dispatch_npu(MemRef2D* input_a, MemRef2D* input_b, MemRef2D* output_ref);

#ifdef __cplusplus
}
#endif

#endif // NPU_DISPATCH_H
