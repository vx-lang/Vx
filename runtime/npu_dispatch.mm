#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#include "npu_dispatch.h"
#include <iostream>

extern "C" bool _mlir_ciface_akar_dispatch_npu(MemRef2D* input_a, MemRef2D* input_b, MemRef2D* output_ref) {
    @autoreleasepool {
        std::cout << "[Akar Hardware Dispatcher] Offloading computation to Apple Silicon accelerators..." << std::endl;
        
        // Ensure dimensions align for Matrix Multiplication (M x K) * (K x N) = (M x N)
        int M = input_a->sizes[0];
        int K = input_a->sizes[1];
        int K_b = input_b->sizes[0];
        int N = input_b->sizes[1];

        if (K != K_b) {
            std::cerr << "[Akar Hardware Dispatcher] Error: Matrix dimensions do not align for SGEMM (" 
                      << M << "x" << K << ") * (" << K_b << "x" << N << ")" << std::endl;
            return false;
        }

        // AMX / Accelerate framework execution
        // cblas_sgemm computes: C = alpha * A * B + beta * C
        // A is M x K, B is K x N, C is M x N
        
        cblas_sgemm(CblasRowMajor, 
                    CblasNoTrans, CblasNoTrans, 
                    M, N, K, 
                    1.0f, // alpha
                    input_a->aligned, input_a->strides[0], // A
                    input_b->aligned, input_b->strides[0], // B
                    0.0f, // beta
                    output_ref->aligned, output_ref->strides[0]); // C

        std::cout << "[Akar Hardware Dispatcher] Accelerator SGEMM execution completed successfully." << std::endl;

        return true;
    }
}
