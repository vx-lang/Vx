#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "npu_dispatch.h"
#include <iostream>

extern "C" int akar_dispatch_amx(float* xout, float* x, float* w, int n, int d) {
    @autoreleasepool {
        // std::cout << "[Akar Dispatcher] Offloading to Apple AMX (Accelerate)..." << std::endl;
        
        // cblas_sgemv computes: y = alpha * A * x + beta * y
        // A is d x n, x is n x 1, y is d x 1
        cblas_sgemv(CblasRowMajor, 
                    CblasNoTrans, 
                    d, n, 
                    1.0f, // alpha
                    w, n, // A, lda
                    x, 1, // x, incx
                    0.0f, // beta
                    xout, 1); // y, incy

        // std::cout << "[Akar Dispatcher] AMX execution completed successfully." << std::endl;
        return 1;
    }
}

extern "C" int akar_dispatch_gpu(float* xout, float* x, float* w, int n, int d) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Failed to acquire Metal device!" << std::endl;
            return 0;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

        // Wrap input arrays in MTLBuffer (using copy for safety as mmap bounds might not be 4K aligned)
        id<MTLBuffer> buf_w = [device newBufferWithBytes:w length:(d * n * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_x = [device newBufferWithBytes:x length:(n * sizeof(float)) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_out = [device newBufferWithLength:(d * sizeof(float)) options:MTLResourceStorageModeShared];

        MPSMatrixDescriptor* desc_w = [MPSMatrixDescriptor matrixDescriptorWithRows:d columns:n rowBytes:(n * sizeof(float)) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_x = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:1 rowBytes:(1 * sizeof(float)) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_out = [MPSMatrixDescriptor matrixDescriptorWithRows:d columns:1 rowBytes:(1 * sizeof(float)) dataType:MPSDataTypeFloat32];

        MPSMatrix* mat_w = [[MPSMatrix alloc] initWithBuffer:buf_w descriptor:desc_w];
        MPSMatrix* mat_x = [[MPSMatrix alloc] initWithBuffer:buf_x descriptor:desc_x];
        MPSMatrix* mat_out = [[MPSMatrix alloc] initWithBuffer:buf_out descriptor:desc_out];

        // C = op(A) * op(B)
        MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                             transposeLeft:false
                                                                            transposeRight:false
                                                                                resultRows:d
                                                                             resultColumns:1
                                                                           interiorColumns:n
                                                                                     alpha:1.0
                                                                                      beta:0.0];

        [mul encodeToCommandBuffer:commandBuffer leftMatrix:mat_w rightMatrix:mat_x resultMatrix:mat_out];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy result back to CPU host memory
        memcpy(xout, [buf_out contents], d * sizeof(float));
        
        return 1;
    }
}


extern "C" int akar_dispatch_ane(float* xout, float* x, float* w, int n, int d) {
    @autoreleasepool {
        // std::cout << "[Akar Dispatcher] Offloading to Apple Neural Engine (ANE)..." << std::endl;
        
        // This is a stub for the actual CoreML integration using a precompiled .mlmodelc
        // In a full implementation, we would map the pointers to MLMultiArray and invoke predictionFromFeatures.
        
        // std::cout << "[Akar Dispatcher] Initializing CoreML MLModel from 'matmul.mlmodelc'..." << std::endl;
        // std::cout << "[Akar Dispatcher] Binding MLMultiArray inputs for MatMul (" << d << "x" << n << ")..." << std::endl;
        
        // Fallback to CPU/AMX for the test stub
        cblas_sgemv(CblasRowMajor, CblasNoTrans, d, n, 1.0f, w, n, x, 1, 0.0f, xout, 1);
        
        // std::cout << "[Akar Dispatcher] CoreML ANE execution completed successfully." << std::endl;
        return 1;
    }
}
