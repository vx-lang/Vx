#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CoreML/CoreML.h>
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
        NSURL *modelURL = [NSURL fileURLWithPath:@"matmul.mlmodelc"];
        NSError *error = nil;
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll; // Prioritize Apple Neural Engine (ANE)
        
        MLModel *model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
        if (!model) {
            std::cerr << "[Akar Dispatcher] Failed to load CoreML model. Falling back to AMX. Error: " << [[error localizedDescription] UTF8String] << std::endl;
            cblas_sgemv(CblasRowMajor, CblasNoTrans, d, n, 1.0f, w, n, x, 1, 0.0f, xout, 1);
            return 1;
        }

        // Wrap inputs in MLMultiArray (Zero-copy wrapping)
        NSArray<NSNumber *> *shapeW = @[@(d), @(n)];
        NSArray<NSNumber *> *stridesW = @[@(n), @(1)];
        MLMultiArray *arrayW = [[MLMultiArray alloc] initWithDataPointer:w shape:shapeW dataType:MLMultiArrayDataTypeFloat32 strides:stridesW deallocator:^(void *bytes) {} error:&error];

        NSArray<NSNumber *> *shapeX = @[@(n), @(1)];
        NSArray<NSNumber *> *stridesX = @[@(1), @(1)];
        MLMultiArray *arrayX = [[MLMultiArray alloc] initWithDataPointer:x shape:shapeX dataType:MLMultiArrayDataTypeFloat32 strides:stridesX deallocator:^(void *bytes) {} error:&error];

        id<MLFeatureProvider> inputFeatures = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"w": arrayW, @"x": arrayX} error:&error];

        id<MLFeatureProvider> outputFeatures = [model predictionFromFeatures:inputFeatures error:&error];
        if (error) {
            std::cerr << "[Akar Dispatcher] CoreML prediction failed! Error: " << [[error localizedDescription] UTF8String] << std::endl;
            return 0;
        }

        MLMultiArray *outArray = [outputFeatures featureValueForName:@"res"].multiArrayValue;
        
        // Copy data back (MLMultiArray might not be contiguous, so copy manually or via memcpy if contiguous)
        // For d x 1, it is essentially 1D contiguous.
        memcpy(xout, outArray.dataPointer, d * sizeof(float));

        return 1;
    }
}
