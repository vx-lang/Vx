#include "npu_dispatch.h"
#import <Accelerate/Accelerate.h>
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <ffi/ffi.h>
#include <iostream>
#include <vector>
#include <cassert>

extern "C" int vx_dispatch_amx(float *xout, float *x, float *w, int n, int d) {
  @autoreleasepool {
    // std::cout << "[Vx Dispatcher] Offloading to Apple AMX (Accelerate)..." <<
    // std::endl;

    // cblas_sgemv computes: y = alpha * A * x + beta * y
    // A is d x n, x is n x 1, y is d x 1
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d, n,
                1.0f,     // alpha
                w, n,     // A, lda
                x, 1,     // x, incx
                0.0f,     // beta
                xout, 1); // y, incy

    // std::cout << "[Vx Dispatcher] AMX execution completed successfully." <<
    // std::endl;
    return 1;
  }
}

extern "C" int vx_dispatch_gpu(float *xout, float *x, float *w, int n, int d) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::cerr << "Failed to acquire Metal device!" << std::endl;
      return 0;
    }

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    // Wrap input arrays in MTLBuffer (using copy for safety as mmap bounds
    // might not be 4K aligned)
    id<MTLBuffer> buf_w =
        [device newBufferWithBytes:w
                            length:(d * n * sizeof(float))
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_x =
        [device newBufferWithBytes:x
                            length:(n * sizeof(float))
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_out =
        [device newBufferWithLength:(d * sizeof(float))
                            options:MTLResourceStorageModeShared];

    MPSMatrixDescriptor *desc_w =
        [MPSMatrixDescriptor matrixDescriptorWithRows:d
                                              columns:n
                                             rowBytes:(n * sizeof(float))
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *desc_x =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                              columns:1
                                             rowBytes:(1 * sizeof(float))
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *desc_out =
        [MPSMatrixDescriptor matrixDescriptorWithRows:d
                                              columns:1
                                             rowBytes:(1 * sizeof(float))
                                             dataType:MPSDataTypeFloat32];

    MPSMatrix *mat_w = [[MPSMatrix alloc] initWithBuffer:buf_w
                                              descriptor:desc_w];
    MPSMatrix *mat_x = [[MPSMatrix alloc] initWithBuffer:buf_x
                                              descriptor:desc_x];
    MPSMatrix *mat_out = [[MPSMatrix alloc] initWithBuffer:buf_out
                                                descriptor:desc_out];

    // C = op(A) * op(B)
    MPSMatrixMultiplication *mul =
        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                          transposeLeft:false
                                         transposeRight:false
                                             resultRows:d
                                          resultColumns:1
                                        interiorColumns:n
                                                  alpha:1.0
                                                   beta:0.0];

    [mul encodeToCommandBuffer:commandBuffer
                    leftMatrix:mat_w
                   rightMatrix:mat_x
                  resultMatrix:mat_out];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Copy result back to CPU host memory
    memcpy(xout, [buf_out contents], d * sizeof(float));

    return 1;
  }
}

extern "C" int vx_dispatch_ane(float *xout, float *x, float *w, int n, int d) {
  @autoreleasepool {
    printf("--- EXECUTING ON APPLE NEURAL ENGINE ---\n");
    static MLModel *model = nil;
    NSError *error = nil;
    if (!model) {
      NSURL *modelURL = [NSURL fileURLWithPath:@"matmul_4x4.mlmodelc"];
      MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
      config.computeUnits = MLComputeUnitsAll;
      model = [MLModel modelWithContentsOfURL:modelURL
                                configuration:config
                                        error:&error];
      if (!model) {
        std::cerr << "[Vx Dispatcher] WARNING: Failed to load ANE model (matmul_4x4.mlmodelc)! Falling back to CPU." << std::endl;
        return 0;
      }
    }

    // Allocate MLMultiArrays directly and copy data into them.
    // CoreML/ANE requires specific memory alignment and mapping (often IOSurface-backed)
    // for DMA access. Using initWithDataPointer on unaligned heap pointers causes
    // the Neural Engine to crash when it attempts to fetch the memory.
    MLMultiArray *arrayW = [[MLMultiArray alloc] initWithShape:@[ @(d), @(n) ]
                                                      dataType:MLMultiArrayDataTypeFloat32
                                                         error:&error];
    if (arrayW) {
        memcpy(arrayW.dataPointer, w, d * n * sizeof(float));
    }

    MLMultiArray *arrayX = [[MLMultiArray alloc] initWithShape:@[ @(n), @(d) ]
                                                      dataType:MLMultiArrayDataTypeFloat32
                                                         error:&error];
    if (arrayX) {
        memcpy(arrayX.dataPointer, x, n * d * sizeof(float));
    }

    id<MLFeatureProvider> inputFeatures = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"w" : arrayW, @"x" : arrayX}
                     error:&error];

    id<MLFeatureProvider> outputFeatures =
        [model predictionFromFeatures:inputFeatures error:&error];
    if (error) {
      std::cerr << "[Vx Dispatcher] CoreML prediction failed! Error: " <<
          [[error localizedDescription] UTF8String] << std::endl;
      assert(false && "CoreML prediction failed on ANE");
      abort();
    }

    // The CoreML model might name the output "var_2", "matmul_0", etc.
    // We just grab the first available output feature.
    NSString *outputName = [outputFeatures.featureNames anyObject];
    MLMultiArray *outArray = [outputFeatures featureValueForName:outputName].multiArrayValue;

    if (!outArray) {
        std::cerr << "[Vx Dispatcher] FATAL: Failed to extract MLMultiArray from output!" << std::endl;
        abort();
    }

    if (outArray.dataPointer) {
        memcpy(xout, outArray.dataPointer, d * n * sizeof(float));
    } else {
        for (int i = 0; i < (d * n); i++) {
            NSNumber *val = [outArray objectAtIndexedSubscript:i];
            xout[i] = [val floatValue];
        }
    }

    return 1;
  }
}

extern "C" int vx_dispatch_ane_affine(float *out, float *x, float alpha, float beta, int length) {
  @autoreleasepool {
    printf("--- EXECUTING ON APPLE NEURAL ENGINE (AFFINE) ---\n");
    static MLModel *affineModel = nil;
    NSError *error = nil;
    if (!affineModel) {
      NSURL *modelURL = [NSURL fileURLWithPath:@"affine_4.mlmodelc"];
      MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
      config.computeUnits = MLComputeUnitsAll;
      affineModel = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
      if (!affineModel) {
        std::cerr << "[Vx Dispatcher] WARNING: Failed to load affine_4.mlmodelc. Falling back to CPU." << std::endl;
        return 0;
      }
    }

    MLMultiArray *arrayX = [[MLMultiArray alloc] initWithShape:@[ @(length) ] dataType:MLMultiArrayDataTypeFloat32 error:&error];
    if (arrayX) memcpy(arrayX.dataPointer, x, length * sizeof(float));

    MLMultiArray *arrayAlpha = [[MLMultiArray alloc] initWithShape:@[ @(1) ] dataType:MLMultiArrayDataTypeFloat32 error:&error];
    if (arrayAlpha) ((float*)arrayAlpha.dataPointer)[0] = alpha;

    MLMultiArray *arrayBeta = [[MLMultiArray alloc] initWithShape:@[ @(1) ] dataType:MLMultiArrayDataTypeFloat32 error:&error];
    if (arrayBeta) ((float*)arrayBeta.dataPointer)[0] = beta;

    id<MLFeatureProvider> inputFeatures = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"a": arrayX, @"alpha": arrayAlpha, @"beta": arrayBeta} error:&error];

    id<MLFeatureProvider> outputFeatures = [affineModel predictionFromFeatures:inputFeatures error:&error];
    if (error) {
      std::cerr << "[Vx Dispatcher] CoreML prediction failed! Error: " << [[error localizedDescription] UTF8String] << std::endl;
      abort();
    }

    NSString *outputName = [outputFeatures.featureNames anyObject];
    MLMultiArray *arrayOut = [outputFeatures featureValueForName:outputName].multiArrayValue;
    if (arrayOut) {
      if (arrayOut.dataPointer) {
          memcpy(out, arrayOut.dataPointer, length * sizeof(float));
      } else {
          for (int i = 0; i < length; i++) {
              out[i] = [[arrayOut objectAtIndexedSubscript:i] floatValue];
          }
      }
    }
    return 1;
  }
}



#include "../include/vx_hardware_runtime.h"
#include "vx_host_call.h"
#include <cstdlib>

extern "C" {
#include <dlfcn.h>

void *vx_plugin_alloc_and_transfer(size_t bytes, void *host_ptr,
                                   uint32_t topology_id) {
  void *ptr = malloc(bytes);
  if (host_ptr) {
    memcpy(ptr, host_ptr, bytes);
  }
  return ptr;
}

// Invoke the JIT-compiled kernel through its MLIR C-interface. The producer
// supplies, per argument, a pointer to that argument's value (device_args) plus
// an ABI type tag (arg_tags). libffi uses the tags to place each argument in
// the correct GP/FP register or stack slot, which a fixed void*-array call
// cannot do (e.g. by-value floats must travel in FP registers). See
// docs/lang/abi.md.


extern "C" uint64_t vx_plugin_dispatch_async(const void *binary_payload,
                                             size_t payload_size,
                                             void **device_args,
                                             const int32_t *arg_tags,
                                             int64_t num_args) {
  const char *kernel_name = (const char *)binary_payload;
  printf("[Vx Dispatcher] Intercepted kernel dispatch: %s with %lld args\n", kernel_name, (long long)num_args);

  std::string kernel_str(kernel_name);
  assert(kernel_str.find("vx_npu_kernel_") == 0 && "Expected kernel name to start with vx_npu_kernel_");

  // Dynamically scan for all memory reference arguments (arg_tags == 0)
  // rather than hardcoding exactly 5 arguments (which fluctuates based on MLIR
  // optimization passes like block size, grid size, or extra scalars).
  // Selection is unchanged -- every kind-0 argument is collected in order, and
  // the heuristic below still picks the last three. What is new is that each
  // one now carries its element type and rank, so the f32/rank-2 layout
  // MemRef2D assumes can be *checked* before the pointer is reinterpreted
  // rather than taken on faith. An f16 matmul does execute (#320) and would
  // otherwise be read as f32 here.
  std::vector<MemRef2D*> memrefs;
  std::vector<int32_t> memref_elems;
  std::vector<int32_t> memref_ranks;
  printf("[Vx Dispatcher] Scanning %lld args:\n", (long long)num_args);
  for (int64_t i = 0; i < num_args; i++) {
    printf("  Arg %lld: tag=%d\n", (long long)i, arg_tags[i]);
    if (VX_ABI_KIND(arg_tags[i]) == VX_ABI_KIND_MEMREF) {
      memrefs.push_back(*(MemRef2D**)device_args[i]);
      memref_elems.push_back(VX_ABI_ELEM(arg_tags[i]));
      memref_ranks.push_back(VX_ABI_RANK(arg_tags[i]));
    }
  }

  // True when the descriptor at `idx` really is the layout MemRef2D describes.
  // Rank 0 with an unknown element type is an opaque pointer rather than a
  // ranked memref: the tag encoding gives both kind 0.
  auto layout_ok = [&](size_t idx) {
    return memref_elems[idx] == VX_DTYPE_F32 && memref_ranks[idx] == 2;
  };
  printf("[Vx Dispatcher] Found %zu MemRefs\n", memrefs.size());

  size_t n_mr = memrefs.size();
  if (n_mr >= 3 && layout_ok(n_mr - 3) && layout_ok(n_mr - 2) &&
      layout_ok(n_mr - 1)) {
    // Conventionally, Vx compiler captures these as [..., res, a, b] based on usage/definition order
    MemRef2D* res = memrefs[memrefs.size() - 3];
    MemRef2D* a = memrefs[memrefs.size() - 2];
    MemRef2D* b = memrefs[memrefs.size() - 1];

    // Ensure pointers are valid before accessing sizes
    if (res && a && b) {
      MemRef2D *actual_res = (MemRef2D *)res->aligned;
      MemRef2D *actual_a = (MemRef2D *)a->aligned;
      MemRef2D *actual_b = (MemRef2D *)b->aligned;

      // Very basic shape heuristic routing to our universal 4x4 ANE primitive
      if (actual_res && actual_a && actual_b &&
          actual_res->sizes[0] == 4 && actual_res->sizes[1] == 4 && actual_a->sizes[0] == 4) {
          if (vx_dispatch_ane(actual_res->aligned, actual_b->aligned, actual_a->aligned, 4, 4)) {
            return 1;
          }
      }
    }
  } else if (n_mr == 2 && memref_elems[0] == VX_DTYPE_F32 &&
             memref_elems[1] == VX_DTYPE_F32) {
    // Affine scalar math pattern: c[i] = a[i] * alpha + beta
    std::vector<float> scalars;
    for (int64_t i = 0; i < num_args; i++) {
      if (VX_ABI_KIND(arg_tags[i]) == VX_ABI_KIND_F32) {
        scalars.push_back(*(float*)device_args[i]);
      }
    }

    if (scalars.size() >= 2) {
      MemRef1D* res = (MemRef1D*)memrefs[0];
      MemRef1D* a = (MemRef1D*)memrefs[1];

      if (res && a) {
        MemRef1D *actual_res = (MemRef1D *)res->aligned;
        MemRef1D *actual_a = (MemRef1D *)a->aligned;

        if (actual_res && actual_a && actual_res->sizes[0] == 4 && actual_a->sizes[0] == 4) {
          if (vx_dispatch_ane_affine(actual_res->aligned, actual_a->aligned, scalars[0], scalars[1], 4)) {
            return 1;
          }
        }
      }
    }
  }

#ifdef VX_ENABLE_CPU_FALLBACK
  printf("[Vx Dispatcher] Hardware backend unsupported for %s. Falling back to CPU libffi dispatch!\n", kernel_name);

  void *kernel = vx_host_kernel_symbol(kernel_name);
  if (!kernel) {
    printf("DEBUG: Could not find JIT kernel %s, skipping execution\n",
           kernel_name);
    return 1;
  }

  if (!vx_host_call_kernel(kernel, device_args, arg_tags, num_args)) {
    printf("ERROR: could not build a call for %s\n", kernel_name);
    return 0;
  }
  return 1;
#else
  std::cerr << "[Vx Dispatcher] FATAL: Kernel " << kernel_name << " not explicitly handled, and CPU fallback is disabled!" << std::endl;
  assert(false && "CPU Fallback Disabled");
  abort();
#endif
}

uint64_t vx_plugin_dispatch_async_flat(float *xout, float *x, float *w, int n,
                                       int d) {
  // Dispatch to Apple Neural Engine payload executor
  vx_dispatch_ane(xout, x, w, n, d);
  return 1;
}

void vx_plugin_await_future(uint64_t future_id) {
  // Dummy synchronous implementation, so we don't need to block
}

int32_t vx_plugin_transfer_device_to_host(void *device_ptr, void *host_ptr,
                                          size_t bytes) {
  memcpy(host_ptr, device_ptr, bytes);
  return 1;
}

void vx_plugin_free(void *device_ptr, uint32_t topology_id) {
  free(device_ptr);
}

void vx_plugin_release_future(uint64_t future_id) {
  // No-op for dummy implementation
}

int32_t vx_plugin_control(uint32_t opcode, void *payload) {
  if (opcode == VX_CTRL_GET_DEVICE_COUNT) {
    return 1; // One Apple NPE device available
  }
  return 0;
}
}
