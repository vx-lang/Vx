#include "npu_dispatch.h"
#import <Accelerate/Accelerate.h>
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <ffi/ffi.h>
#include <iostream>
#include <vector>

/// Whether to narrate what the dispatcher is doing.
///
/// This backend wrote its narration to *stdout*, unconditionally, which for a
/// program whose result is what it prints means the diagnostics and the answer
/// are interleaved character by character and neither can be read. A Llama run
/// on this machine emitted its story shredded through
/// `[Vx Dispatcher] Intercepted kernel dispatch`, so there was no way to check
/// tokens locally -- the check had to be done on rented hardware, where the
/// CUDA backend happens to be quiet.
///
/// Both other backends already had this: runtime/cuda_dispatch.cpp and
/// runtime/host_dispatch_common.h gate on the same variable and write to
/// stderr, the latter with the comment that "an unconditional banner would fail
/// every placed test". This one was the exception.
static bool vx_npu_verbose() {
  static const bool on = [] {
    const char *v = getenv("VX_DISPATCH_VERBOSE");
    return v && v[0] != '\0' && strcmp(v, "0") != 0;
  }();
  return on;
}

/// Narration: stderr, and only when asked for.
#define VX_NPU_LOG(...)                                                        \
  do {                                                                         \
    if (vx_npu_verbose()) {                                                    \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

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
    VX_NPU_LOG("--- EXECUTING ON APPLE NEURAL ENGINE ---\n");
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
        std::cerr << "[Vx Dispatcher] WARNING: Failed to load ANE model "
                     "(matmul_4x4.mlmodelc)! Falling back to CPU."
                  << std::endl;
        return 0;
      }
    }

    // Allocate MLMultiArrays directly and copy data into them.
    // CoreML/ANE requires specific memory alignment and mapping (often
    // IOSurface-backed) for DMA access. Using initWithDataPointer on unaligned
    // heap pointers causes the Neural Engine to crash when it attempts to fetch
    // the memory.
    MLMultiArray *arrayW =
        [[MLMultiArray alloc] initWithShape:@[ @(d), @(n) ]
                                   dataType:MLMultiArrayDataTypeFloat32
                                      error:&error];
    if (arrayW) {
      memcpy(arrayW.dataPointer, w, d * n * sizeof(float));
    }

    MLMultiArray *arrayX =
        [[MLMultiArray alloc] initWithShape:@[ @(n), @(d) ]
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
    MLMultiArray *outArray =
        [outputFeatures featureValueForName:outputName].multiArrayValue;

    if (!outArray) {
      std::cerr << "[Vx Dispatcher] FATAL: Failed to extract MLMultiArray from "
                   "output!"
                << std::endl;
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

extern "C" int vx_dispatch_ane_affine(float *out, float *x, float alpha,
                                      float beta, int length) {
  @autoreleasepool {
    VX_NPU_LOG("--- EXECUTING ON APPLE NEURAL ENGINE (AFFINE) ---\n");
    static MLModel *affineModel = nil;
    NSError *error = nil;
    if (!affineModel) {
      NSURL *modelURL = [NSURL fileURLWithPath:@"affine_4.mlmodelc"];
      MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
      config.computeUnits = MLComputeUnitsAll;
      affineModel = [MLModel modelWithContentsOfURL:modelURL
                                      configuration:config
                                              error:&error];
      if (!affineModel) {
        std::cerr << "[Vx Dispatcher] WARNING: Failed to load "
                     "affine_4.mlmodelc. Falling back to CPU."
                  << std::endl;
        return 0;
      }
    }

    MLMultiArray *arrayX =
        [[MLMultiArray alloc] initWithShape:@[ @(length) ]
                                   dataType:MLMultiArrayDataTypeFloat32
                                      error:&error];
    if (arrayX)
      memcpy(arrayX.dataPointer, x, length * sizeof(float));

    MLMultiArray *arrayAlpha =
        [[MLMultiArray alloc] initWithShape:@[ @(1) ]
                                   dataType:MLMultiArrayDataTypeFloat32
                                      error:&error];
    if (arrayAlpha)
      ((float *)arrayAlpha.dataPointer)[0] = alpha;

    MLMultiArray *arrayBeta =
        [[MLMultiArray alloc] initWithShape:@[ @(1) ]
                                   dataType:MLMultiArrayDataTypeFloat32
                                      error:&error];
    if (arrayBeta)
      ((float *)arrayBeta.dataPointer)[0] = beta;

    id<MLFeatureProvider> inputFeatures =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
          @"a" : arrayX,
          @"alpha" : arrayAlpha,
          @"beta" : arrayBeta
        }
                                                          error:&error];

    id<MLFeatureProvider> outputFeatures =
        [affineModel predictionFromFeatures:inputFeatures error:&error];
    if (error) {
      std::cerr << "[Vx Dispatcher] CoreML prediction failed! Error: " <<
          [[error localizedDescription] UTF8String] << std::endl;
      abort();
    }

    NSString *outputName = [outputFeatures.featureNames anyObject];
    MLMultiArray *arrayOut =
        [outputFeatures featureValueForName:outputName].multiArrayValue;
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
#include "vx_dispatch_plan.h"
#include "vx_host_call.h"
#include "vx_remote_routing.h"
#include <cstdlib>

extern "C" {
#include <dlfcn.h>

void *vx_plugin_alloc_and_transfer(size_t bytes, void *host_ptr,
                                   uint32_t topology_id) {
  // Placement first, hardware second. Without this the whole fleet story is
  // absent from this backend: a program run on Apple silicon with a manifest
  // naming a worker allocated here instead, dispatched here, and said nothing.
  // Not a wrong answer -- the right answer from the wrong machine, which is
  // worse, because a run that never left the host still prints tokens.
  void *remote = nullptr;
  if (vx_routing_try_alloc(bytes, host_ptr, topology_id, &remote)) {
    return remote;
  }
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

  // Before anything local: does this placement name another machine?
  if (vx_routing_try_dispatch(binary_payload, payload_size, device_args,
                              arg_tags, num_args)) {
    return 1;
  }

  VX_NPU_LOG("[Vx Dispatcher] Intercepted kernel dispatch: %s with %lld args\n",
             kernel_name, (long long)num_args);

  std::string kernel_str(kernel_name);
  assert(kernel_str.find("vx_npu_kernel_") == 0 &&
         "Expected kernel name to start with vx_npu_kernel_");

  // A matmul the compiler recognised says so, and says which argument is which
  // (#325). That replaces what stood here: collect every memref argument, take
  // the last three, and call them [result, a, b] by convention. The convention
  // was unfalsifiable -- for square operands every assignment conforms, so a
  // wrong one produced a plausible matrix rather than a failure -- and it was
  // wrong for any kernel whose captures did not land in that order.
  //
  // The 4x4 check that remains is a real constraint rather than a heuristic:
  // the ANE primitive is a CoreML model compiled for exactly that shape
  // (scripts/generate_ane_primitives.py). Anything else has no model to run on.
  vx_gemm_plan plan;
  bool is_gemm = vx_gemm_plan_decode(binary_payload, payload_size, device_args,
                                     arg_tags, num_args, &plan);
  if (is_gemm) {
    VX_NPU_LOG("[Vx Dispatcher] Recognised GEMM %lldx%lldx%lld %s -> %s\n",
               (long long)plan.m, (long long)plan.n, (long long)plan.k,
               vx_dtype_name(plan.dtype),
               plan.out_kind == VX_GEMM_OUT_SLOT ? "slot" : "buffer");
  }

  if (is_gemm && plan.dtype == VX_DTYPE_F32 && plan.m == 4 && plan.n == 4 &&
      plan.k == 4 && plan.a_row_stride == 4 && plan.b_row_stride == 4 &&
      plan.out_row_stride == 4) {
    // Every buffer is 16 contiguous floats, which is what the model copies in
    // and out; a padded row stride would need a strided copy it does not do.
    // The model computes w @ x, so `w` is A and `x` is B.
    float *result = (float *)plan.out_data;
    if (plan.out_kind == VX_GEMM_OUT_SLOT) {
      // Standing in for the kernel means allocating the result it would have
      // allocated, and publishing the descriptor it would have stored.
      result = (float *)malloc(16 * sizeof(float));
    }

    if (result &&
        vx_dispatch_ane(result, (float *)const_cast<void *>(plan.b_data),
                        (float *)const_cast<void *>(plan.a_data), 4, 4)) {
      if (plan.out_kind == VX_GEMM_OUT_SLOT) {
        vx_gemm_publish_slot(&plan, result);
      }
      return 1;
    }
    if (plan.out_kind == VX_GEMM_OUT_SLOT) {
      free(result);
    }
  }

  // The affine pattern (`c[i] = a[i] * alpha + beta`) is not classified, so it
  // is still recognised by shape. What has changed is how each argument is
  // read: an operand may arrive as a buffer or as the slot a local tensor lives
  // in, and vx_operand_desc resolves either to the descriptor itself. The code
  // here previously assumed every argument was a slot and dereferenced
  // unconditionally, which was right only while every tensor involved was a
  // local.
  std::vector<const void *> memrefs;
  std::vector<int32_t> memref_elems;
  VX_NPU_LOG("[Vx Dispatcher] Scanning %lld args:\n", (long long)num_args);
  for (int64_t i = 0; i < num_args; i++) {
    VX_NPU_LOG("  Arg %lld: tag=%d\n", (long long)i, arg_tags[i]);
    if (VX_ABI_KIND(arg_tags[i]) != VX_ABI_KIND_MEMREF) {
      continue;
    }
    const void *desc = vx_operand_desc(device_args, arg_tags, (int)i);
    if (!desc) {
      continue;
    }
    memrefs.push_back(desc);
    memref_elems.push_back(VX_ABI_ELEM(arg_tags[i]));
  }
  VX_NPU_LOG("[Vx Dispatcher] Found %zu MemRefs\n", memrefs.size());

  if (memrefs.size() == 2 && memref_elems[0] == VX_DTYPE_F32 &&
      memref_elems[1] == VX_DTYPE_F32) {
    // Affine scalar math pattern: c[i] = a[i] * alpha + beta
    std::vector<float> scalars;
    for (int64_t i = 0; i < num_args; i++) {
      if (VX_ABI_KIND(arg_tags[i]) == VX_ABI_KIND_F32) {
        scalars.push_back(*(float *)device_args[i]);
      }
    }

    if (scalars.size() >= 2) {
      const int64_t *res_sizes = vx_memref_sizes(memrefs[0]);
      const int64_t *a_sizes = vx_memref_sizes(memrefs[1]);

      if (res_sizes[0] == 4 && a_sizes[0] == 4) {
        if (vx_dispatch_ane_affine((float *)vx_memref_aligned(memrefs[0]),
                                   (float *)vx_memref_aligned(memrefs[1]),
                                   scalars[0], scalars[1], 4)) {
          return 1;
        }
      }
    }
  }

#ifdef VX_ENABLE_CPU_FALLBACK
  VX_NPU_LOG("[Vx Dispatcher] Hardware backend unsupported for %s. Falling "
             "back to CPU libffi dispatch!\n",
             kernel_name);

  // Both failures below mean the kernel did not run. Neither is recoverable:
  // the buffer the caller expected to be filled holds whatever it held before,
  // and the program cannot tell that from an answer. The other two backends
  // abort here; this one returned 1 for the first case -- "skipping execution"
  // reported as success -- and 0 for the second, which no caller checks.
  void *kernel = vx_host_kernel_symbol(kernel_name);
  if (!kernel) {
    fprintf(stderr, "[Vx Dispatcher] FATAL: outlined kernel %s not found\n",
            kernel_name);
    abort();
  }

  if (!vx_host_call_kernel(kernel, device_args, arg_tags, num_args)) {
    fprintf(stderr, "[Vx Dispatcher] FATAL: could not build a call for %s\n",
            kernel_name);
    abort();
  }
  return 1;
#else
  std::cerr << "[Vx Dispatcher] FATAL: Kernel " << kernel_name
            << " not explicitly handled, and CPU fallback is disabled!"
            << std::endl;
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
                                          size_t bytes, uint32_t topology_id) {
  if (vx_routing_try_fetch(device_ptr, host_ptr, bytes, topology_id)) {
    return 1;
  }
  vx_routing_refuse_handle("a read-back", device_ptr, topology_id);
  // The NPE shares the host's memory, so the topology names it and nothing
  // follows from that.
  (void)topology_id;
  memcpy(host_ptr, device_ptr, bytes);
  return 1;
}

void *vx_plugin_transfer_peer(void *src_device_ptr, uint32_t src_topology_id,
                              uint32_t dst_topology_id, size_t bytes) {
  void *routed = nullptr;
  if (vx_routing_try_peer(src_device_ptr, src_topology_id, dst_topology_id,
                          bytes, &routed)) {
    return routed;
  }
  // One memory here too, so a movement between two topologies is a copy. A
  // disaggregated program therefore runs on this backend and produces the same
  // tokens, which is what gives a two-device run something to be checked
  // against (#347).
  vx_routing_refuse_handle("a peer handoff", src_device_ptr, src_topology_id);
  (void)src_topology_id;
  (void)dst_topology_id;
  void *dst = malloc(bytes);
  if (dst && src_device_ptr) {
    memcpy(dst, src_device_ptr, bytes);
  }
  return dst;
}

void vx_plugin_free(void *device_ptr, uint32_t topology_id) {
  if (vx_routing_try_free(device_ptr, topology_id)) {
    return;
  }
  vx_routing_refuse_handle("a free", device_ptr, topology_id);
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
