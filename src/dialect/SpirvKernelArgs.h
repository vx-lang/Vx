#ifndef VX_SPIRV_KERNEL_ARGS_H
#define VX_SPIRV_KERNEL_ARGS_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace mlir {
namespace vx {

/// Rewrite every kernel in `gpuModule` to the flat argument list the launch
/// ABI already passes, so the SPIR-V pipeline can compile it.
///
/// Each memref argument becomes the values `vx_launch_param_width` counts --
/// allocated pointer, aligned pointer, offset, one size and one stride per
/// dimension -- and each memref access becomes address arithmetic on them. An
/// architecture whose device image needs a memref value on the device side
/// cannot use this; SPIR-V's typed pointers reject the descriptor struct the
/// memref would become.
///
/// Returns failure and sets `error` to a message naming the construct when a
/// kernel holds something the rewrite cannot express. It never leaves a kernel
/// half-rewritten: a mistranslated kernel is a wrong answer on a machine the
/// compiler is not running on.
LogicalResult flattenSpirvKernelArgs(gpu::GPUModuleOp gpuModule,
                                     std::string &error);

} // namespace vx
} // namespace mlir

#endif // VX_SPIRV_KERNEL_ARGS_H
