#ifndef VX_SPIRV_KERNEL_ARGS_H
#define VX_SPIRV_KERNEL_ARGS_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>
#include <optional>
#include <string>

namespace mlir {
namespace vx {

/// The reason a placed region cannot become a SPIR-V device image, in the words
/// a warning gives its author, or nothing when it can.
///
/// `op` is an operation in the host form of a placed region, before the region
/// is copied into a device module.
///
/// This is the one answer for the whole SPIR-V path. Two places ask it: the
/// gate that decides whether a region gets an image, and the warning that tells
/// the author why the region runs on the host. The flattening below is the
/// other half of the same rule: it rewrites what this admits, and what it still
/// cannot rewrite is a warning and a host run rather than a failed compile. The
/// two used to carry a rule each, and the day they disagreed a program that
/// used to run stopped compiling.
std::optional<std::string> spirvRefusalOf(Operation *op);

/// Rewrite every kernel in `gpuModule` to the flat argument list the launch
/// ABI already passes, so the SPIR-V pipeline can compile it.
///
/// Each memref argument becomes the values `vx_launch_param_width` counts --
/// allocated pointer, aligned pointer, offset, one size and one stride per
/// dimension -- and each memref access becomes address arithmetic on them. An
/// architecture whose device image needs a memref value on the device side
/// cannot use this. SPIR-V's typed pointers reject the descriptor struct the
/// memref would become.
///
/// Returns failure and sets `error` to a message naming the construct when a
/// kernel holds something the rewrite cannot express. It never leaves a kernel
/// half-rewritten: a mistranslated kernel is a wrong answer on a machine the
/// compiler is not running on.
///
/// `failedKernel` gets the name of the kernel that holds the construct. It
/// stays empty when the construct is outside every kernel, for example in a
/// global or in a function that a kernel calls.
LogicalResult flattenSpirvKernelArgs(gpu::GPUModuleOp gpuModule,
                                     std::string &error,
                                     std::string &failedKernel);

/// The flattening as a pass, for `vx-opt`, which is where the address
/// arithmetic it builds can be read without a GPU.
std::unique_ptr<Pass> createFlattenSpirvKernelArgsPass();

} // namespace vx
} // namespace mlir

#endif // VX_SPIRV_KERNEL_ARGS_H
