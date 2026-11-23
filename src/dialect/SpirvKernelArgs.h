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

/// Inline each function that a kernel in `gpuModule` calls, and erase the
/// functions that no call uses any more. The flattening below rewrites the
/// tensors of a kernel, not those of a function it calls, so this runs first.
/// A call that stays, such as one to an OpenCL built-in, is left as it is.
LogicalResult inlineSpirvKernelCalls(gpu::GPUModuleOp gpuModule,
                                     std::string &error);

/// Give each `index` block argument of a kernel in `gpuModule` the type `i64`,
/// with a cast at each branch to the block and at its start. The gpu ->
/// llvm-spv conversion turns `index` into `i64` in the ops it rewrites, but not
/// in a block argument. A loop of a function that a kernel calls counts in
/// `index`, and its loop becomes such an argument once the image inlines the
/// call.
void widenIndexBlockArguments(gpu::GPUModuleOp gpuModule);

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
/// A kernel that the flattening cannot rewrite, and the reason in the words a
/// warning gives the author.
struct SpirvKernelRefusal {
  std::string kernel;
  std::string reason;
};

/// Returns failure when the module holds something the rewrite cannot express.
/// A kernel that failed may be half rewritten, so the caller must discard the
/// module: a mistranslated kernel is a wrong answer on a machine the compiler
/// is not running on.
///
/// On failure exactly one of the two outputs is set. `refused` gets every
/// kernel that holds such a construct, in the module's order. `error` gets the
/// reason when no kernel holds it, for example a global or a function that a
/// kernel calls.
LogicalResult
flattenSpirvKernelArgs(gpu::GPUModuleOp gpuModule, std::string &error,
                       SmallVectorImpl<SpirvKernelRefusal> &refused);

/// The flattening as a pass, for `vx-opt`, which is where the address
/// arithmetic it builds can be read without a GPU.
std::unique_ptr<Pass> createFlattenSpirvKernelArgsPass();

} // namespace vx
} // namespace mlir

#endif // VX_SPIRV_KERNEL_ARGS_H
