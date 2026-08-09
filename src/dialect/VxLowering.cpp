#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "VxDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include <atomic>

using namespace mlir;
using namespace mlir::vx;

#define DEBUG_TYPE "vx-lowering"

namespace {

// Classify what an outlined region computes, for plugins that route to vendor
// kernels (#325). Returns "" when the region is not a single recognised
// operation, which is the common case and carries no attribute.
//
// Deliberately conservative: it counts the payload operations in the region and
// classifies only when exactly one is present, because a library call replaces
// the whole kernel. A matmul with a bias add fused around it is not something a
// plain GEMM call can stand in for.
static bool isZeroConstant(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getName().getStringRef() != "arith.constant")
    return false;
  auto attr = def->getAttrOfType<FloatAttr>("value");
  return attr && attr.getValue().isZero();
}

static StringRef kernelKindOf(Region &body, Operation **payloadOut = nullptr) {
  if (payloadOut)
    *payloadOut = nullptr;
  Operation *matmul = nullptr;
  Operation *fill = nullptr;
  unsigned otherPayload = 0;

  // Only the region's own operations, not a deep walk: a linalg op carries its
  // arithmetic in a nested region, and descending into it would count that
  // op's `linalg.yield` as separate payload and classify nothing.
  for (Block &block : body) {
    for (Operation &opRef : block) {
      Operation *op = &opRef;
      StringRef name = op->getName().getStringRef();
      // Structural and bookkeeping operations are not the payload.
      if (name == "vx.yield" || name == "vx.return" ||
          name.starts_with("arith.") || name.starts_with("memref.") ||
          name.starts_with("llvm.") || name.starts_with("scf.") ||
          name.starts_with("cf.") || name.starts_with("builtin.")) {
        continue;
      }
      if (name == "linalg.matmul" && !matmul) {
        matmul = op;
        continue;
      }
      if (name == "linalg.fill" && !fill) {
        fill = op;
        continue;
      }
      ++otherPayload;
    }
  }

  if (!matmul || otherPayload > 0)
    return StringRef();

  // `c = a @ b` emits a zero-fill of the accumulator followed by the matmul,
  // which is precisely GEMM with beta = 0 -- a library call subsumes both. The
  // fill has to be zeroing *this* matmul's output for that to hold: filling
  // with a non-zero v computes A*B + v, which no beta = 0 GEMM reproduces.
  if (fill) {
    if (fill->getNumOperands() < 2 || matmul->getNumOperands() < 3)
      return StringRef();
    if (fill->getOperand(1) != matmul->getOperand(2))
      return StringRef();
    if (!isZeroConstant(fill->getOperand(0)))
      return StringRef();
  }

  // Only on success, so a caller cannot read a payload op out of a region that
  // was ultimately rejected.
  if (payloadOut)
    *payloadOut = matmul;

  return "matmul";
}

// Describe which launch operand plays which role in a recognised matmul, as
// `a:<i>,b:<j>,out:<k>`, indexing the operand list the launch is built from.
//
// This is the half that actually resolves the ambiguity #325 exists for. The
// operation name alone tells a plugin it has a GEMM; it does not say which
// buffer is which, and for square operands nothing about the shapes can, since
// every assignment conforms. linalg names them -- `ins` in order, `outs` -- so
// the mapping is read off rather than guessed.
//
// The result buffer is normally *not* a capture: `c = a @ b` allocates inside
// the kernel and stores the pointer through a captured slot, so `out:` names
// that slot and `outkind=slot` says so. A plugin then knows it must publish the
// result by writing a descriptor there rather than filling a buffer it was
// handed. When the buffer is captured directly, `outkind=buffer`.
//
// Returns "" unless every role resolves. A partial mapping is worse than none:
// it invites a plugin to fill in the rest by convention, which is the guessing
// this is meant to replace.
static std::string matmulRolesOf(Operation *matmul, Region &body,
                                 const SetVector<Value> &captures,
                                 StringRef &outKind) {
  if (!matmul || matmul->getNumOperands() < 3)
    return std::string();

  auto indexOf = [&](Value v) -> int {
    for (auto en : llvm::enumerate(captures)) {
      if (en.value() == v)
        return static_cast<int>(en.index());
    }
    return -1;
  };

  int aIdx = indexOf(matmul->getOperand(0));
  int bIdx = indexOf(matmul->getOperand(1));
  int outIdx = indexOf(matmul->getOperand(2));
  outKind = "buffer";

  if (outIdx < 0) {
    // Follow the store that publishes the locally allocated result.
    Value resultBuf = matmul->getOperand(2);
    for (Block &block : body) {
      for (Operation &opRef : block) {
        if (opRef.getName().getStringRef() != "memref.store" ||
            opRef.getNumOperands() < 2) {
          continue;
        }
        if (opRef.getOperand(0) == resultBuf) {
          outIdx = indexOf(opRef.getOperand(1));
          outKind = "slot";
        }
      }
    }
  }

  if (aIdx < 0 || bIdx < 0 || outIdx < 0)
    return std::string();

  return ("a:" + Twine(aIdx) + ",b:" + Twine(bIdx) + ",out:" + Twine(outIdx))
      .str();
}

// Lower `vx.spawn` to `async.execute` for CPU topologies, or an outlined
// kernel + `vx.launch` for NPU/AccCore topologies.
struct SpawnOpLowering : public OpRewritePattern<SpawnOp> {
  using OpRewritePattern<SpawnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SpawnOp op,
                                PatternRewriter &rewriter) const override {
    int32_t topology = op.getTopology();

    if (topology == 0) {
      auto asyncExecuteOp =
          rewriter.create<async::ExecuteOp>(op.getLoc(),
                                            /*resultTypes=*/TypeRange{},
                                            /*dependencies=*/ValueRange{},
                                            /*operands=*/ValueRange{});

      Region &spawnBody = op.getBody();
      Region &asyncBody = asyncExecuteOp.getRegion();

      Block *asyncBlock;
      if (asyncBody.empty()) {
        asyncBlock = rewriter.createBlock(&asyncBody);
        rewriter.create<async::YieldOp>(op.getLoc(), ValueRange{});
      } else {
        asyncBlock = &asyncBody.front();
        if (asyncBlock->empty() || !isa<async::YieldOp>(asyncBlock->back())) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToEnd(asyncBlock);
          rewriter.create<async::YieldOp>(op.getLoc(), ValueRange{});
        }
      }

      SmallVector<Value> yieldedValues;
      if (!spawnBody.empty()) {
        Block &spawnBlock = spawnBody.front();
        if (!spawnBlock.empty() && isa<vx::YieldOp>(spawnBlock.back())) {
          auto yieldOp = cast<vx::YieldOp>(spawnBlock.back());
          for (auto val : yieldOp.getOperands()) {
            yieldedValues.push_back(val);
          }
          rewriter.eraseOp(yieldOp);
        }
        // Move operations from spawnBlock to asyncBlock
        auto &asyncOps = asyncBlock->getOperations();
        auto &spawnOps = spawnBlock.getOperations();
        // Move before the yield (which is at the end of asyncBlock)
        asyncOps.splice(std::prev(asyncOps.end()), spawnOps, spawnOps.begin(),
                        spawnOps.end());
      }

      rewriter.replaceOp(op, yieldedValues);
      return success();
    }

    // For NPU (100) or AccCore (200), we lower to an outlined kernel.
    Region &spawnBody = op.getBody();
    if (spawnBody.empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    SetVector<Value> captures;
    getUsedValuesDefinedAbove(spawnBody, captures);

    // Create the outlined function at the module level
    auto module = op->getParentOfType<ModuleOp>();
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToEnd(module.getBody());

    SmallVector<Type> argTypes;
    for (auto val : captures) {
      argTypes.push_back(val.getType());
    }

    // Determine result types from the yield operation. The yield may live in a
    // nested block (e.g. inside control flow) now that the entire region is
    // inlined, so search the whole region rather than only the entry block's
    // terminator.
    SmallVector<Type> resultTypes;
    vx::YieldOp resultYield;
    spawnBody.walk([&](vx::YieldOp y) {
      if (!resultYield)
        resultYield = y;
    });
    if (resultYield) {
      for (auto val : resultYield.getOperands()) {
        resultTypes.push_back(val.getType());
      }
    }

    // Create the kernel op. The counter is process-global (kernel names must be
    // unique across the module) and the test harness compiles files in
    // parallel, so it must be atomic to avoid a data race / duplicate names.
    auto funcType = rewriter.getFunctionType(argTypes, resultTypes);
    static std::atomic<int> kernelIdx{0};
    std::string funcName =
        "vx_npu_kernel_" + std::to_string(kernelIdx.fetch_add(1));
    auto kernelOp = rewriter.create<vx::KernelOp>(op.getLoc(), funcName,
                                                  funcType, topology);

    // Name what the region computes, so a plugin can route it to a vendor
    // kernel instead of inferring the operation from buffer shapes (#325).
    // Shape conformance alone cannot do it: for square operands every operand
    // assignment conforms, and swapping them yields a plausible matrix of wrong
    // numbers rather than a failure.
    //
    // Matched on the operation name rather than through the Linalg headers, to
    // avoid taking a dialect dependency here for what is a single string
    // comparison. Only a region whose whole job is the one op is classified: a
    // matmul with other computation around it is not a matmul the runtime can
    // hand to cuBLAS wholesale.
    //
    // A kernel that is not recognised carries no attribute and dispatches
    // exactly as before -- routing is an optimisation, and a kernel the
    // compiler cannot classify is not an error.
    Operation *payloadOp = nullptr;
    StringRef kernelKind = kernelKindOf(spawnBody, &payloadOp);
    StringRef outKind;
    std::string kernelRoles =
        matmulRolesOf(payloadOp, spawnBody, captures, outKind);
    if (!kernelKind.empty()) {
      kernelOp->setAttr("vx.kernel_kind", rewriter.getStringAttr(kernelKind));
      if (!kernelRoles.empty()) {
        kernelOp->setAttr("vx.kernel_roles",
                          rewriter.getStringAttr(kernelRoles));
        kernelOp->setAttr("vx.kernel_out_kind",
                          rewriter.getStringAttr(outKind));
      }
    }

    // Clone the entire region to avoid leaving SpawnOp with an invalid empty
    // region
    Region &kernelRegion = kernelOp.getBody();
    rewriter.cloneRegionBefore(spawnBody, kernelRegion, kernelRegion.end());

    // Fix up the entry block arguments to accept the captured variables
    Block &entryBlock = kernelRegion.front();
    for (auto type : argTypes) {
      entryBlock.addArgument(type, op.getLoc());
    }

    // Replace usages of captured variables inside the region with the block
    // arguments
    for (auto [cap, arg] : llvm::zip(captures, entryBlock.getArguments())) {
      Value capVal = cap;
      rewriter.replaceUsesWithIf(capVal, arg, [&](OpOperand &use) {
        return kernelRegion.isAncestor(use.getOwner()->getParentRegion());
      });
    }

    // Replace vx.yield with vx.return
    SmallVector<vx::YieldOp> yieldsToErase;
    kernelRegion.walk(
        [&](vx::YieldOp yieldOp) { yieldsToErase.push_back(yieldOp); });
    for (auto y : yieldsToErase) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(y);
      rewriter.create<vx::ReturnOp>(y.getLoc(), y.getOperands());
      rewriter.eraseOp(y);
    }

    // If the region has no terminator in the last block (e.g. empty spawn), add
    // vx.return
    for (Block &block : kernelRegion) {
      if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>()) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(&block);
        rewriter.create<vx::ReturnOp>(op.getLoc(), ValueRange{});
      }
    }

    // Restore insertion point to replace vx.spawn with vx.launch
    rewriter.restoreInsertionPoint(ip);

    SmallVector<Value> launchOperands(captures.begin(), captures.end());
    auto launchOp = rewriter.create<vx::LaunchOp>(
        op.getLoc(), resultTypes,
        SymbolRefAttr::get(rewriter.getContext(), funcName), launchOperands);

    // Carried on the launch as well as the kernel: the launch is what lowers to
    // the dispatch call, so this is where the fact has to be to reach a plugin.
    if (!kernelKind.empty()) {
      launchOp->setAttr("vx.kernel_kind", rewriter.getStringAttr(kernelKind));
      if (!kernelRoles.empty()) {
        launchOp->setAttr("vx.kernel_roles",
                          rewriter.getStringAttr(kernelRoles));
        launchOp->setAttr("vx.kernel_out_kind",
                          rewriter.getStringAttr(outKind));
      }
    }

    rewriter.replaceOp(op, launchOp.getResults());
    return success();
  }
};

struct TransferOpLowering : public OpRewritePattern<TransferOp> {
  using OpRewritePattern<TransferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getOperand();
    auto srcType = dyn_cast<MemRefType>(src.getType());

    // If it's not a MemRef (e.g., primitive i32/f32), we fail compilation.
    // The user explicitly mandated that implicit conversions/pass-throughs
    // are disabled to enforce strict data layout transitions.
    if (!srcType) {
      llvm::errs() << "[VxLowering] TransferOp srcType is not MemRefType\n";
      op.emitError("vx.transfer currently only supports MemRef types. "
                   "Attempted to transfer a scalar/primitive.");
      return failure();
    }

    auto targetType = cast<MemRefType>(op.getResult().getType());
    LLVM_DEBUG(llvm::errs() << "[VxLowering] TransferOp lowering from "
                            << srcType << " to " << targetType << "\n");

    // Extract dynamic sizes from the source memref
    SmallVector<Value> dynamicSizes;
    for (int i = 0; i < srcType.getRank(); ++i) {
      if (srcType.isDynamicDim(i)) {
        auto indexAttr = rewriter.getIndexAttr(i);
        auto indexVal =
            rewriter.create<arith::ConstantOp>(op.getLoc(), indexAttr);
        auto dimVal =
            rewriter.create<memref::DimOp>(op.getLoc(), src, indexVal);
        dynamicSizes.push_back(dimVal);
      }
    }

    // Emit memref.alloc on target topology with the dynamic sizes
    auto allocOp =
        rewriter.create<memref::AllocOp>(op.getLoc(), targetType, dynamicSizes);

    // Emit memref.copy from src to alloc
    rewriter.create<memref::CopyOp>(op.getLoc(), src, allocOp);

    // Enforce Zero Memory Leaks:
    // We must emit a memref.dealloc at the end of the current scope (block)
    // so that the allocated memory behaves like a C++ RAII object.
    Block *currentBlock = op->getBlock();
    if (!currentBlock->empty() &&
        currentBlock->back().hasTrait<OpTrait::IsTerminator>()) {
      // Temporarily move insertion point to just before the terminator
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(&currentBlock->back());
      // Cast to memory space 0 so memref-to-llvm can use standard `free`
      auto defaultType =
          MemRefType::get(targetType.getShape(), targetType.getElementType());
      auto castOp = rewriter.create<memref::MemorySpaceCastOp>(
          op.getLoc(), defaultType, allocOp);
      rewriter.create<memref::DeallocOp>(op.getLoc(), castOp);
    } else {
      // If there is no terminator yet, just append it to the block
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(currentBlock);
      // Cast to memory space 0 so memref-to-llvm can use standard `free`
      auto defaultType =
          MemRefType::get(targetType.getShape(), targetType.getElementType());
      auto castOp = rewriter.create<memref::MemorySpaceCastOp>(
          op.getLoc(), defaultType, allocOp);
      rewriter.create<memref::DeallocOp>(op.getLoc(), castOp);
    }

    // Replace transfer with the allocated memref
    rewriter.replaceOp(op, allocOp.getResult());
    return success();
  }
};

struct ConvertVxToStandardPass
    : public PassWrapper<ConvertVxToStandardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVxToStandardPass)

  llvm::StringRef getArgument() const override {
    return "convert-vx-to-standard";
  }

  llvm::StringRef getDescription() const override {
    return "Lowers Vx dialect operations to standard MLIR dialects";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<async::AsyncDialect, func::FuncDialect, memref::MemRefDialect,
                arith::ArithDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    getOperation()->emitRemark("Lowering Vx to Standard dialects");
    RewritePatternSet patterns(&getContext());
    patterns.add<SpawnOpLowering, TransferOpLowering>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// ABI type tags for kernel arguments, shared with runtime/npu_dispatch.mm.
// The runtime maps these to libffi types to reconstruct the C calling
// convention. Keep the encoding in sync with vx_abi_ffi_type() there.
//   0=ptr, 1=i1, 2=i8, 3=i16, 4=i32, 5=i64, 6=f32, 7=f64
static int32_t abiTagForType(Type t) {
  if (isa<Float32Type>(t))
    return 6;
  if (isa<Float64Type>(t))
    return 7;
  if (auto it = dyn_cast<IntegerType>(t)) {
    switch (it.getWidth()) {
    case 1:
      return 1;
    case 8:
      return 2;
    case 16:
      return 3;
    case 32:
      return 4;
    default:
      return 5; // i64 (and any wider integer, widened to i64 on the slot)
    }
  }
  return 0; // pointer (memref descriptor) and fallback
}

// Element type code for a memref's element, packed into the high bytes of the
// argument tag. Keep in sync with the VX_DTYPE_* enum in
// include/vx_hardware_runtime.h.
//
// This is what a plugin needs in order to interpret the descriptor it receives:
// without it, a runtime can only assume a layout, which is why the Apple path
// hardcodes float and rank 2 and would silently misread an f16 buffer.
static int32_t elemDtypeCode(Type t) {
  if (isa<Float32Type>(t))
    return 6;
  if (isa<Float64Type>(t))
    return 7;
  if (isa<Float16Type>(t))
    return 8;
  if (isa<BFloat16Type>(t))
    return 9;
  if (auto it = dyn_cast<IntegerType>(t)) {
    switch (it.getWidth()) {
    case 1:
      return 1;
    case 8:
      return 2;
    case 16:
      return 3;
    case 32:
      return 4;
    default:
      return 5;
    }
  }
  return 0; // unknown
}

struct LaunchOpLowering : public OpRewritePattern<vx::LaunchOp> {
  const LLVMTypeConverter &typeConverter;

  LaunchOpLowering(const LLVMTypeConverter &typeConverter, MLIRContext *context)
      : OpRewritePattern<vx::LaunchOp>(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(vx::LaunchOp op,
                                PatternRewriter &rewriter) const override {
    auto &convRewriter = static_cast<ConversionPatternRewriter &>(rewriter);
    Location loc = op.getLoc();
    auto callee = op.getCalleeAttr().getValue();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto llvmI8Type = IntegerType::get(getContext(), 8);
    auto llvmPtrType = LLVM::LLVMPointerType::get(getContext());
    auto llvmI64Type = IntegerType::get(getContext(), 64);
    auto llvmI32Type = IntegerType::get(getContext(), 32);

    // 1. Create the payload: a NUL-separated blob whose first entry is the
    //    kernel name, followed by zero or more `key=value` entries, bounded by
    //    the payload_size argument. Reading it as a `const char *` still yields
    //    the kernel name exactly as before, so a consumer that predates the
    //    extra entries is unaffected -- which is why the name leads rather than
    //    a header. See vx_payload_field() in include/vx_hardware_runtime.h.
    std::string payload = callee.str();
    payload.push_back('\0');
    if (auto kindAttr = op->getAttrOfType<StringAttr>("vx.kernel_kind")) {
      payload += "kind=";
      payload += kindAttr.getValue().str();
      payload.push_back('\0');
    }
    if (auto rolesAttr = op->getAttrOfType<StringAttr>("vx.kernel_roles")) {
      payload += "roles=";
      payload += rolesAttr.getValue().str();
      payload.push_back('\0');
    }
    if (auto outAttr = op->getAttrOfType<StringAttr>("vx.kernel_out_kind")) {
      payload += "outkind=";
      payload += outAttr.getValue().str();
      payload.push_back('\0');
    }

    std::string globalName = (callee + "_str").str();
    LLVM::GlobalOp globalOp = module.lookupSymbol<LLVM::GlobalOp>(globalName);
    if (!globalOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      std::string strWithNull = payload;
      auto arrayTy = LLVM::LLVMArrayType::get(llvmI8Type, strWithNull.size());
      globalOp = rewriter.create<LLVM::GlobalOp>(
          loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal,
          globalName,
          rewriter.getStringAttr(
              StringRef(strWithNull.data(), strWithNull.size())));
    }
    Value globalPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, llvmPtrType, globalName);

    // 2. Allocate the device_args array (one pointer per argument) and a
    //    parallel array of ABI type tags. Every device_args[i] points to the
    //    value of the i-th C-interface argument, so the runtime can rebuild the
    //    platform calling convention via libffi (see docs/lang/abi.md):
    //      - scalar param -> pointer to the scalar value
    //      - memref param -> pointer to the (pointer-to-descriptor)
    int numArgs = op.getNumOperands();
    Value countVal = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type,
        rewriter.getI32IntegerAttr(numArgs > 0 ? numArgs : 1));
    Value argsArray = rewriter.create<LLVM::AllocaOp>(
        loc, llvmPtrType, llvmPtrType, countVal, /*alignment=*/0);
    Value tagsArray = rewriter.create<LLVM::AllocaOp>(
        loc, llvmPtrType, llvmI32Type, countVal, /*alignment=*/0);

    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(1));

    for (auto en : llvm::enumerate(op.getOperands())) {
      Value originalArg = en.value();
      Type originalTy = originalArg.getType();
      Type argTy = typeConverter.convertType(originalTy);

      Value arg = originalArg;
      if (originalTy.isIndex()) {
        arg = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(),
                                                  originalArg);
        argTy = rewriter.getI64Type();
      } else if (argTy && argTy != originalTy) {
        arg =
            rewriter.create<UnrealizedConversionCastOp>(loc, argTy, originalArg)
                .getResult(0);
      }

      Value valuePtr;
      int32_t tag;
      if (isa<MemRefType>(originalTy)) {
        // The C-interface passes a memref as a pointer to its descriptor, so
        // the arg value is that descriptor pointer; device_args[i] points to
        // it.
        Value descAlloc = rewriter.create<LLVM::AllocaOp>(
            loc, llvmPtrType, argTy, one, /*alignment=*/0);
        rewriter.create<LLVM::StoreOp>(loc, arg, descAlloc);
        Value descPtrAlloc = rewriter.create<LLVM::AllocaOp>(
            loc, llvmPtrType, llvmPtrType, one, /*alignment=*/0);
        rewriter.create<LLVM::StoreOp>(loc, descAlloc, descPtrAlloc);
        valuePtr = descPtrAlloc;
        // Kind stays 0 so the calling convention is unchanged; rank and element
        // type ride in the high bytes for plugins that route to vendor kernels.
        auto memrefTy = cast<MemRefType>(originalTy);
        Type elemTy = memrefTy.getElementType();
        int32_t rank = static_cast<int32_t>(memrefTy.getRank());
        int32_t slotBit = 0;

        // A memref whose element is itself a memref is a *slot*: storage
        // holding a descriptor rather than elements. `c = a @ b` allocates its
        // result inside the kernel and stores the descriptor through such a
        // slot, so a plugin standing in for the kernel has to write one there
        // itself.
        //
        // Described verbatim the slot is rank 0 with no scalar element type,
        // which is exactly the encoding of an opaque pointer -- the compiler
        // would be discarding the one fact that makes the argument writable.
        // Describe the pointee instead and set the slot bit to say so. One
        // level only: a slot of slots leaves the element unknown, which is
        // honest.
        if (auto innerTy = dyn_cast<MemRefType>(elemTy)) {
          slotBit = 1 << 24;
          rank = static_cast<int32_t>(innerTy.getRank());
          elemTy = innerTy.getElementType();
        }

        tag = slotBit | (rank << 16) | (elemDtypeCode(elemTy) << 8);
      } else {
        // Scalar: device_args[i] points directly to the value.
        Value scalarAlloc = rewriter.create<LLVM::AllocaOp>(
            loc, llvmPtrType, argTy, one, /*alignment=*/0);
        rewriter.create<LLVM::StoreOp>(loc, arg, scalarAlloc);
        valuePtr = scalarAlloc;
        tag = abiTagForType(argTy);
      }

      Value argSlot =
          rewriter.create<LLVM::GEPOp>(loc, llvmPtrType, llvmPtrType, argsArray,
                                       ArrayRef<LLVM::GEPArg>{en.index()});
      rewriter.create<LLVM::StoreOp>(loc, valuePtr, argSlot);

      Value tagVal = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI32Type, rewriter.getI32IntegerAttr(tag));
      Value tagSlot =
          rewriter.create<LLVM::GEPOp>(loc, llvmPtrType, llvmI32Type, tagsArray,
                                       ArrayRef<LLVM::GEPArg>{en.index()});
      rewriter.create<LLVM::StoreOp>(loc, tagVal, tagSlot);
    }

    // 3. Declare vx_plugin_dispatch_async(name, payload_size, device_args,
    //    arg_tags, num_args)
    StringRef dispatchFuncName = "vx_plugin_dispatch_async";
    LLVM::LLVMFuncOp dispatchFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>(dispatchFuncName);
    if (!dispatchFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto funcType = LLVM::LLVMFunctionType::get(
          llvmI64Type,
          {llvmPtrType, llvmI64Type, llvmPtrType, llvmPtrType, llvmI64Type},
          false);
      dispatchFunc =
          rewriter.create<LLVM::LLVMFuncOp>(loc, dispatchFuncName, funcType);
    }

    // 4. Emit the call. payload_size bounds the blob so a consumer can walk the
    //    entries past the kernel name without running off the end; it was
    //    previously passed as 0.
    Value payloadSize = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Type,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(payload.size())));
    Value numArgsVal = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Type, rewriter.getI64IntegerAttr(numArgs));
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, dispatchFunc,
        ValueRange{globalPtr, payloadSize, argsArray, tagsArray, numArgsVal});

    // 5. Handle return type
    // If the original operation had a result, we must provide it.
    // Since this is a stub for now, we provide a dummy value (e.g., 0) casted
    // to the expected LLVM type.
    if (op.getNumResults() > 0) {
      Type resultTy = typeConverter.convertType(op.getResultTypes()[0]);
      if (resultTy.isInteger(32)) {
        Value zero = rewriter.create<LLVM::ConstantOp>(
            loc, resultTy, rewriter.getI32IntegerAttr(0));
        rewriter.replaceOp(op, zero);
      } else {
        // Fallback to undef for other types (e.g. memref returns)
        Value undef = rewriter.create<LLVM::UndefOp>(loc, resultTy);
        rewriter.replaceOp(op, undef);
      }
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

struct KernelOpLowering : public OpRewritePattern<vx::KernelOp> {
  using OpRewritePattern<vx::KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vx::KernelOp op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = rewriter.create<func::FuncOp>(
        op.getLoc(), op.getSymName(), cast<FunctionType>(op.getFunctionType()));
    funcOp->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());

    // Move the region over
    rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());

    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnOpLowering : public OpRewritePattern<vx::ReturnOp> {
  using OpRewritePattern<vx::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vx::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getOperands());
    return success();
  }
};

struct ConvertVxToLLVMPass
    : public PassWrapper<ConvertVxToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVxToLLVMPass)

  llvm::StringRef getArgument() const override { return "vx-to-llvm"; }

  llvm::StringRef getDescription() const override {
    return "Lowers remaining Vx dialect operations to standard/LLVM dialects";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<vx::VxDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<async::AsyncDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<vx::LaunchOp>();
    target.addIllegalOp<vx::KernelOp>();
    target.addIllegalOp<vx::ReturnOp>();

    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<LaunchOpLowering>(typeConverter, &getContext());
    patterns.add<KernelOpLowering>(&getContext());
    patterns.add<ReturnOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

extern "C" {
void registerVxLoweringPass(MlirContext ctx) {
  // Exposed via registerVxPasses instead.
}

void addVxLoweringPass(MlirPassManager pm) {
  unwrap(pm)->addPass(std::make_unique<ConvertVxToStandardPass>());
}

void addVxToLLVMPass(MlirPassManager pm) {
  unwrap(pm)->addPass(std::make_unique<ConvertVxToLLVMPass>());
}
} // extern "C"

namespace mlir {
namespace vx {
void registerVxPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ConvertVxToStandardPass>();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ConvertVxToLLVMPass>();
  });
}
} // namespace vx
} // namespace mlir
