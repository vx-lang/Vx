#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "VxDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::vx;

namespace {

// Lower `vx.spawn` to `async.execute` for CPU topologies, or a func call for
// NPU.
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

    // Determine result types from the yield operation
    SmallVector<Type> resultTypes;
    SmallVector<Value> yieldedValues;
    Block &spawnBlock = spawnBody.front();
    if (!spawnBlock.empty() && isa<vx::YieldOp>(spawnBlock.back())) {
      auto yieldOp = cast<vx::YieldOp>(spawnBlock.back());
      for (auto val : yieldOp.getOperands()) {
        resultTypes.push_back(val.getType());
      }
    }

    auto funcType = rewriter.getFunctionType(argTypes, resultTypes);
    static int kernelIdx = 0;
    std::string funcName = "vx_npu_kernel_" + std::to_string(kernelIdx++);
    auto funcOp =
        rewriter.create<func::FuncOp>(op.getLoc(), funcName, funcType);
    funcOp->setAttr("vx.kernel", rewriter.getUnitAttr());
    funcOp->setAttr("vx.topology", rewriter.getI32IntegerAttr(topology));

    // Copy the region
    Block *funcBlock = rewriter.createBlock(
        &funcOp.getBody(), funcOp.getBody().end(), argTypes,
        SmallVector<Location>(argTypes.size(), op.getLoc()));

    IRMapping mapping;
    for (auto [cap, arg] : llvm::zip(captures, funcBlock->getArguments())) {
      mapping.map(cap, arg);
    }

    // Clone operations
    for (auto &innerOp : spawnBlock.without_terminator()) {
      rewriter.clone(innerOp, mapping);
    }

    // Handle yield by returning the mapped values
    if (!spawnBlock.empty() && isa<vx::YieldOp>(spawnBlock.back())) {
      auto yieldOp = cast<vx::YieldOp>(spawnBlock.back());
      SmallVector<Value> returnOperands;
      for (auto val : yieldOp.getOperands()) {
        returnOperands.push_back(mapping.lookupOrDefault(val));
      }
      rewriter.create<func::ReturnOp>(op.getLoc(), returnOperands);
    } else {
      rewriter.create<func::ReturnOp>(op.getLoc());
    }

    // Restore insertion point to replace vx.spawn with vx.dispatch
    rewriter.restoreInsertionPoint(ip);

    SmallVector<Value> dispatchOperands(captures.begin(), captures.end());
    auto dispatchOp = rewriter.create<vx::DispatchOp>(
        op.getLoc(), resultTypes,
        SymbolRefAttr::get(rewriter.getContext(), funcName), dispatchOperands);

    rewriter.replaceOp(op, dispatchOp.getResults());
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
    llvm::errs() << "[VxLowering] TransferOp lowering from " << srcType
                 << " to " << targetType << "\n";

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
    RewritePatternSet patterns(&getContext());
    patterns.add<SpawnOpLowering, TransferOpLowering>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct DispatchOpLowering : public ConvertOpToLLVMPattern<vx::DispatchOp> {
  using ConvertOpToLLVMPattern<vx::DispatchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vx::DispatchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto callee = op.getCalleeAttr().getValue();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto llvmI8Type = IntegerType::get(getContext(), 8);
    auto llvmPtrType = LLVM::LLVMPointerType::get(getContext());
    auto llvmI64Type = IntegerType::get(getContext(), 64);
    auto llvmI32Type = IntegerType::get(getContext(), 32);

    // 1. Create the global string for the kernel name
    std::string globalName = (callee + "_str").str();
    LLVM::GlobalOp globalOp = module.lookupSymbol<LLVM::GlobalOp>(globalName);
    if (!globalOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      std::string strWithNull = callee.str() + '\0';
      auto arrayTy = LLVM::LLVMArrayType::get(llvmI8Type, strWithNull.size());
      globalOp = rewriter.create<LLVM::GlobalOp>(
          loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal,
          globalName,
          rewriter.getStringAttr(
              StringRef(strWithNull.data(), strWithNull.size())));
    }
    Value globalPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, llvmPtrType, globalName);

    // 2. Allocate the array of pointers for device_args
    auto operands =
        adaptor.getOperands(); // These are already converted to LLVM types!
    Value numArgs = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(operands.size()));
    Value argsArray = rewriter.create<LLVM::AllocaOp>(
        loc, llvmPtrType, llvmPtrType, numArgs, /*alignment=*/0);

    for (auto en : llvm::enumerate(operands)) {
      Value arg = en.value();
      Type argTy = arg.getType();

      // Allocate space for this argument to get a pointer to it
      Value one = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI32Type, rewriter.getI32IntegerAttr(1));
      Value argAlloc = rewriter.create<LLVM::AllocaOp>(loc, llvmPtrType, argTy,
                                                       one, /*alignment=*/0);
      rewriter.create<LLVM::StoreOp>(loc, arg, argAlloc);

      // Get pointer to argsArray[i]
      Value index = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI32Type, rewriter.getI32IntegerAttr(en.index()));
      Value slotPtr =
          rewriter.create<LLVM::GEPOp>(loc, llvmPtrType, llvmPtrType, argsArray,
                                       ArrayRef<LLVM::GEPArg>{en.index()});

      // Store the argument pointer into argsArray[i]
      rewriter.create<LLVM::StoreOp>(loc, argAlloc, slotPtr);
    }

    // 3. Declare vx_plugin_dispatch_async
    StringRef dispatchFuncName = "vx_plugin_dispatch_async";
    LLVM::LLVMFuncOp dispatchFunc =
        module.lookupSymbol<LLVM::LLVMFuncOp>(dispatchFuncName);
    if (!dispatchFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto funcType = LLVM::LLVMFunctionType::get(
          llvmI64Type, {llvmPtrType, llvmI64Type, llvmPtrType}, false);
      dispatchFunc =
          rewriter.create<LLVM::LLVMFuncOp>(loc, dispatchFuncName, funcType);
    }

    // 4. Emit the call
    Value payloadSize = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Type, rewriter.getI64IntegerAttr(0));
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, dispatchFunc, ValueRange{globalPtr, payloadSize, argsArray});

    // 5. Handle return type
    // If the original operation had a result, we must provide it.
    // Since this is a stub for now, we provide a dummy value (e.g., 0) casted
    // to the expected LLVM type.
    if (op.getNumResults() > 0) {
      Type resultTy = getTypeConverter()->convertType(op.getResultTypes()[0]);
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

    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<DispatchOpLowering>(typeConverter);

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
