#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "VxDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

      if (!spawnBody.empty()) {
        Block &spawnBlock = spawnBody.front();
        if (!spawnBlock.empty() && isa<vx::YieldOp>(spawnBlock.back())) {
          rewriter.eraseOp(&spawnBlock.back());
        }
        // Move operations from spawnBlock to asyncBlock
        auto &asyncOps = asyncBlock->getOperations();
        auto &spawnOps = spawnBlock.getOperations();
        // Move before the yield (which is at the end of asyncBlock)
        asyncOps.splice(std::prev(asyncOps.end()), spawnOps, spawnOps.begin(), spawnOps.end());
      }

      rewriter.eraseOp(op);
      return success();
    }

    // For NPU (100) or AccCore (200), we lower to a gpu.launch.
    Region &spawnBody = op.getBody();
    if (spawnBody.empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    auto constOne = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    auto launchOp = rewriter.create<gpu::LaunchOp>(
        op.getLoc(),
        constOne, constOne, constOne,
        constOne, constOne, constOne);

    Region &launchRegion = launchOp.getBody();
    Block *launchBlock;
    if (launchRegion.empty()) {
      launchBlock = rewriter.createBlock(&launchRegion);
      // gpu.launch takes 12 index arguments (thread/block indices and sizes)
      for (int i = 0; i < 12; ++i) {
        launchBlock->addArgument(rewriter.getIndexType(), op.getLoc());
      }
      rewriter.create<gpu::TerminatorOp>(op.getLoc());
    } else {
      launchBlock = &launchRegion.front();
      if (launchBlock->empty() || !isa<gpu::TerminatorOp>(launchBlock->back())) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(launchBlock);
        rewriter.create<gpu::TerminatorOp>(op.getLoc());
      }
    }

    // Splice operations from spawnBlock to launchBlock
    Block &spawnBlock = spawnBody.front();
    if (!spawnBlock.empty() && isa<vx::YieldOp>(spawnBlock.back())) {
      rewriter.eraseOp(&spawnBlock.back());
    }
    
    auto &launchOps = launchBlock->getOperations();
    auto &spawnOps = spawnBlock.getOperations();
    launchOps.splice(std::prev(launchOps.end()), spawnOps, spawnOps.begin(), spawnOps.end());

    rewriter.eraseOp(op);
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
    llvm::errs() << "[VxLowering] TransferOp lowering from " << srcType << " to " << targetType << "\n";

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
      auto defaultType = MemRefType::get(targetType.getShape(), targetType.getElementType());
      auto castOp = rewriter.create<memref::MemorySpaceCastOp>(op.getLoc(), defaultType, allocOp);
      rewriter.create<memref::DeallocOp>(op.getLoc(), castOp);
    } else {
      // If there is no terminator yet, just append it to the block
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(currentBlock);
      // Cast to memory space 0 so memref-to-llvm can use standard `free`
      auto defaultType = MemRefType::get(targetType.getShape(), targetType.getElementType());
      auto castOp = rewriter.create<memref::MemorySpaceCastOp>(op.getLoc(), defaultType, allocOp);
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
    registry.insert<async::AsyncDialect, func::FuncDialect,
                    memref::MemRefDialect, arith::ArithDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SpawnOpLowering, TransferOpLowering>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

extern "C" {
void registerVxLoweringPass(MlirContext ctx) {
  // Technically, passes are registered globally in MLIR or added to a
  // PassManager. We will expose a C API to add this pass to an existing
  // MlirPassManager instead.
}

void addVxLoweringPass(MlirPassManager pm) {
  unwrap(pm)->addPass(std::make_unique<ConvertVxToStandardPass>());
}
} // extern "C"

namespace mlir {
namespace vx {
void registerVxPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<ConvertVxToStandardPass>();
  });
}
} // namespace vx
} // namespace mlir
