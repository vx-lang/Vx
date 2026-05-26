#include "VxDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Builders.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"

using namespace mlir;
using namespace mlir::vx;

namespace {

// Lower `vx.spawn` to `async.execute` for CPU topologies, or a func call for NPU.
struct SpawnOpLowering : public OpRewritePattern<SpawnOp> {
  using OpRewritePattern<SpawnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SpawnOp op, PatternRewriter &rewriter) const override {
    int32_t topology = op.getTopology();
    
    // Topology 0 == Host CPU. Lower to async.execute
    if (topology == 0) {
      // Create async.execute
      auto asyncExecuteOp = rewriter.create<async::ExecuteOp>(
          op.getLoc(),
          /*resultTypes=*/TypeRange{},
          /*dependencies=*/ValueRange{},
          /*operands=*/ValueRange{});

      // Move the body of vx.spawn into async.execute
      Region &spawnBody = op.getBody();
      Region &asyncBody = asyncExecuteOp.getRegion();
      
      // async.execute expects a block, but we can splice our blocks in.
      // However, async.execute usually expects a specific terminator (async.yield).
      rewriter.inlineRegionBefore(spawnBody, asyncBody, asyncBody.end());

      // If the block is empty or missing a terminator, we must ensure it has async.yield
      if (!asyncBody.empty()) {
        Block &lastBlock = asyncBody.back();
        if (lastBlock.empty() || !lastBlock.back().hasTrait<OpTrait::IsTerminator>()) {
          rewriter.setInsertionPointToEnd(&lastBlock);
          rewriter.create<async::YieldOp>(op.getLoc(), ValueRange{});
        }
      }

      rewriter.eraseOp(op);
      return success();
    }
    
    // For NPU (100) or AccCore (200), we would lower to a hardware dispatch call.
    // For now, we will just inline it sequentially to avoid crashes, as a placeholder
    // for true hardware kernel dispatch.
    Region &spawnBody = op.getBody();
    if (!spawnBody.empty()) {
        Block &bodyBlock = spawnBody.front();
        // Remove terminator if we had one and it's not a standard func return
        if (!bodyBlock.empty() && bodyBlock.back().hasTrait<OpTrait::IsTerminator>()) {
            rewriter.eraseOp(&bodyBlock.back());
        }
        rewriter.inlineBlockBefore(&bodyBlock, op.getOperation());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct TransferOpLowering : public OpRewritePattern<TransferOp> {
  using OpRewritePattern<TransferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferOp op, PatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    auto srcType = dyn_cast<MemRefType>(src.getType());
    
    // If it's not a MemRef (e.g., primitive i32/f32), we fail compilation.
    // The user explicitly mandated that implicit conversions/pass-throughs
    // are disabled to enforce strict data layout transitions.
    if (!srcType) {
      op.emitError("vx.transfer currently only supports MemRef types. Attempted to transfer a scalar/primitive.");
      return failure();
    }

    int32_t topology = op.getTargetTopology();
    auto memorySpace = IntegerAttr::get(IntegerType::get(getContext(), 32), topology);

    auto targetType = MemRefType::get(srcType.getShape(), srcType.getElementType(),
                                      srcType.getLayout(), memorySpace);

    // Emit memref.alloc on target topology
    auto allocOp = rewriter.create<memref::AllocOp>(op.getLoc(), targetType);

    // Emit memref.copy from src to alloc
    rewriter.create<memref::CopyOp>(op.getLoc(), src, allocOp);

    // Enforce Zero Memory Leaks:
    // We must emit a memref.dealloc at the end of the current scope (block)
    // so that the allocated memory behaves like a C++ RAII object.
    Block *currentBlock = op->getBlock();
    if (!currentBlock->empty() && currentBlock->back().hasTrait<OpTrait::IsTerminator>()) {
      // Temporarily move insertion point to just before the terminator
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(&currentBlock->back());
      rewriter.create<memref::DeallocOp>(op.getLoc(), allocOp);
    } else {
      // If there is no terminator yet, just append it to the block
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(currentBlock);
      rewriter.create<memref::DeallocOp>(op.getLoc(), allocOp);
    }

    // Replace transfer with the allocated memref
    rewriter.replaceOp(op, allocOp.getResult());
    return success();
  }
};

struct ConvertVxToStandardPass : public PassWrapper<ConvertVxToStandardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVxToStandardPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SpawnOpLowering, TransferOpLowering>(&getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<async::AsyncDialect, func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalOp<SpawnOp, TransferOp>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

extern "C" {
    void registerVxLoweringPass(MlirContext ctx) {
        // Technically, passes are registered globally in MLIR or added to a PassManager.
        // We will expose a C API to add this pass to an existing MlirPassManager instead.
    }
    
    void addVxLoweringPass(MlirPassManager pm) {
        unwrap(pm)->addPass(std::make_unique<ConvertVxToStandardPass>());
    }
}
