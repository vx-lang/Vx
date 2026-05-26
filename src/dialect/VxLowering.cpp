#include "VxDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

struct ConvertVxToStandardPass : public PassWrapper<ConvertVxToStandardPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVxToStandardPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SpawnOpLowering>(&getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<async::AsyncDialect, func::FuncDialect>();
    target.addIllegalOp<SpawnOp>();
    // We allow transfer op to remain for now or mark it illegal if we implement lowering

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
