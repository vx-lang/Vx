#include "VxDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Remarks.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::vx;

// Include the auto-generated Dialect CPP.
#include "VxDialect.cpp.inc"

// Include the auto-generated Operation CPP.
#define GET_OP_CLASSES
#include "VxOps.cpp.inc"

#define DEBUG_TYPE "vx-dialect"

void VxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "VxOps.cpp.inc"
      >();
}

extern "C" {
void registerVxDialect(MlirContext ctx) {
  mlir::MLIRContext *cppCtx = unwrap(ctx);
  cppCtx->getOrLoadDialect<VxDialect>();
}

void parseCommandLineOptions(int argc, const char *const *argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
}

void mlirEnableOptimizationRemarksForTesting(MlirContext ctx) {
  mlir::MLIRContext *cppCtx = unwrap(ctx);
  mlir::remark::RemarkCategories cats{/*all=*/".*"};
  std::unique_ptr<mlir::remark::RemarkEmittingPolicyAll> policy =
      std::make_unique<mlir::remark::RemarkEmittingPolicyAll>();
  (void)mlir::remark::enableOptimizationRemarks(*cppCtx, nullptr,
                                                std::move(policy), cats, true);
}
}
