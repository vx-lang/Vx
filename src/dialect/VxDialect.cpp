#include "VxDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Remarks.h"

using namespace mlir;
using namespace mlir::vx;

// Include the auto-generated Dialect CPP.
#include "VxDialect.cpp.inc"
#include "VxOpInterfaces.cpp.inc"

// Include the auto-generated Operation CPP.
#define GET_OP_CLASSES
#include "VxOps.cpp.inc"

#define DEBUG_TYPE "vx-dialect"

namespace {
// When MLIR prints a module as text, it asks each loaded dialect for a short
// name for each type. The LLVM dialect gives none for a struct, so MLIR writes
// the whole struct at each use. A struct with many fields then makes text that
// grows with the number of fields times the number of uses. This interface
// names such a struct `!vx_struct`, and MLIR writes the struct once. A struct
// with few fields keeps its full type, which is easier to read.
struct VxOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  static constexpr size_t kFewestFieldsToName = 17;

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    auto structTy = dyn_cast<LLVM::LLVMStructType>(type);
    if (!structTy || structTy.isIdentified() ||
        structTy.getBody().size() < kFewestFieldsToName)
      return AliasResult::NoAlias;
    os << "vx_struct";
    return AliasResult::FinalAlias;
  }
};
} // namespace

void VxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "VxOps.cpp.inc"
      >();
  addInterfaces<VxOpAsmInterface>();
}

extern "C" {
void registerVxDialect(MlirContext ctx) {
  mlir::MLIRContext *cppCtx = unwrap(ctx);
  cppCtx->getOrLoadDialect<VxDialect>();
}

void mlirEnableOptimizationRemarks(MlirContext ctx) {
  mlir::MLIRContext *cppCtx = unwrap(ctx);
  mlir::remark::RemarkCategories cats{/*all=*/".*"};
  std::unique_ptr<mlir::remark::RemarkEmittingPolicyAll> policy =
      std::make_unique<mlir::remark::RemarkEmittingPolicyAll>();
  (void)mlir::remark::enableOptimizationRemarks(*cppCtx, nullptr,
                                                std::move(policy), cats, true);
}
}
