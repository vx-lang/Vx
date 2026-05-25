#include "VxDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/CAPI/IR.h"

using namespace mlir;
using namespace mlir::vx;

// Include the auto-generated Dialect CPP.
#include "VxDialect.cpp.inc"

// Include the auto-generated Operation CPP.
#define GET_OP_CLASSES
#include "VxOps.cpp.inc"

void VxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "VxOps.cpp.inc"
      >();
}

extern "C" {
    void registerVxDialect(MlirContext ctx) {
        mlir::MLIRContext* cppCtx = unwrap(ctx);
        cppCtx->getOrLoadDialect<VxDialect>();
    }
}
