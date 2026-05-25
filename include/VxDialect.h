#ifndef VX_DIALECT_H
#define VX_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir-c/IR.h"

// Include the auto-generated Dialect header.
#include "VxDialect.h.inc"

// Define the operations.
#define GET_OP_CLASSES
#include "VxOps.h.inc"

extern "C" {
    // FFI entry point for Rust / Melior to register the dialect
    void registerVxDialect(MlirContext ctx);
}

#endif // VX_DIALECT_H
