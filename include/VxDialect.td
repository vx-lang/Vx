#ifndef VX_DIALECT
#define VX_DIALECT

include "mlir/IR/OpBase.td"

def Vx_Dialect : Dialect {
    let name = "vx";
    let summary = "The Vx compiler MLIR dialect.";
    let description = [{
        The `vx` dialect models heterogeneous execution and memory spaces.
    }];
    let cppNamespace = "::mlir::vx";
}

// Base class for Vx operations.
class Vx_Op<string mnemonic, list<Trait> traits = []> :
    Op<Vx_Dialect, mnemonic, traits>;

// A dummy operation just to verify things are working.
def Vx_SpawnOp : Vx_Op<"spawn"> {
    let summary = "A placeholder operation for spawning topologies.";
}

#endif // VX_DIALECT
