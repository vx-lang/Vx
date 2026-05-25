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

// A spawn operation to execute a region of code on a specific topology
def Vx_SpawnOp : Vx_Op<"spawn"> {
    let summary = "Spawns execution on a specific topology.";
    let arguments = (ins I32Attr:$topology);
    let regions = (region AnyRegion:$body);
}

// A transfer operation to move memory/execution context between topologies
def Vx_TransferOp : Vx_Op<"transfer"> {
    let summary = "Transfers a tensor to a target topology.";
    let arguments = (ins AnyType:$src, I32Attr:$target_topology);
    let results = (outs AnyType:$dst);
}

#endif // VX_DIALECT
