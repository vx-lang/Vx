#ifndef VX_DIALECT
#define VX_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

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
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region AnyRegion:$body);
    let assemblyFormat = "`topology` `(` $topology `)` $body attr-dict (`:` type($results)^)?";
}

// A transfer operation to move memory/execution context between topologies
def Vx_TransferOp : Vx_Op<"transfer"> {
    let summary = "Transfers a tensor to a target topology.";
    let arguments = (ins AnyType:$src, I32Attr:$target_topology);
    let results = (outs AnyType:$dst);
}

// A yield operation to terminate vx.spawn regions
def Vx_YieldOp : Vx_Op<"yield", [Terminator, ReturnLike]> {
    let summary = "Terminates a vx.spawn region.";
    let arguments = (ins Variadic<AnyType>:$results);
    let assemblyFormat = "($results^ `:` type($results))? attr-dict";
}

// A dispatch operation to execute an outlined kernel
def Vx_DispatchOp : Vx_Op<"dispatch"> {
    let summary = "Dispatches an outlined kernel.";
    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$results);
    let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` functional-type($operands, results)";
}

// A return operation to terminate vx.spawn regions from anywhere
def Vx_ReturnOp : Vx_Op<"return", [Terminator, ReturnLike]> {
    let summary = "Returns from a vx.spawn region.";
    let arguments = (ins Variadic<AnyType>:$results);
    let assemblyFormat = "($results^ `:` type($results))? attr-dict";
}

#endif // VX_DIALECT
