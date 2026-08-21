#ifndef VX_DIALECT
#define VX_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

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

// A launch operation to launch a hardware kernel
def Vx_LaunchOp : Vx_Op<"launch"> {
    let summary = "Launches an outlined hardware kernel.";
    // `topology` is the device this launch targets, the same id `vx.kernel`
    // carries. The launch is what becomes the call into a plugin, so a plugin
    // could not tell one device from another without it -- which is the whole
    // of disaggregated execution (#331).
    let arguments = (ins FlatSymbolRefAttr:$callee, I32Attr:$topology,
                         Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$results);
    let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` functional-type($operands, results)";
}

// A native kernel function operation. Like a function, its body is an
// automatic-allocation scope: `memref.alloca`/`llvm.alloca` for kernel-local
// values (cloned in from the outlined `vx.spawn` region) are scoped to the
// kernel invocation, matching the `func.func` it lowers to.
def Vx_KernelOp : Vx_Op<"kernel", [IsolatedFromAbove, Symbol, AutomaticAllocationScope]> {
    let summary = "A native hardware kernel function.";
    let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$function_type, I32Attr:$topology);
    let regions = (region AnyRegion:$body);
    let assemblyFormat = "$sym_name `(` $topology `)` attr-dict `:` $function_type $body";
}

// Thread ID query
def Vx_ThreadIdOp : Vx_Op<"thread_id", [Pure]> {
    let summary = "Returns the thread ID for a given dimension.";
    let arguments = (ins StrAttr:$dimension);
    let results = (outs Index:$result);
    let assemblyFormat = "$dimension attr-dict `:` type($result)";
}


// Block ID query
def Vx_BlockIdOp : Vx_Op<"block_id", [Pure]> {
    let summary = "Returns the block ID for a given dimension.";
    let arguments = (ins StrAttr:$dimension);
    let results = (outs Index:$result);
    let assemblyFormat = "$dimension attr-dict `:` type($result)";
}

// A return operation to terminate vx.spawn regions from anywhere
def Vx_ReturnOp : Vx_Op<"return", [Terminator, ReturnLike]> {
    let summary = "Returns from a vx.spawn region.";
    let arguments = (ins Variadic<AnyType>:$results);
    let assemblyFormat = "($results^ `:` type($results))? attr-dict";
}

// Block-level synchronization inside a spawn region (Vx#379). The device
// clone rewrites it to gpu.barrier; the host lowering erases it, because a
// serial loop already provides the ordering a barrier asks for. Deliberately
// NOT Pure: a pure no-result op would be folded away before the device clone
// ever saw it.
def Vx_BarrierOp : Vx_Op<"barrier"> {
    let summary = "All threads of a block reach this point before any proceeds.";
    let assemblyFormat = "attr-dict";
}

#endif // VX_DIALECT
