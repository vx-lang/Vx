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
// It reads the source and allocates the copy. The general write is the copy
// into the target's memory, and keeps an unused transfer from being deleted.
// The copy is freed by a `vx.free` that convert-vx-to-standard places after
// its last use, so the buffer-deallocation analysis must not free it too.
def Vx_TransferOp : Vx_Op<"transfer", [MemoryEffects<[MemWrite]>]> {
    let summary = "Transfers a tensor to a target topology.";
    let arguments = (ins Arg<AnyType, "the value to copy", [MemRead]>:$src,
                         I32Attr:$target_topology);
    let results = (outs Res<AnyType, "the copy on the target topology",
                            [MemAlloc]>:$dst);
}

// Frees the buffer a transfer allocated, after its last use. On the host it
// becomes `memref.dealloc`; on a device it becomes `vx_plugin_free`, which must
// run after the work already given to that device.
def Vx_FreeOp : Vx_Op<"free"> {
    let summary = "Frees the buffer a vx.transfer allocated.";
    let arguments = (ins Arg<AnyType, "the buffer to free", [MemFree]>:$buffer,
                         I32Attr:$topology);
    let assemblyFormat = "$buffer attr-dict `:` type($buffer)";
}

// A yield operation to terminate vx.spawn regions
def Vx_YieldOp : Vx_Op<"yield", [Pure, Terminator, ReturnLike]> {
    let summary = "Terminates a vx.spawn region.";
    let arguments = (ins Variadic<AnyType>:$results);
    let assemblyFormat = "($results^ `:` type($results))? attr-dict";
}

// A dispatch operation to execute an outlined kernel
// The kernel may read and write any buffer it is given.
def Vx_DispatchOp : Vx_Op<"dispatch", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Dispatches an outlined kernel.";
    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$results);
    let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` functional-type($operands, results)";
}

// A launch operation to launch a hardware kernel
// The kernel may read and write any buffer it is given.
def Vx_LaunchOp : Vx_Op<"launch", [MemoryEffects<[MemRead, MemWrite]>]> {
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
def Vx_ReturnOp : Vx_Op<"return", [Pure, Terminator, ReturnLike]> {
    let summary = "Returns from a vx.spawn region.";
    let arguments = (ins Variadic<AnyType>:$results);
    let assemblyFormat = "($results^ `:` type($results))? attr-dict";
}

// Block-level synchronization inside a spawn region (Vx#379). The device
// clone rewrites it to gpu.barrier; the host lowering erases it, because a
// serial loop already provides the ordering a barrier asks for. Deliberately
// NOT Pure: a pure no-result op would be folded away before the device clone
// ever saw it. It orders memory, so it reads and writes it.
def Vx_BarrierOp : Vx_Op<"barrier", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "All threads of a block reach this point before any proceeds.";
    let assemblyFormat = "attr-dict";
}

#endif // VX_DIALECT
