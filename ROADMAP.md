# Akar Compiler Roadmap

This document serves as the master tracking sheet for the Akar compiler. It maps our implemented features against the 7 foundational Core Philosophies defined in the project.

## 1. Heterogeneous Compute & Address Spaces
Treating topologies, execution spaces, and disparate memory domains as first-class primitives.
- `[x]` Basic `spawn on(Topology::...)` syntax
- `[x]` Explicit memory space type enforcement (`Memory::NPU_HBM`, `Host_DRAM`)
- `[x]` Data movement explicit primitives (`transfer()`)
- `[ ]` Cross-topology synchronization primitives (Barriers, Mutexes, Atomics)
- `[ ]` Compile-time hardware capability queries and limits

## 2. Ease of Verified Computation
Allowing programmers to safely and mathematically verify computation boundaries.
- `[x]` `Verified<T>` primitive wrapper in the type system
- `[ ]` Hardware-aware Effect tracking in Semantic Analysis
- `[ ]` Formal Pre-condition / Post-condition verification contracts
- `[ ]` Dependent types (e.g., verifying matrix dimensions match at compile time)

## 3. Performance
Bypassing runtime overhead and relying entirely on heavy ahead-of-time (AOT) optimizations.
- `[x]` Direct lowering to MLIR and LLVM IR for execution
- `[x]` High-performance heterogeneous JIT compilation via `lli` integration
- `[ ]` Core MLIR optimization passes (Loop Unrolling, LICM)
- `[ ]` Auto-Vectorization passes
- `[ ]` Dead Code Elimination (DCE)

## 4. Deterministic Memory Control
Total programmatic control over memory lifetimes and representations; absolute avoidance of garbage collection.
- `[x]` No mandatory Garbage Collector implemented
- `[x]` Raw Pointers (`*mut T`, `*const T`) and pointer arithmetic
- `[x]` Custom User Structs and Unions
- `[ ]` Explicit layout controls (`#pragma pack`, explicit alignments)
- `[x]` Memory allocation lifecycles (`malloc`/`free` equivalents or borrowing semantics)

## 5. Predictable Execution (Zero-Cost Abstractions)
Code must run as efficiently as hand-written assembly; abstractions must vanish during compilation.
- `[ ]` Monomorphized Generics
- `[ ]` Traits / Interfaces utilizing purely static dispatch
- `[ ]` Zero-cost Iterators mapped to loops

## 6. Direct Hardware Access
Unimpeded access to the lowest levels of the underlying execution silicon.
- `[ ]` Inline Assembly Blocks (`asm! { ... }`)
- `[ ]` Volatile memory operations for Memory-Mapped I/O (MMIO)
- `[ ]` Intrinsics for CPU registers and SIMD instructions
- `[ ]` Hardware trap and Interrupt Handler integration

## 7. Strong System Interoperability
Clean interoperability with the pre-existing low-level world (Operating Systems, Kernels, POSIX).
- `[ ]` Foreign Function Interface (FFI) for importing `extern "C"` functions
- `[ ]` Akar function exporting formatted to the standard C ABI
- `[ ]` Static and Dynamic linking capabilities against host OS binaries
