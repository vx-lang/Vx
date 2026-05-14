# Hardware Topologies & Memory Contexts

Vx's distinguishing feature is the elevation of distributed hardware and heterogeneous memory directly into the type system and operational semantics. We will now fully implement the **Topology and Context Engine**.

## Goal Description

We will extend the compiler frontend (Parser/Sema) and backend (MLIR) to strictly track the current topological execution context and the location of data in memory. This will prevent illegal memory access across disjoint memory spaces (e.g., the CPU trying to read an un-transferred pointer in NPU HBM).

## User Review Required

> [!IMPORTANT]
> **MLIR Representation of `spawn on`**
> Currently, our MLIR backend uses standard dialects (`func`, `arith`, `scf`). To represent `spawn on(Topology) { ... }`, we can emit an `scf.execute_region` block and use MLIR `memref` memory space integers (e.g., `memref<?x?xf32, 1>` where `1` represents NPU HBM) to track memory location.
> Alternatively, we could start introducing a custom `vx` dialect, though that requires a heavier LLVM setup.
> **Proposal:** We map memory spaces to MLIR `memref` space integers (`0` = Host, `1` = NPU HBM, `2` = LocalSRAM), and map `spawn on` to `scf.execute_region` (or just inline it with memory space annotations for now). Do you agree?

## Open Questions

> [!WARNING]
> **Transfer Semantics**
> Should moving data from `HostDRAM` to `NPUHBM` require an explicit function like `transfer(tensor, NPU[0])` or an intrinsic method like `tensor.to_device(NPU[0])`?
> **Proposal:** Implement an intrinsic method `.to_device(Topology)` and `.to_host()` which consumes the tensor and returns a `Pinned<Tensor, Topology>` type.

## Proposed Changes

### 1. Lexer & Parser (`src/parser.rs`)
#### [MODIFY] `src/parser.rs`
- Add a dedicated `parse_spawn_on()` function to correctly parse the `spawn on Topology { Statement }` block.
- Parse topologies fully: `NPU[0]`, `Host`.

### 2. Semantic Analysis (`src/sema.rs`)
#### [MODIFY] `src/sema.rs`
- **Context Tracking:** Add `active_topology: Topology` to the `TypeChecker`.
- **Memory Affinity:** When defining a variable inside a `spawn on` block, its type will automatically map to the active topology's memory space.
- **Access Safety:** When reading an `Identifier`, if its `MemorySpace` does not align with the `active_topology`, emit a strict semantic error: `Semantic Error: Cannot access Host memory from NPU context. Use .to_device()`.

### 3. MLIR Code Generation (`src/codegen.rs`)
#### [MODIFY] `src/codegen.rs`
- **Memory Spaces:** Update the MLIR type stringifier to output `<... x f32, 1>` when compiling `MemorySpace::NPUHBM`.
- **Transfers:** Implement the intrinsic `.to_device()` by emitting an explicit memory copy instruction (e.g., `memref.alloc` on device + `memref.copy` from host) in MLIR.

## Verification Plan
1. **Semantic Checks:** Write tests proving that Vx blocks illegal memory accesses between `Host` and `NPU`.
2. **MLIR Output:** Verify `FileCheck` matches `memref<?x?xf32, 1>` and `memref.copy` instructions when `.to_device()` is used.
