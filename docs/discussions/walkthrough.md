# MLIR Port: `spawn on` and `transfer()`

This document summarizes the changes made to port the `spawn on` blocks and `transfer()` calls from our legacy AST strings-based backend into our new Melior-based MLIR backend.

## What was changed

- Defined new MLIR operations in our custom TableGen dialect (`include/VxDialect.td`).
  - Added `Vx_SpawnOp` (`vx.spawn`) which takes a single `I32Attr` for the target topology and has one attached region for its body.
  - Added `Vx_TransferOp` (`vx.transfer`) which takes an `AnyType` value, an `I32Attr` for the target topology, and returns an `AnyType` result.
- Included the new MLIR Bytecode interfaces (`BytecodeOpInterface.h`) in our C++ wrapper to enable proper property serialization for the MLIR ops.
- Implemented a topology mapping function `topology_to_i32()` inside `src/melior_codegen.rs` to convert logical topology locations (like `NPU` and `AccCore`) into simple integers that cross the C++ boundary cleanly.
  - `Host` -> `0`
  - `NPU(x)` -> `100 + x`
  - `AccCore(x)` -> `200 + x`
- Hooked up `LowerToMelior` for both `SpawnOnStmt` and `TransferExpr`.
  - `SpawnOnStmt` lowers recursively on its interior statements, adding them to an MLIR region, and attaching it to the `vx.spawn` op.
  - `TransferExpr` parses its interior expression and outputs `vx.transfer`- Handled proper mapping of nested AST constructs to `Location` boundaries.

### Formalized Topological Memory Graph (v3.0)

- **`HardwareGraph` Restructure**: Transformed the static boolean match logic in `src/arch.rs` into a rigorous stateful `HardwareGraph`.
- **Graph Edges**: Explicitly added `transfer_edges` for Memory-to-Memory transfers and `visibility_edges` for Topology-to-Memory visibility.
- **BFS Multi-Hop Pathfinding**: Converted `can_transfer` into a dynamic Breadth-First Search (BFS) pathfinding algorithm, enabling automatic multi-hop transfer validation (e.g. `LocalSRAM <-> HostDRAM` routing via `NPUHBM`).
- **Integration**: Plumbed the stateful `HardwareGraph` through the `TypeChecker` in `src/sema.rs`, successfully catching all cross-topology boundary violations exactly as before, but with data-driven architecture rules.
- **Test Expectations**: Fixed tests and assertions so that they properly initialize and query the `HardwareGraph` instance.

## Testing & Validation

- Regenerated `VxOps.h.inc` and `VxOps.cpp.inc` via `mlir-tblgen` by running `cargo build`.
- Automatically updated our `middle_end/pass` checks using the test script: `cargo run update_mlir_test_checks -- tests/middle_end/pass/*.vx tests/middle_end/pass/*.mlr`.

### Lexical Borrow Checker (v3.0)

- Implemented **Strict Aliasing** rules embedded directly into the Semantic Analyzer (`src/sema.rs`).
- `TypeChecker` now maintains an `active_borrows` hash map that tracks Lexical Lifetimes of references.
- Verified Shared XOR Mutable properties:
  - Added compile-time errors to reject creating multiple mutable borrows of the same memory (`borrow_mut_twice.vx`).
  - Added compile-time errors to reject taking an immutable borrow when a mutable borrow exists (`borrow_mut_imm.vx`).
- Cleaned up reference lifetimes correctly on block scope exit, fully passing `borrow_lexical.vx`.
- Ran and passed `cargo test test_middle_end`, validating that the new operations generate valid MLIR text!

Below is an example of what our newly ported IR looks like when `vxc --emit-mlir` compiles code with memory transfers and execution placement:

```mlir
// For: `let t_device = t_host.to_device();`
%0 = "vx.transfer"(%arg0) <{target_topology = 100 : i32}> : (memref<?x?xf32>) -> memref<?x?xf32>

// For: `spawn on (Topology::NPU[0]) { ... }`
"vx.spawn"() <{topology = 100 : i32}> ({
^bb0:
}) : () -> ()
```

## What's Next

- Expanding `Vx_SpawnOp` to optionally accept the dynamic device index as an SSA value instead of embedding it into the static attribute.
- Moving forward with our v3 release roadmap items, such as the Topological Memory Algebra layout graph verification.
