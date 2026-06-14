# High-Level Intermediate Representation (HIR) Bytecode Packing

## Overview

The Vx compiler architecture leverages a flattened, zero-allocation High-Level Intermediate Representation (HIR) bytecode stream as a bridge between the deeply recursive Abstract Syntax Tree (AST) and the highly structured MLIR generation layer.

This data-oriented design represents the program as a contiguous array of `HirInstruction` structures.

## Instruction Layout

Originally, the `HirInstruction` was a hyper-compact 16-byte structure consisting of 4 `u32` fields:

- `opcode: Opcode`
- `operand1: u32`
- `operand2: u32`
- `type_idx: u32`

While this 16-byte layout provided an extremely cache-friendly footprint, it lacked the ability to store immediate scalar constants (e.g. `f32`, `f64`, `i32`, `i64`) inline. Without an inline payload, the compiler would be forced to maintain a separate side-band constant pool, indirection into which would cause cache misses and pipeline stalls.

## The 24-Byte Expansion

To achieve maximum performance for mathematical evaluation, we expanded the `HirInstruction` struct to **24 bytes** by appending a single `u64` payload:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct HirInstruction {
    pub opcode: Opcode,
    pub operand1: u32,
    pub operand2: u32,
    pub type_idx: u32,
    pub imm: u64,
}
```

### Tradeoffs & Benefits

1. **Increased Cache Pressure**: Expanding the struct by 50% reduces the number of instructions that fit into a CPU cache line (64 bytes). Previously, 4 instructions fit exactly into a single cache line. Now, only 2 instructions fit cleanly (with 16 bytes spilling into the next).
1. **Elimination of the Constant Pool**: The `imm: u64` payload allows us to embed arbitrary 64-bit scalars inline. When a `Const` instruction is encountered, the generator can simply read `f64::from_bits(instruction.imm)` directly out of the flat instruction stream.
1. **Data Locality**: This entirely eliminates the need for side-band lookups for scalar literals, drastically reducing pointer chasing and improving memory locality during the linear scan of the bytecode stream.

The decision to sacrifice 8 bytes of padding for pure, self-contained bytecode stream generation aligns perfectly with the Vx V3 design philosophy: flat, array-based, contiguous execution.
