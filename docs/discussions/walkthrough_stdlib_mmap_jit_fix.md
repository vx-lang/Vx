# HIR Refactoring Walkthrough

I have implemented the improvements detailed in the `docs/discussions/Vx_src_hir.md` proposal.

## Changes Made

### Strongly-Typed Opcodes

The previous loose `u32` constant defines (`pub const OP_ADD: u32 = 4`, etc.) have been removed. We now have a robust `#[repr(u32)] pub enum Opcode` which guarantees exhaustiveness checking across the compiler when processing opcodes during lowering.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Opcode {
    Nop = 0,
    Const = 1,
    Load = 2,
    Store = 3,
    Add = 4,
    Sub = 5,
    Mul = 6,
    Div = 7,
    Call = 8,
    Ret = 9,
    Matmul = 10,
}
```

### Immediate Scalar Payloads

I expanded `HirInstruction` to 24 bytes by introducing a `pub imm: u64` payload.

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

This avoids side-band constant pools for scalars (e.g., users can simply `f64::from_bits(imm)` directly out of the flat instruction stream), aligning beautifully with the data-oriented design!

## Validation Results

Since the legacy HIR stream generation was recently decoupled from the `TypeChecker`, usages of `HirInstruction::new` and old `OP_` constants inside the frontend were already eliminated! The refactor integrated instantly.
`cargo test` confirmed that the structural size change broke zero tests. All pre-commit formatting and testing passed flawlessly.
