//===- hir.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file defines the High-Level Intermediate Representation (HIR) for Vx.
// It provides a flattened, bytecode-like representation that bridges the gap
// between the verbose Abstract Syntax Tree and the highly structured MLIR dialects,
// facilitating easier analysis and optimization.
//
//===----------------------------------------------------------------------===//
/// High-Level Intermediate Representation (HIR)
/// Flat Array Bytecode replacing the AST.

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
    /// Scalar comparison; the relation (0=Eq,1=Ne,2=Lt,3=Gt,4=Le,5=Ge) is carried in `imm`, result
    /// is a `bool`.
    Cmp = 11,
    /// Scalar conversion (`as`); `type_idx` is the *target* type, `operand1` the source value.
    Cast = 12,
    /// Arithmetic negation (`-x`); `operand1` the source value.
    Neg = 13,
    /// Logical/bitwise not (`!x`); `operand1` the source value.
    Not = 14,
    /// Allocate a stack slot for a named local; the result register is the slot handle, `type_idx`
    /// its element type (matches the AST codegen's `alloca`-backed locals). Used only when a
    /// function has control flow, so values survive across basic blocks via memory.
    Alloca = 15,
    /// Load a value from a slot: `operand1` the slot handle; the result is the loaded value.
    SlotLoad = 16,
    /// Unconditional branch; `imm` is the target basic-block id.
    Br = 17,
    /// Conditional branch on `operand1` (a bool); `imm` packs the two targets as
    /// `then_block | (else_block << 32)`.
    CondBr = 18,
    /// Marks the entry of basic block `imm`. Branch targets are block ids; codegen turns each
    /// `BlockStart` into an MLIR block.
    BlockStart = 19,
    /// Opens a `vx.spawn` region on the topology whose dispatch id is `imm`
    /// (`arch::topology_dispatch_id`). The instructions up to the matching `SpawnEnd` form the
    /// spawned kernel body.
    Spawn = 20,
    /// Closes the `vx.spawn` region opened by the nearest preceding `Spawn` (maps to `vx.yield`).
    SpawnEnd = 21,
    /// Load a struct field: `operand1` is the aggregate's slot handle (from `Alloca`), `imm` the
    /// field's byte offset (from the registry layout), and `type_idx` the field's type. The result
    /// is the loaded field value — the field-addressed counterpart of `SlotLoad`.
    FieldLoad = 22,
    /// Store into a struct field (no result): `operand1` is the aggregate's slot handle, `operand2`
    /// the value to store, and `imm` the field's byte offset. Used to initialize a struct field by
    /// field — the field-addressed counterpart of `Store`.
    FieldStore = 23,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Register(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeIdx(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct HirInstruction {
    /// The specific operation (e.g. Add, Call, Store, Branch)
    pub opcode: Opcode,
    /// Register index for the first operand
    pub operand1: Register,
    /// Register index for the second operand
    pub operand2: Register,
    /// Lightweight 32-bit index pointing into the `LOCAL_TYPE_STREAM`
    /// to fetch the resolved 256-bit GID for this instruction's type.
    pub type_idx: TypeIdx,
    /// Inline scalar immediate value (e.g. f64 or i64)
    pub imm: u64,
}

impl HirInstruction {
    pub fn new(
        opcode: Opcode,
        operand1: Register,
        operand2: Register,
        type_idx: TypeIdx,
        imm: u64,
    ) -> Self {
        Self {
            opcode,
            operand1,
            operand2,
            type_idx,
            imm,
        }
    }
}
