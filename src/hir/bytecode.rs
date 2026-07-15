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
