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
#[repr(C)]
pub struct HirInstruction {
    /// The specific operation (e.g. Add, Call, Store, Branch)
    pub opcode: u32,
    /// Register index for the first operand
    pub operand1: u32,
    /// Register index for the second operand
    pub operand2: u32,
    /// Lightweight 32-bit index pointing into the `LOCAL_TYPE_STREAM`
    /// to fetch the resolved 256-bit GID for this instruction's type.
    pub type_idx: u32,
}

impl HirInstruction {
    pub fn new(opcode: u32, operand1: u32, operand2: u32, type_idx: u32) -> Self {
        Self {
            opcode,
            operand1,
            operand2,
            type_idx,
        }
    }
}

pub const OP_NOP: u32 = 0;
pub const OP_CONST: u32 = 1;
pub const OP_LOAD: u32 = 2;
pub const OP_STORE: u32 = 3;
pub const OP_ADD: u32 = 4;
pub const OP_SUB: u32 = 5;
pub const OP_MUL: u32 = 6;
pub const OP_DIV: u32 = 7;
pub const OP_MATMUL: u32 = 10;
pub const OP_CALL: u32 = 8;
pub const OP_RET: u32 = 9;
