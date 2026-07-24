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
    /// Call a function. `type_idx` is the callee's GID (its identity -- the flat codegen resolves the
    /// name and the result type from it); `imm` is the argument count `N`; the arguments are the
    /// `operand1`s of the N `Arg` instructions immediately preceding this `Call`. The result value is
    /// this instruction's register.
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
    /// Index a tensor along its outermost dimension: `operand1` is the base tensor, `operand2` the
    /// (scalar) index. Rank-reducing — `type_idx` is the result type: a rank-1-smaller tensor (a
    /// row/sub-view) or, when the last dimension is indexed, the scalar element. Chained for
    /// multi-dimensional access (`q[i][j]`). `imm` selects value vs. place: `0` = a value (the read
    /// form — codegen loads the scalar element / views the sub-view); `1` = an element **place** on
    /// the left of an assignment, so the following `TensorStore` writes into it instead of loading.
    TensorIndex = 24,
    /// Reduce a rank-1 tensor slice to a scalar. `operand1` is the slice (and `operand2` a second
    /// slice for `dot`, else unused); `imm` is the reduction kind (0 = dot, 1 = sum, 2 = max,
    /// 3 = min); `type_idx` is the scalar element type. Lowers to `vector.reduction` (with an
    /// elementwise `mulf` first for `dot`).
    Reduce = 25,
    /// Allocate storage for a tensor (`Tensor<T>([..])`): `type_idx` is the tensor type, `imm` its
    /// static byte size (element size × the product of the dims) — so the receiving side of a store
    /// has enough room. The result is the tensor buffer.
    TensorAlloc = 26,
    /// Store a value into a tensor place (no result): `operand1` is the destination `TensorIndex`
    /// place, `operand2` the value. The place type selects the store: a row/sub-view place takes a
    /// slice (`o[i] = <slice>`); a scalar-element place (a `TensorIndex` with `imm = 1`) takes a
    /// scalar (`q[i][j] = <scalar>`).
    TensorStore = 27,
    /// Move a tensor to a memory space: `operand1` is the source, `imm` the target space's dispatch
    /// id (`arch::memory_space_dispatch_id`), `type_idx` the result tensor (same element + shape, so
    /// the destination is sized to hold it). Backs `transfer(src, Memory::X)`.
    Transfer = 28,
    /// One argument of the following `Call`: `operand1` is the argument's value register. The N-ary
    /// argument list a fixed two-operand instruction can't hold is spelled as N `Arg`s immediately
    /// preceding the `Call`, in order.
    Arg = 29,
    /// Print a value (`print(x)`, no result): `operand1` is the argument's value register, `type_idx`
    /// its type. A scalar routes to the `print_*` FFI helper; a tensor is cast to an unranked memref
    /// and routed to `printMemref*` — the same runtime helpers the AST path calls.
    Print = 30,
}

impl Opcode {
    /// Recover an opcode from its `#[repr(u32)]` discriminant, e.g. when decoding a serialized flat
    /// HIR body (`crate::metadata`). Returns `None` for an out-of-range value -- a checked conversion,
    /// never a `transmute`, so a corrupt/forward-version stream is rejected rather than UB.
    pub fn from_u32(v: u32) -> Option<Self> {
        use Opcode::*;
        Some(match v {
            0 => Nop,
            1 => Const,
            2 => Load,
            3 => Store,
            4 => Add,
            5 => Sub,
            6 => Mul,
            7 => Div,
            8 => Call,
            9 => Ret,
            10 => Matmul,
            11 => Cmp,
            12 => Cast,
            13 => Neg,
            14 => Not,
            15 => Alloca,
            16 => SlotLoad,
            17 => Br,
            18 => CondBr,
            19 => BlockStart,
            20 => Spawn,
            21 => SpawnEnd,
            22 => FieldLoad,
            23 => FieldStore,
            24 => TensorIndex,
            25 => Reduce,
            26 => TensorAlloc,
            27 => TensorStore,
            28 => Transfer,
            29 => Arg,
            30 => Print,
            _ => return None,
        })
    }
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
