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
// A top-level module with no imports of its own. registry, session, and metadata all store and
// replay these instructions, while the checker and the flattener that produce them depend on
// registry and session -- so this has to sit below all of them rather than inside hir/.
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
    /// `imm` is normally 0. `IMM_PARALLEL_INIT` marks the one store `lower_for` emits to
    /// initialize the induction variable of a loop `parallel_outer_for` proved disjoint — codegen
    /// renders it with a `vx.parallel_init` attribute so the device clone can offset it by the
    /// global thread id (#251). Inert everywhere else.
    Store = 3,
    /// `imm` is normally 0. `IMM_PARALLEL_STEP` marks the latch increment of the same loop —
    /// rendered with `vx.parallel_step` so the device clone can widen the step to the grid stride
    /// (#251). Inert everywhere else.
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
    /// `imm` is the trip count of the region's outermost loop when `parallel_outer_for` proved its
    /// iterations disjoint (0 otherwise) — codegen stamps it on the `vx.spawn` op as
    /// `vx_parallel_trip`, and the device pipeline grid-strides the loop and sizes the launch from
    /// it. The host path ignores it entirely (#251). Under `SPAWN_TWO_LEVEL` (bit 62, Vx#379) the
    /// low bits are a BLOCK count and bits 32..47 the thread width; `SPAWN_COOP` (bit 61, stage C)
    /// additionally marks the thread loop as cooperative — barriers inside it, no serial
    /// schedule, so the host refuses the region instead of running it.
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
    /// Print a string literal (no result): `imm` is the index into this function's string side table
    /// (`LocalWorkerState::local_string_table`). The flat codegen emits an `llvm.mlir.global internal
    /// constant` holding the (null-terminated) bytes and calls `@print_str(!llvm.ptr) -> i32` — the
    /// same runtime helper the AST path uses for a string `print!` argument.
    PrintStr = 31,
    /// A string literal in *value* position (`let s = "…"`, a string function argument): `imm` is the
    /// index into this function's string side table. The flat codegen emits the same
    /// `llvm.mlir.global internal constant` as `PrintStr`, then `llvm.mlir.addressof @".str.<n>"` to
    /// yield a first-class `!llvm.ptr` value — matching the AST path's `StringLiteralExpr`. The
    /// register's result type is a pointer (`LoweredTy::Ptr`). (#231)
    StringConst = 32,
    /// Index a raw pointer (`p[i]` where `p : *mut T`/`*const T`): `operand1` is the base pointer,
    /// `operand2` the (scalar) index, `type_idx` the pointee element's scalar type (so codegen knows
    /// the GEP stride + load type). `imm` selects value vs. place: `0` = a value read (GEP then
    /// `llvm.load`); `1` = an element **place** on the left of an assignment (GEP only), consumed by
    /// the following `PtrStore`. This is `Vec`'s `self.data[i]` — the raw-pointer analogue of
    /// `TensorIndex`, GEP-addressed instead of memref-addressed. (#242)
    PtrIndex = 33,
    /// Store a value into a raw-pointer place (no result): `operand1` is the destination `PtrIndex`
    /// place (an `imm = 1` `PtrIndex`), `operand2` the value. The pointee element type comes from the
    /// place. The raw-pointer analogue of `TensorStore` — `Vec`'s `self.data[i] = val`. (#242)
    PtrStore = 34,
    /// Materialize a function pointer (`!llvm.ptr`) for a named function: `type_idx` is the target
    /// function's GID (resolved to symbol + signature via the callee/func-sig maps). Codegen emits
    /// `func.constant @name : (params)->ret` then a cast to `!llvm.ptr`. This is a bare function name
    /// used as a value (`apply_func(square, 5)`) or a closure's call function inside the
    /// `Closure_N -> ClosureK` adapter. (#242)
    FuncConst = 35,
    /// An indirect call through a function-pointer value (`f(x)` where `f : fn(..)->R`): `operand1` is
    /// the callee `!llvm.ptr`, `imm` the argument count (taken from the preceding `Arg`s, exactly like
    /// `Call`), and this instruction's own `type_idx` the scalar return type. Codegen reconstructs the
    /// function type `(arg types)->ret` from the actual argument registers, casts the pointer to it, and
    /// emits `call_indirect`. Backs `Closure1`'s `f_ptr(env, item)` and bare-fn-pointer calls. (#242)
    CallIndirect = 36,
    /// The address of a *by-value nested-aggregate field* (`&outer.inner`): `operand1` is the enclosing
    /// aggregate's slot/pointer, `imm` the field's byte offset, and this instruction's own `type_idx`
    /// the nested aggregate's layout GID. Codegen GEPs to the field and yields the pointer, tracked as an
    /// aggregate slot so a chained access (`outer.inner.a`) or a method receiver (`self.iter.next()`)
    /// addresses through it. The nested-aggregate analogue of `FieldLoad` that stops at the pointer. (#242)
    FieldAddr = 37,
    /// CONDITIONAL abort: terminate unless `operand1` (a bool) holds. `imm` indexes this
    /// function's string side table for the message, exactly as `PrintStr` does.
    ///
    /// Terminating is the primitive and everything else is a use of it: `assert(c, m)` is this
    /// with the programmer's condition, and `abort()` is this with a constant `false`. Calling
    /// it is SAFE -- ending a process violates no memory-safety property, the same reason
    /// `std::process::abort` is safe in Rust.
    ///
    /// Emitted as `cf.assert`, which is target-portable in a way a hand-written branch onto
    /// libc `abort` is not: on the host `convert-cf-to-llvm` expands it to `puts` + `abort` +
    /// `unreachable`, and INSIDE A KERNEL `convert-gpu-to-nvvm` expands it to `__assertfail`
    /// with the message, file, line and a `noreturn` attribute. Both passes are already in the
    /// pipelines. An earlier draft desugared this by hand into `print_str` + `abort` calls,
    /// which are `func` ops -- not device-lowerable, so a kernel containing one was dropped
    /// from GPU compilation entirely (Vx#362).
    Abort = 38,
    /// `barrier()` inside a spawn region (Vx#379): all threads of a block reach this point
    /// before any proceeds. On the device clone it becomes `gpu.barrier`; on the host path a
    /// serial loop already IS the barrier's ordering, so it lowers to a dead store the code
    /// around it never reads. Codegen carries it as a `vx.barrier`-tagged op so the device
    /// rewrite can find it after cloning (no operands, no result).
    Barrier = 39,
    /// `matmul_into(&mut dst, &a, &b)`: fill `dst` with zero and accumulate `a @ b` into it, in
    /// place. `operand1` = a, `operand2` = b, and — the one opcode that does this — `imm` is the
    /// DESTINATION register, because an effect instruction has only two operand fields and the
    /// destination is a third. Codegen emits `linalg.fill` + `linalg.matmul`, the same pair the
    /// AST path builds, which is exactly the shape `kernelKindOf` classifies for cuBLAS routing:
    /// this op exists so a matmul region no longer evicts the whole program from the flat path
    /// (and with it every other region's parallel proof).
    MatmulInto = 40,
    /// `flash_attention_into(&mut o, &q, &k, &v, scale)`: fused scaled-dot-product attention
    /// written in place, `o = softmax(q @ k^T * scale) @ v` over rank-2 f16 tensors. Five values
    /// through two operand fields: `operand1` = q, `operand2` = k, and the imm packs the rest --
    /// `o | v << 16 | scale << 32` (register indices are far below 2^16). Codegen emits a serial
    /// fallback nest plus a `vx.attention_note` naming the roles, which `kernelKindOf` classifies
    /// as `kind=attention` so the runtime can route the region to a vendor flash kernel; a
    /// runtime that refuses runs the nest as written -- slower, never wrong.
    FlashAttnInto = 41,
    /// A differentiated call: `grad(f, x)`, `vjp(f, x, v)` or `jvp(f, x, v)`. `type_idx` is the
    /// *target* function's GID (its name and signature come from there), and `imm` packs the
    /// argument count in the low 32 bits with the mode above them -- `AUTODIFF_REVERSE` or
    /// `AUTODIFF_FORWARD`. The arguments are the `Arg` instructions immediately preceding, as for
    /// `Call`; forward mode carries its tangent as the last of them.
    ///
    /// Codegen materializes the target as a function constant and calls the Enzyme wrapper with
    /// it as the first argument. `vjp` has no mode of its own: the flattener lowers it as reverse
    /// mode followed by an ordinary multiply against the cotangent.
    AutoDiff = 42,
    /// Read a rank-0 tensor's element (`memref.load %t[]`): `operand1` is the tensor, the result
    /// the scalar. Emitted where a rank-0 value sits in scalar position — the arithmetic and
    /// comparison operands of a `let t : Tensor<el, [?, ?]> = <scalar>` local (Vx#396). Shaped tensors
    /// never emit this; their reads index.
    TensorLoad = 43,
    /// The runtime extent of one dimension of a tensor (`t.shape[k]` -> `memref.dim`):
    /// `operand1` the tensor, `operand2` the dimension index, the result an `i32`.
    TensorDim = 44,
    /// Zero a freshly allocated tensor (`Tensor<T, [..]>::new()`, no result): `operand1` is the
    /// buffer. Lowers to `linalg.fill` with a zero of the element type -- the same fill
    /// `MatmulInto` emits before accumulating, so there is one way a tensor gets zeroed.
    ///
    /// A separate instruction rather than a flag on `TensorAlloc` because the alloc's two operand
    /// fields are spoken for (`operand2` carries a memory-space dispatch id) and a register field
    /// pressed into service as a boolean is a decoding hazard, not a saving.
    TensorZero = 45,
    /// Fill a freshly allocated tensor with a value (`Tensor<T, [..]>::fill(v)`, no result):
    /// `operand1` is the buffer, `operand2` the value. Lowers to `linalg.fill`, the same fill
    /// `TensorZero` emits, with the value the source wrote instead of a zero.
    TensorFill = 46,
}

/// Reverse mode for `Opcode::AutoDiff`: the gradient, through `__enzyme_autodiff_grad_*`.
pub const AUTODIFF_REVERSE: u64 = 0;
/// Forward mode for `Opcode::AutoDiff`: the tangent, through `__enzyme_fwddiff_jvp_*`.
pub const AUTODIFF_FORWARD: u64 = 1;
/// Where the mode sits in `Opcode::AutoDiff`'s `imm`; the argument count is below it.
pub const AUTODIFF_MODE_SHIFT: u32 = 32;

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
            31 => PrintStr,
            32 => StringConst,
            33 => PtrIndex,
            34 => PtrStore,
            35 => FuncConst,
            36 => CallIndirect,
            37 => FieldAddr,
            38 => Abort,
            39 => Barrier,
            40 => MatmulInto,
            41 => FlashAttnInto,
            42 => AutoDiff,
            43 => TensorLoad,
            44 => TensorDim,
            45 => TensorZero,
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

/// `imm` tag on the induction-variable init `Store` of a grid-stridable loop (see `Store`'s doc).
/// A distinct constant per site, even though both are 1 today, so a reader grepping either name
/// finds the emitting and the consuming side rather than a bare literal.
pub const IMM_PARALLEL_INIT: u64 = 1;
/// `imm` tag on the latch `Add` of the same loop (see `Add`'s doc).
pub const IMM_PARALLEL_STEP: u64 = 1;

/// `imm` tags for the two-level mapping (Vx#379): the induction-variable init `Store` and latch
/// `Add` of a BLOCK-mapped loop -- offset by `blockIdx.x`, stride `gridDim.x`. Every thread of a
/// block walks the same block iterations (redundant execution on identical values is the
/// two-level model's block-scope semantics); only thread-mapped loops partition within a block.
pub const IMM_BLOCK_INIT: u64 = 2;
pub const IMM_BLOCK_STEP: u64 = 2;
/// The same pair for a THREAD-mapped loop -- offset by `threadIdx.x`, stride `blockDim.x`. At a
/// 1x1x1 launch both mappings collapse to the serial loop, which is what keeps the launcher's
/// EXPECT degeneracy and the host path exact.
pub const IMM_THREAD_INIT: u64 = 3;
pub const IMM_THREAD_STEP: u64 = 3;
