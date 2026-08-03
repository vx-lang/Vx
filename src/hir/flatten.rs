//===- flatten.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// HIR lowering (#197): the instruction-selection pass that flattens a
// type-checked function *body* from the tree AST into the worker's
// `local_hir_stream` (`Vec<HirInstruction>`), the flat bytecode Phase-7 codegen
// consumes. Distinct from `hir/lower_ast.rs`, which builds an arena *tree* HIR.
//
// SSA-by-position: the instruction at index `i` defines `Register(i)`; operands
// name earlier instructions by index; `type_idx` indexes `local_type_stream`.
// See docs/discussions/implementation_plans/hir_flattening.md.
//
//===----------------------------------------------------------------------===//
use crate::gid::TypeId;
use crate::hir::bytecode::{HirInstruction, Opcode, Register, TypeIdx};
use crate::layout::FieldTy;
use crate::registry::ImmutableGlobalRegistry;
use crate::session::LocalWorkerState;
use crate::symbol::Symbol;
use crate::syntax::{
    BinaryOp, ElementType, Expr, Function, LogicalOp, NumberExpr, RelationalOp, Statement, Type,
    UnaryOp,
};
use std::collections::{HashMap, HashSet};

/// The stable GID of a primitive scalar type: module 0 (builtin) + a content hash of the element
/// name. Single source of truth so a scalar has the *same* GID whether it appears in a signature
/// (`pipeline.rs`) or a lowered body — the flat type stream must agree on identity.
pub fn scalar_gid(elem: &ElementType) -> TypeId {
    let sym = crate::hash::DefPath::Named(&format!("$prim::{elem:?}")).compute_symbol_hash();
    TypeId::new(0, sym, 0, 0)
}

/// The stable GID of a tensor type: module 0 (builtin) + a content hash of its element type and
/// canonical shape, so `Tensor<f32,[2,4]>` and `Tensor<f32,[4,4]>` are distinct and a tensor has the
/// *same* identity in a signature (`pipeline.rs`) and a lowered body — the flat type stream must
/// agree on identity, exactly as for scalars.
pub fn tensor_gid(elem: &ElementType, shape: &[String]) -> TypeId {
    let sym = crate::hash::DefPath::Named(&format!("$tensor::{elem:?}::[{}]", shape.join(",")))
        .compute_symbol_hash();
    TypeId::new(0, sym, 0, 0)
}

/// The stable GID of a raw pointer type (`!llvm.ptr`): module 0 (builtin) + a content hash of a fixed
/// name. Every pointer — a string value, a `*mut i8`/`*const u8` FFI argument or result — is the same
/// opaque `!llvm.ptr`, so one GID identifies them all (matching MLIR's opaque pointer model). Kept
/// distinct from any scalar/tensor GID so `elem_of_gid` never mistakes a pointer for a scalar. (#231)
pub fn ptr_gid() -> TypeId {
    let sym = crate::hash::DefPath::Named("$prim::ptr").compute_symbol_hash();
    TypeId::new(0, sym, 0, 0)
}

/// The stable per-instance GID of a monomorphized data-carrying enum (`Option<i32>`): module 0 +
/// a content hash of the base name and mangled type arguments, so `Option<i32>` and `Option<i64>` are
/// distinct aggregates and the same instance hashes identically in a signature and a body. Distinct
/// from any struct GID (those carry a real module hash), so it never collides with a registry layout.
/// (#242)
pub fn enum_instance_gid(base: &str, args: &[Type]) -> TypeId {
    use crate::syntax::types::Mangle;
    let mangled: Vec<String> = args.iter().map(|a| a.mangle()).collect();
    let sym = crate::hash::DefPath::Named(&format!("$enum::{base}<{}>", mangled.join(",")))
        .compute_symbol_hash();
    TypeId::new(0, sym, 0, 0)
}

/// The tensor GID of a `Type::Tensor`, or `None` if it is not a tensor, its element is generic, or a
/// dim is not a literal or a plain name (canonicalizing an arbitrary expression would not be stable).
pub fn tensor_gid_of(ty: &Type) -> Option<TypeId> {
    let (elem, shape) = tensor_elem_shape(ty)?;
    Some(tensor_gid(&elem, &shape))
}

/// The element type + canonical shape of a `Type::Tensor`, or `None` if it is unmodelled (generic
/// element, or a dim that isn't a literal/name). The shape is what the flat lowerer rank-reduces on
/// indexing.
fn tensor_elem_shape(ty: &Type) -> Option<(ElementType, Vec<String>)> {
    let Type::Tensor(elem, dims, _) = ty else {
        return None;
    };
    if matches!(elem, ElementType::Generic(_)) {
        return None;
    }
    let shape: Vec<String> = dims.iter().map(tensor_dim_string).collect::<Option<_>>()?;
    Some((elem.clone(), shape))
}

/// Canonicalize a tensor dimension for the GID: a numeric literal by value, a const/generic name by
/// its name. Anything else declines (so the tensor stays unmodelled rather than hashing unstably).
fn tensor_dim_string(e: &Expr) -> Option<String> {
    match e {
        Expr::Number(n) => Some(n.value.as_ref().to_string()),
        Expr::Identifier(id) => Some(id.name.as_ref().to_string()),
        _ => None,
    }
}

/// `type_idx` sentinel for *effect* instructions (`Store`/`Br`/`CondBr`/`BlockStart`) that produce
/// no result value and therefore have no result type.
const NO_TYPE: u32 = u32::MAX;

/// The `Expr` variant name, for the `VX_FLAT_DBG` decline survey (which construct is unsupported).
fn expr_kind(e: &Expr) -> &'static str {
    match e {
        Expr::MethodCall(_) => "MethodCall",
        Expr::Array(_) => "Array",
        Expr::Closure(_) => "Closure",
        Expr::Range(_) => "Range",
        Expr::SpawnOn(_) => "SpawnOn",
        Expr::ComptimeBlock(_) => "ComptimeBlock",
        Expr::TransferPredicate(_) => "TransferPredicate",
        Expr::LogicalOp(_) => "LogicalOp",
        Expr::Borrow(_) => "Borrow",
        Expr::Dereference(_) => "Dereference",
        Expr::Match(_) => "Match",
        Expr::If(_) => "If",
        Expr::IndirectCall(_) => "IndirectCall",
        Expr::VecMacro(_) => "VecMacro",
        Expr::MacroCall(_) => "MacroCall",
        Expr::EnumVariant(_) => "EnumVariant",
        Expr::Grad(_) => "Grad",
        Expr::Vjp(_) => "Vjp",
        Expr::Jvp(_) => "Jvp",
        Expr::StringLiteral(_) => "StringLiteral",
        Expr::MemorySpace(_) => "MemorySpace",
        Expr::Topology(_) => "Topology",
        _ => "other-expr",
    }
}

/// The `Statement` variant name, for the `VX_FLAT_DBG` decline survey.
fn stmt_kind(s: &Statement) -> &'static str {
    match s {
        Statement::Loop(_) => "Loop",
        Statement::Assert(_) => "Assert",
        Statement::MacroCall(_) => "MacroCall",
        Statement::ExprStmt(_) => "ExprStmt",
        _ => "other-stmt",
    }
}

/// The type of a lowered value: a primitive scalar, or an aggregate nominal (struct/enum) named by
/// its GID. Aggregates size their `Alloca` from the frozen-registry layout (#199); scalar ops
/// (`Add`/`Cmp`/`Cast`) apply only to `Scalar`. This is what the flat *type stream* carries per
/// value instruction.
#[derive(Clone)]
enum LoweredTy {
    Scalar(ElementType),
    Aggregate(TypeId),
    /// A tensor by element type + canonical shape. Passed by reference (a memref descriptor), so it
    /// lives as an SSA value — never `Alloca`'d into a slot. The shape is carried (not just the GID)
    /// so indexing can rank-reduce it to a row/sub-view or the scalar element.
    Tensor {
        elem: ElementType,
        shape: Vec<String>,
    },
    /// A raw pointer (`!llvm.ptr`): a string value, or a `*const`/`*mut` FFI argument or result. All
    /// pointers share one opaque type (matching MLIR), so no element is carried. Stored in a slot via
    /// `llvm.alloca` in memory mode; a value otherwise. (#231/#235)
    Ptr,
}

impl LoweredTy {
    /// The GID this type contributes to `local_type_stream`.
    fn gid(&self) -> TypeId {
        match self {
            LoweredTy::Scalar(e) => scalar_gid(e),
            LoweredTy::Aggregate(id) => *id,
            LoweredTy::Tensor { elem, shape } => tensor_gid(elem, shape),
            LoweredTy::Ptr => ptr_gid(),
        }
    }
}

/// A lowered value: the SSA register holding it and its type.
#[derive(Clone)]
struct Val {
    reg: Register,
    ty: LoweredTy,
}

/// How an in-scope name is materialized.
#[derive(Clone)]
enum Binding {
    /// Straight-line SSA: the name aliases an existing value register (no control flow).
    Reg(Val),
    /// Memory model: the name is a stack slot (`Alloca`); reads emit `SlotLoad`, writes `Store`, so
    /// the value survives across basic blocks. `ty` is the slot's type.
    Slot { reg: Register, ty: LoweredTy },
    /// A non-escaping reference bound as a symbolic *place* (#275, §5): the reference names a location
    /// — the **place expression** that was borrowed — whose address is never materialized. A read of
    /// `*r` re-lowers `place` as a value (`Identifier(x)` reads the local directly, so `let r = &x;
    /// *r` needs no `alloca` — §5 Example A; `MemberAccess(p.x)` `FieldLoad`s the field — Example B/C);
    /// a write `*r = v` re-lowers `place` as an assignment target. `ty` is the pointee type. Created
    /// only when escape analysis proves the reference never needs a real address — an escaping borrow
    /// (call arg, return, struct field) instead materializes its base and binds a `Ptr`, as before.
    ///
    /// Storing the borrowed expression (rather than a base + interned projection list) keeps the
    /// variable-length path in the AST it already lives in and lets `*r` reuse the existing field
    /// read/store lowering wholesale (§5.5: a non-escaping place is pure lowerer data). The projection
    /// path §5.4's disjointness needs is recovered from `place` on demand.
    Place { place: Expr },
}

/// Per-function lowering accumulator. Instructions and their result-type GIDs are built into local
/// buffers and only committed to the worker on full success, so a partial (aborted) lowering leaves
/// no trace.
struct Lowerer<'r> {
    code: Vec<HirInstruction>,
    types: Vec<TypeId>,
    scope: HashMap<Symbol, Binding>,
    /// The frozen nominal-type registry — the source of aggregate layouts (size/align/field offsets)
    /// used to size an aggregate's `Alloca` and, later, resolve field offsets.
    registry: &'r ImmutableGlobalRegistry,
    /// When set, named locals live in memory (`Alloca`/`Store`/`SlotLoad`) so values survive across
    /// basic blocks — chosen for functions with control flow (or an aggregate local, which must sit
    /// in a slot to be addressable), matching the AST codegen's `alloca`-backed locals. Straight-line
    /// scalar functions stay pure-SSA (bindings are `Reg`).
    memory: bool,
    /// Next basic-block id to hand out (0 is the entry block).
    next_block: u32,
    /// Enclosing loops: `(continue_target, break_target)` block ids. `continue` branches to the
    /// first (the header for `loop`, the increment latch for `for`), `break` to the second (exit).
    loop_stack: Vec<(u32, u32)>,
    /// Tensor-type side table `GID -> (element, shape)`, recorded as tensor-typed values are emitted.
    /// Tensor GIDs are content hashes, so codegen recovers a tensor's memref shape from here rather
    /// than by inverting the GID (which is impossible). Committed onto the worker.
    tensor_types: Vec<(TypeId, ElementType, Vec<String>)>,
    /// String side table: the bytes for each `PrintStr` emitted, in emission order. A `PrintStr`'s
    /// `imm` indexes here; codegen emits an `llvm.mlir.global` per entry. Committed onto the worker.
    strings: Vec<String>,
    /// The concrete AST type of each in-scope name (params + `let` locals) — the flat-path analogue of
    /// the AST codegen's identifier→type env. It is the only source of a *pointer's pointee element
    /// type*, which the frozen `layouts` erase (a pointer field is `Opaque`): recovering `self.data`'s
    /// `*mut i32` for a `self.data[i]` index needs `self : &mut Vec<i32>`'s type substituted into the
    /// base struct's `data : *mut T` field. See `infer_ast_type` (#242).
    ast_types: HashMap<Symbol, Type>,
    /// Synthetic aggregate layouts for monomorphized data-carrying enum instances (`Option<i32>` ->
    /// `{ i32 tag, i32 payload }`), keyed by a per-instance GID: `(gid, field offsets, field MLIR
    /// types)`. Such a layout is instance-dependent (the by-value payload varies with `T`), so it
    /// isn't in the frozen registry; synthesized here as an enum is constructed/matched and committed
    /// so codegen can address it (the tagged-union analogue of `tensor_types`). (#242)
    agg_layouts: Vec<(TypeId, Vec<u64>, Vec<String>)>,
    /// The function's lowered return type, for a `return <match>` whose arms `return` themselves:
    /// the match's fall-through merge block needs a terminator, so it returns a default (zero) value of
    /// this type — mirroring the AST codegen's default-return merge block (`Option::unwrap`). (#242)
    ret_ty: Option<LoweredTy>,
    /// Base locals whose address must be **materialized** — a borrow of them *escapes* (a call arg, a
    /// return, a struct field, or a reference local used as anything but `*r`). Such a local lives in an
    /// addressable `Slot` so `&x` has a real pointer to yield. This refines the old "any `&x` slots x"
    /// rule (§9): a `&x` that only feeds a local `*r` never materializes — its base stays a register and
    /// the reference is a `Binding::Place` (§5 Example A). Computed by the escape pre-pass. (#230/#275)
    materialized: HashSet<Symbol>,
    /// Locals reassigned after their `let` (`x = ..`, `x += ..`). A *scalar* such local needs a memory
    /// slot to carry its new value across a block boundary — but **only when the function has control
    /// flow**; a straight-line reassignment stays a pure-SSA rebind. This is the per-local half of the
    /// step-2 refinement (§3.2): a non-mutated scalar keeps a register even under control flow, where
    /// the old function-global rule slotted *every* local. (#230)
    mutated: HashSet<Symbol>,
    /// Whether the function has control flow (`if`/`loop`/`for`/`match`/logical op) — the precondition
    /// for the mutated-scalar slot rule above. Distinct from `memory` (which also turns on for an
    /// aggregate param / struct construction to drive the block model): a mutated scalar in a
    /// *straight-line* struct-constructing function does not need a slot. (#230)
    has_control_flow: bool,
    /// Ref-locals the escape analysis realized as symbolic places (§5): a `let r = &<lvalue>` in this
    /// set binds `r` to a `Binding::Place` over the borrowed lvalue rather than materializing an
    /// address. (#275)
    place_bindings: HashSet<Symbol>,
    /// Set while lowering a place-write `*r = v` (M2b-2): the borrowed place's `(root local, field
    /// path)`. The `FieldStore` the write lowers to reads this to tag itself as a place-write for
    /// alias-scope metadata; a direct `p.x = v` leaves it `None` and its store is untagged. (#275, §5.4)
    pending_place_write: Option<(Symbol, Vec<Symbol>)>,
    /// Place-write field stores collected during lowering, as `(stream position, borrowed root, field
    /// path)`. Post-lowering these reduce to a numeric group/sibling table (`reduce_place_alias`) so
    /// codegen can attach `alias_scopes`/`noalias_scopes` — carrying the borrow checker's disjointness
    /// of simultaneously live `&mut o.field` borrows into the IR. (#275, §5.4)
    place_field_stores: Vec<(usize, Symbol, Vec<Symbol>)>,
}

impl<'r> Lowerer<'r> {
    fn new(registry: &'r ImmutableGlobalRegistry) -> Self {
        Self {
            code: Vec::new(),
            types: Vec::new(),
            scope: HashMap::new(),
            registry,
            memory: false,
            next_block: 1,
            loop_stack: Vec::new(),
            tensor_types: Vec::new(),
            strings: Vec::new(),
            ast_types: HashMap::new(),
            agg_layouts: Vec::new(),
            ret_ty: None,
            materialized: HashSet::new(),
            mutated: HashSet::new(),
            has_control_flow: false,
            place_bindings: HashSet::new(),
            pending_place_write: None,
            place_field_stores: Vec::new(),
        }
    }

    /// Emit a *value* instruction defining a fresh SSA register (= its stream position) with result
    /// type `ty` (scalar or aggregate), pushing its GID onto the type stream.
    fn emit_typed(
        &mut self,
        opcode: Opcode,
        o1: Register,
        o2: Register,
        ty: LoweredTy,
        imm: u64,
    ) -> Val {
        let type_idx = TypeIdx(self.types.len() as u32);
        self.types.push(ty.gid());
        // Record tensor types in the side table so codegen can recover the memref shape by GID.
        if let LoweredTy::Tensor { elem, shape } = &ty {
            self.tensor_types
                .push((ty.gid(), elem.clone(), shape.clone()));
        }
        let reg = Register(self.code.len() as u32);
        self.code
            .push(HirInstruction::new(opcode, o1, o2, type_idx, imm));
        Val { reg, ty }
    }

    /// Convenience for the common scalar case: emit a value with a scalar result type.
    fn emit_value(
        &mut self,
        opcode: Opcode,
        o1: Register,
        o2: Register,
        ty: ElementType,
        imm: u64,
    ) -> Val {
        self.emit_typed(opcode, o1, o2, LoweredTy::Scalar(ty), imm)
    }

    /// Emit an `Alloca` slot for a value of type `ty`. `type_idx` is the slot's element/aggregate
    /// GID; for an aggregate `imm` carries its byte size from the registry layout (0 for a scalar,
    /// whose element type already implies its size). The result register is the slot handle.
    fn emit_alloca(&mut self, ty: LoweredTy) -> Val {
        let imm = match &ty {
            // A tensor is a reference (memref), not a stack value, so it is never `Alloca`'d — but
            // keep the match total; if one ever reaches here its size is left unencoded. A pointer
            // slot is a single `!llvm.ptr` cell, so its size is likewise implied by its type.
            LoweredTy::Scalar(_) | LoweredTy::Tensor { .. } | LoweredTy::Ptr => 0,
            LoweredTy::Aggregate(id) => self
                .registry
                .layouts
                .get(id)
                .map(|d| d.size_bytes as u64)
                .unwrap_or(0),
        };
        self.emit_typed(Opcode::Alloca, Register(0), Register(0), ty, imm)
    }

    /// Emit an *effect* instruction (no result value): `type_idx` is the [`NO_TYPE`] sentinel.
    fn emit_effect(&mut self, opcode: Opcode, o1: Register, o2: Register, imm: u64) {
        self.code
            .push(HirInstruction::new(opcode, o1, o2, TypeIdx(NO_TYPE), imm));
    }

    fn new_block(&mut self) -> u32 {
        let b = self.next_block;
        self.next_block += 1;
        b
    }

    /// Whether the last emitted instruction is a block terminator — so we don't append a second one
    /// (e.g. a `Br` to the merge after a branch already `return`ed).
    fn block_terminated(&self) -> bool {
        matches!(
            self.code.last().map(|i| i.opcode),
            Some(Opcode::Ret | Opcode::Br | Opcode::CondBr)
        )
    }

    /// Bind a fresh local name to a value: an SSA alias in straight-line mode, or an `Alloca` slot
    /// (+ initializing `Store`) in memory mode.
    /// Emit a `Print` for a value: `type_idx` carries the value's type (scalar or tensor) so codegen
    /// routes to the right `print_*` / `printMemref*` runtime helper.
    fn emit_print(&mut self, v: Val) {
        let type_idx = TypeIdx(self.types.len() as u32);
        self.types.push(v.ty.gid());
        if let LoweredTy::Tensor { elem, shape } = &v.ty {
            self.tensor_types
                .push((v.ty.gid(), elem.clone(), shape.clone()));
        }
        self.code.push(HirInstruction::new(
            Opcode::Print,
            v.reg,
            Register(0),
            type_idx,
            0,
        ));
    }

    /// Emit a `PrintStr` for a string literal (no result): record the bytes in the string side table
    /// and carry that entry's index in `imm`. Codegen emits an `llvm.mlir.global` for the bytes and
    /// calls `@print_str` — matching the AST path's string `print!` argument.
    fn emit_print_str(&mut self, s: &str) {
        let imm = self.strings.len() as u64;
        self.strings.push(s.to_string());
        self.emit_effect(Opcode::PrintStr, Register(0), Register(0), imm);
    }

    /// Emit a `StringConst` for a string literal in *value* position: record the bytes in the string
    /// side table and carry that entry's index in `imm`. The result is an `!llvm.ptr` value (codegen
    /// emits the global + `addressof`), which a `let`/argument can carry — matching the AST path's
    /// `StringLiteralExpr`. (#231)
    fn emit_string_const(&mut self, s: &str) -> Val {
        let imm = self.strings.len() as u64;
        self.strings.push(s.to_string());
        self.emit_typed(
            Opcode::StringConst,
            Register(0),
            Register(0),
            LoweredTy::Ptr,
            imm,
        )
    }

    /// Lower one `print!`/`println!` argument: a string literal emits a `PrintStr`; any other argument
    /// is lowered as a value and `print`ed (routing to the scalar/tensor `print_*` helper by type).
    fn lower_print_arg(&mut self, arg: &Expr) -> Option<()> {
        if let Expr::StringLiteral(sl) = arg {
            self.emit_print_str(sl.value.as_ref());
        } else {
            let v = self.lower_expr(arg)?;
            self.emit_print(v);
        }
        Some(())
    }

    fn bind_local(&mut self, name: Symbol, v: Val) {
        let materialized_scalar =
            matches!(v.ty, LoweredTy::Scalar(_)) && self.materialized.contains(&name);
        // The per-local slot decision (§3.2 step 2 + §5 escape refinement):
        //   - an aggregate always needs a slot (its fields must be `getelementptr`'d, #215);
        //   - a scalar needs one only if its address is *materialized* (an escaping `&x`, #275) or it is
        //     reassigned in a control-flow function (the new value must cross a block boundary). A
        //     non-materialized, non-mutated scalar stays a dominating SSA register — and a scalar
        //     borrowed only into a non-escaping place is *not* materialized, so `let r = &x; *r` needs
        //     no slot at all (§5 Example A);
        //   - a *pointer* local gets the same per-local rule as a scalar (M3b, §11's remainder): a slot
        //     when it is *address-taken* (`&p` — a pointer-to-pointer, so `&p` has a real `!llvm.ptr`
        //     cell to hand out; this is how nested `&mut &mut T` materializes, #278) or reassigned in a
        //     control-flow function (the new pointer value must cross a block boundary). A non-reassigned,
        //     non-address-taken pointer — e.g. a materialized `&mut x` used as `*p` in a branch — stays a
        //     dominating SSA register, one fewer `alloca`;
        //   - a tensor local keeps the coarser function-global model.
        // Safety: a register only ever holds a *single-definition* local's value (the right one), and
        // any read outside that definition's dominance is rejected by the MLIR verifier — so the flat
        // path declines to the AST oracle rather than ever silently miscompiling.
        let needs_slot = match &v.ty {
            LoweredTy::Aggregate(_) => true,
            LoweredTy::Scalar(_) => {
                materialized_scalar || (self.has_control_flow && self.mutated.contains(&name))
            }
            LoweredTy::Ptr => {
                self.materialized.contains(&name)
                    || (self.has_control_flow && self.mutated.contains(&name))
            }
            _ => self.memory,
        };
        if needs_slot {
            // A materialized scalar needs an `llvm.alloca` (a real `!llvm.ptr`) so `&x` yields a
            // pointer, not a rank-0 `memref` (which cannot be `getelementptr`'d). Signalled to codegen
            // by `imm = 1` on the `Alloca`; a memory-mode-but-never-borrowed scalar keeps the memref
            // (`imm = 0`). This matches the AST codegen, which allocas every mutable scalar as `!llvm.ptr`.
            let slot = if materialized_scalar {
                self.emit_typed(Opcode::Alloca, Register(0), Register(0), v.ty.clone(), 1)
            } else {
                self.emit_alloca(v.ty.clone())
            };
            self.emit_effect(Opcode::Store, slot.reg, v.reg, 0);
            self.scope.insert(
                name,
                Binding::Slot {
                    reg: slot.reg,
                    ty: v.ty,
                },
            );
        } else {
            self.scope.insert(name, Binding::Reg(v));
        }
    }

    /// Assign to an already-bound name: a `Store` to its slot (memory mode) or an SSA rebind
    /// (straight-line). `None` if the name is unbound.
    fn assign_local(&mut self, name: &Symbol, v: Val) -> Option<()> {
        match self.scope.get(name)?.clone() {
            Binding::Slot { reg, .. } => {
                self.emit_effect(Opcode::Store, reg, v.reg, 0);
                Some(())
            }
            Binding::Reg(_) => {
                self.scope.insert(name.clone(), Binding::Reg(v));
                Some(())
            }
            // A place-bound reference reassigned by name (`r = ..`) is an escape the analysis routes to
            // materialization; if one reaches here, decline to the AST oracle. (#275)
            Binding::Place { .. } => None,
        }
    }

    /// Lower an expression to the register holding its result (emitting instructions as needed).
    /// Returns `None` for anything outside the current subset — the caller then aborts.
    fn lower_expr(&mut self, e: &Expr) -> Option<Val> {
        match e {
            Expr::Number(n) => {
                let elem = number_elem(n)?;
                let imm = encode_imm(n.value.as_ref(), &elem)?;
                Some(self.emit_value(Opcode::Const, Register(0), Register(0), elem, imm))
            }
            // A name read: an SSA alias (no instruction) or a `SlotLoad` from its memory slot. A name
            // that isn't a local but names a registered function is a *function pointer* value (a bare
            // `square` passed to `apply_func`), lowered to a `FuncConst`. (#242)
            Expr::Identifier(id) => {
                // Boolean literals parse as the identifiers `true`/`false` (matching the AST codegen);
                // lower to a `bool` (i1) constant, `1` or `0`.
                if id.name.as_ref() == "true" || id.name.as_ref() == "false" {
                    let v = (id.name.as_ref() == "true") as u64;
                    return Some(self.emit_value(
                        Opcode::Const,
                        Register(0),
                        Register(0),
                        ElementType::Bool,
                        v,
                    ));
                }
                match self.scope.get(&id.name).cloned() {
                    Some(Binding::Reg(v)) => Some(v),
                    Some(Binding::Slot { reg, ty }) => {
                        Some(self.emit_typed(Opcode::SlotLoad, reg, Register(0), ty, 0))
                    }
                    // Reading a place-bound reference *as a value* (`f(r)`, `return r`) is an escape —
                    // the analysis materializes those, so this arm is the safety-net decline. (#275)
                    Some(Binding::Place { .. }) => None,
                    None => self.lower_func_const(&id.name),
                }
            }
            Expr::BinaryOp(b) => {
                let l = self.lower_expr(&b.lhs)?;
                let r = self.lower_expr(&b.rhs)?;
                let op = binop_opcode(&b.op)?;
                // Operands are type-checked *assignable* but not necessarily identical; the result
                // carries the left operand's type. When either operand is a tensor the op is
                // *elementwise* and the result is the tensor type (a scalar operand broadcasts) -- an
                // arith opcode with a tensor result type is the flat HIR's elementwise form, mirroring
                // `arith.mulf` on a vector in codegen.
                let result_ty = match (&l.ty, &r.ty) {
                    (LoweredTy::Tensor { .. }, _) => l.ty.clone(),
                    (_, LoweredTy::Tensor { .. }) => r.ty.clone(),
                    _ => l.ty.clone(),
                };
                // Two scalar operands already share a type: the checker types both to the same scalar
                // and rejects a genuine mismatch (no implicit conversion, #240), so no coercion here.
                Some(self.emit_typed(op, l.reg, r.reg, result_ty, 0))
            }
            // A comparison yields a `bool`; the relation is carried in `imm`.
            Expr::RelationalOp(r) => {
                let l = self.lower_expr(&r.lhs)?;
                let rhs = self.lower_expr(&r.rhs)?;
                // Both operands already share a type (the checker reconciles them and rejects a
                // genuine mismatch, #240), so `Cmp` compares them directly.
                Some(self.emit_value(
                    Opcode::Cmp,
                    l.reg,
                    rhs.reg,
                    ElementType::Bool,
                    rel_code(&r.op),
                ))
            }
            Expr::UnaryOp(u) => {
                let v = self.lower_expr(&u.expr)?;
                let op = match u.op {
                    UnaryOp::Neg => Opcode::Neg,
                    UnaryOp::Not => Opcode::Not,
                };
                Some(self.emit_typed(op, v.reg, Register(0), v.ty, 0))
            }
            // A scalar `as` cast: the result carries the (scalar) target type.
            Expr::AsCast(c) => {
                let v = self.lower_expr(&c.expr)?;
                let target = scalar_of(&c.target_ty)?;
                Some(self.emit_value(Opcode::Cast, v.reg, Register(0), target, 0))
            }
            // A struct field read `base.member`. `base` is either a local aggregate slot (`p.x`,
            // #215) or a pointer to an aggregate (`self.len`/`self.data` where `self : &mut Vec`,
            // #242) — `lower_agg_base` resolves both to the `!llvm.ptr` addressing the struct plus its
            // layout GID. The field's offset + type come from the registry layout: a scalar field
            // yields its element, a pointer field (`Vec`'s `data`) an opaque `!llvm.ptr`, and a by-value
            // nested-aggregate field (`VecMap`'s `f : Closure1`) the whole `!llvm.struct` value (#242).
            Expr::MemberAccess(m) => {
                let (base_reg, gid) = self.lower_agg_base(&m.base)?;
                let field = self
                    .registry
                    .layouts
                    .get(&gid)?
                    .fields
                    .iter()
                    .find(|f| f.name.as_ref() == m.member.as_ref())?;
                let offset = field.offset as u64;
                let result_ty = match &field.ty {
                    FieldTy::Scalar(e) => LoweredTy::Scalar(e.clone()),
                    FieldTy::Opaque => LoweredTy::Ptr,
                    FieldTy::Nominal(nested_gid) => LoweredTy::Aggregate(*nested_gid),
                };
                Some(self.emit_typed(Opcode::FieldLoad, base_reg, Register(0), result_ty, offset))
            }
            // Indexing `base[index]`. A *tensor* base rank-reduces along its outermost dimension (a
            // remaining shape yields a row/sub-view tensor, an empty one the scalar element; chained
            // `q[i][j]` recurses). A *raw-pointer* base (`self.data[i]` where `self.data : *mut T`)
            // GEP-loads the pointee element — the element type recovered from the base's AST type,
            // since the layout erases a pointer's pointee (#242).
            Expr::IndexAccess(ix) => {
                let base = self.lower_expr(&ix.base)?;
                match &base.ty {
                    LoweredTy::Tensor { elem, shape } => {
                        let (elem, shape) = (elem.clone(), shape.clone());
                        if shape.is_empty() {
                            return None; // cannot index a rank-0 value
                        }
                        let index = self.lower_expr(&ix.index)?;
                        if !matches!(index.ty, LoweredTy::Scalar(_)) {
                            return None; // index must be a scalar
                        }
                        let reduced: Vec<String> = shape[1..].to_vec();
                        let result_ty = if reduced.is_empty() {
                            LoweredTy::Scalar(elem)
                        } else {
                            LoweredTy::Tensor {
                                elem,
                                shape: reduced,
                            }
                        };
                        Some(self.emit_typed(
                            Opcode::TensorIndex,
                            base.reg,
                            index.reg,
                            result_ty,
                            0,
                        ))
                    }
                    LoweredTy::Ptr => {
                        let elem = pointer_elem_ty(&self.infer_ast_type(&ix.base)?, self.registry)?;
                        let index = self.lower_expr(&ix.index)?;
                        if !matches!(index.ty, LoweredTy::Scalar(_)) {
                            return None; // index must be a scalar
                        }
                        Some(self.emit_typed(Opcode::PtrIndex, base.reg, index.reg, elem, 0))
                    }
                    _ => None,
                }
            }
            // `transfer(src, Memory::X)`: re-home a tensor into another memory space. The result has
            // the same element + shape, so the destination is sized to hold the source ("enough
            // storage on the receiving side").
            Expr::Transfer(t) => {
                let src = self.lower_expr(&t.expr)?;
                let LoweredTy::Tensor { elem, shape } = &src.ty else {
                    return None; // only tensors transfer
                };
                let result_ty = LoweredTy::Tensor {
                    elem: elem.clone(),
                    shape: shape.clone(),
                };
                let mem_id = crate::arch::memory_space_dispatch_id(&t.space) as u64;
                Some(self.emit_typed(Opcode::Transfer, src.reg, Register(0), result_ty, mem_id))
            }
            // Slice reductions `dot`/`sum`/`max`/`min` over rank-1 tensor slices -> a scalar. `dot`
            // takes two slices (fused multiply then reduce-add); the rest take one. `Tensor<T>([..])`
            // allocates a buffer. Any other name is an ordinary function call.
            Expr::FunctionCall(fc) => {
                if fc.name.as_ref() == "Tensor" {
                    return self.lower_tensor_alloc(fc);
                }
                let kind = match fc.name.as_ref() {
                    "dot" => 0u64,
                    "sum" => 1,
                    "max" => 2,
                    "min" => 3,
                    _ => return self.lower_call(fc),
                };
                let arity = if kind == 0 { 2 } else { 1 };
                if fc.args.len() != arity {
                    return None;
                }
                let mut regs = [Register(0); 2];
                let mut elem: Option<ElementType> = None;
                for (n, arg) in fc.args.iter().enumerate() {
                    let v = self.lower_expr(arg)?;
                    let LoweredTy::Tensor { elem: e, shape } = &v.ty else {
                        return None; // reductions are over tensor slices
                    };
                    if shape.len() != 1 {
                        return None; // a rank-1 slice reduces to a scalar; higher ranks don't
                    }
                    elem = Some(e.clone());
                    regs[n] = v.reg;
                }
                let elem = elem?;
                Some(self.emit_typed(
                    Opcode::Reduce,
                    regs[0],
                    regs[1],
                    LoweredTy::Scalar(elem),
                    kind,
                ))
            }
            // `unsafe { .. }` in value position (e.g. a stdlib wrapper's `return unsafe { sqrtf(self) }`).
            // Safety was checked upstream, so `unsafe` is transparent to lowering: run the block's
            // statements, then yield its trailing value expression.
            Expr::UnsafeBlock(u) => {
                for s in &u.stmts {
                    self.lower_stmt(s)?;
                }
                self.lower_expr(u.ret.as_deref()?)
            }
            // A struct literal in value position (e.g. `return P { .. }`, #215): construct it in a slot
            // (the `let x = P { .. }` form is handled directly in `lower_stmt`). The `Val` is the slot,
            // which a `Ret` loads + returns by value.
            Expr::StructInit(si) => self.lower_struct_init(si),
            // `t.with_memory(Memory::X)` annotates a tensor's home memory space for the seam/type
            // analysis but emits no op — the AST codegen treats it the same way (`with_memory` returns
            // its receiver, `codegen/lower/expr.rs`). So it's transparent to lowering: yield the
            // receiver tensor and drop the memory-space argument. Device-placement *transfers*
            // (`to_device`/`to_host`/…) are already rewritten to `Expr::Transfer` by the type checker,
            // so they never reach here as a method. Any other method declines (#226).
            Expr::MethodCall(mc) if mc.method_name.as_ref() == "with_memory" => {
                self.lower_expr(&mc.base)
            }
            // A `comptime { .. }` block: the AST codegen lowers it *transparently* (its `sizeof<T>()`
            // folds to a constant and its `assert`s are runtime no-ops), so at runtime a compile-time
            // block produces no observable effect. Mirror that — lower the inner statements, then the
            // trailing value (or a discarded dummy for a statement-position block). #228
            Expr::ComptimeBlock(cb) => {
                for s in &cb.stmts {
                    self.lower_stmt(s)?;
                }
                match &cb.ret {
                    Some(r) => self.lower_expr(r),
                    None => Some(self.emit_value(
                        Opcode::Const,
                        Register(0),
                        Register(0),
                        ElementType::I32,
                        0,
                    )),
                }
            }
            // `sizeof<T>()`: a compile-time constant `i64` of `T`'s byte size. Scalars/pointers get
            // their precise size (as the AST codegen does), and an *aggregate* gets its **real layout
            // size** from the registry — NOT the oracle's `SizeOfExpr::lower` `_ => 8` fallback, which
            // is a genuine bug: an element buffer sized by `sizeof<Vec<i32>>()` would under-allocate
            // (`8` vs the real `16`) and write structs out of bounds (UB that only "works" by heap
            // slack). So for `Vec<struct>` the flat path is deliberately *more correct* than the
            // oracle and is validated **standalone** (the oracle can't be a reference for a construct
            // it mis-sizes). Any other unmodelled non-scalar (a tensor, `i128`) still falls back to
            // `8`, matching the oracle where it isn't demonstrably broken. (#242)
            Expr::SizeOf(s) => {
                let size = sizeof_bytes(&s.target_ty)
                    .or_else(|| {
                        agg_gid_of_ty(&s.target_ty, self.registry)
                            .and_then(|g| self.registry.layouts.get(&g))
                            .map(|d| d.size_bytes as u64)
                    })
                    .unwrap_or(8);
                Some(self.emit_value(
                    Opcode::Const,
                    Register(0),
                    Register(0),
                    ElementType::I64,
                    size,
                ))
            }
            // Construct an enum value. A payload-free (C-like) variant (`Color::Green`, #227) is a bare
            // `i32` discriminant. A data-carrying variant (`Option<i32>::Some(30)` / `::None`, #242) is
            // a `{ i32 tag, payload }` aggregate — see `lower_enum_construct`.
            Expr::EnumVariant(ev) => {
                if let Some(variants) = self.registry.enum_variants.get(&ev.enum_name) {
                    if ev.payload.as_ref().is_none_or(|p| p.is_empty()) {
                        let ordinal = variants.iter().position(|v| v == &ev.variant_name)? as u64;
                        return Some(self.emit_value(
                            Opcode::Const,
                            Register(0),
                            Register(0),
                            ElementType::I32,
                            ordinal,
                        ));
                    }
                }
                self.lower_enum_construct(ev)
            }
            // `&<expr>`: a borrow. A tensor is a memref — already a reference value — so borrowing it
            // is transparent: yield the tensor itself, matching the AST codegen (`BorrowExpr` returns
            // the memref for an allocated tensor identifier). This backs `print(&t)`. Borrowing a
            // *local* that lives in a slot (`&v` where `v` is a struct slot, or `&x` where `x` is an
            // address-taken scalar demoted to a slot by the pre-pass, #230) yields the slot's
            // `!llvm.ptr` — the pointer a `&Vec<T>` method receives so `v.len()` lowers (#242), or the
            // `&i32` a `pick(a : &i32, ..)` argument passes. Only scalars/aggregates are ever slotted,
            // so the slot register is the pointee's address in both cases.
            Expr::Borrow(b) => {
                if let Expr::Identifier(id) = &*b.expr {
                    if let Some(Binding::Slot { reg, .. }) = self.scope.get(&id.name).cloned() {
                        return Some(Val {
                            reg,
                            ty: LoweredTy::Ptr,
                        });
                    }
                }
                if let Expr::MemberAccess(m) = &*b.expr {
                    // `&outer.inner`: the address of a by-value nested-aggregate field — a method
                    // receiver (`self.iter.next()` -> `&self.iter`) or a nested `&o.inner`.
                    // `lower_agg_base` GEPs to the field (a `FieldAddr`); the pointer is the borrow. (#242)
                    if let Some((reg, _)) = self.lower_agg_base(&b.expr) {
                        return Some(Val {
                            reg,
                            ty: LoweredTy::Ptr,
                        });
                    }
                    // `&param.scalar` — the address of a *scalar* field (a reference return
                    // `probe(m : &Map) -> &i32 { return &m.slot; }`, #275 M3b). `lower_agg_base` only
                    // addresses nested-aggregate fields; a scalar field GEPs to its element pointer here.
                    // Safety is the borrow checker's (return-provenance, #243): the flat path only emits
                    // the address the frontend already proved outlives the callee.
                    if let Some(val) = self.lower_scalar_field_addr(m) {
                        return Some(val);
                    }
                }
                let v = self.lower_expr(&b.expr)?;
                if matches!(v.ty, LoweredTy::Tensor { .. }) {
                    Some(v)
                } else {
                    None
                }
            }
            // A value-position `if` in expression context: nested (`if c { if d { .. } else { .. } }
            // else { .. }`), a call argument, an implicit return (the parser rewrites a trailing `if`
            // to `return if ..`), or a compound-assign RHS (#229). Infer the result type from the
            // then-branch's trailing value, allocate a slot, store each branch's value into it, and
            // load the result. (The annotated `let v: T = if ..` form uses its annotation directly in
            // `lower_stmt`, a more precise path that this does not replace.)
            Expr::If(if_expr) => {
                let result_ty = self.infer_block_ty(&if_expr.then_block)?;
                let slot = self.emit_alloca(result_ty.clone());
                self.lower_if_into_slot(if_expr, slot.reg)?;
                Some(self.emit_typed(Opcode::SlotLoad, slot.reg, Register(0), result_ty, 0))
            }
            // A string literal in value position (`let s = "…"`, a string function argument): emit the
            // module-level global + `addressof`, yielding a first-class `!llvm.ptr` value — the same
            // shape the AST path's `StringLiteralExpr` produces. Backs the string-passing FFI programs
            // (`vx_stdout_write(msg, 14)`). A print-*position* string never reaches here; it takes the
            // `PrintStr` effect path in `lower_print_arg`. (#231)
            Expr::StringLiteral(sl) => Some(self.emit_string_const(sl.value.as_ref())),
            // `*p`: a raw-pointer dereference read, lowered as `p[0]` (`let v = *p`, a `Box`'s heap
            // cell). The store form (`*p = val`) is in `lower_stmt`'s assignment. (#242)
            Expr::Dereference(d) => {
                // `*r` where `r` is a non-escaping place resolves by *re-lowering the borrowed place
                // expression* as a value — no address is materialized (§5): `Identifier(x)` reads the
                // local directly (Example A), `MemberAccess(p.x)` `FieldLoad`s the field (Example B/C).
                // Any other `*p` is a real pointer deref. (#275)
                if let Expr::Identifier(id) = &*d.expr {
                    if let Some(Binding::Place { place, .. }) = self.scope.get(&id.name).cloned() {
                        return self.lower_expr(&place);
                    }
                }
                self.lower_ptr_deref(&d.expr, false)
            }
            // Short-circuit `&&` / `||` (#239): a branch skeleton producing a `bool`.
            Expr::LogicalOp(l) => self.lower_logical(l),
            // A value array literal `[a, b, c]` (#239): a rank-1 tensor buffer with the elements
            // stored into it. (A `Tensor<T>([…])` shape argument is consumed by `lower_tensor_alloc`.)
            Expr::Array(arr) => self.lower_array(arr),
            other => {
                if std::env::var("VX_FLAT_DBG").is_ok() {
                    eprintln!("[flat-dbg]   unsupported expr: {}", expr_kind(other));
                }
                None
            }
        }
    }

    /// Infer the [`LoweredTy`] of an expression from the AST + current scope, *without emitting* — so
    /// the result slot of a value-position `if` can be sized before its branches are lowered (#229).
    /// Covers the scalar-producing forms that appear as a branch's trailing value; anything else
    /// returns `None`, declining the value-`if`.
    fn infer_expr_ty(&self, e: &Expr) -> Option<LoweredTy> {
        match e {
            Expr::Number(n) => Some(LoweredTy::Scalar(number_elem(n)?)),
            Expr::Identifier(id) => match self.scope.get(&id.name)? {
                Binding::Reg(v) => Some(v.ty.clone()),
                Binding::Slot { ty, .. } => Some(ty.clone()),
                // A place read as a value has no value type here — decline the inference. (#275)
                Binding::Place { .. } => None,
            },
            Expr::BinaryOp(b) => {
                // Mirror `lower_expr`: the result is the tensor side if either operand is a tensor,
                // else the left operand's type.
                let l = self.infer_expr_ty(&b.lhs)?;
                if matches!(l, LoweredTy::Tensor { .. }) {
                    return Some(l);
                }
                let r = self.infer_expr_ty(&b.rhs)?;
                if matches!(r, LoweredTy::Tensor { .. }) {
                    return Some(r);
                }
                Some(l)
            }
            Expr::UnaryOp(u) => self.infer_expr_ty(&u.expr),
            Expr::RelationalOp(_) => Some(LoweredTy::Scalar(ElementType::Bool)),
            Expr::AsCast(c) => Some(LoweredTy::Scalar(scalar_of(&c.target_ty)?)),
            Expr::FunctionCall(fc) => {
                let sig = self.registry.fn_sigs.get(&fc.name)?;
                lowered_ty(&sig.ret_ty, self.registry)
            }
            Expr::If(iff) => self.infer_block_ty(&iff.then_block),
            Expr::UnsafeBlock(ub) => self.infer_expr_ty(ub.ret.as_deref()?),
            Expr::ComptimeBlock(cb) => self.infer_expr_ty(cb.ret.as_deref()?),
            _ => None,
        }
    }

    /// The inferred type of a block's trailing semicolon-less value expression (a branch's value).
    fn infer_block_ty(&self, stmts: &[Statement]) -> Option<LoweredTy> {
        match stmts.last()? {
            Statement::ExprStmt(es) if !es.has_semi => self.infer_expr_ty(&es.expr),
            _ => None,
        }
    }

    /// Recover the concrete AST `Type` of an expression from the scope's `ast_types` + the registry's
    /// base struct fields — the flat-path port of the AST codegen's `infer_ast_type`, restricted to
    /// the forms a *pointer element type* flows through (`self.data`, an identifier, an `unsafe`/`as`
    /// wrapper). The load-bearing case is a member access through a monomorphized generic aggregate:
    /// substitute the instance's type arguments into the base struct's declared field type, so
    /// `self.data` on `self : &mut Vec<i32>` resolves to `*mut i32`. `None` for an unhandled form —
    /// the caller then declines the construct, keeping the AST path the oracle. (#242)
    fn infer_ast_type(&self, e: &Expr) -> Option<Type> {
        match e {
            Expr::Identifier(id) => self.ast_types.get(&id.name).cloned(),
            // A numeric literal's own scalar type (`let x = 5` -> `i32`), so an unannotated `let`
            // records its type in `ast_types` and a later chain resolves — `&x : &i32`, `&&x : &&i32`,
            // and a nested `**rr` can recover the pointee element. Without this the chain breaks at the
            // literal and only single-level refs (which take the place shortcut) worked. (#275 nested refs)
            Expr::Number(n) => Some(Type::Scalar(number_elem(n)?)),
            Expr::MemberAccess(m) => {
                let base_ty = self.infer_ast_type(&m.base)?;
                let pointee = deref_to_pointee(&base_ty);
                let (_base_name, args) = nominal_name_and_args(pointee)?;
                // GID-routed since #291 (attached GID, else unambiguous bare name) — an ambiguous
                // name declines here, keeping the AST path the oracle.
                let decl = self.registry.struct_fields_of(pointee)?;
                let (_, fty) = decl
                    .fields
                    .iter()
                    .find(|(n, _)| n.as_ref() == m.member.as_ref())?;
                Some(substitute_generics(fty, &decl.generics, &args))
            }
            // A struct literal has its named type (`let adder = |x| ..` -> the generated
            // `Closure_N` struct), so a closure local passed to `.map` is recognized as a closure. (#242)
            Expr::StructInit(si) => Some(Type::Struct(si.name.clone(), si.type_id)),
            Expr::UnsafeBlock(u) => self.infer_ast_type(u.ret.as_deref()?),
            Expr::AsCast(c) => Some(c.target_ty.clone()),
            // `&e` has a borrow type over `e`'s type (`&x` where `x : i32` -> `&i32`), so an
            // unannotated `let r = &x` types `r` as `&i32` and a later `*r` recovers `i32`. (#230)
            Expr::Borrow(b) => Some(Type::Borrow {
                inner: Box::new(self.infer_ast_type(&b.expr)?),
                mem_space: None,
                is_mut: false,
                region_id: 0,
            }),
            // `*p` has the pointee type (`p : *mut i32` -> `i32`).
            Expr::Dereference(d) => Some(deref_to_pointee(&self.infer_ast_type(&d.expr)?).clone()),
            // A call's type is its callee's return type (`v.iter()` -> `Vec$iter`'s `VecIter<i32>`),
            // for typing a `for x in v.iter()` iterable.
            Expr::FunctionCall(fc) => self
                .registry
                .fn_sigs
                .get(fc.name.as_ref())
                .map(|s| s.ret_ty.clone()),
            _ => None,
        }
    }

    /// Resolve a member-access base to the `!llvm.ptr` register addressing the aggregate plus its
    /// layout GID: either a local bound to an aggregate *slot* (`let p = Point { .. }`, #215) or a
    /// *pointer to* an aggregate (`self : &mut Vec<i32>`, #242). Both are an `!llvm.ptr` to the
    /// struct, so a `FieldLoad`/`FieldStore` addresses them identically. `None` for any other base.
    fn lower_agg_base(&mut self, base: &Expr) -> Option<(Register, TypeId)> {
        if let Expr::Identifier(id) = base {
            if let Some(Binding::Slot {
                reg,
                ty: LoweredTy::Aggregate(gid),
            }) = self.scope.get(&id.name).cloned()
            {
                return Some((reg, gid));
            }
        }
        // `(*p).field`: a field access through a *dereferenced* pointer (`(*self.vec).len` in
        // `VecIter::next`). The deref is transparent in field-access position — yield the pointer `p`
        // itself (loading `p`, e.g. `self.vec`, a `*const Vec<T>` field, to an `!llvm.ptr`) plus the
        // pointee aggregate's layout GID, so the following field op GEPs through it. (#242)
        if let Expr::Dereference(d) = base {
            let base_ty = self.infer_ast_type(base)?; // the pointee (Vec<T>)
            let gid = agg_gid_of_ty(&base_ty, self.registry)?;
            let v = self.lower_expr(&d.expr)?; // lower the pointer, not the deref
            return matches!(v.ty, LoweredTy::Ptr).then_some((v.reg, gid));
        }
        // A by-value nested-aggregate field as a base (`outer.inner.a`, or `&self.iter` as a method
        // receiver): recurse to the enclosing aggregate's pointer, then GEP to the nested field via a
        // `FieldAddr`. The nested field must itself be a modelled aggregate. (#242)
        if let Expr::MemberAccess(m) = base {
            if let Some((parent_reg, parent_gid)) = self.lower_agg_base(&m.base) {
                let field = self
                    .registry
                    .layouts
                    .get(&parent_gid)?
                    .fields
                    .iter()
                    .find(|f| f.name.as_ref() == m.member.as_ref())?;
                if let FieldTy::Nominal(nested_gid) = field.ty {
                    let offset = field.offset as u64;
                    let type_idx = TypeIdx(self.types.len() as u32);
                    self.types.push(nested_gid);
                    let reg = Register(self.code.len() as u32);
                    self.code.push(HirInstruction::new(
                        Opcode::FieldAddr,
                        parent_reg,
                        Register(0),
                        type_idx,
                        offset,
                    ));
                    return Some((reg, nested_gid));
                }
            }
        }
        // A pointer to an aggregate: the base lowers to a pointer value; its pointee layout GID comes
        // from the base's AST type (the layout the frozen registry keyed under the base nominal).
        let base_ty = self.infer_ast_type(base)?;
        let gid = agg_gid_of_ty(&base_ty, self.registry)?;
        let v = self.lower_expr(base)?;
        matches!(v.ty, LoweredTy::Ptr).then_some((v.reg, gid))
    }

    /// The address of a **scalar** field as an element pointer (`&param.slot`): GEP the parent aggregate
    /// (resolved by `lower_agg_base`, so a param `&Map`, a local slot, or a nested aggregate all work) to
    /// the field and yield a `!llvm.ptr`. The result type GID is the field's *scalar* GID so codegen
    /// tracks it as a plain pointer (not an aggregate slot). `None` when the parent doesn't resolve or the
    /// field is not scalar (a nested-aggregate field is the `lower_agg_base` path instead). (#275 M3b)
    fn lower_scalar_field_addr(&mut self, m: &crate::syntax::MemberAccessExpr) -> Option<Val> {
        let (parent_reg, parent_gid) = self.lower_agg_base(&m.base)?;
        // Snapshot the field's offset + element so the immutable registry borrow ends before we emit.
        let (offset, elem) = {
            let field = self
                .registry
                .layouts
                .get(&parent_gid)?
                .fields
                .iter()
                .find(|f| f.name.as_ref() == m.member.as_ref())?;
            match &field.ty {
                FieldTy::Scalar(e) => (field.offset as u64, e.clone()),
                _ => return None,
            }
        };
        let type_idx = TypeIdx(self.types.len() as u32);
        self.types.push(scalar_gid(&elem));
        let reg = Register(self.code.len() as u32);
        self.code.push(HirInstruction::new(
            Opcode::FieldAddr,
            parent_reg,
            Register(0),
            type_idx,
            offset,
        ));
        Some(Val {
            reg,
            ty: LoweredTy::Ptr,
        })
    }

    /// Lower an `if`/`else` statement to basic blocks + branches (memory mode only, so mutated or
    /// cross-block locals are already in slots). Early `return` in a branch is honored: the trailing
    /// `Br` to the merge is skipped when the branch already terminated.
    fn lower_if(&mut self, e: &crate::syntax::IfExpr) -> Option<()> {
        let cond = self.lower_expr(&e.cond)?;
        let then_b = self.new_block();
        let (else_b, merge_b) = match &e.else_block {
            Some(_) => (self.new_block(), self.new_block()),
            None => {
                let m = self.new_block();
                (m, m) // no else: the "else" edge goes straight to the merge block
            }
        };
        self.emit_effect(
            Opcode::CondBr,
            cond.reg,
            Register(0),
            pack_targets(then_b, else_b),
        );

        // then block
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), then_b as u64);
        for s in &e.then_block {
            self.lower_stmt(s)?;
        }
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), merge_b as u64);
        }

        // else block (only when distinct from the merge)
        if let Some(else_stmts) = &e.else_block {
            self.emit_effect(Opcode::BlockStart, Register(0), Register(0), else_b as u64);
            for s in else_stmts {
                self.lower_stmt(s)?;
            }
            if !self.block_terminated() {
                self.emit_effect(Opcode::Br, Register(0), Register(0), merge_b as u64);
            }
        }

        // merge block — subsequent statements continue here
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), merge_b as u64);
        Some(())
    }

    /// Lower a *statement-form* `match <subject> { <arms> }` over a payload-free enum (#227). The
    /// subject is an `i32` discriminant; each `EnumVariant` arm becomes a `cmp subject == ordinal` +
    /// conditional branch to the arm body (taken) or the next arm's test (else) — the same eq-compare
    /// chain the AST codegen emits. A `Wildcard` arm is the unconditional default. Data-carrying
    /// patterns (payload bindings), literal/identifier patterns, and value-producing `match` decline.
    fn lower_match(&mut self, m: &crate::syntax::MatchExpr) -> Option<()> {
        // A data-carrying enum match (`match o { Option<i32>::Some(v) => .. None => .. }`, #242): the
        // subject is a `{ tag, payload }` aggregate, dispatched on its tag field. Detected from the
        // first `EnumVariant` pattern naming an enum with a payload-carrying variant.
        if let Some(pattern_en) = m.arms.iter().find_map(|a| match &a.pattern {
            crate::syntax::Pattern::EnumVariant(en, _, _) => {
                let (base, _) = parse_enum_instance(en);
                let has_payload = self
                    .registry
                    .enum_data
                    .get(base.as_str())
                    .is_some_and(|d| d.variants.iter().any(|(_, p)| !p.is_empty()));
                has_payload.then(|| en.to_string())
            }
            _ => None,
        }) {
            // The concrete instance name comes from the *subject's* type (`match *self` on
            // `&Option<i32>` -> `Option<i32>`): monomorphization substitutes the receiver's type args
            // in the signature but not in the body's match patterns, which keep the generic spelling
            // (`Option<T>`). Fall back to the pattern's spelling when the subject type is unavailable.
            let enum_name = self
                .infer_ast_type(&m.expr)
                .map(|t| deref_to_pointee(&t).to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or(pattern_en);
            return self.lower_data_match(m, &enum_name);
        }
        let subj = self.lower_expr(&m.expr)?;
        if !matches!(subj.ty, LoweredTy::Scalar(ElementType::I32)) {
            return None; // only payload-free enums (a bare i32 discriminant)
        }
        let merge = self.new_block();
        for arm in &m.arms {
            match &arm.pattern {
                crate::syntax::Pattern::EnumVariant(enum_name, variant, payload) => {
                    if payload.as_ref().is_some_and(|p| !p.is_empty()) {
                        return None; // tagged-union payload binding not modelled
                    }
                    // Absent from `enum_variants` => a data-carrying (or generic) enum: decline.
                    let variants = self.registry.enum_variants.get(enum_name)?;
                    let ordinal = variants.iter().position(|v| v == variant)? as u64;
                    let tag = self.emit_value(
                        Opcode::Const,
                        Register(0),
                        Register(0),
                        ElementType::I32,
                        ordinal,
                    );
                    let cond = self.emit_value(
                        Opcode::Cmp,
                        subj.reg,
                        tag.reg,
                        ElementType::Bool,
                        rel_code(&RelationalOp::Eq),
                    );
                    let body_b = self.new_block();
                    let next_b = self.new_block();
                    self.emit_effect(
                        Opcode::CondBr,
                        cond.reg,
                        Register(0),
                        pack_targets(body_b, next_b),
                    );
                    self.emit_effect(Opcode::BlockStart, Register(0), Register(0), body_b as u64);
                    for s in &arm.body {
                        self.lower_stmt(s)?;
                    }
                    if !self.block_terminated() {
                        self.emit_effect(Opcode::Br, Register(0), Register(0), merge as u64);
                    }
                    // Subsequent arm tests continue in the else block.
                    self.emit_effect(Opcode::BlockStart, Register(0), Register(0), next_b as u64);
                }
                crate::syntax::Pattern::Wildcard => {
                    for s in &arm.body {
                        self.lower_stmt(s)?;
                    }
                    if !self.block_terminated() {
                        self.emit_effect(Opcode::Br, Register(0), Register(0), merge as u64);
                    }
                    // A wildcard is the default; any later arm is unreachable.
                    break;
                }
                _ => return None, // literal / identifier patterns not supported
            }
        }
        // No wildcard matched: the final else block falls through to the merge.
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), merge as u64);
        }
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), merge as u64);
        Some(())
    }

    /// Resolve the subject of a data-carrying-enum `match` to a pointer addressing its `{ tag, payload }`
    /// aggregate. Three shapes: `match o` (a local aggregate slot, via `lower_agg_base`); `match *self`
    /// (an `&Option<T>` receiver in an `Option` method — the pointer value *is* the slot, and the emitter
    /// tags the pointer param's pointee with the synthetic enum GID); and `match <value>` (a by-value
    /// enum — a call result or a by-value `self` — spilled to a fresh slot so its fields are addressable).
    /// The `*self`/by-value forms are what let `Option::is_none`/`is_some`/`unwrap` lower on flat. (#242)
    fn lower_data_match_slot(&mut self, subj: &Expr, gid: TypeId) -> Option<Register> {
        // `match *self`: the dereferenced pointer *is* the aggregate slot.
        if let Expr::Dereference(d) = subj {
            let v = self.lower_expr(&d.expr)?;
            return matches!(v.ty, LoweredTy::Ptr).then_some(v.reg);
        }
        // `match o`: a local aggregate slot (or a self-pointer to one).
        if let Some((slot, _)) = self.lower_agg_base(subj) {
            return Some(slot);
        }
        // `match <value>`: a by-value enum aggregate — spill it to a slot to address its fields.
        let v = self.lower_expr(subj)?;
        if matches!(v.ty, LoweredTy::Aggregate(_)) {
            let slot = self.emit_alloca(LoweredTy::Aggregate(gid));
            self.emit_effect(Opcode::Store, slot.reg, v.reg, 0);
            return Some(slot.reg);
        }
        None
    }

    /// Lower a statement-form `match` over a *data-carrying* enum aggregate (`Option<i32>`): resolve
    /// the subject to its `{ tag, payload }` slot, then for each `EnumVariant` arm compare the loaded
    /// tag against the variant's ordinal and, in the taken block, bind each payload pattern to the
    /// loaded payload field before running the arm body — the same tag-dispatch + `extractvalue` the
    /// AST codegen emits, but through the flat aggregate machinery. (#242)
    fn lower_data_match(&mut self, m: &crate::syntax::MatchExpr, enum_name: &str) -> Option<()> {
        let (gid, offsets, payload_types) = self.enum_instance_layout(enum_name)?;
        let slot = self.lower_data_match_slot(&m.expr, gid)?;
        let tag_off = *offsets.first()?;
        let (base, _) = parse_enum_instance(enum_name);
        let data = self.registry.enum_data.get(base.as_str())?.clone();
        let merge = self.new_block();
        for arm in &m.arms {
            match &arm.pattern {
                crate::syntax::Pattern::EnumVariant(_, variant, payload_pats) => {
                    let ordinal = data
                        .variants
                        .iter()
                        .position(|(n, _)| n.as_ref() == variant.as_ref())?
                        as u64;
                    let tag = self.emit_value(
                        Opcode::FieldLoad,
                        slot,
                        Register(0),
                        ElementType::I32,
                        tag_off,
                    );
                    let tagc = self.emit_value(
                        Opcode::Const,
                        Register(0),
                        Register(0),
                        ElementType::I32,
                        ordinal,
                    );
                    let cond = self.emit_value(
                        Opcode::Cmp,
                        tag.reg,
                        tagc.reg,
                        ElementType::Bool,
                        rel_code(&RelationalOp::Eq),
                    );
                    let body_b = self.new_block();
                    let next_b = self.new_block();
                    self.emit_effect(
                        Opcode::CondBr,
                        cond.reg,
                        Register(0),
                        pack_targets(body_b, next_b),
                    );
                    self.emit_effect(Opcode::BlockStart, Register(0), Register(0), body_b as u64);
                    // Bind each payload pattern to its field (`Some(v)` -> `v = <payload>`).
                    if let Some(pats) = payload_pats {
                        for (i, pat) in pats.iter().enumerate() {
                            if let crate::syntax::Pattern::Identifier(pname) = pat {
                                let poff = *offsets.get(i + 1)?;
                                let lty = lowered_ty(payload_types.get(i)?, self.registry)?;
                                let pval = self.emit_typed(
                                    Opcode::FieldLoad,
                                    slot,
                                    Register(0),
                                    lty,
                                    poff,
                                );
                                self.bind_local(pname.clone(), pval);
                            }
                        }
                    }
                    for s in &arm.body {
                        self.lower_stmt(s)?;
                    }
                    if !self.block_terminated() {
                        self.emit_effect(Opcode::Br, Register(0), Register(0), merge as u64);
                    }
                    self.emit_effect(Opcode::BlockStart, Register(0), Register(0), next_b as u64);
                }
                crate::syntax::Pattern::Wildcard => {
                    for s in &arm.body {
                        self.lower_stmt(s)?;
                    }
                    if !self.block_terminated() {
                        self.emit_effect(Opcode::Br, Register(0), Register(0), merge as u64);
                    }
                    break;
                }
                _ => return None,
            }
        }
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), merge as u64);
        }
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), merge as u64);
        Some(())
    }

    /// Lower a value-position `if` into `slot`: each branch stores its trailing value into `slot`,
    /// then the merge block continues (a following `SlotLoad` yields the result). A value `if` must be
    /// total, so an `else` is required. (#201)
    fn lower_if_into_slot(&mut self, e: &crate::syntax::IfExpr, slot: Register) -> Option<()> {
        let else_stmts = e.else_block.as_ref()?;
        let cond = self.lower_expr(&e.cond)?;
        let then_b = self.new_block();
        let else_b = self.new_block();
        let merge_b = self.new_block();
        self.emit_effect(
            Opcode::CondBr,
            cond.reg,
            Register(0),
            pack_targets(then_b, else_b),
        );

        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), then_b as u64);
        self.lower_block_into_slot(&e.then_block, slot)?;
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), merge_b as u64);
        }

        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), else_b as u64);
        self.lower_block_into_slot(else_stmts, slot)?;
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), merge_b as u64);
        }

        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), merge_b as u64);
        Some(())
    }

    /// Lower a branch block whose trailing semicolon-less expression is the branch's value, stored into
    /// `slot`. Leading statements lower normally; declines if the block has no trailing value. Both
    /// branches of a value-`if` already agree on the result type — the checker reconciles them and
    /// types a literal branch to the result (no implicit conversion, #240) — so the value is stored
    /// directly.
    fn lower_block_into_slot(&mut self, stmts: &[Statement], slot: Register) -> Option<()> {
        let n = stmts.len();
        for (i, s) in stmts.iter().enumerate() {
            if i + 1 == n {
                if let Statement::ExprStmt(es) = s {
                    if !es.has_semi {
                        let v = self.lower_expr(&es.expr)?;
                        self.emit_effect(Opcode::Store, slot, v.reg, 0);
                        return Some(());
                    }
                }
                return None; // last stmt isn't a trailing value expression
            }
            self.lower_stmt(s)?;
        }
        None // empty branch has no value
    }

    /// Lower a short-circuit logical op (`a && b`, `a || b`) to the same branch skeleton the AST
    /// codegen uses (#239): evaluate the left operand, and only evaluate the right when it can change
    /// the result — otherwise take the short-circuit constant. The `bool` result flows through a slot
    /// (as the value-`if` does), so this needs the memory model; `body_has_control_flow` forces a
    /// function containing a logical op into it, so the `!self.memory` guard is a defensive decline.
    fn lower_logical(&mut self, e: &crate::syntax::LogicalOpExpr) -> Option<Val> {
        if !self.memory {
            return None;
        }
        let slot = self.emit_alloca(LoweredTy::Scalar(ElementType::Bool));
        let lhs = self.lower_expr(&e.lhs)?;
        let rhs_b = self.new_block();
        let short_b = self.new_block();
        let merge_b = self.new_block();
        // `&&`: left true -> evaluate right; left false -> short-circuit to `false`.
        // `||`: left true -> short-circuit to `true`; left false -> evaluate right.
        let (then_b, else_b) = match e.op {
            LogicalOp::And => (rhs_b, short_b),
            LogicalOp::Or => (short_b, rhs_b),
        };
        self.emit_effect(
            Opcode::CondBr,
            lhs.reg,
            Register(0),
            pack_targets(then_b, else_b),
        );

        // The right operand determines the result.
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), rhs_b as u64);
        let rhs = self.lower_expr(&e.rhs)?;
        self.emit_effect(Opcode::Store, slot.reg, rhs.reg, 0);
        self.emit_effect(Opcode::Br, Register(0), Register(0), merge_b as u64);

        // The short-circuit constant: `false` for `&&`, `true` for `||`.
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), short_b as u64);
        let imm = match e.op {
            LogicalOp::And => 0,
            LogicalOp::Or => 1,
        };
        let konst = self.emit_value(
            Opcode::Const,
            Register(0),
            Register(0),
            ElementType::Bool,
            imm,
        );
        self.emit_effect(Opcode::Store, slot.reg, konst.reg, 0);
        self.emit_effect(Opcode::Br, Register(0), Register(0), merge_b as u64);

        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), merge_b as u64);
        Some(self.emit_value(
            Opcode::SlotLoad,
            slot.reg,
            Register(0),
            ElementType::Bool,
            0,
        ))
    }

    /// Lower an infinite `loop { body }`: a header block the body branches back to, plus an exit
    /// block that `break` targets. `continue` re-enters the header.
    fn lower_loop(&mut self, body: &[Statement]) -> Option<()> {
        let header = self.new_block();
        let exit = self.new_block();
        self.emit_effect(Opcode::Br, Register(0), Register(0), header as u64);
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), header as u64);
        self.loop_stack.push((header, exit)); // continue -> header, break -> exit
        for s in body {
            self.lower_stmt(s)?;
        }
        self.loop_stack.pop();
        // Back-edge, unless the body already terminated every path (e.g. ended in `break`/`return`).
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), header as u64);
        }
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), exit as u64);
        Some(())
    }

    /// Lower `for i in a..b { body }` over a scalar exclusive range. The induction variable and the
    /// (once-evaluated) bound live in slots so they cross blocks; `continue` targets the increment
    /// latch (so it doesn't skip the step), `break` the exit.
    fn lower_for(&mut self, f: &crate::syntax::ForLoopStmt) -> Option<()> {
        let Expr::Range(range) = &*f.iterable else {
            // A non-range iterable is an iterator (`for x in v.iter()`): the sugar over
            // `loop { match it.next() { Some(x) => body, None => break } }`. (#242)
            return self.lower_for_iterator(f);
        };
        let start = self.lower_expr(&range.start)?;
        let end = self.lower_expr(&range.end)?;
        let elem = match &start.ty {
            LoweredTy::Scalar(e) => e.clone(),
            // ranges are over scalars
            LoweredTy::Aggregate(_) | LoweredTy::Tensor { .. } | LoweredTy::Ptr => return None,
        };
        // Induction variable `i` and the loop bound both need to survive across blocks -> slots.
        let i_slot = self.emit_alloca(LoweredTy::Scalar(elem.clone()));
        self.emit_effect(Opcode::Store, i_slot.reg, start.reg, 0);
        let end_slot = self.emit_alloca(LoweredTy::Scalar(elem.clone()));
        self.emit_effect(Opcode::Store, end_slot.reg, end.reg, 0);
        self.scope.insert(
            f.iter.as_str().into(),
            Binding::Slot {
                reg: i_slot.reg,
                ty: LoweredTy::Scalar(elem.clone()),
            },
        );

        let header = self.new_block();
        let body_b = self.new_block();
        let latch = self.new_block();
        let exit = self.new_block();

        self.emit_effect(Opcode::Br, Register(0), Register(0), header as u64);
        // header: cond = i < end
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), header as u64);
        let i_val = self.emit_value(Opcode::SlotLoad, i_slot.reg, Register(0), elem.clone(), 0);
        let end_val = self.emit_value(Opcode::SlotLoad, end_slot.reg, Register(0), elem.clone(), 0);
        let cond = self.emit_value(
            Opcode::Cmp,
            i_val.reg,
            end_val.reg,
            ElementType::Bool,
            rel_code(&RelationalOp::Lt),
        );
        self.emit_effect(
            Opcode::CondBr,
            cond.reg,
            Register(0),
            pack_targets(body_b, exit),
        );

        // body
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), body_b as u64);
        self.loop_stack.push((latch, exit)); // continue -> latch, break -> exit
        for s in &f.body {
            self.lower_stmt(s)?;
        }
        self.loop_stack.pop();
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), latch as u64);
        }

        // latch: i = i + 1; back to header
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), latch as u64);
        let i2 = self.emit_value(Opcode::SlotLoad, i_slot.reg, Register(0), elem.clone(), 0);
        let one = self.emit_value(Opcode::Const, Register(0), Register(0), elem.clone(), 1);
        let inc = self.emit_value(Opcode::Add, i2.reg, one.reg, elem, 0);
        self.emit_effect(Opcode::Store, i_slot.reg, inc.reg, 0);
        self.emit_effect(Opcode::Br, Register(0), Register(0), header as u64);

        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), exit as u64);
        Some(())
    }

    /// Lower `for x in <iterator> { body }` — the sugar over
    /// `loop { match it.next() { Some(x) => body; None => break } }`. The iterator is spilled to a
    /// slot (so `next(&mut it)` can mutate it across iterations); each round calls the monomorphized
    /// `next`, spills its `Option<Element>` result, loads the tag, and dispatches: on `Some` it binds
    /// `x` to the payload and runs the body (branching back to the header), on `None` it exits. This is
    /// the flat-model counterpart of the AST codegen's generic-iterator loop (#242).
    fn lower_for_iterator(&mut self, f: &crate::syntax::ForLoopStmt) -> Option<()> {
        // The iterator value (`v.iter()` -> `VecIter<i32>`) and its monomorphized `next`.
        let iter_ast_ty = self.infer_ast_type(&f.iterable)?;
        let (next_gid, opt_ty) = self.find_iterator_next(&iter_ast_ty)?;
        let (enum_gid, offsets, payload_types) = self.enum_instance_layout(&opt_ty.to_string())?;
        // `Some`'s discriminant ordinal (the payload-carrying variant).
        let (base, _) = parse_enum_instance(&opt_ty.to_string());
        let data = self.registry.enum_data.get(base.as_str())?.clone();
        let some_ord = data.variants.iter().position(|(_, p)| !p.is_empty())? as u64;

        let iter_val = self.lower_expr(&f.iterable)?;
        if !matches!(iter_val.ty, LoweredTy::Aggregate(_)) {
            return None;
        }
        let it_slot = self.emit_alloca(iter_val.ty.clone());
        self.emit_effect(Opcode::Store, it_slot.reg, iter_val.reg, 0);

        let header = self.new_block();
        let body_b = self.new_block();
        let exit = self.new_block();
        self.emit_effect(Opcode::Br, Register(0), Register(0), header as u64);
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), header as u64);

        // `opt = next(&mut it)`: the iterator slot pointer is the `&mut self` argument.
        self.emit_effect(Opcode::Arg, it_slot.reg, Register(0), 0);
        let type_idx = TypeIdx(self.types.len() as u32);
        self.types.push(next_gid);
        let call_reg = Register(self.code.len() as u32);
        self.code.push(HirInstruction::new(
            Opcode::Call,
            Register(0),
            Register(0),
            type_idx,
            1,
        ));
        // Spill the returned `Option` to a slot so its tag/payload fields are addressable.
        let opt_slot = self.emit_alloca(LoweredTy::Aggregate(enum_gid));
        self.emit_effect(Opcode::Store, opt_slot.reg, call_reg, 0);

        // Dispatch on the tag: `Some` -> body, else -> exit.
        let tag = self.emit_value(
            Opcode::FieldLoad,
            opt_slot.reg,
            Register(0),
            ElementType::I32,
            *offsets.first()?,
        );
        let some_c = self.emit_value(
            Opcode::Const,
            Register(0),
            Register(0),
            ElementType::I32,
            some_ord,
        );
        let cond = self.emit_value(
            Opcode::Cmp,
            tag.reg,
            some_c.reg,
            ElementType::Bool,
            rel_code(&RelationalOp::Eq),
        );
        self.emit_effect(
            Opcode::CondBr,
            cond.reg,
            Register(0),
            pack_targets(body_b, exit),
        );

        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), body_b as u64);
        // Bind the loop variable to the payload.
        let lty = lowered_ty(payload_types.first()?, self.registry)?;
        let x = self.emit_typed(
            Opcode::FieldLoad,
            opt_slot.reg,
            Register(0),
            lty,
            *offsets.get(1)?,
        );
        self.bind_local(f.iter.as_str().into(), x);
        self.loop_stack.push((header, exit)); // continue -> header, break -> exit
        for s in &f.body {
            self.lower_stmt(s)?;
        }
        self.loop_stack.pop();
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), header as u64);
        }
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), exit as u64);
        Some(())
    }

    /// Find the monomorphized `next` for an iterator type (`VecIter<i32>` -> `VecIter$i32$next$i32`):
    /// its callee GID and `Option<Element>` return type. Matches a `fn_sig` whose name shares the
    /// iterator's base and carries a `next` method (the mangler uses `$`). (#242)
    fn find_iterator_next(&self, iter_ty: &Type) -> Option<(TypeId, Type)> {
        let (base, _) = nominal_name_and_args(deref_to_pointee(iter_ty))?;
        for (name, sig) in &self.registry.fn_sigs {
            let n = name.as_ref();
            if n.starts_with(base.as_ref()) && (n.contains("$next$") || n.ends_with("$next")) {
                return Some((sig.gid, sig.ret_ty.clone()));
            }
        }
        None
    }

    /// Lower `spawn on (<topology>) { body }` into a `Spawn`/`SpawnEnd`-delimited region carrying the
    /// topology dispatch id. The body may use control flow (`for`/`loop`/`if`) and the enclosing
    /// function may be in memory mode — the flat emitter materializes the body as the `vx.spawn` op's
    /// nested MLIR region, so the region's blocks are self-contained (#226). A *value-producing* spawn
    /// (a yielded result) still declines: it needs `vx.yield` with a result plus threading the spawn's
    /// result value, which the statement-form device corpus doesn't use.
    fn lower_spawn(&mut self, s: &crate::syntax::SpawnOnExpr) -> Option<()> {
        if s.ret.is_some() {
            return None;
        }
        let top_id = crate::arch::topology_dispatch_id(&s.top);
        self.emit_effect(Opcode::Spawn, Register(0), Register(0), top_id as u64);
        for stmt in &s.stmts {
            self.lower_stmt(stmt)?;
        }
        self.emit_effect(Opcode::SpawnEnd, Register(0), Register(0), 0);
        Some(())
    }

    /// Construct a struct literal into a fresh stack slot: `Alloca` the aggregate (sized from its
    /// layout) then `FieldStore` each field at its layout offset. Returns the slot as an aggregate
    /// `Val`. First cut: scalar fields only, and the struct's GID must be annotated (by the type
    /// checker) and its layout computed — otherwise the construction is declined.
    /// Construct a data-carrying enum value (`Option<i32>::Some(30)` / `::None`) as a `{ i32 tag,
    /// payload }` aggregate: `Alloca` the instance's synthesized layout, `FieldStore` the variant's
    /// discriminant ordinal into the tag, then `FieldStore` each payload value into its field (a
    /// payload-free variant like `None` stores only the tag, leaving the payload undefined — as the AST
    /// codegen does). Returns the slot as an aggregate `Val`. (#242)
    fn lower_enum_construct(&mut self, ev: &crate::syntax::EnumVariantExpr) -> Option<Val> {
        let (base, _) = parse_enum_instance(&ev.enum_name);
        let data = self.registry.enum_data.get(base.as_str())?;
        let ordinal =
            data.variants
                .iter()
                .position(|(n, _)| n.as_ref() == ev.variant_name.as_ref())? as u64;
        let (gid, offsets, _) = self.enum_instance_layout(&ev.enum_name)?;
        let slot = self.emit_alloca(LoweredTy::Aggregate(gid));
        let tag = self.emit_value(
            Opcode::Const,
            Register(0),
            Register(0),
            ElementType::I32,
            ordinal,
        );
        self.emit_effect(Opcode::FieldStore, slot.reg, tag.reg, *offsets.first()?);
        if let Some(payload) = &ev.payload {
            for (i, pexpr) in payload.iter().enumerate() {
                let v = self.lower_expr(pexpr)?;
                self.emit_effect(Opcode::FieldStore, slot.reg, v.reg, *offsets.get(i + 1)?);
            }
        }
        Some(slot)
    }

    /// Like the free `lowered_ty`, but for a *data-carrying enum instance* (`Option<i32>`) it
    /// synthesizes and records the instance's `{ tag, payload }` layout (which the free function can't,
    /// lacking `&mut self`) and returns `Aggregate(gid)`. Used where a type may be such an enum — a
    /// call's return (`VecIter::next -> Option<i32>`), a `let` binding. Falls back to `lowered_ty`
    /// for everything else. (#242)
    fn lower_ty_synth(&mut self, ty: &Type) -> Option<LoweredTy> {
        if let Type::GenericInstance(base, _) = ty {
            if let Type::Enum(n, _) | Type::Struct(n, _) = base.as_ref() {
                let is_data = self
                    .registry
                    .enum_data
                    .get(n.as_ref())
                    .is_some_and(|d| d.variants.iter().any(|(_, p)| !p.is_empty()));
                if is_data {
                    let (gid, _, _) = self.enum_instance_layout(&ty.to_string())?;
                    return Some(LoweredTy::Aggregate(gid));
                }
            }
        }
        lowered_ty(ty, self.registry)
    }

    /// Synthesize (once) and record the aggregate layout of a monomorphized data-carrying enum
    /// instance (`"Option<i32>"` -> `{ i32 tag @0, i32 payload @4 }`), returning its per-instance GID
    /// and the field byte offsets. The payload is the first non-empty variant's payload types
    /// (`Option`'s `Some(T)`) substituted with the instance args, laid after the `i32` tag with
    /// natural alignment — matching the AST codegen's `{ i32, <payload> }`. `None` for a payload-free
    /// enum (a bare `i32` discriminant, not an aggregate) or an unmodelled payload type. (#242)
    fn enum_instance_layout(&mut self, enum_name: &str) -> Option<(TypeId, Vec<u64>, Vec<Type>)> {
        let (base, args) = parse_enum_instance(enum_name);
        let data = self.registry.enum_data.get(base.as_str())?;
        let mut mapping = HashMap::new();
        for (g, a) in data.generics.iter().zip(&args) {
            mapping.insert(g.clone(), a.clone());
        }
        let payload: Vec<Type> = data
            .variants
            .iter()
            .find(|(_, p)| !p.is_empty())?
            .1
            .iter()
            .map(|t| t.substitute(&mapping))
            .collect();
        let mut offsets = vec![0u64];
        let mut field_tys = vec!["i32".to_string()]; // the discriminant tag
        let mut off = 4u64;
        for pt in &payload {
            let (sz, al, mlir) = enum_payload_field(pt)?;
            off = crate::layout::align_up(off as usize, al as usize) as u64;
            offsets.push(off);
            field_tys.push(mlir);
            off += sz;
        }
        let gid = enum_instance_gid(&base, &args);
        if !self.agg_layouts.iter().any(|(g, _, _)| *g == gid) {
            self.agg_layouts.push((gid, offsets.clone(), field_tys));
        }
        Some((gid, offsets, payload))
    }

    fn lower_struct_init(&mut self, si: &crate::syntax::StructInitExpr) -> Option<Val> {
        // The struct's layout GID: the checker-attached `type_id` when present, else resolved by
        // name. A *monomorphized generic* construction (`Vec<i32> { .. }`) carries no `type_id` (the
        // instance identity isn't a plain module symbol), so fall back to the base nominal's layout
        // by name — its layout is instance-independent (every generic parameter is behind a pointer),
        // exactly the `lowered_ty(GenericInstance)` rule (#242).
        let gid = match si.type_id {
            Some(g) => g,
            None => self.struct_gid_by_name(&si.name)?,
        };
        let def = self.registry.layouts.get(&gid)?;
        if def.align_bytes == 0 {
            return None; // layout not modelled yet
        }
        // Snapshot (name, offset, type) so the immutable registry borrow ends before we emit.
        let field_layouts: Vec<(Symbol, u64, FieldTy)> = def
            .fields
            .iter()
            .map(|f| (f.name.clone(), f.offset as u64, f.ty.clone()))
            .collect();

        let slot = self.emit_alloca(LoweredTy::Aggregate(gid));
        for (name, offset, _fty) in field_layouts {
            // Scalar, pointer, or by-value nested-aggregate fields — a nested aggregate is stored as a
            // whole `!llvm.struct` value (`VecMap { iter: VecIter, f: Closure1 }`), the store type coming
            // from the layout's `field_tys` at emit (#242).
            let (_, init_expr) = si
                .fields
                .iter()
                .find(|(n, _)| n.as_ref() == name.as_ref())?;
            // The initializer already carries the field's declared type — the checker infers a literal
            // to the field type and rejects a genuine mismatch (no implicit conversion, #240).
            let mut v = self.lower_expr(init_expr)?;
            // A by-value nested-aggregate field (`Outer { inner : Inner { .. } }`): the initializer
            // yields the inner construction *slot* (a pointer), but the field must hold the struct
            // *value*. Load it so the `FieldStore` stores the value, not the address (#277) — the same
            // fix `lower_assign` applies to an aggregate-construction RHS. First cut per §16.1 option 1;
            // constructing in place (option 2) is the destination-passing follow-up.
            if matches!(init_expr, Expr::StructInit(_) | Expr::EnumVariant(_))
                && matches!(v.ty, LoweredTy::Aggregate(_))
            {
                v = self.emit_typed(Opcode::SlotLoad, v.reg, Register(0), v.ty.clone(), 0);
            }
            self.emit_effect(Opcode::FieldStore, slot.reg, v.reg, offset);
        }
        Some(slot)
    }

    /// The layout GID of a struct by name — the fallback for a monomorphized generic construction
    /// (`Vec<i32> { .. }`) whose `StructInit` carries no checker-attached `type_id`. Resolves to the
    /// *base* nominal's modelled layout (the display name a monomorphized instance renders under);
    /// declines if the name is ambiguous (two distinct GIDs) so a wrong layout is never chosen. (#242)
    fn struct_gid_by_name(&self, name: &Symbol) -> Option<TypeId> {
        struct_layout_gid_by_name(self.registry, name.as_ref())
    }

    /// Lower a tensor allocation `Tensor<T>([d0, d1, ...])` (or `Tensor<T>(d0, d1)`): a `TensorAlloc`
    /// whose `imm` is the static byte size, so the buffer has room for every element. Declines a
    /// dynamic/symbolic shape (its byte size isn't statically known) or a non-scalar element.
    fn lower_tensor_alloc(&mut self, fc: &crate::syntax::FunctionCallExpr) -> Option<Val> {
        let elem = fc.type_args.as_ref()?.first().and_then(scalar_of)?;
        // `Tensor<T>([d0, d1])` passes the shape as one array arg; `Tensor<T>(d0, d1)` as bare args.
        let dims: &[Expr] = match fc.args.first() {
            Some(Expr::Array(arr)) if fc.args.len() == 1 => &arr.elements,
            _ => &fc.args,
        };
        let bytes = crate::hir::memory::static_tensor_bytes(&elem, dims)?;
        let shape: Vec<String> = dims.iter().map(tensor_dim_string).collect::<Option<_>>()?;
        let ty = LoweredTy::Tensor { elem, shape };
        Some(self.emit_typed(Opcode::TensorAlloc, Register(0), Register(0), ty, bytes))
    }

    /// Lower a *value* array literal `[a, b, c]` (#239): allocate a rank-1 tensor buffer and store
    /// each element at its index, yielding the tensor. This is the flat path's memref analogue of the
    /// AST codegen's `tensor.from_elements`; every element is a scalar of one type (the checker
    /// reconciles them), so a following `arr[i]` reads through the same `TensorIndex` path as any
    /// tensor. Declines an empty literal or a non-scalar element. (A `Tensor<T>([…])` shape argument
    /// never reaches here — `lower_tensor_alloc` reads it as dimensions directly.)
    fn lower_array(&mut self, arr: &crate::syntax::ArrayExpr) -> Option<Val> {
        if arr.elements.is_empty() {
            return None; // an empty array has no element type to size the buffer
        }
        let mut vals = Vec::with_capacity(arr.elements.len());
        for el in &arr.elements {
            vals.push(self.lower_expr(el)?);
        }
        let elem = match &vals[0].ty {
            LoweredTy::Scalar(e) => e.clone(),
            _ => return None, // only scalar-element arrays are modelled
        };
        let n = vals.len();
        let bytes = (crate::hir::memory::element_bits(&elem)? * n as u64).div_ceil(8);
        let buf = self.emit_typed(
            Opcode::TensorAlloc,
            Register(0),
            Register(0),
            LoweredTy::Tensor {
                elem: elem.clone(),
                shape: vec![n.to_string()],
            },
            bytes,
        );
        for (k, v) in vals.iter().enumerate() {
            let idx = self.emit_value(
                Opcode::Const,
                Register(0),
                Register(0),
                ElementType::I32,
                k as u64,
            );
            let place = self.emit_typed(
                Opcode::TensorIndex,
                buf.reg,
                idx.reg,
                LoweredTy::Scalar(elem.clone()),
                1,
            );
            self.emit_effect(Opcode::TensorStore, place.reg, v.reg, 0);
        }
        Some(buf)
    }

    /// Lower an ordinary fixed-arity call `f(a, b, ...)`: resolve the callee via the frozen registry
    /// (its GID + return type), lower each argument, mark them with `Arg` instructions in order, then
    /// emit `Call` (callee GID in `type_idx`, arg count in `imm`). Declines an unknown callee (or one
    /// ambiguous across modules) and a void/unmodelled return -- for now only value-returning calls.
    /// If `arg` is a closure literal environment (`Closure_N`), materialize the nominal `ClosureK`
    /// fat struct `{ env, func }` the stdlib API expects (`.map`'s `f : Closure1<T, NewItem>`): `env`
    /// is the address of the closure's environment aggregate (its captures), `func` a `FuncConst`
    /// pointer to the generated `Closure_N_call`. Every `ClosureK` layout is structurally `{ ptr, ptr }`
    /// (the flat aggregates are anonymous structs), so any is a valid target — the arity only matters at
    /// the eventual `CallIndirect`, which rebuilds the function type from the actual arguments. Returns
    /// the adapted value (passed by value), or `None` if `arg` isn't a closure. (#242)
    fn try_adapt_closure_arg(&mut self, arg: &Expr) -> Option<Val> {
        let cn_name = match self.infer_ast_type(arg)? {
            Type::Struct(name, _) if name.as_ref().starts_with("Closure_") => {
                name.as_ref().to_string()
            }
            _ => return None,
        };
        // env = the address of the closure's environment aggregate (its captured variables).
        let (env_ptr, _cn_gid) = self.lower_agg_base(arg)?;
        // func = a pointer to the closure's generated call function `Closure_N_call`.
        let call_name: Symbol = format!("{cn_name}_call").into();
        let fnptr = self.lower_func_const(&call_name)?;
        // Target `ClosureK` layout `{ env: ptr, func: ptr }` — structurally identical for every arity.
        let ck_gid = struct_layout_gid_by_name(self.registry, "Closure1")?;
        let (env_off, func_off) = {
            let fields = &self.registry.layouts.get(&ck_gid)?.fields;
            (fields.first()?.offset as u64, fields.get(1)?.offset as u64)
        };
        let slot = self.emit_alloca(LoweredTy::Aggregate(ck_gid));
        self.emit_effect(Opcode::FieldStore, slot.reg, env_ptr, env_off);
        self.emit_effect(Opcode::FieldStore, slot.reg, fnptr.reg, func_off);
        // Passed by value: load the completed fat struct.
        Some(self.emit_typed(
            Opcode::SlotLoad,
            slot.reg,
            Register(0),
            LoweredTy::Aggregate(ck_gid),
            0,
        ))
    }

    /// Materialize a function pointer for a registered function name (`FuncConst`): the result is an
    /// opaque `!llvm.ptr`, and `type_idx` carries the target's GID so codegen can emit
    /// `func.constant @name : sig`. Declines for a name that isn't a registered function. (#242)
    fn lower_func_const(&mut self, name: &Symbol) -> Option<Val> {
        let gid = self.registry.fn_sigs.get(name.as_ref())?.gid;
        let type_idx = TypeIdx(self.types.len() as u32);
        self.types.push(gid);
        let reg = Register(self.code.len() as u32);
        self.code.push(HirInstruction::new(
            Opcode::FuncConst,
            Register(0),
            Register(0),
            type_idx,
            0,
        ));
        Some(Val {
            reg,
            ty: LoweredTy::Ptr,
        })
    }

    /// Lower an indirect call `f(args)` where `f` is a local (a fn-pointer parameter or a `Closure1`'s
    /// loaded `func` field), not a registered function. The callee's function type — hence the scalar
    /// return — comes from `f`'s AST type; args are emitted as `Arg`s exactly like a direct call, and a
    /// `CallIndirect` carries the callee pointer register + arg count + return type. Declines for a
    /// non-scalar return or an unknown callee type. (#242)
    fn lower_indirect_call(
        &mut self,
        fc: &crate::syntax::FunctionCallExpr,
        callee: Binding,
    ) -> Option<Val> {
        // Load the callee function pointer (an SSA alias, or a `SlotLoad` from its slot).
        let fnptr = match callee {
            Binding::Reg(v) => v,
            Binding::Slot { reg, ty } => self.emit_typed(Opcode::SlotLoad, reg, Register(0), ty, 0),
            // A place-bound reference is not a function pointer — decline. (#275)
            Binding::Place { .. } => return None,
        };
        if !matches!(fnptr.ty, LoweredTy::Ptr) {
            return None;
        }
        // The callee's return type comes from `f`'s AST function type.
        let ret_elem = match self.ast_types.get(fc.name.as_ref())? {
            Type::Function(_, ret) | Type::Closure(_, ret) => scalar_of(ret)?,
            _ => return None,
        };
        let mut arg_regs = Vec::with_capacity(fc.args.len());
        for arg in &fc.args {
            arg_regs.push(self.lower_expr(arg)?.reg);
        }
        for reg in arg_regs {
            self.emit_effect(Opcode::Arg, reg, Register(0), 0);
        }
        let ty = LoweredTy::Scalar(ret_elem);
        let type_idx = TypeIdx(self.types.len() as u32);
        self.types.push(ty.gid());
        let reg = Register(self.code.len() as u32);
        self.code.push(HirInstruction::new(
            Opcode::CallIndirect,
            fnptr.reg,
            Register(0),
            type_idx,
            fc.args.len() as u64,
        ));
        Some(Val { reg, ty })
    }

    fn lower_call(&mut self, fc: &crate::syntax::FunctionCallExpr) -> Option<Val> {
        // An indirect call: the callee name is a local holding a function pointer (a fn-pointer
        // parameter, or a `Closure1`'s loaded `func` field), not a registered function. (#242)
        if !self.registry.fn_sigs.contains_key(fc.name.as_ref()) {
            if let Some(binding) = self.scope.get(&fc.name).cloned() {
                return self.lower_indirect_call(fc, binding);
            }
        }
        let sig = match self.registry.fn_sigs.get(fc.name.as_ref()) {
            Some(s) => s.clone(),
            None => {
                // An unresolved callee (an ambiguous-across-modules name dropped from `fn_sigs`, or an
                // unregistered symbol) — decline, so the AST path stays the oracle for the call.
                if std::env::var("VX_FLAT_DBG").is_ok() {
                    eprintln!("[flat-dbg]   call: no fn_sig for {}", fc.name.as_ref());
                }
                return None;
            }
        };
        // A void callee (`bump(&mut x) -> void`, a `&mut` mutator) has no result value; it appears only
        // in statement position, where the returned `Val` is discarded. Give it a placeholder scalar
        // type — never read — rather than declining the call. (#230)
        let void_ret = crate::codegen::flat::is_void_ty(&sig.ret_ty);
        let ret_ty = if void_ret {
            LoweredTy::Scalar(ElementType::I32)
        } else {
            self.lower_ty_synth(&sig.ret_ty)?
        };
        let mut arg_regs = Vec::with_capacity(fc.args.len());
        for arg in &fc.args {
            // A closure literal passed where a nominal `ClosureK` is expected (`.map(adder)`) is
            // adapted to the `{ env, func }` fat struct; any other argument lowers normally. (#242)
            let v = match self.try_adapt_closure_arg(arg) {
                Some(v) => v,
                None => self.lower_expr(arg)?,
            };
            arg_regs.push(v.reg);
        }
        for reg in arg_regs {
            self.emit_effect(Opcode::Arg, reg, Register(0), 0);
        }
        let type_idx = TypeIdx(self.types.len() as u32);
        self.types.push(sig.gid);
        let reg = Register(self.code.len() as u32);
        self.code.push(HirInstruction::new(
            Opcode::Call,
            Register(0),
            Register(0),
            type_idx,
            fc.args.len() as u64,
        ));
        Some(Val { reg, ty: ret_ty })
    }

    /// Lower an assignable tensor place `base[index]` (an lvalue for a following `TensorStore`). The
    /// outer indices produce sub-view tensors exactly as a read does, but the *final scalar* index
    /// yields an element **place** — a `TensorIndex` with `imm = 1` — so codegen addresses the element
    /// and stores into it instead of loading its value. A still-nonempty shape yields a row/sub-view
    /// place (`imm = 0`, identical to the read form; a slice store writes through it). `None` for any
    /// non-tensor-index place.
    /// Lower `*p` as `p[0]`: a `PtrIndex` at a constant zero index — a value read (`is_place=false`)
    /// or an element place (`is_place=true`, consumed by a `PtrStore`). The pointee element type comes
    /// from `p`'s AST type. Backs `let v = *p` / `*p = val` (`Box`'s heap cell). A non-pointer, or a
    /// pointer whose element isn't a scalar/aggregate, declines. (#242)
    fn lower_ptr_deref(&mut self, ptr_expr: &Expr, is_place: bool) -> Option<Val> {
        let base = self.lower_expr(ptr_expr)?;
        if !matches!(base.ty, LoweredTy::Ptr) {
            return None;
        }
        let elem = pointer_elem_ty(&self.infer_ast_type(ptr_expr)?, self.registry)?;
        let zero = self.emit_value(Opcode::Const, Register(0), Register(0), ElementType::I32, 0);
        Some(self.emit_typed(
            Opcode::PtrIndex,
            base.reg,
            zero.reg,
            elem,
            if is_place { 1 } else { 0 },
        ))
    }

    fn lower_place(&mut self, e: &Expr) -> Option<Val> {
        // A raw-pointer dereference place `*p = val`: `p[0]`.
        if let Expr::Dereference(d) = e {
            return self.lower_ptr_deref(&d.expr, true);
        }
        let Expr::IndexAccess(ix) = e else {
            return None;
        };
        let base = self.lower_expr(&ix.base)?;
        // A raw-pointer place (`self.data[i] = val`): a `PtrIndex` with `imm = 1` (an element
        // pointer), consumed by a `PtrStore`. The element type comes from the base's AST type (#242).
        if matches!(base.ty, LoweredTy::Ptr) {
            let elem = pointer_elem_ty(&self.infer_ast_type(&ix.base)?, self.registry)?;
            let index = self.lower_expr(&ix.index)?;
            if !matches!(index.ty, LoweredTy::Scalar(_)) {
                return None;
            }
            return Some(self.emit_typed(Opcode::PtrIndex, base.reg, index.reg, elem, 1));
        }
        let (elem, shape) = match &base.ty {
            LoweredTy::Tensor { elem, shape } => (elem.clone(), shape.clone()),
            _ => return None,
        };
        if shape.is_empty() {
            return None; // cannot index a rank-0 value
        }
        let index = self.lower_expr(&ix.index)?;
        if !matches!(index.ty, LoweredTy::Scalar(_)) {
            return None; // the index must be a scalar
        }
        let reduced: Vec<String> = shape[1..].to_vec();
        if reduced.is_empty() {
            // A scalar element place: the final index, marked `imm = 1` so codegen stores into the
            // element rather than loading it (`q[i][j] = <scalar>`).
            Some(self.emit_typed(
                Opcode::TensorIndex,
                base.reg,
                index.reg,
                LoweredTy::Scalar(elem),
                1,
            ))
        } else {
            // A row/sub-view place: an addressable sub-view, same shape as the read form
            // (`o[i] = <slice>`).
            Some(self.emit_typed(
                Opcode::TensorIndex,
                base.reg,
                index.reg,
                LoweredTy::Tensor {
                    elem,
                    shape: reduced,
                },
                0,
            ))
        }
    }

    /// Lower an assignment `lhs = rhs`, dispatching on the place kind. Factored out of `lower_stmt` so a
    /// write through a symbolic place (`*r = v` where `r` is a `Binding::Place`) re-dispatches as a
    /// write to the borrowed place expression itself (§5 Example C). (#242/#275)
    fn lower_assign(&mut self, lhs: &Expr, rhs: &Expr) -> Option<()> {
        // `*r = v` where `r` is a non-escaping place: re-lower as a write to the borrowed lvalue —
        // `p.x = v` `FieldStore`s the field, `x = v` rebinds the local. No pointer is materialized. (#275)
        if let Expr::Dereference(d) = lhs {
            if let Expr::Identifier(id) = &*d.expr {
                if let Some(Binding::Place { place }) = self.scope.get(&id.name).cloned() {
                    // Tag the write with the borrowed place so its `FieldStore` records alias-scope
                    // metadata (M2b-2): a set of simultaneously live `&mut o.field` borrows the checker
                    // admitted is pairwise disjoint, which the store carries as `noalias` scopes.
                    // Restore the prior tag afterwards so a nested/adjacent write isn't mis-tagged. (#275)
                    let prev = self.pending_place_write.take();
                    self.pending_place_write = crate::hir::places::base_and_path(&place);
                    let r = self.lower_assign(&place, rhs);
                    self.pending_place_write = prev;
                    return r;
                }
            }
        }
        // An indexed place store `place[i] = value`. A *tensor* place is a `TensorIndex` (a row/sub-view
        // `o[i] = <slice>` or a scalar element `q[i][j] = <scalar>`, the final index marked `imm = 1`)
        // written by a `TensorStore`; a *raw-pointer* place (`self.data[i] = val`) is a `PtrIndex` place
        // written by a `PtrStore` (#242). The base's AST type selects the store — a tensor local isn't
        // in `ast_types`, so it reads as non-pointer.
        if let Expr::IndexAccess(ix) = lhs {
            let is_ptr = self
                .infer_ast_type(&ix.base)
                .and_then(|t| pointer_elem_ty(&t, self.registry))
                .is_some();
            let place = self.lower_place(lhs)?;
            // The stored scalar already matches the place's element type (the checker types a literal
            // RHS to the element and rejects a genuine mismatch, #240).
            let value = self.lower_expr(rhs)?;
            let store = if is_ptr {
                Opcode::PtrStore
            } else {
                Opcode::TensorStore
            };
            self.emit_effect(store, place.reg, value.reg, 0);
            return Some(());
        }
        // A raw-pointer dereference store `*p = val` (`Box`'s `*p = val`): the place is `p[0]` (always a
        // pointer, so always a `PtrStore`). (#242)
        if let Expr::Dereference(_) = lhs {
            let place = self.lower_place(lhs)?;
            let value = self.lower_expr(rhs)?;
            self.emit_effect(Opcode::PtrStore, place.reg, value.reg, 0);
            return Some(());
        }
        // A field store `base.member = value` through an aggregate slot or a `self` pointer
        // (`self.len = self.len + 1`, `self.data = grow(..)`, #242). A by-value nested-aggregate field
        // store isn't modelled.
        if let Expr::MemberAccess(m) = lhs {
            let (base_reg, gid) = self.lower_agg_base(&m.base)?;
            let field = self
                .registry
                .layouts
                .get(&gid)?
                .fields
                .iter()
                .find(|f| f.name.as_ref() == m.member.as_ref())?;
            let offset = field.offset as u64;
            if matches!(field.ty, FieldTy::Nominal(_)) {
                return None;
            }
            let v = self.lower_expr(rhs)?;
            let pos = self.code.len();
            self.emit_effect(Opcode::FieldStore, base_reg, v.reg, offset);
            // A place-write field store (`*r = v` through a `&mut o.field` place): record it for
            // alias-scope metadata. A direct `p.x = v` leaves `pending_place_write` unset. (#275, §5.4)
            if let Some((root, path)) = self.pending_place_write.take() {
                self.place_field_stores.push((pos, root, path));
            }
            return Some(());
        }
        // `name = expr` (simple identifier target). The value already matches the slot's type (the
        // checker types a literal RHS to the target and rejects a mismatch, #240).
        let name = simple_ident(lhs)?;
        let mut v = self.lower_expr(rhs)?;
        // An aggregate *construction* RHS (`ret = Some(val)`, `p = Point { .. }`) yields the
        // construction *slot* (a pointer), but the assignment must copy the struct *value* into the
        // target slot — load it first, else the slot pointer is stored as a struct (#242).
        if matches!(rhs, Expr::EnumVariant(_) | Expr::StructInit(_))
            && matches!(v.ty, LoweredTy::Aggregate(_))
        {
            v = self.emit_typed(Opcode::SlotLoad, v.reg, Register(0), v.ty.clone(), 0);
        }
        self.assign_local(&name, v)
    }

    /// Lower a statement. `None` aborts the whole function's lowering.
    fn lower_stmt(&mut self, s: &Statement) -> Option<()> {
        match s {
            Statement::LetDecl(l) => {
                // Record the local's concrete AST type for `infer_ast_type` (a pointer local like
                // `let ptr : *mut T = ...` -> its pointee element for a later index, #242). The
                // annotation is authoritative; else fall back to inferring the initializer's type.
                if let Some(t) = l.ty_ann.clone().or_else(|| self.infer_ast_type(&l.expr)) {
                    self.ast_types.insert(l.name.clone(), t);
                }
                // `let r = &<lvalue>` the escape analysis realized as a place (§5): bind `r` to the
                // borrowed *place expression* itself, so `*r` re-lowers it (a bare local reads directly
                // — Example A; a field access `FieldLoad`s — Example B/C) and no address is
                // materialized. Escaping / disqualified borrows are absent from `place_bindings` and
                // fall through to the general path (a real pointer). (#275)
                if self.place_bindings.contains(&l.name) {
                    if let Expr::Borrow(b) = &l.expr {
                        self.scope.insert(
                            l.name.clone(),
                            Binding::Place {
                                place: (*b.expr).clone(),
                            },
                        );
                        return Some(());
                    }
                }
                // A struct literal is constructed *in place* into its own slot; the local is that
                // slot (binding it directly avoids re-`Alloca`ing and storing the slot handle).
                if let Expr::StructInit(si) = &l.expr {
                    let slot = self.lower_struct_init(si)?;
                    self.scope.insert(
                        l.name.clone(),
                        Binding::Slot {
                            reg: slot.reg,
                            ty: slot.ty,
                        },
                    );
                    return Some(());
                }
                // A data-carrying enum construction (`let o = Option<i32>::Some(30)`) also builds its
                // aggregate in place; bind the local to that slot directly (else `bind_local` would
                // re-`Alloca` and store the slot *pointer*, not the value). A payload-free variant is a
                // scalar and falls through to the general path. (#242)
                if let Expr::EnumVariant(ev) = &l.expr {
                    let (base, _) = parse_enum_instance(&ev.enum_name);
                    let is_data = self
                        .registry
                        .enum_data
                        .get(base.as_str())
                        .is_some_and(|d| d.variants.iter().any(|(_, p)| !p.is_empty()));
                    if is_data {
                        let slot = self.lower_enum_construct(ev)?;
                        self.scope.insert(
                            l.name.clone(),
                            Binding::Slot {
                                reg: slot.reg,
                                ty: slot.ty,
                            },
                        );
                        return Some(());
                    }
                }
                // A value-position `if` (`let v: T = if c { .. } else { .. }`): allocate a result slot,
                // have each branch store its trailing value into it, and bind the local to the slot
                // (the merge block loads it). The slot type comes from the `let`'s annotation (#201).
                if let Expr::If(if_expr) = &l.expr {
                    let ty_ann = l.ty_ann.as_ref()?;
                    let result_ty = lowered_ty(ty_ann, self.registry)?;
                    let slot = self.emit_alloca(result_ty.clone());
                    self.lower_if_into_slot(if_expr, slot.reg)?;
                    self.scope.insert(
                        l.name.clone(),
                        Binding::Slot {
                            reg: slot.reg,
                            ty: result_ty,
                        },
                    );
                    return Some(());
                }
                let v = self.lower_expr(&l.expr)?;
                // A tensor local is a reference (memref) — bind it as an SSA register, not a slot;
                // stores write through the descriptor to the buffer.
                if matches!(v.ty, LoweredTy::Tensor { .. }) {
                    self.scope.insert(l.name.clone(), Binding::Reg(v));
                } else {
                    // The initializer already carries the local's type: an annotated `let` types its
                    // literal initializer to the annotation, and a genuine mismatch is rejected (no
                    // implicit conversion, #240), so the value is bound directly.
                    self.bind_local(l.name.clone(), v);
                }
                Some(())
            }
            Statement::Return(r) => {
                // `return <match>` (a value-position match whose arms `return` themselves, e.g.
                // `Option::unwrap`): lower the match as a statement — each arm emits its own `Ret` — then
                // give the fall-through merge block a terminator, a default (zero) return of the
                // function's type, exactly as the AST codegen's merge block does. (#242)
                if let Expr::Match(m) = &r.expr {
                    self.lower_match(m)?;
                    if !self.block_terminated() {
                        let rty = self.ret_ty.clone()?;
                        let zero = match &rty {
                            LoweredTy::Scalar(e) => self.emit_value(
                                Opcode::Const,
                                Register(0),
                                Register(0),
                                e.clone(),
                                0,
                            ),
                            _ => return None, // a non-scalar default return isn't modelled
                        };
                        self.emit_typed(Opcode::Ret, zero.reg, Register(0), rty, 0);
                    }
                    return Some(());
                }
                // The returned value already carries the function's declared return type — the checker
                // types a literal to it and rejects a genuine mismatch (#240).
                let v = self.lower_expr(&r.expr)?;
                self.emit_typed(Opcode::Ret, v.reg, Register(0), v.ty, 0);
                Some(())
            }
            Statement::Assign(a) => self.lower_assign(&a.lhs, &a.rhs),
            // Compound assignment `lhs op= rhs` desugars to `lhs = (lhs op rhs)`: read the current
            // value of the place, combine it with the right side, and store back.
            Statement::CompoundAssign(a) => {
                let cur = self.lower_expr(&a.lhs)?;
                let rhs = self.lower_expr(&a.rhs)?;
                let op = binop_opcode(&a.op)?;
                // Elementwise if either side is a tensor (the arith opcode carries the tensor result
                // type), else the scalar result type — mirroring `BinaryOp` in `lower_expr`.
                let result_ty = match (&cur.ty, &rhs.ty) {
                    (LoweredTy::Tensor { .. }, _) => cur.ty.clone(),
                    (_, LoweredTy::Tensor { .. }) => rhs.ty.clone(),
                    _ => cur.ty.clone(),
                };
                // Both scalar operands already share a type (`x += 1` types `1` to `x`'s type; a
                // genuine mismatch is rejected, #240), so they combine directly.
                let combined = self.emit_typed(op, cur.reg, rhs.reg, result_ty, 0);
                if matches!(&a.lhs, Expr::IndexAccess(_)) {
                    let place = self.lower_place(&a.lhs)?;
                    self.emit_effect(Opcode::TensorStore, place.reg, combined.reg, 0);
                    Some(())
                } else {
                    let name = simple_ident(&a.lhs)?;
                    self.assign_local(&name, combined)
                }
            }
            // `assert(cond, msg)` is a *compile-time* fact (used for seam certificates); the AST codegen
            // emits no runtime check for it (`generator.rs`, `Statement::Assert`). Match that exactly:
            // lower it to nothing, so flat and AST agree at runtime.
            Statement::Assert(_) => Some(()),
            Statement::ExprStmt(e) => match &e.expr {
                Expr::If(iff) => self.lower_if(iff),
                Expr::Match(m) => self.lower_match(m),
                Expr::SpawnOn(sp) => self.lower_spawn(sp),
                // A statement-position `unsafe { … }` (an FFI program's `unsafe { … }` wrapper with no
                // trailing value): safety was checked upstream, so `unsafe` is transparent — lower the
                // inner statements, and its trailing value expression if any. (The value-position form,
                // `return unsafe { … }`, is the `Expr::UnsafeBlock` arm in `lower_expr`.)
                Expr::UnsafeBlock(ub) => {
                    for s in &ub.stmts {
                        self.lower_stmt(s)?;
                    }
                    if let Some(r) = &ub.ret {
                        self.lower_expr(r)?;
                    }
                    Some(())
                }
                // `print(x)` is a statement-level effect (no result): lower its one argument and emit
                // a `Print`, whose `type_idx` carries the argument's type (scalar or tensor) so codegen
                // routes to the right `print_*`/`printMemref*` runtime helper.
                Expr::FunctionCall(fc) if fc.name.as_ref() == "print" && fc.args.len() == 1 => {
                    let v = self.lower_expr(&fc.args[0])?;
                    self.emit_print(v);
                    Some(())
                }
                // The `print!` macro form (`Expr::Print`): prints each argument in sequence via the same
                // `print_*` / `print_str` helpers, no separators — matching the AST codegen. A
                // `StringLiteral` arg emits a `PrintStr`; any other arg is lowered and `print`ed.
                Expr::Print(p) => {
                    for arg in &p.args {
                        self.lower_print_arg(arg)?;
                    }
                    Some(())
                }
                // The `println!` macro form (`Expr::Println`): print each argument (as `print!`), then a
                // trailing newline. The AST path calls a `println()` runtime helper for the newline;
                // printing a `"\n"` string is byte-identical, so reuse `PrintStr` and add no new helper.
                Expr::Println(p) => {
                    for arg in &p.args {
                        self.lower_print_arg(arg)?;
                    }
                    self.emit_print_str("\n");
                    Some(())
                }
                other => {
                    self.lower_expr(other)?;
                    Some(())
                }
            },
            Statement::Loop(l) => self.lower_loop(&l.body),
            Statement::ForLoop(f) => self.lower_for(f),
            // `break`/`continue` branch to the enclosing loop's exit/continue target.
            Statement::Break(_) => {
                let (_, brk) = *self.loop_stack.last()?;
                self.emit_effect(Opcode::Br, Register(0), Register(0), brk as u64);
                Some(())
            }
            Statement::Continue(_) => {
                let (cont, _) = *self.loop_stack.last()?;
                self.emit_effect(Opcode::Br, Register(0), Register(0), cont as u64);
                Some(())
            }
            other => {
                if std::env::var("VX_FLAT_DBG").is_ok() {
                    eprintln!("[flat-dbg]   unsupported stmt: {}", stmt_kind(other));
                }
                None
            }
        }
    }

    /// Append the built stream onto the worker: body types extend `local_type_stream` (after any
    /// signature types already there), and each value instruction's local `type_idx` is rebased to
    /// the absolute index in that stream (effect instructions keep the [`NO_TYPE`] sentinel).
    fn commit(self, worker: &mut LocalWorkerState) {
        let base = worker.local_type_stream.len() as u32;
        worker.local_type_stream.extend(self.types);
        // The tensor side table is keyed by (content-hash) GID, which `commit` does not rebase, so it
        // transfers as-is.
        worker.local_tensor_types.extend(self.tensor_types);
        // Synthetic enum-instance layouts (keyed by content-hash GID, not rebased) transfer as-is.
        worker.local_agg_layouts.extend(self.agg_layouts);
        // The string side table is indexed by each `PrintStr`'s `imm`; a fresh worker lowers exactly
        // one function, so the indices need no rebasing (they start at 0 per function).
        worker.local_string_table.extend(self.strings);
        // Place-write alias table (M2b-2): reduce collected field stores to `(position, group, siblings)`.
        // Positions are stream-relative; a fresh worker lowers one function, so they need no rebasing.
        worker
            .local_place_alias_stores
            .extend(reduce_place_alias(&self.place_field_stores));
        for mut ins in self.code {
            if ins.type_idx.0 != NO_TYPE {
                ins.type_idx = TypeIdx(ins.type_idx.0 + base);
            }
            worker.local_hir_stream.push(ins);
        }
    }
}

/// Lower a whole function body to flat HIR bytecode on `worker`. **Atomic**: returns `false` and
/// leaves the worker untouched if any construct is outside the current subset, so
/// `local_hir_stream` is only ever a complete, correct lowering or empty (keep-green). Returns
/// `true` when the full body lowered.
pub fn lower_function_to_hir(func: &Function, worker: &mut LocalWorkerState) -> bool {
    // Cheap `Arc` clone so the borrow of the registry doesn't collide with the later `&mut worker`
    // in `commit`; the frozen registry is immutable, so this is a pure reference bump.
    let registry = worker.global.registry.clone();
    match try_lower(func, &registry) {
        Some(lw) => {
            lw.commit(worker);
            true
        }
        None => false,
    }
}

fn try_lower<'r>(func: &Function, registry: &'r ImmutableGlobalRegistry) -> Option<Lowerer<'r>> {
    let mut lw = Lowerer::new(registry);
    // Local-usage pre-pass (#230): one syntactic walk collecting the locals whose address is taken
    // (`&x`) and those reassigned (`x = ..`), so `bind_local` can slot exactly the locals that need it.
    // Must run before params bind, since a param can be address-taken or reassigned too.
    let uses = analyze_local_uses(&func.body);
    lw.materialized = uses.materialized;
    lw.mutated = uses.mutated;
    lw.place_bindings = uses.place_bindings;
    // Control flow drives the block model (entry block + branch skeletons) and is the precondition for
    // slotting a *mutated* scalar (it must cross a block boundary). An aggregate param or a struct
    // construction also turns on the memory model (an aggregate must live in an addressable slot), but
    // *not* the per-scalar rule — a mutated scalar in a straight-line function stays a pure-SSA rebind.
    lw.has_control_flow = body_has_control_flow(&func.body);
    let has_aggregate_param = func
        .params
        .iter()
        .any(|(_, ty)| matches!(lowered_ty(ty, registry), Some(LoweredTy::Aggregate(_))));
    lw.memory = lw.has_control_flow || has_aggregate_param || body_constructs_struct(&func.body);
    lw.ret_ty = lw.lower_ty_synth(&func.return_type);
    if lw.memory {
        lw.emit_effect(Opcode::BlockStart, Register(0), Register(0), 0); // entry block
    }
    // Parameters: materialize the incoming value (`Load` imm = index), then bind (a slot in memory
    // mode, an SSA register otherwise). A tensor is a reference value (memref), so it always binds
    // as an SSA register — never `Alloca`'d into a slot.
    for (i, (name, ty)) in func.params.iter().enumerate() {
        // `lower_ty_synth` (not the free `lowered_ty`) so a by-value data-carrying enum parameter
        // (`self : Option<T>` in `Option::unwrap`) synthesizes its `{ tag, payload }` instance layout
        // and binds as an aggregate rather than declining. (#242)
        let lty = lw.lower_ty_synth(ty)?;
        // Record the param's concrete AST type so `infer_ast_type` can recover a pointer field's
        // pointee element (`self : &mut Vec<i32>` -> `self.data : *mut i32`, #242).
        lw.ast_types.insert(name.clone(), ty.clone());
        let incoming = lw.emit_typed(Opcode::Load, Register(0), Register(0), lty, i as u64);
        // A tensor (memref) or a pointer to a modelled aggregate (`self : &mut Vec`) is a reference
        // value that binds as an SSA register even in memory mode: it is a block argument that
        // dominates every block, and mutation flows through the pointer to the pointee, not to the
        // register (so it never needs a slot). (#242)
        if matches!(incoming.ty, LoweredTy::Tensor { .. }) || is_ptr_to_agg(ty, registry) {
            lw.scope.insert(name.clone(), Binding::Reg(incoming));
        } else {
            lw.bind_local(name.clone(), incoming);
        }
    }
    for (si, stmt) in func.body.iter().enumerate() {
        if lw.lower_stmt(stmt).is_none() {
            if std::env::var("VX_FLAT_DBG").is_ok() {
                eprintln!(
                    "[flat-dbg] fn {} declined at stmt #{si}",
                    func.name.as_ref()
                );
            }
            return None;
        }
    }
    Some(lw)
}

/// The flat-HIR type of an AST type: a scalar, or an aggregate nominal (struct/enum) whose layout
/// the frozen registry has actually computed. Returns `None` for a type outside the modelled subset
/// (generic, tensor, pointer, closure) or an aggregate whose layout is the not-yet-computed 0/0 stub
/// (`align_bytes == 0`) — the caller then aborts, keeping the lowering atomic.
fn lowered_ty(ty: &Type, registry: &ImmutableGlobalRegistry) -> Option<LoweredTy> {
    if let Some(elem) = scalar_of(ty) {
        return Some(LoweredTy::Scalar(elem));
    }
    if let Some((elem, shape)) = tensor_elem_shape(ty) {
        return Some(LoweredTy::Tensor { elem, shape });
    }
    // A raw pointer (`*const T`/`*mut T`, `&T`) is an opaque `!llvm.ptr` — the ABI of the string-value
    // and FFI-pointer programs (`vx_stdout_write(buffer: *const u8, …)`, an extern returning
    // `*mut i8`). Matches the AST codegen's `lower_type` for `Type::Pointer`/`Type::Borrow`. (#231/#235)
    // A function/closure type (`fn(i32)->i32`, a `Closure1`'s `func` field) is also an opaque
    // `!llvm.ptr` — a materialized function pointer, called via `CallIndirect`. (#242)
    if matches!(
        ty,
        Type::Pointer(..) | Type::Borrow { .. } | Type::Function(..) | Type::Closure(..)
    ) {
        return Some(LoweredTy::Ptr);
    }
    match ty {
        // A payload-free (C-like) enum is a bare `i32` discriminant, not an aggregate (#227). The
        // resolver may spell an enum type as `Type::Enum` or `Type::Struct`, so match on the name; a
        // data-carrying enum is absent from `enum_variants` and falls through to the aggregate case.
        Type::Enum(name, _) | Type::Struct(name, _)
            if registry.enum_variants.contains_key(name) =>
        {
            Some(LoweredTy::Scalar(ElementType::I32))
        }
        Type::Struct(name, id) | Type::Enum(name, id) => {
            // The attached GID when name resolution modelled it (non-stub), else by name — a
            // monomorphized cross-module signature may carry an unresolved base (`Struct("Vec", None)`).
            let gid = id
                .filter(|g| registry.layouts.get(g).is_some_and(|d| d.align_bytes != 0))
                .or_else(|| struct_layout_gid_by_name(registry, name.as_ref()))?;
            Some(LoweredTy::Aggregate(gid))
        }
        // A monomorphized generic struct instance (`Vec<i32>`): its layout is the base nominal's when
        // that layout is *instance-independent* — every generic parameter appears only behind a
        // pointer (a pointer field is 8 bytes for any `T`), as in `Vec<T> { data: *mut T, len, cap }`.
        // A by-value generic field (`Box<T> { value: T }`) leaves the base layout the 0/0 stub, so
        // this resolves exactly the pointer-backed containers and declines the rest. The base may be
        // an unresolved `Struct(_, None)` (a cross-module mono), handled by the nominal arm's
        // name fallback.
        Type::GenericInstance(base, _) => lowered_ty(base, registry),
        _ => None,
    }
}

/// Whether the (top-level) body contains control flow (`if`/`loop`/`for`) — the trigger for the
/// memory model, so mutated or loop-carried locals survive across basic blocks. Nested control flow
/// rides on its enclosing top-level construct, and the `lower_*` helpers recurse in memory mode.
fn body_has_control_flow(stmts: &[Statement]) -> bool {
    stmts.iter().any(|s| match s {
        Statement::Loop(_) | Statement::ForLoop(_) => true,
        Statement::ExprStmt(e) => {
            matches!(e.expr, Expr::If(_) | Expr::Match(_)) || expr_has_logical(&e.expr)
        }
        // A value-position `if` (`let v = if .. { .. } else { .. }`, #201) lowers to blocks + a result
        // slot, which needs the memory model too — as does a short-circuit `&&`/`||` (#239).
        Statement::LetDecl(l) => matches!(l.expr, Expr::If(_)) || expr_has_logical(&l.expr),
        Statement::Return(r) => matches!(r.expr, Expr::If(_)) || expr_has_logical(&r.expr),
        Statement::Assign(a) => matches!(a.rhs, Expr::If(_)) || expr_has_logical(&a.rhs),
        _ => false,
    })
}

/// Whether an expression contains a short-circuit logical op (`&&`/`||`) that forces the memory
/// model — the operator lowers to a branch skeleton with a cross-block result slot (#239). Recurses
/// the compound forms a logical op realistically nests in; a leaf (or an unhandled variant) is
/// `false`, so at worst such a function declines rather than lowering incorrectly.
fn expr_has_logical(e: &Expr) -> bool {
    match e {
        Expr::LogicalOp(_) => true,
        Expr::BinaryOp(b) => expr_has_logical(&b.lhs) || expr_has_logical(&b.rhs),
        Expr::RelationalOp(r) => expr_has_logical(&r.lhs) || expr_has_logical(&r.rhs),
        Expr::UnaryOp(u) => expr_has_logical(&u.expr),
        Expr::AsCast(c) => expr_has_logical(&c.expr),
        _ => false,
    }
}

/// Whether the (top-level) body constructs a struct into a local (`let x = S { .. }`) — the trigger
/// for the memory model, since the constructed aggregate must live in an addressable slot.
fn body_constructs_struct(stmts: &[Statement]) -> bool {
    stmts
        .iter()
        .any(|s| matches!(s, Statement::LetDecl(l) if matches!(l.expr, Expr::StructInit(_))))
}

/// Per-body local-usage facts driving slot allocation (#230/#275): which base locals must have their
/// address **materialized** (a borrow of them escapes — §5), and which are reassigned (need a memory
/// slot to carry the value across a block boundary under control flow — §3.2 step 2).
///
/// The place-vs-materialize split is the #275 refinement of the old "any `&x` slots x" rule: a `&x`
/// that only feeds a local `*r` never needs a real address, so its base stays a register and the
/// reference is a symbolic `Binding::Place` (Example A). A borrow that *escapes* — a call argument, a
/// return, a struct field, or a reference local used as anything but `*r` — still materializes.
///
/// Missing a fact is *safe*: an under-collected escape leaves a base a register, and the borrow that
/// actually escapes then hits the `Expr::Borrow` arm with a non-slot base and **declines** to the AST
/// oracle (the MLIR verifier is the backstop) — never a silent miscompile. The walk still covers every
/// form the flat subset lowers, to avoid needless declines. Body-local, no fixpoint (§3.4).
#[derive(Default)]
struct LocalUses {
    materialized: HashSet<Symbol>,
    mutated: HashSet<Symbol>,
    /// Ref-locals realized as symbolic `Binding::Place`s — a `let r = &<lvalue>` whose reference does
    /// not escape. A scalar place (`&x`, empty path — Example A) additionally requires immutability,
    /// read-only use, and an un-materialized base; a field place (`&[mut] p.x` — Example B/C) has no
    /// such restriction (`*r`/`*r = v` resolve to a field load/store). (#275)
    place_bindings: HashSet<Symbol>,
}

/// A `let r = &[mut] <lvalue>` place candidate: the root local of the borrowed lvalue, whether the
/// lvalue is a field access (a member chain) or the bare local, and the borrow's mutability. (#275)
struct Candidate {
    base: Symbol,
    is_field: bool,
    is_mut: bool,
}

/// The root local of an lvalue path — the local a `&x` / `&p.x.y` ultimately borrows. `None` for a
/// non-lvalue (a call result, an index, …), which is therefore never a place candidate. (#275)
fn place_root(e: &Expr) -> Option<Symbol> {
    match e {
        Expr::Identifier(id) => Some(id.name.clone()),
        Expr::MemberAccess(m) => place_root(&m.base),
        _ => None,
    }
}

/// Index of the `(root, path)` key in `keys`, or `None` if absent — the group-id lookup for the alias
/// reduction. (#275)
fn place_group_of(keys: &[(Symbol, Vec<Symbol>)], root: &Symbol, path: &[Symbol]) -> Option<usize> {
    keys.iter()
        .position(|(r, p)| r == root && p.as_slice() == path)
}

/// Reduce collected place-write field stores to codegen's numeric alias table: assign each distinct
/// `(root, path)` a group id (first-appearance order) and, per store, the group ids of its disjoint
/// siblings (same root, non-overlapping path). Codegen turns a group into a `distinct[]` alias scope
/// and its siblings into `noalias_scopes`. Stores to the *same* field share a group (they alias); to
/// disjoint fields become mutual `noalias` siblings. (#275, §5.4)
fn reduce_place_alias(stores: &[(usize, Symbol, Vec<Symbol>)]) -> Vec<(usize, usize, Vec<usize>)> {
    let mut keys: Vec<(Symbol, Vec<Symbol>)> = Vec::new();
    for (_, root, path) in stores {
        if place_group_of(&keys, root, path).is_none() {
            keys.push((root.clone(), path.clone()));
        }
    }
    stores
        .iter()
        .map(|(pos, root, path)| {
            let own = place_group_of(&keys, root, path).unwrap();
            let siblings = keys
                .iter()
                .enumerate()
                .filter(|(gi, (r, p))| {
                    *gi != own && r == root && !crate::hir::places::paths_may_alias(path, p)
                })
                .map(|(gi, _)| gi)
                .collect();
            (*pos, own, siblings)
        })
        .collect()
}

fn analyze_local_uses(stmts: &[Statement]) -> LocalUses {
    // Pass 1: direct reassignments, `let r = &<lvalue>` place candidates, and every *other* (raw)
    // borrow base.
    let mut scan = BorrowScan::default();
    scan.block(stmts);
    // Pass 2: which candidate ref-locals escape (used as anything but `*r`) or are written through.
    let mut refs = RefUseScan {
        candidates: &scan.place_candidates,
        escaping: HashSet::new(),
        written: HashSet::new(),
    };
    refs.block(stmts);
    // A base is materialized if it is raw-borrowed, or if a *scalar* candidate over it cannot be a
    // place (mutable, escaping, or written — Example A is immutable read-only). A field candidate's base
    // is an aggregate that is already addressable, so it never forces materialization this way.
    let mut materialized = scan.raw_borrowed;
    for (r, c) in &scan.place_candidates {
        let cannot_place = c.is_mut || refs.escaping.contains(r) || refs.written.contains(r);
        if !c.is_field && cannot_place {
            materialized.insert(c.base.clone());
        }
    }
    // Realize the places: a field candidate whenever it does not escape (writes are fine — they resolve
    // to field stores); a scalar candidate under the Example A restrictions.
    let mut place_bindings = HashSet::new();
    for (r, c) in &scan.place_candidates {
        let realized = if c.is_field {
            !refs.escaping.contains(r)
        } else {
            !c.is_mut
                && !refs.escaping.contains(r)
                && !refs.written.contains(r)
                && !materialized.contains(&c.base)
        };
        if realized {
            place_bindings.insert(r.clone());
        }
    }
    LocalUses {
        materialized,
        mutated: scan.mutated,
        place_bindings,
    }
}

/// Pass-1 accumulator: reassigned locals, `let r = &<lvalue>` place candidates, and the root bases of
/// every *other* borrow (which forces materialization).
#[derive(Default)]
struct BorrowScan {
    mutated: HashSet<Symbol>,
    place_candidates: HashMap<Symbol, Candidate>,
    raw_borrowed: HashSet<Symbol>,
}

impl BorrowScan {
    fn block(&mut self, stmts: &[Statement]) {
        for s in stmts {
            match s {
                // `let r = &[mut] <lvalue>` is a place candidate — record it and do *not* count the
                // borrow as raw. The lvalue is an identifier (scalar place) or a member chain rooted at
                // one (field place). Any other initializer is scanned normally.
                Statement::LetDecl(l) => {
                    if let Expr::Borrow(b) = &l.expr {
                        if let Some(base) = place_root(&b.expr) {
                            self.place_candidates.insert(
                                l.name.clone(),
                                Candidate {
                                    base,
                                    is_field: matches!(&*b.expr, Expr::MemberAccess(_)),
                                    is_mut: b.is_mut,
                                },
                            );
                            continue;
                        }
                    }
                    self.expr(&l.expr);
                }
                Statement::Return(r) => self.expr(&r.expr),
                Statement::ExprStmt(e) => self.expr(&e.expr),
                Statement::Assign(a) => {
                    if let Expr::Identifier(id) = &a.lhs {
                        self.mutated.insert(id.name.clone());
                    }
                    self.expr(&a.lhs);
                    self.expr(&a.rhs);
                }
                Statement::CompoundAssign(a) => {
                    if let Expr::Identifier(id) = &a.lhs {
                        self.mutated.insert(id.name.clone());
                    }
                    self.expr(&a.lhs);
                    self.expr(&a.rhs);
                }
                Statement::ForLoop(f) => {
                    self.expr(&f.iterable);
                    self.block(&f.body);
                }
                Statement::Loop(l) => self.block(&l.body),
                Statement::Assert(a) => self.expr(&a.expr),
                _ => {}
            }
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            // A borrow reached here (not a `let r = &<lvalue>` candidate) is a raw/escaping borrow: its
            // root base must be materialized.
            Expr::Borrow(b) => {
                if let Some(base) = place_root(&b.expr) {
                    self.raw_borrowed.insert(base);
                }
                self.expr(&b.expr);
            }
            Expr::Dereference(d) => self.expr(&d.expr),
            Expr::BinaryOp(b) => {
                self.expr(&b.lhs);
                self.expr(&b.rhs);
            }
            Expr::RelationalOp(r) => {
                self.expr(&r.lhs);
                self.expr(&r.rhs);
            }
            Expr::LogicalOp(l) => {
                self.expr(&l.lhs);
                self.expr(&l.rhs);
            }
            Expr::UnaryOp(un) => self.expr(&un.expr),
            Expr::AsCast(c) => self.expr(&c.expr),
            Expr::FunctionCall(fc) => fc.args.iter().for_each(|a| self.expr(a)),
            Expr::MethodCall(mc) => {
                self.expr(&mc.base);
                mc.args.iter().for_each(|a| self.expr(a));
            }
            Expr::MemberAccess(m) => self.expr(&m.base),
            Expr::IndexAccess(ix) => {
                self.expr(&ix.base);
                self.expr(&ix.index);
            }
            Expr::Array(arr) => arr.elements.iter().for_each(|el| self.expr(el)),
            Expr::If(i) => {
                self.expr(&i.cond);
                self.block(&i.then_block);
                if let Some(eb) = &i.else_block {
                    self.block(eb);
                }
            }
            Expr::Match(m) => {
                self.expr(&m.expr);
                m.arms.iter().for_each(|arm| self.block(&arm.body));
            }
            Expr::UnsafeBlock(ub) => {
                self.block(&ub.stmts);
                if let Some(r) = &ub.ret {
                    self.expr(r);
                }
            }
            Expr::ComptimeBlock(c) => {
                self.block(&c.stmts);
                if let Some(r) = &c.ret {
                    self.expr(r);
                }
            }
            // A borrow inside an aggregate literal (`Holder { r : &x }`, a reference-typed field — #275
            // M4) escapes into the aggregate, so its base must materialize. Descend into the field/payload
            // values so the `&x` is seen (else `x` stays a register and the `&x` declines at lowering).
            Expr::StructInit(si) => si.fields.iter().for_each(|(_, e)| self.expr(e)),
            Expr::EnumVariant(ev) => {
                if let Some(p) = &ev.payload {
                    p.iter().for_each(|e| self.expr(e));
                }
            }
            _ => {}
        }
    }
}

/// Pass-2 scan: classify each place-candidate ref-local. A ref `r` **escapes** if it appears as
/// anything but the operand of a `*r` (a call arg, a return, an rvalue, a re-borrow); it is **written**
/// if `*r = ..`. Either disqualifies its base from staying a register.
struct RefUseScan<'a> {
    candidates: &'a HashMap<Symbol, Candidate>,
    escaping: HashSet<Symbol>,
    written: HashSet<Symbol>,
}

impl RefUseScan<'_> {
    fn block(&mut self, stmts: &[Statement]) {
        for s in stmts {
            match s {
                // `*r = v` writes through the ref; the deref-lhs is a write use (not an escape), the rhs
                // is scanned normally.
                Statement::Assign(a) => {
                    if let Some(r) = deref_ident(&a.lhs) {
                        if self.candidates.contains_key(&r) {
                            self.written.insert(r);
                        } else {
                            self.expr(&a.lhs);
                        }
                    } else {
                        self.expr(&a.lhs);
                    }
                    self.expr(&a.rhs);
                }
                Statement::CompoundAssign(a) => {
                    if let Some(r) = deref_ident(&a.lhs) {
                        if self.candidates.contains_key(&r) {
                            self.written.insert(r);
                        } else {
                            self.expr(&a.lhs);
                        }
                    } else {
                        self.expr(&a.lhs);
                    }
                    self.expr(&a.rhs);
                }
                Statement::LetDecl(l) => self.expr(&l.expr),
                Statement::Return(r) => self.expr(&r.expr),
                Statement::ExprStmt(e) => self.expr(&e.expr),
                Statement::ForLoop(f) => {
                    self.expr(&f.iterable);
                    self.block(&f.body);
                }
                Statement::Loop(l) => self.block(&l.body),
                Statement::Assert(a) => self.expr(&a.expr),
                _ => {}
            }
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            // `*r` is the one non-escaping use — do not descend into `r`. Any *other* mention of a
            // ref-local is an escape.
            Expr::Dereference(d) => {
                if let Expr::Identifier(id) = &*d.expr {
                    if self.candidates.contains_key(&id.name) {
                        return;
                    }
                }
                self.expr(&d.expr);
            }
            Expr::Identifier(id) if self.candidates.contains_key(&id.name) => {
                self.escaping.insert(id.name.clone());
            }
            Expr::Borrow(b) => self.expr(&b.expr),
            Expr::BinaryOp(b) => {
                self.expr(&b.lhs);
                self.expr(&b.rhs);
            }
            Expr::RelationalOp(r) => {
                self.expr(&r.lhs);
                self.expr(&r.rhs);
            }
            Expr::LogicalOp(l) => {
                self.expr(&l.lhs);
                self.expr(&l.rhs);
            }
            Expr::UnaryOp(un) => self.expr(&un.expr),
            Expr::AsCast(c) => self.expr(&c.expr),
            Expr::FunctionCall(fc) => fc.args.iter().for_each(|a| self.expr(a)),
            Expr::MethodCall(mc) => {
                self.expr(&mc.base);
                mc.args.iter().for_each(|a| self.expr(a));
            }
            Expr::MemberAccess(m) => self.expr(&m.base),
            Expr::IndexAccess(ix) => {
                self.expr(&ix.base);
                self.expr(&ix.index);
            }
            Expr::Array(arr) => arr.elements.iter().for_each(|el| self.expr(el)),
            Expr::If(i) => {
                self.expr(&i.cond);
                self.block(&i.then_block);
                if let Some(eb) = &i.else_block {
                    self.block(eb);
                }
            }
            Expr::Match(m) => {
                self.expr(&m.expr);
                m.arms.iter().for_each(|arm| self.block(&arm.body));
            }
            Expr::UnsafeBlock(ub) => {
                self.block(&ub.stmts);
                if let Some(r) = &ub.ret {
                    self.expr(r);
                }
            }
            Expr::ComptimeBlock(c) => {
                self.block(&c.stmts);
                if let Some(r) = &c.ret {
                    self.expr(r);
                }
            }
            // A ref-local mentioned inside an aggregate literal escapes into it (a non-`*r` use), so it
            // can't stay a symbolic place — descend to catch it, matching pass 1. (#275 M4)
            Expr::StructInit(si) => si.fields.iter().for_each(|(_, e)| self.expr(e)),
            Expr::EnumVariant(ev) => {
                if let Some(p) = &ev.payload {
                    p.iter().for_each(|e| self.expr(e));
                }
            }
            _ => {}
        }
    }
}

/// `*r` -> `Some(r)` when the dereferenced expression is a plain identifier (a ref-local write target).
fn deref_ident(e: &Expr) -> Option<Symbol> {
    match e {
        Expr::Dereference(d) => match &*d.expr {
            Expr::Identifier(id) => Some(id.name.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn simple_ident(e: &Expr) -> Option<Symbol> {
    match e {
        Expr::Identifier(id) => Some(id.name.clone()),
        _ => None,
    }
}

/// Parse a monomorphized enum instance name into its base name + type arguments: `"Option<i32>"` ->
/// `("Option", [i32])`, `"Color"` -> `("Color", [])`. Type args are parsed as scalars (else a nominal
/// `Struct`), matching the AST codegen's string-keyed approach. (#242)
fn parse_enum_instance(name: &str) -> (String, Vec<Type>) {
    let Some(lt) = name.find('<') else {
        return (name.to_string(), Vec::new());
    };
    let base = name[..lt].to_string();
    let inner = &name[lt + 1..name.rfind('>').unwrap_or(name.len())];
    let args = inner
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| parse_scalar_type_arg(s.trim()))
        .collect();
    (base, args)
}

/// Parse a type-argument string to a `Type`: a scalar spelling to its `ElementType`, anything else to
/// a nominal `Struct` (name resolution is unavailable here; substitution only needs the leaf identity).
fn parse_scalar_type_arg(s: &str) -> Type {
    use ElementType::*;
    let e = match s {
        "i8" => I8,
        "u8" => U8,
        "i16" => I16,
        "u16" => U16,
        "i32" => I32,
        "u32" => U32,
        "i64" => I64,
        "u64" => U64,
        "f16" => F16,
        "bf16" => BF16,
        "f32" => F32,
        "f64" => F64,
        "bool" | "Bool" => Bool,
        other => return Type::Struct(other.to_string().into(), None),
    };
    Type::Scalar(e)
}

/// The byte size, alignment, and MLIR type string of an enum payload field — a scalar or a pointer.
/// `None` for a by-value aggregate/tensor/generic payload (not modelled). (#242)
fn enum_payload_field(ty: &Type) -> Option<(u64, u64, String)> {
    match ty {
        Type::Scalar(ElementType::Generic(_)) => None,
        Type::Scalar(e) => {
            let (s, a) = crate::layout::scalar_size_align(e)?;
            Some((s as u64, a as u64, element_mlir(e)?.to_string()))
        }
        Type::Pointer(..) | Type::Borrow { .. } | Type::Ref(..) => {
            Some((8, 8, "!llvm.ptr".to_string()))
        }
        _ => None,
    }
}

/// The MLIR scalar type string for an element type (`i32`, `f32`, `i1` for bool, …) — the flat-lowerer
/// counterpart of codegen's `mlir_scalar`, used to spell a synthesized enum-instance field. (#242)
fn element_mlir(e: &ElementType) -> Option<&'static str> {
    use ElementType::*;
    Some(match e {
        F16 => "f16",
        F32 => "f32",
        F64 => "f64",
        BF16 => "bf16",
        I8 | U8 => "i8",
        I16 | U16 => "i16",
        I32 | U32 => "i32",
        I64 | U64 => "i64",
        I128 | U128 => "i128",
        I4 | U4 => "i4",
        Bool => "i1",
        // fp8 is capacity/declaration-only for now: the JIT has no fp8 arithmetic,
        // so the flat path declines. Compute support is #249.
        F8E4M3 | F8E5M2 => return None,
        Generic(_) => return None,
    })
}

/// Strip one borrow/pointer/ref wrapper, yielding the pointee (or the type itself if not a
/// reference) — used to look through `self : &mut Vec<i32>` to the `Vec<i32>` it points at. (#242)
fn deref_to_pointee(ty: &Type) -> &Type {
    match ty {
        Type::Borrow { inner, .. } | Type::Pointer(inner, ..) | Type::Ref(inner, ..) => inner,
        other => other,
    }
}

/// The base struct name + type-argument list of a nominal type, looking through a `GenericInstance`
/// (`Vec<i32>` -> `("Vec", [i32])`) or a plain nominal (`Point` -> `("Point", [])`). `None` for a
/// non-nominal. The name keys the registry's base struct fields (#242).
fn nominal_name_and_args(ty: &Type) -> Option<(Symbol, Vec<Type>)> {
    match ty {
        Type::GenericInstance(base, args) => {
            let (name, _) = nominal_name_and_args(base)?;
            Some((name, args.clone()))
        }
        Type::Struct(name, _) | Type::Enum(name, _) => Some((name.clone(), Vec::new())),
        _ => None,
    }
}

/// Substitute an instance's type arguments into a base struct's generic field type: build
/// `{param -> arg}` from the generic parameter names (declaration order) and apply it. With no
/// arguments (a non-generic struct) the field type passes through unchanged. This is how
/// `Vec<T>`'s `data : *mut T` becomes `*mut i32` for a `Vec<i32>` access (#242).
fn substitute_generics(fty: &Type, generics: &[Symbol], args: &[Type]) -> Type {
    if args.is_empty() {
        return fty.clone();
    }
    let mut mapping = HashMap::new();
    for (g, a) in generics.iter().zip(args) {
        mapping.insert(g.clone(), a.clone());
    }
    fty.substitute(&mapping)
}

/// The lowered element type of a raw pointer, for lowering a raw-pointer index `p[i]`:
/// `*mut i32` -> `Scalar(i32)` (`Vec<i32>`'s `self.data[i]`), `*mut Vec<i32>` -> `Aggregate(gid)`
/// (`Vec<Vec<i32>>`'s element, stored/loaded as a whole `!llvm.struct` by value). Only scalar and
/// aggregate elements are modelled — a pointer-to-pointer or pointer-to-tensor element declines. (#242)
fn pointer_elem_ty(ty: &Type, registry: &ImmutableGlobalRegistry) -> Option<LoweredTy> {
    let inner = match ty {
        Type::Pointer(inner, ..) | Type::Borrow { inner, .. } | Type::Ref(inner, ..) => {
            inner.as_ref()
        }
        _ => return None,
    };
    match lowered_ty(inner, registry)? {
        // A pointer whose pointee is itself a pointer (`&&T`, `*mut *mut T`): the deref loads/stores a
        // bare `!llvm.ptr` element. This is what makes a *materialized* nested reference (`&mut &mut i32`,
        // whose `rr` is a real ptr-to-ptr rather than a symbolic place) deref through `PtrIndex`. (#278)
        e @ (LoweredTy::Scalar(_) | LoweredTy::Aggregate(_) | LoweredTy::Ptr) => Some(e),
        _ => None,
    }
}

/// The modelled layout GID of a struct/enum by its base name — the fallback for when name resolution
/// left a *monomorphized cross-module* instance's base GID unattached (`Struct("Vec", None)`, as a
/// mono's substituted signature carries). Searches the frozen layouts for a non-stub definition of
/// that base name (`Vec<i32>` -> `Vec`); declines on an ambiguous name (two distinct GIDs) so a wrong
/// layout is never chosen. (#242)
fn struct_layout_gid_by_name(registry: &ImmutableGlobalRegistry, name: &str) -> Option<TypeId> {
    let base = name.split('<').next().unwrap_or(name);
    let mut found: Option<TypeId> = None;
    for def in registry.layouts.values() {
        if def.name == base && def.align_bytes != 0 {
            if found.is_some_and(|g| g != def.id) {
                return None; // ambiguous name across modules — decline, keep the AST oracle
            }
            found = Some(def.id);
        }
    }
    found
}

/// The layout GID of the aggregate a (borrow/pointer-to-)nominal type names, resolving a
/// monomorphized generic instance to its base nominal (`&mut Vec<i32>` / `Vec<i32>` -> the `Vec`
/// layout GID). Uses the attached GID when name resolution modelled it, else resolves by name (a
/// monomorphized cross-module instance's base may carry no GID). Requires a modelled (non-stub)
/// layout. (#242)
fn agg_gid_of_ty(ty: &Type, registry: &ImmutableGlobalRegistry) -> Option<TypeId> {
    let pointee = deref_to_pointee(ty);
    // A data-carrying enum instance (`Option<i32>`, as `self : &Option<T>` in an `Option` method) has a
    // *synthesized* per-instance `{ tag, payload }` layout keyed by `enum_instance_gid`, not a registry
    // layout — resolve it directly so the pointer binds as an aggregate reference (`match *self`). (#242)
    if let Type::GenericInstance(base, args) = pointee {
        if let Type::Enum(n, _) | Type::Struct(n, _) = base.as_ref() {
            if registry
                .enum_data
                .get(n.as_ref())
                .is_some_and(|d| d.variants.iter().any(|(_, p)| !p.is_empty()))
            {
                return Some(enum_instance_gid(n.as_ref(), args));
            }
        }
    }
    let nominal = match pointee {
        Type::GenericInstance(base, _) => base.as_ref(),
        other => other,
    };
    let (name, gid_opt) = match nominal {
        Type::Struct(n, id) | Type::Enum(n, id) => (n.as_ref(), *id),
        _ => return None,
    };
    gid_opt
        .filter(|id| registry.layouts.get(id).is_some_and(|d| d.align_bytes != 0))
        .or_else(|| struct_layout_gid_by_name(registry, name))
}

/// Whether a parameter type is a pointer/borrow to a modelled aggregate (`self : &mut Vec<i32>`) —
/// such a param binds as an SSA register even in memory mode (a block argument that dominates all
/// blocks; mutation flows through the pointer to the pointee, not to the register). (#242)
fn is_ptr_to_agg(ty: &Type, registry: &ImmutableGlobalRegistry) -> bool {
    matches!(ty, Type::Borrow { .. } | Type::Pointer(..)) && agg_gid_of_ty(ty, registry).is_some()
}

/// Pack an `if`'s two branch targets into `CondBr`'s `imm`: `then | (else << 32)`.
fn pack_targets(then_b: u32, else_b: u32) -> u64 {
    (then_b as u64) | ((else_b as u64) << 32)
}

/// The scalar element type of a parameter type, or `None` for non-scalars / generic scalars (which
/// the flat lowering does not model).
fn scalar_of(ty: &Type) -> Option<ElementType> {
    match ty {
        Type::Scalar(ElementType::Generic(_)) => None,
        Type::Scalar(e) => Some(e.clone()),
        _ => None,
    }
}

/// The byte size a `sizeof<T>()` folds to, matching the AST codegen (`SizeOfExpr`) for the scalar and
/// pointer types they agree on. Returns `None` for anything else (struct/enum/tensor/`i128`), so the
/// enclosing `comptime` block declines to the AST path rather than risk a divergent size. (#228)
fn sizeof_bytes(ty: &Type) -> Option<u64> {
    use ElementType::*;
    Some(match ty {
        Type::Scalar(F32 | I32 | U32) => 4,
        Type::Scalar(F64 | I64 | U64) => 8,
        Type::Scalar(I8 | U8 | Bool) => 1,
        Type::Scalar(I16 | U16 | BF16 | F16) => 2,
        Type::Pointer(..) | Type::Borrow { .. } | Type::Ref(..) => 8,
        _ => return None,
    })
}

fn binop_opcode(op: &BinaryOp) -> Option<Opcode> {
    Some(match op {
        BinaryOp::Add => Opcode::Add,
        BinaryOp::Sub => Opcode::Sub,
        BinaryOp::Mul => Opcode::Mul,
        BinaryOp::Div => Opcode::Div,
        BinaryOp::MatMul => Opcode::Matmul,
    })
}

/// The `Cmp` relation code stored in `imm` (kept in sync with codegen's decoding).
fn rel_code(op: &RelationalOp) -> u64 {
    match op {
        RelationalOp::Eq => 0,
        RelationalOp::NotEq => 1,
        RelationalOp::Lt => 2,
        RelationalOp::Gt => 3,
        RelationalOp::Le => 4,
        RelationalOp::Ge => 5,
    }
}

/// The element type of a numeric literal: its checked annotation when concrete, else inferred from
/// the spelling.
fn number_elem(n: &NumberExpr) -> Option<ElementType> {
    match &n.ty {
        Some(ElementType::Generic(_)) | None => infer_elem(n.value.as_ref()),
        Some(e) => Some(e.clone()),
    }
}

/// The default element type for an *untyped* literal reaching the flat lowerer — shares the type
/// checker's fallback so both backends agree on an un-annotated literal (#240). In practice the
/// checker types every value-position literal before lowering, so this is only exercised by the
/// checker-less flat unit tests.
fn infer_elem(s: &str) -> Option<ElementType> {
    Some(crate::parser::expr::default_number_elem(s))
}

fn is_float(e: &ElementType) -> bool {
    e.is_float()
}

/// Encode a literal's raw 64-bit `imm`: float bit-pattern for float types, else the integer value.
fn encode_imm(s: &str, elem: &ElementType) -> Option<u64> {
    if is_float(elem) {
        Some(s.parse::<f64>().ok()?.to_bits())
    } else if *elem == ElementType::Bool {
        match s {
            "true" => Some(1),
            "false" => Some(0),
            _ => s.parse::<u64>().ok(),
        }
    } else if let Ok(i) = s.parse::<i64>() {
        Some(i as u64)
    } else {
        s.parse::<u64>().ok()
    }
}

/// Debug-only structural check on a worker's flat HIR: every value `type_idx` is in-bounds (effect
/// instructions carry the [`NO_TYPE`] sentinel), every operand an opcode reads names a
/// strictly-earlier instruction (temporaries are block-local, so the linear check still captures SSA
/// dominance), and every branch targets a declared block. One worker holds exactly one function's
/// stream.
///
/// Gated on `any(debug_assertions, test)`, not `debug_assertions` alone: the production callers in
/// `pipeline.rs` are `#[cfg(debug_assertions)]`, but the `#[cfg(test)]` unit tests below call it
/// directly, so it must also exist under `cargo test --release` (test on, debug-assertions off).
#[cfg(any(debug_assertions, test))]
pub fn verify_hir_stream(worker: &LocalWorkerState) {
    let n_types = worker.local_type_stream.len() as u32;
    // Declared basic blocks (a `BlockStart` per block id) — branch targets must land in this set.
    let blocks: std::collections::HashSet<u64> = worker
        .local_hir_stream
        .iter()
        .filter(|ins| ins.opcode == Opcode::BlockStart)
        .map(|ins| ins.imm)
        .collect();

    for (i, ins) in worker.local_hir_stream.iter().enumerate() {
        let i = i as u32;
        if ins.type_idx.0 != NO_TYPE {
            assert!(
                ins.type_idx.0 < n_types,
                "HIR type_idx {} out of bounds ({}) at instruction {i}",
                ins.type_idx.0,
                n_types
            );
        }
        match ins.opcode {
            // Binary: both operands read.
            Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Matmul
            | Opcode::Cmp
            | Opcode::Store
            | Opcode::FieldStore
            | Opcode::TensorIndex
            | Opcode::TensorStore
            // `PtrIndex` reads base+index; `PtrStore` reads place+value (#242).
            | Opcode::PtrIndex
            | Opcode::PtrStore
            // `Reduce`'s operand2 is a real slice for `dot`, else the dominated `Register(0)`.
            | Opcode::Reduce => {
                assert!(
                    ins.operand1.0 < i && ins.operand2.0 < i,
                    "HIR operand not dominated at instruction {i}"
                );
            }
            // Unary: operand1 read.
            Opcode::Ret
            | Opcode::Cast
            | Opcode::Neg
            | Opcode::Not
            | Opcode::SlotLoad
            | Opcode::FieldLoad
            | Opcode::Transfer
            | Opcode::Print
            | Opcode::Arg => assert!(
                ins.operand1.0 < i,
                "HIR operand not dominated at instruction {i}"
            ),
            Opcode::CondBr => {
                assert!(
                    ins.operand1.0 < i,
                    "HIR cond not dominated at instruction {i}"
                );
                let then_b = ins.imm & 0xFFFF_FFFF;
                let else_b = ins.imm >> 32;
                assert!(
                    blocks.contains(&then_b) && blocks.contains(&else_b),
                    "HIR CondBr targets undeclared block(s) at instruction {i}"
                );
            }
            Opcode::Br => assert!(
                blocks.contains(&ins.imm),
                "HIR Br targets an undeclared block at instruction {i}"
            ),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::GlobalSession;
    use std::sync::Arc;

    fn parse_fn(src: &str) -> Function {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let prog = parser.parse().expect("parse failed");
        prog.functions.into_iter().next().expect("expected one fn")
    }

    fn worker() -> LocalWorkerState {
        LocalWorkerState::new(Arc::new(GlobalSession::new(1)))
    }

    fn opcodes(w: &LocalWorkerState) -> Vec<Opcode> {
        w.local_hir_stream.iter().map(|i| i.opcode).collect()
    }

    fn count(w: &LocalWorkerState, op: Opcode) -> usize {
        w.local_hir_stream.iter().filter(|i| i.opcode == op).count()
    }

    /// Parse a whole program, resolve names, build the *real* frozen registry (so aggregate layouts
    /// exist), type-check (so `StructInit`s get their GID annotated), and attempt to lower `fn_name`.
    /// Returns `(did_lower, worker)`.
    fn lower_with_registry(src: &str, fn_name: &str) -> (bool, LocalWorkerState) {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let mut prog = parser.parse().expect("parse failed");
        prog.module_path = "crate::t".into();
        let mut mods = vec![prog];
        let symbol_map = crate::resolver::build_symbol_map(&mods);
        mods[0].resolve_names(&symbol_map);
        let registry = crate::pipeline::build_frozen_registry(&mods).expect("registry builds");
        let mut worker = LocalWorkerState::new(Arc::new(GlobalSession::with_registry(1, registry)));

        // Type-check so the type checker annotates each `StructInit` with its struct GID.
        let env_mods: Vec<_> = mods.clone();
        let env = crate::hir::GlobalAstEnv::build(&env_mods);
        {
            let mut checker = crate::hir::TypeChecker::new(&env, &mut worker);
            for f in &mut mods[0].functions {
                checker.check_function(f);
            }
        }

        let func = mods
            .into_iter()
            .next()
            .unwrap()
            .functions
            .into_iter()
            .find(|f| f.name.as_ref() == fn_name)
            .expect("fn present");
        let did = lower_function_to_hir(&func, &mut worker);
        (did, worker)
    }

    #[test]
    fn struct_param_lowers_as_aggregate_slot() {
        // A struct parameter forces the memory model: its `Alloca` is sized from the registry
        // layout (Point = two i32s = 8 bytes) and its aggregate GID flows through the type stream.
        let (did, w) = lower_with_registry(
            "struct Point { x: i32, y: i32 }\nfn id(p: Point) -> Point { return p; }",
            "id",
        );
        assert!(did, "struct-param identity fn should lower");
        let alloca = w
            .local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::Alloca)
            .expect("an Alloca slot for the aggregate param");
        assert_eq!(alloca.imm, 8, "Point Alloca sized from its 8-byte layout");
        // A scalar's GID lives in module 0; an aggregate GID carries the (nonzero) module hash, so
        // its presence proves the aggregate type reached the flat type stream.
        assert!(
            w.local_type_stream.iter().any(|t| t.module_id() != 0),
            "the Point aggregate GID is in the type stream"
        );
        assert_eq!(count(&w, Opcode::Store), 1, "the incoming struct is stored");
        assert!(count(&w, Opcode::SlotLoad) >= 1, "return reads the slot");
        assert_eq!(count(&w, Opcode::Ret), 1);
        verify_hir_stream(&w);
    }

    #[test]
    fn generic_aggregate_param_resolves_via_base_layout() {
        // A monomorphized generic struct instance (`Wrap<i32>`) resolves to its base nominal's layout
        // when that layout is instance-independent — a pointer field is 8 bytes for any `T`, so
        // `Wrap<T> { data: *mut T, n: i32 }` has a concrete layout regardless of `T`. So a function
        // taking and returning `Wrap<i32>` HIR-lowers as an aggregate: the type-level foundation the
        // `Vec<T>` surface builds on. (The flat *emitter* still declines an aggregate with a pointer
        // field — a later layer — so this asserts the HIR lowering only.)
        let (did, w) = lower_with_registry(
            "struct Wrap<T> { data: *mut T, n: i32 }\n\
             fn id(w: Wrap<i32>) -> Wrap<i32> { return w; }",
            "id",
        );
        assert!(
            did,
            "generic-aggregate identity fn should HIR-lower via the base layout"
        );
        // The aggregate GID (nonzero module hash) reached the type stream — the generic instance
        // resolved to a concrete layout rather than declining.
        assert!(
            w.local_type_stream.iter().any(|t| t.module_id() != 0),
            "the Wrap<i32> aggregate GID is in the type stream"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn pointer_field_and_raw_index_lower_through_self_pointer() {
        // The `Vec` surface in miniature (#242): a `&mut Buf` self pointer whose fields are read/
        // written through the pointer (`FieldLoad`/`FieldStore`) and whose raw-pointer field is
        // indexed (`PtrIndex` place + `PtrStore`). `b.data[0] = v` is a raw-pointer element store;
        // `b.len = b.len + 1` reads then writes the scalar field.
        let (did, w) = lower_with_registry(
            "struct Buf { data: *mut i32, len: i32 }\n\
             fn bump(b: &mut Buf, v: i32) -> i32 { unsafe { b.data[0] = v; } \
             b.len = b.len + 1; return 0; }",
            "bump",
        );
        assert!(
            did,
            "field access + raw-pointer index through a self pointer should lower"
        );
        // `b.data` (a pointer field) -> a `FieldLoad`; `b.data[0] = v` -> a `PtrIndex` place + a
        // `PtrStore`; `b.len` read -> a `FieldLoad`; `b.len = ...` -> a `FieldStore`.
        assert_eq!(
            count(&w, Opcode::PtrIndex),
            1,
            "one raw-pointer element place"
        );
        assert_eq!(
            count(&w, Opcode::PtrStore),
            1,
            "one raw-pointer element store"
        );
        assert!(
            count(&w, Opcode::FieldLoad) >= 2,
            "b.data and b.len are read"
        );
        assert_eq!(count(&w, Opcode::FieldStore), 1, "b.len is written");
        // The self pointer binds as an SSA register (no aggregate `Alloca` for it), so the only
        // slots are the memory-mode scalar locals — never the `&mut Buf` itself.
        verify_hir_stream(&w);
    }

    #[test]
    fn struct_field_read_lowers_to_field_load() {
        // `p.y` becomes a `FieldLoad` off the aggregate slot at y's layout offset (4).
        let (did, w) = lower_with_registry(
            "struct Point { x: i32, y: i32 }\nfn gety(p: Point) -> i32 { return p.y; }",
            "gety",
        );
        assert!(did, "a scalar field read should lower");
        let fl = w
            .local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::FieldLoad)
            .expect("a FieldLoad instruction");
        assert_eq!(fl.imm, 4, "y is at byte offset 4 in {{x:i32, y:i32}}");
        assert_eq!(count(&w, Opcode::FieldLoad), 1);
        assert_eq!(count(&w, Opcode::Ret), 1);
        verify_hir_stream(&w);
    }

    #[test]
    fn field_read_offset_honours_alignment_padding() {
        // Rec { a: i8, b: i32 } -> b sits at offset 4 (3 bytes of padding after `a`); the FieldLoad
        // must use that padded offset, proving it comes from the real layout, not field order.
        let (did, w) = lower_with_registry(
            "struct Rec { a: i8, b: i32 }\nfn getb(r: Rec) -> i32 { return r.b; }",
            "getb",
        );
        assert!(did);
        let fl = w
            .local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::FieldLoad)
            .expect("a FieldLoad instruction");
        assert_eq!(fl.imm, 4, "b is at offset 4 after i8 + 3 bytes padding");
        verify_hir_stream(&w);
    }

    #[test]
    fn struct_construction_lowers_to_alloca_and_field_stores() {
        // `let p = Point { .. }` constructs in place: one sized Alloca + a FieldStore per field at
        // its layout offset; the later `p.y` is a FieldLoad off the same slot.
        let (did, w) = lower_with_registry(
            "struct Point { x: i32, y: i32 }\n\
             fn build() -> i32 { let p = Point { x: 7i32, y: 9i32 }; return p.y; }",
            "build",
        );
        assert!(did, "struct construction + field read should lower");
        assert_eq!(count(&w, Opcode::Alloca), 1, "one aggregate slot for p");
        let alloca = w
            .local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::Alloca)
            .unwrap();
        assert_eq!(alloca.imm, 8, "Point slot sized from its layout");
        assert_eq!(count(&w, Opcode::FieldStore), 2, "x and y stored");
        let store_offsets: Vec<u64> = w
            .local_hir_stream
            .iter()
            .filter(|i| i.opcode == Opcode::FieldStore)
            .map(|i| i.imm)
            .collect();
        assert!(
            store_offsets.contains(&0) && store_offsets.contains(&4),
            "fields stored at layout offsets 0 and 4, got {store_offsets:?}"
        );
        assert_eq!(count(&w, Opcode::FieldLoad), 1, "p.y read");
        verify_hir_stream(&w);
    }

    #[test]
    fn tensor_param_binds_as_ssa_reg_with_tensor_gid() {
        // A tensor parameter is a reference value (memref): it binds as an SSA register (no Alloca),
        // and its element+shape GID enters the flat type stream.
        let f = parse_fn("fn f(q: Tensor<f32, [2, 4]>) -> i32 { return 0; }");
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "tensor-param fn should lower"
        );
        assert_eq!(
            count(&w, Opcode::Alloca),
            0,
            "a tensor param is a reference, not stack-allocated"
        );
        let expected = tensor_gid(&ElementType::F32, &["2".to_string(), "4".to_string()]);
        assert!(
            w.local_type_stream.contains(&expected),
            "the tensor GID is in the type stream"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn tensor_full_index_yields_scalar_element() {
        // `q[0][1]` on a rank-2 tensor rank-reduces twice: [2,4] -> [4] -> scalar f32.
        let f = parse_fn("fn f(q: Tensor<f32, [2, 4]>) -> f32 { return q[0][1]; }");
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "tensor element read should lower"
        );
        assert_eq!(
            count(&w, Opcode::TensorIndex),
            2,
            "two rank-reducing indexes"
        );
        assert_eq!(count(&w, Opcode::Alloca), 0);
        // The intermediate row type and the final scalar element are both in the type stream.
        let row = tensor_gid(&ElementType::F32, &["4".to_string()]);
        assert!(
            w.local_type_stream.contains(&row),
            "row (rank-1) tensor GID present"
        );
        assert!(
            w.local_type_stream.contains(&scalar_gid(&ElementType::F32)),
            "scalar element GID present"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn tensor_partial_index_yields_row_view() {
        // `q[0]` on [2,4] yields a rank-1 row view [4] (one TensorIndex, tensor result).
        let f = parse_fn("fn f(q: Tensor<f32, [2, 4]>) -> Tensor<f32, [4]> { return q[0]; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w), "row view should lower");
        assert_eq!(count(&w, Opcode::TensorIndex), 1);
        let row = tensor_gid(&ElementType::F32, &["4".to_string()]);
        assert!(w.local_type_stream.contains(&row), "row tensor GID present");
        verify_hir_stream(&w);
    }

    fn reduce_imm(w: &LocalWorkerState) -> Option<u64> {
        w.local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::Reduce)
            .map(|i| i.imm)
    }

    /// The result-type GID of the first instruction with `op`.
    fn result_gid(w: &LocalWorkerState, op: Opcode) -> TypeId {
        let ins = w
            .local_hir_stream
            .iter()
            .find(|i| i.opcode == op)
            .expect("op present");
        w.local_type_stream[ins.type_idx.0 as usize]
    }

    #[test]
    fn tensor_elementwise_mul_yields_a_tensor() {
        // `a * b` on two tensors is elementwise: a `Mul` whose *result type* is the tensor.
        let f = parse_fn(
            "fn f(a: Tensor<f32, [4]>, b: Tensor<f32, [4]>) -> Tensor<f32, [4]> { return a * b; }",
        );
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "elementwise mul should lower"
        );
        assert_eq!(count(&w, Opcode::Mul), 1);
        assert_eq!(
            result_gid(&w, Opcode::Mul),
            tensor_gid(&ElementType::F32, &["4".to_string()]),
            "elementwise result is the tensor type"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn scalar_times_tensor_broadcasts_to_a_tensor() {
        // `s * a` (scalar on the left) still yields the tensor type -- the result-type pick must
        // follow the tensor operand, not the lhs.
        let f = parse_fn("fn f(a: Tensor<f32, [4]>, s: f32) -> Tensor<f32, [4]> { return s * a; }");
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "scalar*tensor should lower"
        );
        assert_eq!(
            result_gid(&w, Opcode::Mul),
            tensor_gid(&ElementType::F32, &["4".to_string()]),
            "scalar broadcasts: result is the tensor type"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn slice_dot_reduces_two_slices_to_a_scalar() {
        // `dot(q, k)` over two rank-1 slices -> one Reduce (kind 0 = dot), scalar f32 result.
        let f = parse_fn(
            "fn dotp(q: Tensor<f32, [4]>, k: Tensor<f32, [4]>) -> f32 { return dot(q, k); }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w), "dot should lower");
        assert_eq!(count(&w, Opcode::Reduce), 1);
        assert_eq!(reduce_imm(&w), Some(0), "dot kind");
        assert!(
            w.local_type_stream.contains(&scalar_gid(&ElementType::F32)),
            "scalar result GID present"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn slice_sum_reduces_one_slice_to_a_scalar() {
        let f = parse_fn("fn s(q: Tensor<f32, [4]>) -> f32 { return sum(q); }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w), "sum should lower");
        assert_eq!(count(&w, Opcode::Reduce), 1);
        assert_eq!(reduce_imm(&w), Some(1), "sum kind");
        verify_hir_stream(&w);
    }

    #[test]
    fn dot_of_indexed_rows_lowers() {
        // The FlashAttention shape: `dot(q[i], k[j])` -> index each rank-2 tensor to a row, then
        // reduce. Two TensorIndex feed one Reduce.
        let f = parse_fn(
            "fn score(q: Tensor<f32, [2, 4]>, k: Tensor<f32, [2, 4]>) -> f32 { return dot(q[0], k[0]); }",
        );
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "dot of rows should lower"
        );
        assert_eq!(count(&w, Opcode::TensorIndex), 2, "one index per operand");
        assert_eq!(count(&w, Opcode::Reduce), 1);
        assert_eq!(reduce_imm(&w), Some(0));
        verify_hir_stream(&w);
    }

    #[test]
    fn flashattention_write_path_composes() {
        // Allocate the output, weight v by a scaled score, and store the row back -- alloc + index +
        // dot(reduce) + elementwise + store, the whole non-scalar surface in one flat stream.
        let f = parse_fn(
            "fn attn(q: Tensor<f32, [2, 4]>, k: Tensor<f32, [2, 4]>, v: Tensor<f32, [4]>, scale: f32) \
             -> Tensor<f32, [2, 4]> \
             { let o = Tensor<f32>([2, 4]); o[0] = v * (dot(q[0], k[0]) * scale); return o; }",
        );
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "the FA write path should lower"
        );
        assert_eq!(count(&w, Opcode::TensorAlloc), 1, "output buffer");
        assert_eq!(count(&w, Opcode::TensorIndex), 3, "q[0], k[0], o[0]");
        assert_eq!(count(&w, Opcode::Reduce), 1, "the dot");
        assert_eq!(count(&w, Opcode::Mul), 2, "scale the score, then weight v");
        assert_eq!(count(&w, Opcode::TensorStore), 1, "o[0] = ...");
        verify_hir_stream(&w);
    }

    #[test]
    fn flashattention_score_expression_composes() {
        // The FA inner score `dot(q[i], k[j]) * scale`: index -> reduce -> scalar multiply, proving
        // the tensor pieces compose end to end into one flat stream.
        let f = parse_fn(
            "fn score(q: Tensor<f32, [2, 4]>, k: Tensor<f32, [2, 4]>, scale: f32) -> f32 \
             { return dot(q[0], k[0]) * scale; }",
        );
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "the FA score expression should lower"
        );
        assert_eq!(count(&w, Opcode::TensorIndex), 2, "q[0] and k[0]");
        assert_eq!(count(&w, Opcode::Reduce), 1, "the dot");
        assert_eq!(count(&w, Opcode::Mul), 1, "the scale multiply");
        assert_eq!(
            result_gid(&w, Opcode::Mul),
            scalar_gid(&ElementType::F32),
            "the score is a scalar"
        );
        verify_hir_stream(&w);
    }

    fn op_imm(w: &LocalWorkerState, op: Opcode) -> Option<u64> {
        w.local_hir_stream
            .iter()
            .find(|i| i.opcode == op)
            .map(|i| i.imm)
    }

    #[test]
    fn tensor_alloc_sizes_storage_for_the_receiver() {
        // `Tensor<f32>([2, 4])` allocates a buffer sized for every element (2*4*4 = 32 bytes), so a
        // later store has room.
        let f =
            parse_fn("fn f() -> Tensor<f32, [2, 4]> { let o = Tensor<f32>([2, 4]); return o; }");
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "tensor alloc should lower"
        );
        assert_eq!(count(&w, Opcode::TensorAlloc), 1);
        assert_eq!(
            op_imm(&w, Opcode::TensorAlloc),
            Some(32),
            "byte size = 2*4*4"
        );
        assert_eq!(
            result_gid(&w, Opcode::TensorAlloc),
            tensor_gid(&ElementType::F32, &["2".to_string(), "4".to_string()]),
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn tensor_row_store_writes_through_the_slice() {
        // `o[0] = v` stores the row slice `v` into the allocated buffer `o`: alloc + index + store.
        let f = parse_fn(
            "fn f(v: Tensor<f32, [4]>) -> Tensor<f32, [2, 4]> \
             { let o = Tensor<f32>([2, 4]); o[0] = v; return o; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w), "row store should lower");
        assert_eq!(count(&w, Opcode::TensorAlloc), 1, "the destination buffer");
        assert_eq!(count(&w, Opcode::TensorIndex), 1, "the row place o[0]");
        assert_eq!(count(&w, Opcode::TensorStore), 1, "the store into it");
        verify_hir_stream(&w);
    }

    #[test]
    fn scalar_element_store_into_rank1_marks_a_place() {
        // `q[0] = 1.0` into a rank-1 tensor: the whole tensor is the base, the index is a scalar-
        // element *place* (imm 1), and one `TensorStore` writes the scalar through it.
        let f = parse_fn("fn f() -> f32 { let q = Tensor<f32>([4]); q[0] = 1.0; return sum(q); }");
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "scalar-element store should lower"
        );
        assert_eq!(count(&w, Opcode::TensorStore), 1);
        let places = w
            .local_hir_stream
            .iter()
            .filter(|i| i.opcode == Opcode::TensorIndex && i.imm == 1)
            .count();
        assert_eq!(places, 1, "the element index is a place (imm = 1)");
        verify_hir_stream(&w);
    }

    #[test]
    fn scalar_element_store_rank2_indexes_row_then_element_place() {
        // `q[0][0] = 1.0`: `q[0]` is a value sub-view index (imm 0), the final `[0]` an element
        // place (imm 1); exactly one index carries the place flag.
        let f = parse_fn("fn f(q: Tensor<f32, [2, 4]>) -> f32 { q[0][0] = 1.0; return q[1][1]; }");
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "nested scalar-element store should lower"
        );
        assert_eq!(count(&w, Opcode::TensorStore), 1);
        let places = w
            .local_hir_stream
            .iter()
            .filter(|i| i.opcode == Opcode::TensorIndex && i.imm == 1)
            .count();
        assert_eq!(places, 1, "exactly one element place among the indices");
        verify_hir_stream(&w);
    }

    #[test]
    fn transfer_rehomes_a_tensor_to_a_memory_space() {
        // `transfer(a, Memory::NPU_HBM)` -> a Transfer carrying the space's dispatch id (100);
        // the result keeps the shape, so the receiving buffer is sized to hold it.
        let f = parse_fn(
            "fn f(a: Tensor<f32, [2, 4]>) -> Tensor<f32, [2, 4]> { return transfer(a, Memory::NPU_HBM); }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w), "transfer should lower");
        assert_eq!(count(&w, Opcode::Transfer), 1);
        assert_eq!(
            op_imm(&w, Opcode::Transfer),
            Some(100),
            "NPU_HBM dispatch id"
        );
        assert_eq!(
            result_gid(&w, Opcode::Transfer),
            tensor_gid(&ElementType::F32, &["2".to_string(), "4".to_string()]),
            "same shape -> the destination holds the source",
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn tensor_gid_distinguishes_element_and_shape() {
        let a = tensor_gid(&ElementType::F32, &["2".to_string(), "4".to_string()]);
        let b = tensor_gid(&ElementType::F32, &["4".to_string(), "4".to_string()]);
        let c = tensor_gid(&ElementType::F64, &["2".to_string(), "4".to_string()]);
        assert_ne!(a, b, "different shape => different GID");
        assert_ne!(a, c, "different element => different GID");
        // Deterministic, and distinct from the scalar element GID.
        assert_eq!(
            a,
            tensor_gid(&ElementType::F32, &["2".to_string(), "4".to_string()])
        );
        assert_ne!(a, scalar_gid(&ElementType::F32));
    }

    #[test]
    fn compound_assign_desugars_to_op_and_store() {
        // `s += a` == `s = s + a`: the current value is read, combined, and stored back. In memory
        // mode (the loop forces it) that's a `SlotLoad` + `Add` + `Store`.
        let f =
            parse_fn("fn f(a: i32) -> i32 { let mut s = 0; for i in 0..a { s += a; } return s; }");
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "compound assign should lower"
        );
        assert!(
            count(&w, Opcode::Add) >= 1,
            "the += combine (plus the loop step)"
        );
        assert!(count(&w, Opcode::Store) >= 1, "s is stored back");
        verify_hir_stream(&w);
    }

    #[test]
    fn fixed_arity_call_lowers_to_args_and_call() {
        // `add(3, 4)` -> two `Arg`s then a `Call` whose type_idx is the callee's GID.
        let (did, w) = lower_with_registry(
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n\
             fn main() -> i32 { return add(3i32, 4i32); }",
            "main",
        );
        assert!(did, "a call to a known scalar-returning fn should lower");
        assert_eq!(count(&w, Opcode::Arg), 2, "two arguments");
        assert_eq!(count(&w, Opcode::Call), 1);
        let call = w
            .local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::Call)
            .unwrap();
        assert_eq!(call.imm, 2, "arg count in imm");
        // The Call's type_idx is the callee GID (a real function identity: nonzero module hash),
        // not a scalar type (whose GID lives in module 0).
        let callee = w.local_type_stream[call.type_idx.0 as usize];
        assert_ne!(callee.module_id(), 0, "type_idx is the callee's GID");
        verify_hir_stream(&w);
    }

    #[test]
    fn call_to_unknown_fn_declines() {
        // `mystery` isn't a known function -> not in the registry's fn_sigs -> the call declines,
        // so the whole function declines (atomic no-op).
        let (did, w) = lower_with_registry("fn main() -> i32 { return mystery(1i32); }", "main");
        assert!(!did);
        assert!(w.local_hir_stream.is_empty());
    }

    #[test]
    fn unmodelled_aggregate_param_is_declined() {
        // `Buf` has a tensor field, so its layout is not modelled (the 0/0 stub); a function taking
        // it by value cannot be sized, so lowering is declined atomically (worker untouched).
        let (did, w) = lower_with_registry(
            "struct Buf { data: Tensor<f32, [4]> }\nfn f(b: Buf) -> i32 { return 0; }",
            "f",
        );
        assert!(
            !did,
            "an unmodelled aggregate param should decline lowering"
        );
        assert!(
            w.local_hir_stream.is_empty(),
            "no partial lowering committed"
        );
    }

    #[test]
    fn if_else_slots_only_the_mutated_local() {
        // Per-local rule (step 2): control flow no longer slots *every* local. `x` is reassigned in
        // both branches (its value must cross the merge), so it gets a slot; the param `a` is only read
        // (a dominating entry value), so it stays a register. One `Alloca`, not two — strictly less
        // memory traffic than the old function-global memory mode. (#230)
        let f = parse_fn(
            "fn c(a: i32) -> i32 { let mut x = a; if a < 0 { x = 0; } else { x = 1; } return x; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(
            count(&w, Opcode::Alloca),
            1,
            "only the mutated `x` gets a slot"
        );
        assert!(
            count(&w, Opcode::SlotLoad) >= 1,
            "`x` is read back after the merge"
        );
        assert_eq!(count(&w, Opcode::CondBr), 1);
        assert_eq!(
            count(&w, Opcode::Br),
            2,
            "then and else each branch to merge"
        );
        assert_eq!(count(&w, Opcode::BlockStart), 4, "entry, then, else, merge");
        verify_hir_stream(&w);
    }

    #[test]
    fn if_without_else_targets_merge_directly() {
        let f = parse_fn("fn c(a: i32) -> i32 { let mut x = a; if a < 0 { x = 0; } return x; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(count(&w, Opcode::CondBr), 1);
        assert_eq!(
            count(&w, Opcode::Br),
            1,
            "then branches to merge; else edge is merge"
        );
        assert_eq!(count(&w, Opcode::BlockStart), 3, "entry, then, merge");
        verify_hir_stream(&w);
    }

    #[test]
    fn early_return_in_branch_emits_no_trailing_branch() {
        let f = parse_fn("fn c(a: i32) -> i32 { if a < 0 { return 0; } return a; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        // The then-block terminates with Ret, so no `Br` to the merge is appended.
        assert_eq!(
            count(&w, Opcode::Br),
            0,
            "returning branch emits no trailing Br"
        );
        assert_eq!(count(&w, Opcode::Ret), 2);
        assert_eq!(count(&w, Opcode::CondBr), 1);
        verify_hir_stream(&w);
    }

    #[test]
    fn straight_line_reassignment_stays_pure_ssa() {
        // No control flow -> SSA mode: reassignment is a rebind, no memory ops or blocks.
        let f = parse_fn("fn c(a: i32) -> i32 { let mut x = a; x = a + a; return x; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(count(&w, Opcode::Alloca), 0);
        assert_eq!(count(&w, Opcode::Store), 0);
        assert_eq!(count(&w, Opcode::BlockStart), 0);
        assert_eq!(
            opcodes(&w),
            vec![Opcode::Load, Opcode::Add, Opcode::Ret],
            "a materialized once, x rebinds to a+a"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn non_escaping_borrow_binds_a_place_with_no_alloca() {
        // §5 Example A (#275): `let r = &x; return *r` where the borrow never escapes. `r` binds as a
        // symbolic `Binding::Place` aliasing `x`, `*r` reads `x` directly, and `x` — no longer forced
        // to materialize an address — stays a register. No `Alloca`, no `Store`, no `PtrIndex`: the
        // whole reference round-trip compiles to `const 5; ret`. This is the alloca that should not
        // exist, gone.
        let f = parse_fn("fn f() -> i32 { let x : i32 = 5; let r : &i32 = &x; return *r; }");
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "non-escaping scalar borrow+deref lowers"
        );
        assert_eq!(
            count(&w, Opcode::Alloca),
            0,
            "the base stays a register — no materialized address"
        );
        assert_eq!(
            count(&w, Opcode::Store),
            0,
            "no store: nothing spilled to memory"
        );
        assert_eq!(
            count(&w, Opcode::PtrIndex),
            0,
            "`*r` reads the local directly, not via a pointer"
        );
        assert_eq!(count(&w, Opcode::Ret), 1);
        verify_hir_stream(&w);
    }

    #[test]
    fn escaping_borrow_materializes_a_flagged_slot() {
        // The counterpart: when the reference *escapes* (here as a call argument), the base must
        // materialize a real address — an `llvm.alloca` flagged `imm = 1`, and `&x` a `PtrIndex`-able
        // pointer. `r` is used as `id(r)`, not `*r`, so the escape analysis keeps `x` materialized. (#275)
        let (did, w) = lower_with_registry(
            "fn id(p : &i32) -> i32 { return *p; }\n\
             fn f() -> i32 { let x : i32 = 5; let r : &i32 = &x; return id(r); }",
            "f",
        );
        assert!(did, "the escaping borrow lowers via materialization");
        let alloca = w
            .local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::Alloca)
            .expect("the escaping base gets a materialized slot");
        assert_eq!(alloca.imm, 1, "an addressable `llvm.alloca`, not a memref");
    }

    #[test]
    fn borrow_inside_a_struct_literal_materializes_its_base() {
        // #275 M4: a `&x` inside an aggregate literal (`Holder { r : &x }`, a reference-typed field)
        // escapes into the struct, so the escape scan must descend into the literal and materialize `x`
        // — an addressable `llvm.alloca` (`imm = 1`). Without the StructInit arm in the scan, `x` stayed
        // a register and the `&x` declined. The whole reference-field program then lowers.
        let (did, w) = lower_with_registry(
            "struct Holder { r : &i32 }\n\
             fn main() -> i32 { let x = 5; let h = Holder { r : &x }; return *h.r; }",
            "main",
        );
        assert!(did, "the reference-typed-field program lowers");
        let addressable = w
            .local_hir_stream
            .iter()
            .any(|i| i.opcode == Opcode::Alloca && i.imm == 1);
        assert!(
            addressable,
            "`&x` inside the struct literal materializes `x` as an addressable slot"
        );
    }

    #[test]
    fn field_places_resolve_writes_to_field_stores() {
        // §5 Example C (#275): `&mut p.x` / `&mut p.y` bind *field places*; each `*b = v` re-lowers as a
        // `FieldStore` through `p` — no pointer materialized, no `PtrStore`. The disjoint borrows lower.
        let (did, w) = lower_with_registry(
            "struct Point { x : i32, y : i32 }\n\
             fn update(p : &mut Point) -> void { let bx = &mut p.x; let by = &mut p.y; *bx = 1; *by = 2; }",
            "update",
        );
        assert!(
            did,
            "the disjoint field-borrow mutator lowers on the flat path"
        );
        assert_eq!(
            count(&w, Opcode::FieldStore),
            2,
            "each `*b = v` is a FieldStore through p"
        );
        assert_eq!(
            count(&w, Opcode::PtrStore),
            0,
            "the field places materialize no pointer"
        );
    }

    #[test]
    fn disjoint_place_writes_reduce_to_mutual_noalias_siblings() {
        // §5.4 (M2b-2): the two disjoint place-writes above reduce to a numeric alias table — each store
        // gets its own group, and (same base `p`, disjoint fields `x`/`y`) each lists the other as a
        // sibling. The two FieldStores sit at distinct stream positions; the table pairs them mutually.
        let (did, w) = lower_with_registry(
            "struct Point { x : i32, y : i32 }\n\
             fn update(p : &mut Point) -> void { let bx = &mut p.x; let by = &mut p.y; *bx = 1; *by = 2; }",
            "update",
        );
        assert!(did, "the mutator lowers");
        let table = &w.local_place_alias_stores;
        assert_eq!(table.len(), 2, "both place-writes are tagged: {table:?}");
        // Two distinct groups (`p.x` and `p.y` are different fields).
        let groups: std::collections::HashSet<usize> = table.iter().map(|(_, g, _)| *g).collect();
        assert_eq!(
            groups.len(),
            2,
            "distinct fields => distinct groups: {table:?}"
        );
        // Each store names the *other* store's group as its lone disjoint sibling.
        for (_, own, sibs) in table {
            assert_eq!(sibs.len(), 1, "one disjoint sibling per store: {table:?}");
            assert_ne!(sibs[0], *own, "a store is not its own noalias sibling");
        }
        // The two positions are the two FieldStores in the stream.
        let fs_positions: Vec<usize> = w
            .local_hir_stream
            .iter()
            .enumerate()
            .filter(|(_, i)| i.opcode == Opcode::FieldStore)
            .map(|(p, _)| p)
            .collect();
        let tagged: std::collections::HashSet<usize> = table.iter().map(|(p, _, _)| *p).collect();
        assert_eq!(
            tagged,
            fs_positions.into_iter().collect(),
            "tagged positions are exactly the FieldStores"
        );
    }

    #[test]
    fn a_lone_place_write_has_no_noalias_sibling() {
        // A single place-write has nothing proven disjoint from it: it still gets its own group, but an
        // empty sibling set — the reduction only pairs stores the frontend actually proved disjoint.
        let (did, w) = lower_with_registry(
            "struct P { x : i32, y : i32 }\n\
             fn main() -> i32 { let mut p = P { x : 1, y : 2 }; let r = &mut p.x; *r = 42; return *r; }",
            "main",
        );
        assert!(did, "the single-field mutator lowers");
        assert_eq!(w.local_place_alias_stores.len(), 1, "one tagged store");
        assert!(
            w.local_place_alias_stores[0].2.is_empty(),
            "a lone place-write has no disjoint sibling"
        );
    }

    #[test]
    fn pointer_local_under_control_flow_stays_a_register() {
        // M3b (§11's remainder): a `*mut` pointer local under control flow that is not reassigned binds
        // as an SSA register, not a slot — the same per-local rule scalars got. `ptr` here is the
        // `let ptr = self.data` shape from `Vec::push`'s grow branch: a raw-pointer field read inside an
        // `if`, read once (`b.data = ptr`), never reassigned. It gets no `Alloca`.
        let (did, w) = lower_with_registry(
            "struct Buf { data : *mut i32, n : i32 }\n\
             fn grow(b : &mut Buf, c : i32) -> i32 { if c > 0 { let ptr : *mut i32 = b.data; b.data = ptr; } return 0; }",
            "grow",
        );
        assert!(did, "the raw-pointer mutator lowers");
        assert_eq!(
            count(&w, Opcode::Alloca),
            0,
            "the non-reassigned pointer local under control flow stays a register, no slot"
        );
    }

    #[test]
    fn direct_field_assignment_is_not_tagged() {
        // A *direct* `p.x = v` (not through a `&mut` place) is a plain field store, not a disjoint
        // place-write, so it carries no alias metadata — only reference-mediated writes are tagged.
        let (did, w) = lower_with_registry(
            "struct P { x : i32, y : i32 }\n\
             fn main() -> i32 { let mut p = P { x : 0, y : 0 }; p.x = 1; p.y = 2; return p.x + p.y; }",
            "main",
        );
        assert!(did, "the direct-assignment fn lowers");
        assert!(
            w.local_place_alias_stores.is_empty(),
            "direct field assignments are untagged: {:?}",
            w.local_place_alias_stores
        );
    }

    #[test]
    fn non_address_taken_scalar_stays_a_register() {
        // The demotion is *scoped*: a scalar local whose address is never taken stays pure-SSA (no
        // slot), so the pre-pass does not regress straight-line functions into memory traffic. (#230)
        let f = parse_fn("fn g() -> i32 { let x : i32 = 5; return x; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(
            count(&w, Opcode::Alloca),
            0,
            "no `&x`, so `x` stays a register"
        );
    }

    #[test]
    fn reference_param_derefs_without_a_slot() {
        // A `&i32` parameter arrives already materialized (it crossed a call boundary), so it is a
        // pointer register on entry — never demoted. `*a` is a `PtrIndex` straight off the param, with
        // no `Alloca`. (design doc §3.2 NOTE: only *locals* are demoted.) (#230)
        let f = parse_fn("fn load(a : &i32) -> i32 { return *a; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w), "reference param derefs");
        assert_eq!(
            count(&w, Opcode::Alloca),
            0,
            "the param is a pointer register, not a slot"
        );
        assert_eq!(
            count(&w, Opcode::PtrIndex),
            1,
            "`*a` is a PtrIndex read off the param"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn void_call_lowers_and_mutates_through_a_reference() {
        // A `&mut` mutator returns `void`; calling it in statement position (`bump(&mut x);`) lowers
        // rather than declining — the call is a pure effect whose result is discarded. `x` is
        // address-taken, so it is an `llvm.alloca` slot (imm = 1) the pointer mutates; `return x`
        // reads the mutated value back through the same slot. (#230)
        let (did, w) = lower_with_registry(
            "fn bump(p : &mut i32) -> void { *p = *p + 1; }\n\
             fn main() -> i32 { let mut x = 41; bump(&mut x); return x; }",
            "main",
        );
        assert!(did, "the void call + read-back lowers on the flat path");
        assert_eq!(count(&w, Opcode::Call), 1, "one call to the void mutator");
        let alloca = w
            .local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::Alloca)
            .expect("the address-taken `x` gets a slot");
        assert_eq!(alloca.imm, 1, "x is an address-taken llvm.alloca slot");
    }

    #[test]
    fn for_range_loop_lowers_header_body_latch_exit() {
        let f = parse_fn(
            "fn sum(n: i32) -> i32 { let mut s = 0; for i in 0..n { s = s + i; } return s; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(count(&w, Opcode::CondBr), 1, "loop condition test");
        assert_eq!(
            count(&w, Opcode::BlockStart),
            5,
            "entry, header, body, latch, exit"
        );
        // entry->header, body->latch, latch->header
        assert_eq!(count(&w, Opcode::Br), 3);
        // The induction var increments: an Add feeding a Store in the latch.
        assert!(
            count(&w, Opcode::Add) >= 2,
            "body add + induction increment"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn infinite_loop_with_break_verifies() {
        let f = parse_fn(
            "fn f(a: i32) -> i32 { let mut x = a; loop { x = x - 1; if x < 0 { break; } } return x; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert!(count(&w, Opcode::CondBr) >= 1, "the if condition");
        // loop header/exit + entry + the if's then/merge blocks.
        assert!(count(&w, Opcode::BlockStart) >= 4);
        verify_hir_stream(&w); // every branch (incl. the break) targets a declared block
    }

    #[test]
    fn for_loop_with_continue_verifies() {
        let f = parse_fn(
            "fn f(n: i32) -> i32 { let mut s = 0; for i in 0..n { if i < 2 { continue; } s = s + i; } return s; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        verify_hir_stream(&w);
    }

    #[test]
    fn for_over_non_range_aborts() {
        // A non-range iterable (here a call) is outside the supported subset -> atomic abort.
        let f = parse_fn("fn f(a: i32) -> i32 { for i in gen() { } return a; }");
        let mut w = worker();
        assert!(!lower_function_to_hir(&f, &mut w));
        assert!(w.local_hir_stream.is_empty());
    }

    #[test]
    fn break_outside_loop_aborts() {
        let f = parse_fn("fn f() -> i32 { break; return 0; }");
        let mut w = worker();
        assert!(!lower_function_to_hir(&f, &mut w));
        assert!(w.local_hir_stream.is_empty());
    }

    #[test]
    fn spawn_wraps_body_in_region_markers() {
        let f = parse_fn(
            "fn k(a: i32) -> i32 { spawn on (Topology::GPU) { let x = a + 1; } return a; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(count(&w, Opcode::Spawn), 1);
        assert_eq!(count(&w, Opcode::SpawnEnd), 1);
        // The Spawn carries the topology dispatch id, and the body (`a + 1`) lowered between the
        // region markers.
        let spawn = w
            .local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::Spawn)
            .unwrap();
        assert_eq!(
            spawn.imm,
            crate::arch::topology_dispatch_id(&crate::syntax::Topology::GPU) as u64
        );
        assert!(
            count(&w, Opcode::Add) >= 1,
            "body arithmetic lowered inside the region"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn value_producing_spawn_aborts() {
        // A spawn that yields a value (no trailing `;`) is deferred -> atomic abort.
        let f = parse_fn("fn k(a: i32) -> i32 { spawn on (Topology::GPU) { a + 1 } }");
        let mut w = worker();
        assert!(!lower_function_to_hir(&f, &mut w));
        assert!(w.local_hir_stream.is_empty());
    }

    #[test]
    fn lowers_params_arithmetic_and_return() {
        let f = parse_fn("fn add(a: i32, b: i32) -> i32 { return a + b; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));

        // Load a, Load b, Add(a,b), Ret(add).
        assert_eq!(
            opcodes(&w),
            vec![Opcode::Load, Opcode::Load, Opcode::Add, Opcode::Ret]
        );
        let add = w.local_hir_stream[2];
        assert_eq!(
            (add.operand1.0, add.operand2.0),
            (0, 1),
            "Add reads the two params"
        );
        assert_eq!(
            w.local_hir_stream[3].operand1.0, 2,
            "Ret reads the Add result"
        );
        // One type per instruction, all in bounds.
        assert_eq!(w.local_type_stream.len(), 4);
        verify_hir_stream(&w);
    }

    #[test]
    fn lowers_let_binding_and_reuse() {
        // y = x * x; return y + x  ->  Load x, Mul(0,0), (let y=1), Add(1,0), Ret(2)
        let f = parse_fn("fn sq(x: i32) -> i32 { let y = x * x; return y + x; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(
            opcodes(&w),
            vec![Opcode::Load, Opcode::Mul, Opcode::Add, Opcode::Ret]
        );
        let mul = w.local_hir_stream[1];
        assert_eq!(
            (mul.operand1.0, mul.operand2.0),
            (0, 0),
            "x * x reads param twice"
        );
        let add = w.local_hir_stream[2];
        assert_eq!(
            (add.operand1.0, add.operand2.0),
            (1, 0),
            "y + x reads Mul result and x"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn lowers_integer_literal_immediate() {
        // The literal lowers to a single `Const` (no coercion in the flat lowerer anymore, #240): in
        // real compilation the checker would type `7` to the `i64` return, but this test lowers the
        // raw parsed AST without the checker, so the `Const` keeps its spelling default and `Ret`
        // carries it directly.
        let f = parse_fn("fn seven() -> i64 { return 7; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(opcodes(&w), vec![Opcode::Const, Opcode::Ret]);
        assert_eq!(
            w.local_hir_stream[0].imm, 7,
            "Const carries the literal value"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn lowers_float_literal_as_bit_pattern() {
        let f = parse_fn("fn half() -> f64 { return 0.5; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(w.local_hir_stream[0].imm, 0.5f64.to_bits());
    }

    #[test]
    fn lowers_comparison_to_bool() {
        let f = parse_fn("fn lt(a: i32, b: i32) -> bool { return a < b; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(
            opcodes(&w),
            vec![Opcode::Load, Opcode::Load, Opcode::Cmp, Opcode::Ret]
        );
        let cmp = w.local_hir_stream[2];
        assert_eq!((cmp.operand1.0, cmp.operand2.0), (0, 1));
        assert_eq!(cmp.imm, 2, "Lt relation code");
        // The Cmp result type is bool.
        assert_eq!(
            w.local_type_stream[cmp.type_idx.0 as usize],
            scalar_gid(&ElementType::Bool)
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn lowers_scalar_cast_with_target_type() {
        let f = parse_fn("fn widen(a: i32) -> i64 { return a as i64; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(opcodes(&w), vec![Opcode::Load, Opcode::Cast, Opcode::Ret]);
        let cast = w.local_hir_stream[1];
        assert_eq!(cast.operand1.0, 0, "cast reads the source");
        assert_eq!(
            w.local_type_stream[cast.type_idx.0 as usize],
            scalar_gid(&ElementType::I64),
            "cast result carries the target type"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn lowers_unary_negation() {
        let f = parse_fn("fn neg(a: i32) -> i32 { return -a; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(opcodes(&w), vec![Opcode::Load, Opcode::Neg, Opcode::Ret]);
        assert_eq!(w.local_hir_stream[1].operand1.0, 0);
        verify_hir_stream(&w);
    }

    #[test]
    fn aborts_atomically_on_unsupported_construct() {
        // A call is outside the supported subset -> abort, worker untouched.
        let f = parse_fn("fn f(a: i32) -> i32 { return g(a); }");
        let mut w = worker();
        assert!(!lower_function_to_hir(&f, &mut w));
        assert!(w.local_hir_stream.is_empty(), "no partial stream on abort");
        assert!(w.local_type_stream.is_empty(), "no partial types on abort");
    }

    #[test]
    fn aborts_on_non_scalar_parameter() {
        let f = parse_fn("fn f(s: Widget) -> i32 { return 0; }");
        let mut w = worker();
        assert!(!lower_function_to_hir(&f, &mut w));
        assert!(w.local_hir_stream.is_empty());
    }

    #[test]
    fn unsafe_block_lowers_transparently() {
        // `unsafe` is transparent to lowering (safety was checked upstream): `return unsafe { a * a }`
        // lowers exactly as `return a * a`. This is what lets a stdlib wrapper body like
        // `return unsafe { sqrtf(self) }` lower through the flat path (#217).
        let f = parse_fn("fn sq(a: f32) -> f32 { return unsafe { a * a }; }");
        let mut w = worker();
        assert!(
            lower_function_to_hir(&f, &mut w),
            "unsafe-block body lowers"
        );
        assert!(w.local_hir_stream.iter().any(|i| i.opcode == Opcode::Mul));
        assert_eq!(w.local_hir_stream.last().unwrap().opcode, Opcode::Ret);
        verify_hir_stream(&w);
    }
}
