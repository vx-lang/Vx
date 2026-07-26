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
    BinaryOp, ElementType, Expr, Function, NumberExpr, RelationalOp, Statement, Type, UnaryOp,
};
use std::collections::HashMap;

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
}

impl LoweredTy {
    /// The GID this type contributes to `local_type_stream`.
    fn gid(&self) -> TypeId {
        match self {
            LoweredTy::Scalar(e) => scalar_gid(e),
            LoweredTy::Aggregate(id) => *id,
            LoweredTy::Tensor { elem, shape } => tensor_gid(elem, shape),
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
            // keep the match total; if one ever reaches here its size is left unencoded.
            LoweredTy::Scalar(_) | LoweredTy::Tensor { .. } => 0,
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
        // An aggregate must live in an addressable slot so its fields can be `getelementptr`'d, even in
        // a straight-line function (e.g. binding a struct-returning call result, #215) — so it always
        // takes the memory path, not just when `self.memory` is set for control flow.
        if self.memory || matches!(v.ty, LoweredTy::Aggregate(_)) {
            let slot = self.emit_alloca(v.ty.clone());
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
            // A name read: an SSA alias (no instruction) or a `SlotLoad` from its memory slot.
            Expr::Identifier(id) => match self.scope.get(&id.name)?.clone() {
                Binding::Reg(v) => Some(v),
                Binding::Slot { reg, ty } => {
                    Some(self.emit_typed(Opcode::SlotLoad, reg, Register(0), ty, 0))
                }
            },
            Expr::BinaryOp(b) => {
                let l = self.lower_expr(&b.lhs)?;
                let r = self.lower_expr(&b.rhs)?;
                let op = binop_opcode(&b.op)?;
                // Operands are type-checked to a common type; the result carries that type. When
                // either operand is a tensor the op is *elementwise* and the result is the tensor
                // type (a scalar operand broadcasts) -- an arith opcode with a tensor result type is
                // the flat HIR's elementwise form, mirroring `arith.mulf` on a vector in codegen.
                let result_ty = match (&l.ty, &r.ty) {
                    (LoweredTy::Tensor { .. }, _) => l.ty.clone(),
                    (_, LoweredTy::Tensor { .. }) => r.ty.clone(),
                    _ => l.ty.clone(),
                };
                Some(self.emit_typed(op, l.reg, r.reg, result_ty, 0))
            }
            // A comparison yields a `bool`; the relation is carried in `imm`.
            Expr::RelationalOp(r) => {
                let l = self.lower_expr(&r.lhs)?;
                let rhs = self.lower_expr(&r.rhs)?;
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
            // A struct field read `base.member`: `base` must be a name bound to an aggregate slot;
            // the field's offset + type come from the registry layout. First cut: scalar fields
            // only (a nested-aggregate or pointer field is declined).
            Expr::MemberAccess(m) => {
                let base = simple_ident(&m.base)?;
                let (slot, gid) = match self.scope.get(&base)?.clone() {
                    Binding::Slot {
                        reg,
                        ty: LoweredTy::Aggregate(gid),
                    } => (reg, gid),
                    _ => return None,
                };
                let field = self
                    .registry
                    .layouts
                    .get(&gid)?
                    .fields
                    .iter()
                    .find(|f| f.name.as_ref() == m.member.as_ref())?;
                let elem = match &field.ty {
                    FieldTy::Scalar(e) => e.clone(),
                    // Nested-aggregate / pointer fields need addressed sub-views — not yet.
                    FieldTy::Nominal(_) | FieldTy::Opaque => return None,
                };
                let offset = field.offset as u64;
                Some(self.emit_value(Opcode::FieldLoad, slot, Register(0), elem, offset))
            }
            // Tensor indexing `base[index]`: rank-reduces the base along its outermost dimension.
            // A remaining shape yields a row/sub-view tensor; an empty one yields the scalar element.
            // Chained access (`q[i][j]`) recurses through the nested `IndexAccess`.
            Expr::IndexAccess(ix) => {
                let base = self.lower_expr(&ix.base)?;
                let (elem, shape) = match &base.ty {
                    LoweredTy::Tensor { elem, shape } => (elem.clone(), shape.clone()),
                    _ => return None, // only tensor indexing for now
                };
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
                Some(self.emit_typed(Opcode::TensorIndex, base.reg, index.reg, result_ty, 0))
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
            // `sizeof<T>()`: a compile-time constant `i64` of `T`'s byte size (matching the AST
            // codegen), used inside comptime blocks. Only the scalar / pointer sizes the AST agrees on
            // are emitted; a struct/enum/tensor `sizeof` declines to the AST path (#228).
            Expr::SizeOf(s) => {
                let size = sizeof_bytes(&s.target_ty)?;
                Some(self.emit_value(
                    Opcode::Const,
                    Register(0),
                    Register(0),
                    ElementType::I64,
                    size,
                ))
            }
            // Construct a payload-free (C-like) enum value (`Color::Green`, #227): the value *is* the
            // variant's discriminant ordinal, a bare `i32` constant (matching the AST codegen). A
            // data-carrying variant (a non-empty payload, or an enum absent from `enum_variants`)
            // declines to the AST path.
            Expr::EnumVariant(ev) => {
                if ev.payload.as_ref().is_some_and(|p| !p.is_empty()) {
                    return None; // tagged-union payload not modelled yet
                }
                let variants = self.registry.enum_variants.get(&ev.enum_name)?;
                let ordinal = variants.iter().position(|v| v == &ev.variant_name)? as u64;
                Some(self.emit_value(
                    Opcode::Const,
                    Register(0),
                    Register(0),
                    ElementType::I32,
                    ordinal,
                ))
            }
            other => {
                if std::env::var("VX_FLAT_DBG").is_ok() {
                    eprintln!("[flat-dbg]   unsupported expr: {}", expr_kind(other));
                }
                None
            }
        }
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
    /// `slot`. Leading statements lower normally; declines if the block has no trailing value.
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
            return None; // only integer ranges for now
        };
        let start = self.lower_expr(&range.start)?;
        let end = self.lower_expr(&range.end)?;
        let elem = match &start.ty {
            LoweredTy::Scalar(e) => e.clone(),
            // ranges are over scalars
            LoweredTy::Aggregate(_) | LoweredTy::Tensor { .. } => return None,
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
    fn lower_struct_init(&mut self, si: &crate::syntax::StructInitExpr) -> Option<Val> {
        let gid = si.type_id?;
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
        for (name, offset, fty) in field_layouts {
            // Only scalar fields for now (nested aggregates need addressed sub-views).
            if !matches!(fty, FieldTy::Scalar(_)) {
                return None;
            }
            let (_, init_expr) = si
                .fields
                .iter()
                .find(|(n, _)| n.as_ref() == name.as_ref())?;
            let v = self.lower_expr(init_expr)?;
            self.emit_effect(Opcode::FieldStore, slot.reg, v.reg, offset);
        }
        Some(slot)
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

    /// Lower an ordinary fixed-arity call `f(a, b, ...)`: resolve the callee via the frozen registry
    /// (its GID + return type), lower each argument, mark them with `Arg` instructions in order, then
    /// emit `Call` (callee GID in `type_idx`, arg count in `imm`). Declines an unknown callee (or one
    /// ambiguous across modules) and a void/unmodelled return -- for now only value-returning calls.
    fn lower_call(&mut self, fc: &crate::syntax::FunctionCallExpr) -> Option<Val> {
        let sig = self.registry.fn_sigs.get(fc.name.as_ref())?.clone();
        let ret_ty = lowered_ty(&sig.ret_ty, self.registry)?;
        let mut arg_regs = Vec::with_capacity(fc.args.len());
        for arg in &fc.args {
            arg_regs.push(self.lower_expr(arg)?.reg);
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
    fn lower_place(&mut self, e: &Expr) -> Option<Val> {
        let Expr::IndexAccess(ix) = e else {
            return None;
        };
        let base = self.lower_expr(&ix.base)?;
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

    /// Lower a statement. `None` aborts the whole function's lowering.
    fn lower_stmt(&mut self, s: &Statement) -> Option<()> {
        match s {
            Statement::LetDecl(l) => {
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
                    self.bind_local(l.name.clone(), v);
                }
                Some(())
            }
            Statement::Return(r) => {
                let v = self.lower_expr(&r.expr)?;
                self.emit_typed(Opcode::Ret, v.reg, Register(0), v.ty, 0);
                Some(())
            }
            Statement::Assign(a) => {
                // A tensor place store `place[i] = value`. The left side lowers to a `TensorIndex`
                // place: a row/sub-view (`o[i] = <slice>`) or a scalar element (`q[i][j] = <scalar>`,
                // the final index marked `imm = 1`). The store kind is recovered from the place type
                // in codegen; the right side is the value stored through it.
                if matches!(&a.lhs, Expr::IndexAccess(_)) {
                    let place = self.lower_place(&a.lhs)?;
                    let value = self.lower_expr(&a.rhs)?;
                    self.emit_effect(Opcode::TensorStore, place.reg, value.reg, 0);
                    return Some(());
                }
                // `name = expr` (simple identifier target).
                let name = simple_ident(&a.lhs)?;
                let v = self.lower_expr(&a.rhs)?;
                self.assign_local(&name, v)
            }
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
        // The string side table is indexed by each `PrintStr`'s `imm`; a fresh worker lowers exactly
        // one function, so the indices need no rebasing (they start at 0 per function).
        worker.local_string_table.extend(self.strings);
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
    // Control flow forces the memory model so locals survive across basic blocks (like the AST
    // codegen); an aggregate parameter also forces it, since an aggregate must live in an
    // addressable slot. Straight-line scalar functions stay pure-SSA.
    let has_aggregate_param = func
        .params
        .iter()
        .any(|(_, ty)| matches!(lowered_ty(ty, registry), Some(LoweredTy::Aggregate(_))));
    lw.memory = body_has_control_flow(&func.body)
        || has_aggregate_param
        || body_constructs_struct(&func.body);
    if lw.memory {
        lw.emit_effect(Opcode::BlockStart, Register(0), Register(0), 0); // entry block
    }
    // Parameters: materialize the incoming value (`Load` imm = index), then bind (a slot in memory
    // mode, an SSA register otherwise). A tensor is a reference value (memref), so it always binds
    // as an SSA register — never `Alloca`'d into a slot.
    for (i, (name, ty)) in func.params.iter().enumerate() {
        let lty = lowered_ty(ty, registry)?;
        let incoming = lw.emit_typed(Opcode::Load, Register(0), Register(0), lty, i as u64);
        if matches!(incoming.ty, LoweredTy::Tensor { .. }) {
            lw.scope.insert(name.clone(), Binding::Reg(incoming));
        } else {
            lw.bind_local(name.clone(), incoming);
        }
    }
    for stmt in &func.body {
        lw.lower_stmt(stmt)?;
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
    match ty {
        // A payload-free (C-like) enum is a bare `i32` discriminant, not an aggregate (#227). The
        // resolver may spell an enum type as `Type::Enum` or `Type::Struct`, so match on the name; a
        // data-carrying enum is absent from `enum_variants` and falls through to the aggregate case.
        Type::Enum(name, _) | Type::Struct(name, _)
            if registry.enum_variants.contains_key(name) =>
        {
            Some(LoweredTy::Scalar(ElementType::I32))
        }
        Type::Struct(_, Some(id)) | Type::Enum(_, Some(id)) => {
            let def = registry.layouts.get(id)?;
            if def.align_bytes == 0 {
                return None; // layout not modelled yet (the 0/0 stub)
            }
            Some(LoweredTy::Aggregate(*id))
        }
        _ => None,
    }
}

/// Whether the (top-level) body contains control flow (`if`/`loop`/`for`) — the trigger for the
/// memory model, so mutated or loop-carried locals survive across basic blocks. Nested control flow
/// rides on its enclosing top-level construct, and the `lower_*` helpers recurse in memory mode.
fn body_has_control_flow(stmts: &[Statement]) -> bool {
    stmts.iter().any(|s| match s {
        Statement::Loop(_) | Statement::ForLoop(_) => true,
        Statement::ExprStmt(e) => matches!(e.expr, Expr::If(_) | Expr::Match(_)),
        // A value-position `if` (`let v = if .. { .. } else { .. }`, #201) lowers to blocks + a result
        // slot, which needs the memory model too.
        Statement::LetDecl(l) => matches!(l.expr, Expr::If(_)),
        Statement::Return(r) => matches!(r.expr, Expr::If(_)),
        Statement::Assign(a) => matches!(a.rhs, Expr::If(_)),
        _ => false,
    })
}

/// Whether the (top-level) body constructs a struct into a local (`let x = S { .. }`) — the trigger
/// for the memory model, since the constructed aggregate must live in an addressable slot.
fn body_constructs_struct(stmts: &[Statement]) -> bool {
    stmts
        .iter()
        .any(|s| matches!(s, Statement::LetDecl(l) if matches!(l.expr, Expr::StructInit(_))))
}

fn simple_ident(e: &Expr) -> Option<Symbol> {
    match e {
        Expr::Identifier(id) => Some(id.name.clone()),
        _ => None,
    }
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

fn infer_elem(s: &str) -> Option<ElementType> {
    if s.contains('.') || s.contains('e') || s.contains('E') {
        Some(ElementType::F64)
    } else {
        Some(ElementType::I64)
    }
}

fn is_float(e: &ElementType) -> bool {
    matches!(
        e,
        ElementType::F16 | ElementType::F32 | ElementType::F64 | ElementType::BF16
    )
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
#[cfg(debug_assertions)]
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
    fn if_else_lowers_to_basic_blocks_and_memory_locals() {
        let f = parse_fn(
            "fn c(a: i32) -> i32 { let mut x = a; if a < 0 { x = 0; } else { x = 1; } return x; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        // Control flow forced the memory model: slots for `a` and `x`.
        assert!(count(&w, Opcode::Alloca) >= 2, "alloca slots for a and x");
        assert!(count(&w, Opcode::SlotLoad) >= 1);
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
