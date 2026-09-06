//===- flat.rs - Vx Compiler ---------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Flat codegen (#200): lower a function's flat HIR stream (`local_hir_stream`
// + `local_type_stream`, produced by `hir/flatten.rs`) to MLIR, driven by the
// instruction array instead of an AST walk (doc §Phase 7 "O(1) array codegen").
//
// Current subset: scalar arithmetic (params, const, add/sub/mul/div, compare,
// return); intra-function control flow (basic-block markers, (conditional)
// branches, and `alloca`/store/load'd scalar locals — the memory model the flat
// HIR uses so values cross blocks); fixed-arity scalar calls (`func.call`, callee
// via the registry's `fn_sigs`); all-scalar-field structs (`llvm.alloca`/GEP +
// `llvm.load`/`store`); and the tensor surface (`memref` alloc, element + row
// access, sub-views via `reinterpret_cast`, `vector.reduction`, elementwise
// `vector` ops, row `vector.store`, and `vx.transfer`; tensor shapes come from a
// side table the lowerer fills — see the C2 plan). `emit_function_mlir` emits one
// `func.func` as text (SSA name = producing instruction's register; blocks become
// `^bbN:` labels); `emit_module_mlir` emits a whole program (needed for calls).
// The caller parses + verifies with melior. The AST path stays the oracle: the
// emitter returns `None` for any stream (or module) using opcodes outside this
// subset, so nothing half-lowered is ever emitted.
//
//===----------------------------------------------------------------------===//
use crate::bytecode::{HirInstruction, Opcode};
use crate::decline::{Decline, Lowered};
use crate::gid::TypeId;
use crate::hir::flatten::{ptr_gid, scalar_gid, tensor_gid_of, DYN_DIM};
use crate::mlir_ty::mlir_scalar;
use crate::registry::ImmutableGlobalRegistry;
use crate::syntax::scalar_of;
use crate::syntax::{is_void_ty, Dim, ElementType, Function, Type};
use rayon::prelude::*;
use std::collections::HashMap;

mod emit;

/// The MLIR type string for a scalar element type. Integers are signless (signedness lives in the
/// op, e.g. `divsi`/`divui`); `bool` is `i1`.
/// Format a float so MLIR can parse it back: Rust's `{:?}` prints `0.00001` as `1e-5`,
/// and MLIR needs a decimal point before the exponent (Vx#384).
fn mlir_float_literal(value: f64) -> String {
    let text = format!("{value:?}");
    match text.split_once(['e', 'E']) {
        // `1e-5` -> `1.0e-5`. A mantissa that already has a point, and anything without an
        // exponent at all (`288.0`, `inf`, `NaN`), is left exactly as Rust printed it.
        Some((mantissa, exponent)) if !mantissa.contains('.') => format!("{mantissa}.0e{exponent}"),
        _ => text,
    }
}

/// The inverse of [`mlir_scalar`]: recover an [`ElementType`] from an MLIR scalar type string.
/// Integers map to the signed variant (`mlir_scalar` is many-to-one on signedness); that is enough
/// for choosing a conversion op, since sign-extension keys off the *source* type. Used to coerce a
/// stored value to a tensor's element type at a `memref.store`.
fn elem_from_mlir_scalar(s: &str) -> Option<ElementType> {
    use ElementType::*;
    Some(match s {
        "f16" => F16,
        "bf16" => BF16,
        "f32" => F32,
        "f64" => F64,
        "i1" => Bool,
        "i4" => I4,
        "i8" => I8,
        "i16" => I16,
        "i32" => I32,
        "i64" => I64,
        "i128" => I128,
        _ => return None,
    })
}

fn is_signed(e: &ElementType) -> bool {
    use ElementType::*;
    matches!(e, I4 | I8 | I16 | I32 | I64 | I128)
}

fn int_bits(e: &ElementType) -> Option<u32> {
    use ElementType::*;
    Some(match e {
        Bool => 1,
        I4 | U4 => 4,
        I8 | U8 => 8,
        I16 | U16 => 16,
        I32 | U32 => 32,
        I64 | U64 => 64,
        I128 | U128 => 128,
        _ => return None,
    })
}

fn float_bits(e: &ElementType) -> Option<u32> {
    use ElementType::*;
    Some(match e {
        F16 | BF16 => 16,
        F32 => 32,
        F64 => 64,
        _ => return None,
    })
}

/// The `arith` conversion op for a scalar `as` cast (#214), or `Some("")` when the cast is a no-op
/// (same MLIR type — e.g. `i32 as u32`, which only reinterprets signedness). `None` declines an
/// unsupported pair. Signedness of the *integer* side selects the sign-aware op.
fn cast_op(src: &ElementType, tgt: &ElementType) -> Option<&'static str> {
    if mlir_scalar(src)? == mlir_scalar(tgt)? {
        return Some(""); // same underlying type -> reinterpret, no op
    }
    match (src.is_float(), tgt.is_float()) {
        (false, false) => {
            let (sb, tb) = (int_bits(src)?, int_bits(tgt)?);
            Some(if tb < sb {
                "arith.trunci"
            } else if is_signed(src) {
                "arith.extsi"
            } else {
                "arith.extui"
            })
        }
        (false, true) => Some(if is_signed(src) {
            "arith.sitofp"
        } else {
            "arith.uitofp"
        }),
        (true, false) => Some(if is_signed(tgt) {
            "arith.fptosi"
        } else {
            "arith.fptoui"
        }),
        (true, true) => {
            let (sb, tb) = (float_bits(src)?, float_bits(tgt)?);
            Some(if tb > sb {
                "arith.extf"
            } else {
                "arith.truncf"
            })
        }
    }
}

/// Recover the element type a type-stream GID stands for. The stream stores content-hash GIDs
/// (`scalar_gid`), so we invert by testing the finite set of scalar variants — the flat-driven
/// counterpart of reading a scalar type off the AST.
fn elem_of_gid(gid: TypeId) -> Option<ElementType> {
    use ElementType::*;
    [
        F16, F32, F64, BF16, I4, U4, I8, U8, I16, U16, I32, U32, I64, U64, I128, U128, Bool,
    ]
    .into_iter()
    .find(|e| scalar_gid(e) == gid)
}

/// The layout GID of the aggregate a pointer/borrow points *to* (`self : &mut Vec<i32>` → the
/// `Vec` layout GID), when that aggregate is modelled in `ctx.aggs`. This is what lets a
/// `FieldLoad`/`FieldStore` through a `self` pointer GEP the field — the pointer register is tracked
/// in `agg_of` exactly as an aggregate slot is. `None` for a non-pointer, or a pointee whose layout
/// isn't modelled. (#242)
fn pointee_agg_gid(ty: &Type, ctx: &EmitCtx) -> Option<TypeId> {
    let inner = match ty {
        Type::Borrow { inner, .. } => inner.as_ref(),
        Type::Pointer(inner, ..) => inner.as_ref(),
        _ => return None,
    };
    ctx.agg_gid(inner)
}

/// Whether a type lowers to an opaque `!llvm.ptr` — a `*const T`/`*mut T`, a `&T` borrow, or a
/// function/closure type (a materialized function pointer). The ABI of string values, FFI pointer
/// arguments/results, and function pointers (matching the AST codegen's `lower_type`). (#231/#235/#242)
/// Strip the checker-level wrappers (`Verified<T>`, `Ref<T, Memory>`, `Pinned<T, Topology>`)
/// down to the runtime type, as the AST codegen's `lower_type` does.
fn peel_wrappers(ty: &Type) -> &Type {
    match ty {
        Type::Verified(inner) => peel_wrappers(inner),
        Type::Ref(inner, _) | Type::Pinned(inner, _) => peel_wrappers(inner),
        _ => ty,
    }
}

/// The memref spelling of a tensor type (rank-0 included), peeling wrappers first:
/// `Tensor<f32, [2, 3]>` is `memref<2x3xf32>` and `Tensor<f32, [?, ?]>` is `memref<?x?xf32>`, the rank
/// the oracle assumes for a shape it does not know (Vx#404).
///
/// `None` for a non-tensor, or for a `Tensor` dimension that is not a literal. A shaped value
/// reaching a dynamic position is cast to it rather than spelled as one, so a signature written
/// here never contradicts what the body produces.
fn tensor_memref_of_type(ty: &Type) -> Option<String> {
    match peel_wrappers(ty) {
        Type::Tensor(elem, dims, _) => {
            let mut shape = Vec::with_capacity(dims.len());
            for d in dims {
                shape.push(match d {
                    Dim::Dyn => DYN_DIM.to_string(),
                    _ => d.literal()?.parse::<i64>().ok()?.to_string(),
                });
            }
            tensor_memref_ty(elem, &shape)
        }
        _ => None,
    }
}

fn is_ptr_ty(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Pointer(..) | Type::Borrow { .. } | Type::Function(..) | Type::Closure(..)
    )
}

/// The MLIR argument-attribute suffix carrying the borrow checker's aliasing guarantee into the
/// function signature (#275, §5.4) — the disjointness LLVM cannot re-derive on its own. A `&mut T`
/// parameter is an *exclusive* borrow (the borrow checker forbids any other reference to the same
/// memory for its duration), so it earns `llvm.noalias`. A `&T` shared reference cannot be written
/// *through* (type-guaranteed), so it earns `llvm.readonly` — but *not* `noalias`, since two `&T` may
/// legitimately alias and only the no-write guarantee is sound. Raw `*mut`/`*const` pointers carry no
/// such guarantee and get nothing.
///
/// This is the guarantee an `-O0` differential cannot see (attributes only bite under optimization), so
/// it rests on the borrow checker's soundness rather than on a runtime oracle — hence the deliberately
/// conservative choice above (exactly what rustc emits for `&mut`/`&`). Rendered with a leading space
/// for direct concatenation after the param type, or `""` for none.
fn param_alias_attrs(ty: &Type) -> &'static str {
    match ty {
        Type::Borrow { is_mut: true, .. } => " {llvm.noalias}",
        Type::Borrow { is_mut: false, .. } => " {llvm.readonly}",
        _ => "",
    }
}

/// The `llvm.store` attribute dict for a disjoint place-write (M2b-2): the store belongs to alias scope
/// `own` and does not alias the `siblings` scopes (fields the borrow checker proved disjoint). All
/// scopes share the module domain `distinct[0]`; each scope is `distinct[k]`. Rendered with a leading
/// space for direct concatenation after the store's value/pointer operands, or `""` (via the caller's
/// `unwrap_or_default`) when the store isn't a tagged place-write.
///
/// Like `param_alias_attrs`, an `-O0` differential can't see this (alias metadata only bites under
/// optimization), so its soundness rests on the borrow checker + the structural field-disjointness the
/// lowerer computed, not on a runtime oracle. (#275, §5.4)
fn alias_store_attrs(own: u32, siblings: &[u32]) -> String {
    let scope = |k: u32| {
        format!(
            "#llvm.alias_scope<id = distinct[{k}]<>, domain = #llvm.alias_scope_domain<id = distinct[0]<>>>"
        )
    };
    if siblings.is_empty() {
        format!(" {{alias_scopes = [{}]}}", scope(own))
    } else {
        let noalias = siblings
            .iter()
            .map(|s| scope(*s))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            " {{alias_scopes = [{}], noalias_scopes = [{noalias}]}}",
            scope(own)
        )
    }
}

/// The arith op mnemonic for a binary opcode at a given element type.
fn arith_op(op: Opcode, e: &ElementType) -> Option<&'static str> {
    let f = e.is_float();
    Some(match op {
        Opcode::Add => {
            if f {
                "arith.addf"
            } else {
                "arith.addi"
            }
        }
        Opcode::Sub => {
            if f {
                "arith.subf"
            } else {
                "arith.subi"
            }
        }
        Opcode::Mul => {
            if f {
                "arith.mulf"
            } else {
                "arith.muli"
            }
        }
        Opcode::Div => {
            if f {
                "arith.divf"
            } else if is_signed(e) {
                "arith.divsi"
            } else {
                "arith.divui"
            }
        }
        _ => return None,
    })
}

/// The `arith.cmp{i,f}` op + textual predicate for a `Cmp` on operands of element type `e`, given
/// the relation code stored in the instruction's `imm` (0=Eq,1=Ne,2=Lt,3=Gt,4=Le,5=Ge — kept in
/// sync with `flatten::rel_code`). Integers use signed vs. unsigned predicates by the element's
/// signedness; floats use the ordered predicates.
fn cmp_op(rel: u64, e: &ElementType) -> Option<(&'static str, &'static str)> {
    if e.is_float() {
        let pred = match rel {
            0 => "oeq",
            1 => "one",
            2 => "olt",
            3 => "ogt",
            4 => "ole",
            5 => "oge",
            _ => return None,
        };
        Some(("arith.cmpf", pred))
    } else {
        let s = is_signed(e);
        let pred = match rel {
            0 => "eq",
            1 => "ne",
            2 => {
                if s {
                    "slt"
                } else {
                    "ult"
                }
            }
            3 => {
                if s {
                    "sgt"
                } else {
                    "ugt"
                }
            }
            4 => {
                if s {
                    "sle"
                } else {
                    "ule"
                }
            }
            5 => {
                if s {
                    "sge"
                } else {
                    "uge"
                }
            }
            _ => return None,
        };
        Some(("arith.cmpi", pred))
    }
}

/// A resolved callee for the flat emitter: the MLIR symbol name to `func.call`, and its scalar
/// return element type (`None` for a non-scalar/void return, which this subset declines). Keyed by
/// the callee's GID — the identity a `Call` instruction carries in its `type_idx`.
pub struct Callee {
    pub name: String,
    pub ret: Option<ElementType>,
    /// The GID of the callee's return type when it is a nominal aggregate (struct/enum) -- the call
    /// then returns an `!llvm.struct` by value, spilled to a slot at the call site (#215).
    pub ret_agg: Option<TypeId>,
    /// Whether the callee returns an opaque `!llvm.ptr` (a `*const`/`*mut`/`&` return, e.g. an FFI
    /// allocator). The call's result is then a pointer value tracked in `ptr_of`. (#235)
    pub ret_ptr: bool,
    /// Whether the callee returns `void`. The call emits `func.call @name(..) : (..) -> ()` and binds
    /// no result register — the statement-position form (`bump(&mut x);`) used by `&mut` mutators. (#230)
    pub ret_void: bool,
    /// The memref spelling when the callee returns a statically shaped tensor (wrappers peeled) --
    /// the call's result is then a memref value tracked in `mem_of`.
    pub ret_tensor: Option<String>,
}

/// GID → callee: the reverse of the registry's name-keyed `fn_sigs`. A `Call`'s `type_idx` resolves
/// to a callee GID; this map recovers the symbol name (for `func.call @name`) and the return type
/// (for the call's result type) without a name→AST walk.
pub type CalleeMap = HashMap<TypeId, Callee>;

/// Resolve a nominal type to a *modelled* aggregate layout GID: the attached GID when name resolution
/// set it (present in `aggs`), else the base name's GID (`agg_names`) — a monomorphized cross-module
/// signature carries an unresolved base (`GenericInstance(Struct("Vec", None), ..)` / `Struct("Vec",
/// None)`), so the layout must be recovered by name. Resolves a generic instance through its base.
/// (#242)
fn resolve_agg_gid(
    ty: &Type,
    aggs: &AggMap,
    agg_names: &HashMap<String, TypeId>,
) -> Option<TypeId> {
    // A data-carrying enum instance (`Option<i32>`) has a *synthesized* per-instance layout keyed by
    // `enum_instance_gid` (the lowerer recorded it in the side table now folded into `aggs`); try that
    // first. Its `gid` is distinct from any registry layout, so a struct instance (`Vec<i32>`) misses
    // here and falls through to the nominal resolution below. (#242)
    if let Type::GenericInstance(base, args) = ty {
        if let Type::Enum(name, _) | Type::Struct(name, _) = base.as_ref() {
            let gid = crate::hir::flatten::enum_instance_gid(name.as_ref(), args);
            if aggs.contains_key(&gid) {
                return Some(gid);
            }
        }
    }
    let nominal = match ty {
        Type::GenericInstance(base, _) => base.as_ref(),
        other => other,
    };
    match nominal {
        Type::Struct(name, id) | Type::Enum(name, id) => id
            .filter(|g| aggs.contains_key(g))
            .or_else(|| agg_names.get(name.as_ref()).copied()),
        _ => None,
    }
}

/// Base struct/enum name → modelled layout GID, for resolving a monomorphized cross-module aggregate
/// whose signature carries an unresolved base GID (`Struct("Vec", None)`). Only names that are (a)
/// modelled in `aggs` and (b) unambiguous (a single GID) are listed; an ambiguous name is dropped so
/// a wrong layout is never chosen. (#242)
fn build_agg_names(registry: &ImmutableGlobalRegistry, aggs: &AggMap) -> HashMap<String, TypeId> {
    let mut map: HashMap<String, TypeId> = HashMap::new();
    let mut ambiguous: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (gid, def) in &registry.layouts {
        if !aggs.contains_key(gid) {
            continue;
        }
        match map.get(&def.name) {
            Some(g) if g != gid => {
                ambiguous.insert(def.name.clone());
            }
            _ => {
                map.insert(def.name.clone(), *gid);
            }
        }
    }
    for n in ambiguous {
        map.remove(&n);
    }
    map
}

/// Build the GID→callee map from the frozen registry's function signatures. A callee returning a
/// nominal aggregate resolves its layout GID via `aggs`/`agg_names` (name fallback for a cross-module
/// mono).
pub fn build_callee_map(
    registry: &ImmutableGlobalRegistry,
    aggs: &AggMap,
    agg_names: &HashMap<String, TypeId>,
    sched: crate::config::Schedule,
) -> CalleeMap {
    // A pure map keyed by the callee's GID, so the result does not depend on the order entries are
    // produced -- which is just as well, since it is built from a `HashMap` whose iteration order was
    // never defined to begin with. Parallel because this and `build_agg_map` are the last of the
    // serial codegen prologue (#314): both are folds over the whole frozen registry, on the critical
    // path, ahead of everything else codegen does.
    let one = |(name, sig): (&crate::symbol::Symbol, &crate::registry::FnSig)| {
        (
            sig.gid,
            Callee {
                name: name.to_string(),
                ret: scalar_of(&sig.ret_ty),
                ret_agg: resolve_agg_gid(&sig.ret_ty, aggs, agg_names),
                ret_ptr: is_ptr_ty(&sig.ret_ty),
                ret_void: is_void_ty(&sig.ret_ty),
                ret_tensor: tensor_memref_of_type(&sig.ret_ty),
            },
        )
    };
    if sched == crate::config::Schedule::Sequential {
        registry.fn_sigs.iter().map(one).collect()
    } else {
        registry.fn_sigs.par_iter().map(one).collect()
    }
}

/// The MLIR shape of an aggregate (struct) whose fields are all scalar or pointer: the
/// `!llvm.struct<(...)>` type (for the slot `llvm.alloca` and field `getelementptr`), each field's
/// byte offset in declaration order — so a `FieldLoad`/`FieldStore`, which carries a byte offset,
/// recovers the GEP field index by `offsets.position(|o| o == offset)` — and each field's MLIR type
/// string, so a field op prints the right load/store type (a scalar's element or `!llvm.ptr`).
pub struct AggLayout {
    pub struct_ty: String,
    pub offsets: Vec<u64>,
    /// Each field's MLIR type (`"i32"`, `"!llvm.ptr"`, …) in declaration order, index-aligned with
    /// `offsets`. A field op loads/stores at `field_tys[field_idx]`.
    pub field_tys: Vec<String>,
    /// For each pointer field, the layout GID of the aggregate it points *to*, when that pointee's
    /// layout is instance-independent (`VecIter`'s `vec : *const Vec<T>` -> the `Vec` layout GID, the
    /// same for any `T`). `None` for a scalar field or a pointer to a scalar/unmodelled type. A
    /// `FieldLoad` of such a field tags its result register with this GID, so a *chained* field access
    /// through it (`(*self.vec).len`) can GEP the pointee. (#242)
    pub field_pointee: Vec<Option<TypeId>>,
    /// For each field that is a *by-value nested aggregate* (`VecMap { iter: VecIter, f: Closure1 }`),
    /// the nested aggregate's own layout GID. `None` for a scalar/pointer field. A `FieldLoad` of such
    /// a field loads the whole `!llvm.struct` value (tracked in `agg_val_of`), a `FieldStore` stores it,
    /// and taking its address (`&self.iter` for a method receiver) GEPs to it as an aggregate slot. (#242)
    pub field_agg: Vec<Option<TypeId>>,
}

/// GID → aggregate layout, for the structs a stream constructs/reads. Structs whose fields are all
/// scalar or pointer (`Opaque`) are modelled — a pointer field lowers to `!llvm.ptr`, which is what
/// makes the `Vec<T> { data: *mut T, len, cap }` aggregate emittable (#242). A struct with a
/// by-value nominal (nested-aggregate) field, or an unmodelled (0-align stub) layout, is skipped;
/// an aggregate absent from the map declines, keeping the AST path the oracle.
pub type AggMap = HashMap<TypeId, AggLayout>;

/// Build the GID→aggregate-layout map from the frozen registry's nominal layouts. Skips a struct
/// with any by-value nominal field or an unmodelled (0-align stub) layout; a pointer (`Opaque`)
/// field is modelled as `!llvm.ptr`.
pub fn build_agg_map(registry: &ImmutableGlobalRegistry, sched: crate::config::Schedule) -> AggMap {
    use crate::layout::FieldTy;
    // Name -> layout GID for every modelled nominal, to resolve a pointer field's pointee aggregate
    // (its layout is instance-independent when the field is behind a pointer, so the base name suffices
    // — `VecIter`'s `vec : *const Vec<T>` resolves to the `Vec` layout GID for any `T`).
    let name_to_gid: HashMap<&str, TypeId> = registry
        .layouts
        .iter()
        .filter(|(_, d)| d.align_bytes != 0 && !d.fields.is_empty())
        .map(|(g, d)| (d.name.as_str(), *g))
        .collect();
    // Each layout's entry depends only on the registry and `name_to_gid`, and the key is the layout's
    // own GID, so the map is the same whatever order the entries are produced in. (See
    // `build_callee_map` for why this is worth parallelising.)
    let layout_of = |(gid, def): (&TypeId, &crate::registry::TypeDefinition)| {
        if def.align_bytes == 0 || def.fields.is_empty() {
            return None; // unmodelled stub, or an enum/field-less type (no struct body to emit)
        }
        // The declared field types (with generic pointees intact) for pointee resolution; the frozen
        // `layouts` erase them to `Opaque`. GID-keyed since #291 — this loop iterates `layouts`, so
        // the definition's own GID is the key.
        let decl_fields = registry.structs.get(gid);
        let mut field_tys = Vec::with_capacity(def.fields.len());
        let mut offsets = Vec::with_capacity(def.fields.len());
        let mut field_pointee = Vec::with_capacity(def.fields.len());
        let mut field_agg: Vec<Option<TypeId>> = Vec::with_capacity(def.fields.len());
        let mut modelled = true;
        for (fi, f) in def.fields.iter().enumerate() {
            match &f.ty {
                FieldTy::Scalar(e) => match mlir_scalar(e) {
                    Some(mt) => {
                        field_tys.push(mt.to_string());
                        field_pointee.push(None);
                        field_agg.push(None);
                    }
                    None => {
                        modelled = false;
                        break;
                    }
                },
                // A pointer field (`*mut T`/`&T`, layout-erased to `Opaque`) is an opaque `!llvm.ptr`
                // — the shape of `Vec`'s `data` field (#242). Recover its pointee aggregate (if any)
                // from the declared field type so a chained field access through it resolves.
                FieldTy::Opaque => {
                    field_tys.push("!llvm.ptr".to_string());
                    let pointee = decl_fields
                        .and_then(|sf| sf.fields.get(fi))
                        .and_then(|(_, ft)| pointer_pointee_name(ft))
                        .and_then(|n| name_to_gid.get(n.as_str()).copied());
                    field_pointee.push(pointee);
                    field_agg.push(None);
                }
                // A by-value nested-aggregate field (`VecMap { iter: VecIter, .. }`): its MLIR type is
                // the nested aggregate's `!llvm.struct` (recursively resolved), stored/loaded whole. The
                // nested layout must itself be fully modelled (all scalar/pointer/nested fields), else the
                // whole enclosing struct declines. (#242)
                FieldTy::Nominal(nested_gid) => {
                    match agg_struct_ty_of(*nested_gid, registry, &mut Vec::new()) {
                        Ok(nested_ty) => {
                            field_tys.push(nested_ty);
                            field_pointee.push(None);
                            field_agg.push(Some(*nested_gid));
                        }
                        Err(_) => {
                            modelled = false;
                            break;
                        }
                    }
                }
            }
            offsets.push(f.offset as u64);
        }
        if !modelled {
            return None;
        }
        Some((
            *gid,
            AggLayout {
                struct_ty: format!("!llvm.struct<({})>", field_tys.join(", ")),
                offsets,
                field_tys,
                field_pointee,
                field_agg,
            },
        ))
    };
    if sched == crate::config::Schedule::Sequential {
        registry.layouts.iter().filter_map(layout_of).collect()
    } else {
        registry.layouts.par_iter().filter_map(layout_of).collect()
    }
}

/// The `!llvm.struct<(...)>` MLIR type of a modelled aggregate layout, resolved recursively so a
/// by-value nested-aggregate field expands to its nested struct type. `None` if the layout is a stub,
/// field-less, or has any unmodelled field (a non-lowerable scalar, or a nested aggregate that itself
/// fails). `visiting` guards against a cyclic layout (which would be infinite-size anyway). (#242)
fn agg_struct_ty_of(
    gid: TypeId,
    registry: &ImmutableGlobalRegistry,
    visiting: &mut Vec<TypeId>,
) -> Lowered<String> {
    use crate::layout::FieldTy;
    if visiting.contains(&gid) {
        return Err(Decline::TypeNotModelled {
            what: "an aggregate whose layout is cyclic",
        });
    }
    let def = registry.layouts.get(&gid).ok_or(crate::emitter_gap!())?;
    if def.align_bytes == 0 || def.fields.is_empty() {
        return Err(Decline::TypeNotModelled {
            what: "an aggregate with no fields or alignment",
        });
    }
    visiting.push(gid);
    let mut field_tys = Vec::with_capacity(def.fields.len());
    for f in &def.fields {
        let ft = match &f.ty {
            FieldTy::Scalar(e) => mlir_scalar(e).ok_or(crate::emitter_gap!())?.to_string(),
            FieldTy::Opaque => "!llvm.ptr".to_string(),
            FieldTy::Nominal(n) => agg_struct_ty_of(*n, registry, visiting)?,
        };
        field_tys.push(ft);
    }
    visiting.pop();
    Ok(format!("!llvm.struct<({})>", field_tys.join(", ")))
}

/// The base nominal name a pointer/borrow points *to* (`*const Vec<T>` -> `"Vec"`), for resolving a
/// pointer field's pointee aggregate. `None` for a pointer to a non-nominal (a scalar, `*mut T`). (#242)
fn pointer_pointee_name(ty: &Type) -> Option<String> {
    let inner = match ty {
        Type::Pointer(inner, ..) | Type::Borrow { inner, .. } | Type::Ref(inner, ..) => {
            inner.as_ref()
        }
        _ => return None,
    };
    match inner {
        Type::GenericInstance(base, _) => match base.as_ref() {
            Type::Struct(n, _) | Type::Enum(n, _) => Some(n.as_ref().to_string()),
            _ => None,
        },
        Type::Struct(n, _) | Type::Enum(n, _) => Some(n.as_ref().to_string()),
        _ => None,
    }
}

/// A tensor type recovered by GID: its element and shape (dim expressions, numeric or symbolic).
/// The tensor analogue of `AggLayout`, but sourced from the lowerer's side table rather than the
/// registry (tensor types are structural, not nominal). See the C2 plan's tensor-type design note.
pub type TensorMap = HashMap<TypeId, (ElementType, Vec<String>)>;

/// The resolution context a flat stream references by GID: callee signatures (for `Call`), aggregate
/// layouts (for struct ops), and tensor types (for tensor ops). Bundling them keeps the emitter
/// signature stable as more non-scalar families land. `Default` is the empty context for streams that
/// reference none (scalar/control-flow-only functions).
#[derive(Default)]
pub struct EmitCtx {
    pub callees: CalleeMap,
    pub aggs: AggMap,
    /// Base struct/enum name → modelled layout GID, for resolving a monomorphized cross-module
    /// aggregate whose signature carries an unresolved base GID (`Struct("Vec", None)`). (#242)
    pub agg_names: HashMap<String, TypeId>,
    pub tensors: TensorMap,
    /// Names of payload-free (C-like) enums — an enum-typed value/param/return is a bare `i32`
    /// discriminant, not an aggregate (#227). Mirrors `ImmutableGlobalRegistry::enum_variants`.
    pub enums: std::collections::HashSet<String>,
    /// Function GID → (parameter MLIR types, return MLIR type), for a `FuncConst` that materializes a
    /// pointer to a named function: `func.constant @name : (params)->ret` needs the target's exact
    /// signature. Populated from the module's own function list (`emit_module_mlir`), since the frozen
    /// `fn_sigs` carries only the return type. (#242)
    pub func_sigs: HashMap<TypeId, (Vec<String>, String)>,
    /// Memory-space dispatch id → its declared sub-space descriptor, so an `Opcode::Transfer` (whose
    /// `imm` is that dispatch id) can re-attach the scheduling attributes (`space`/`within`/`granule`/
    /// `capacity`/`scope` + the bump-allocated `offset`/`slots`) the AST path emits. The frozen registry
    /// carries no memory decls, so this is threaded in from the per-compilation env. Empty for a program
    /// with no declared sub-spaces (P0-1).
    pub subspaces: HashMap<u64, SubspaceInfo>,
    /// Declared topology arch by dispatch id, so a `vx.spawn` can carry the arch its machine file
    /// declared and the device pipeline can gate on the DECLARATION rather than on the dispatch-id
    /// band -- which a custom topology can never enter (declared ids start at 3000 by
    /// construction, and the GPU band is [500, 600)). Built-in topologies are not in this map and
    /// keep riding the band.
    pub topo_archs: HashMap<i64, String>,
}

/// The declared properties of a memory sub-space the flat emitter re-attaches to a `vx.transfer`,
/// keyed in `EmitCtx.subspaces` by the space's dispatch id. Mirrors the fields the AST path reads off
/// `MemoryDecl` ([`src/codegen/lower/tensors.rs`]). The frozen registry has no memory decls, so this is
/// built in `build_flat_module` from the per-compilation env and passed into `emit_module_mlir`. (P0-1)
#[derive(Clone, Debug, Default)]
pub struct SubspaceInfo {
    pub dispatch_id: u64,
    pub name: String,
    pub within: Option<String>,
    pub granule: Option<u64>,
    pub capacity: Option<u64>,
    pub scope: Option<String>,
    /// "explicit" or "cached". Decides whether the host may read this space, and
    /// so whether a kernel that fails to route may fall back to the host at all
    /// -- see `diagnoseUnrunnableSpawns` in src/dialect/VxLowering.cpp. Unlike
    /// every other field here it is not descriptive: dropping it silently turns
    /// a compile error into a segmentation fault on a GPU (#251, #348).
    pub managed: Option<String>,
}

/// Runtime helpers the JIT links rather than the module defining: a body that calls one gets a
/// `func.func private` declaration prepended. Order is load-bearing — the emit records which of
/// these a function called as a bitmask over this array's indices, so inserting in the middle
/// renumbers existing entries.
const RUNTIME_HELPERS: [(&str, &str); 11] = [
    ("printMemrefF32", "(memref<*xf32>)"),
    ("printMemrefF64", "(memref<*xf64>)"),
    ("printMemrefI32", "(memref<*xi32>)"),
    ("printMemrefI64", "(memref<*xi64>)"),
    ("printMemrefBF16", "(memref<*xbf16>)"),
    ("print_f32", "(f32) -> i32"),
    ("print_f64", "(f64) -> i32"),
    ("print_i32", "(i32) -> i32"),
    ("print_i64", "(i64) -> i32"),
    ("print_str", "(!llvm.ptr) -> i32"),
    ("vx_init_signals", "()"),
];

/// Recover the declared memory sub-spaces from a compilation's env, in the shape
/// [`emit_module_mlir`] wants.
///
/// The frozen registry carries no memory declarations, so the env is the only place a flat compile
/// can get them; without this the `space`/`within`/`granule`/`capacity`/`scope` attributes an
/// `Opcode::Transfer` should carry are silently dropped on the flat path. Shared by `vxc`'s
/// `build_flat_module` and the parallel pipeline's codegen phase, so the two cannot disagree about
/// what a `vx.transfer` is annotated with (P0-1, #311).
/// The declared-arch table for `EmitCtx::topo_archs`, from the per-compilation env: one entry per
/// user-declared topology that states an `arch:`. A topology that declines to declare one gets no
/// entry, no `arch` attribute, and therefore no device compilation -- refusing to guess is the
/// same policy `fleet/m4-uma.vx` documents for its own arch field.
pub fn topo_archs_from_env(env: &crate::hir::GlobalAstEnv) -> Vec<(i64, String)> {
    let mut out: Vec<(i64, String)> = env
        .topologies
        .values()
        .filter_map(|decl| {
            let arch = decl.descriptor.arch.as_ref()?;
            let id = crate::arch::topology_dispatch_id(&crate::syntax::Topology::Custom(
                decl.name.clone(),
            ));
            Some((id as i64, arch.as_ref().to_string()))
        })
        .collect();
    // Sorted, because `env.topologies` is a HashMap and iteration order would otherwise decide
    // which entry wins an id collision -- the coin-flip E6016 refuses upstream, kept impossible
    // here too so the emitted MLIR is byte-reproducible (the freeze protocol depends on that).
    out.sort();
    out
}

pub fn subspaces_from_env(env: &crate::hir::GlobalAstEnv) -> Vec<SubspaceInfo> {
    let mut out: Vec<SubspaceInfo> = env
        .memories
        .values()
        .map(|decl| {
            let space = crate::syntax::MemorySpace::from_name(decl.name.as_ref());
            SubspaceInfo {
                dispatch_id: crate::arch::memory_space_dispatch_id(&space) as u64,
                name: space.name(),
                within: decl.parent.as_ref().map(|p| p.name()),
                granule: decl.granule.as_ref().map(|g| g.0),
                capacity: decl.capacity.as_ref().map(|c| c.0),
                scope: decl.scope.as_ref().map(|s| {
                    match s {
                        crate::syntax::Scope::Device => "device",
                        crate::syntax::Scope::Sm => "sm",
                        crate::syntax::Scope::Cta => "cta",
                        crate::syntax::Scope::Thread => "thread",
                    }
                    .to_string()
                }),
                managed: Some(
                    match decl.managed {
                        crate::syntax::Management::Explicit => "explicit",
                        crate::syntax::Management::Cached => "cached",
                    }
                    .to_string(),
                ),
            }
        })
        .collect();
    // Sorted for the same reason `topo_archs_from_env` is: `env.memories` is a HashMap, so its
    // walk order would otherwise decide which descriptor wins a dispatch-id collision -- and that
    // order varies per process, which made the emitted MLIR differ between runs of the same file.
    // E6016 refuses the collision upstream now; the sort keeps the emit order stable regardless.
    out.sort_by(|a, b| (a.dispatch_id, &a.name).cmp(&(b.dispatch_id, &b.name)));
    out
}

impl EmitCtx {
    /// Build the callee + struct-layout maps from the registry. The tensor map is *not* in the
    /// registry (tensor types are structural); populate it separately from the lowerer's side table.
    pub fn from_registry(
        registry: &ImmutableGlobalRegistry,
        sched: crate::config::Schedule,
    ) -> Self {
        let aggs = build_agg_map(registry, sched);
        let agg_names = build_agg_names(registry, &aggs);
        let callees = build_callee_map(registry, &aggs, &agg_names, sched);
        Self {
            callees,
            aggs,
            agg_names,
            tensors: TensorMap::new(),
            enums: registry
                .enum_variants
                .keys()
                .map(|s| s.as_ref().to_string())
                .collect(),
            func_sigs: HashMap::new(),
            subspaces: HashMap::new(),
            topo_archs: HashMap::new(),
        }
    }

    /// The modelled layout GID of a nominal type — the attached GID (in `aggs`) or, when a
    /// monomorphized cross-module signature left it unresolved, the base name's GID (`agg_names`).
    /// (#242)
    fn agg_gid(&self, ty: &Type) -> Option<TypeId> {
        resolve_agg_gid(ty, &self.aggs, &self.agg_names)
    }
}

/// The MLIR scalar type of a payload-free enum (a bare `i32` discriminant), if `ty` names one in
/// `ctx.enums`. The resolver may spell an enum as `Type::Enum` or `Type::Struct`, so match on the
/// name. `None` for anything else.
fn enum_scalar(ty: &Type, ctx: &EmitCtx) -> Option<&'static str> {
    let name = match ty {
        Type::Enum(name, _) | Type::Struct(name, _) => name.as_ref(),
        _ => return None,
    };
    if ctx.enums.contains(name) {
        Some("i32")
    } else {
        None
    }
}

/// Byte size of a statically-shaped flat tensor (its shape strings are all integer literals):
/// `ceil(element_bits × Π(dims) / 8)`. `None` when the shape is empty or any dim is symbolic — the
/// flat-emitter analogue of the AST codegen's `static_tensor_bytes`, over the lowerer's `Vec<String>`
/// shape, so both paths size a tile identically for the sub-space bump allocator. (P0-1)
fn static_tile_bytes(elem: &ElementType, shape: &[String]) -> Option<u64> {
    if shape.is_empty() {
        return None;
    }
    let mut count: u64 = 1;
    for d in shape {
        count = count.checked_mul(d.parse::<u64>().ok()?)?;
    }
    Some(
        crate::hir::memory::element_bits(elem)?
            .checked_mul(count)?
            .div_ceil(8),
    )
}

/// The MLIR type string for an AST type in a function signature position (a parameter or return): a
/// scalar's element, a payload-free enum's `i32`, an opaque `!llvm.ptr` (pointer / fn-pointer), a
/// tensor's memref, or a by-value aggregate's `!llvm.struct`. `None` for a void / unmodelled type.
/// The single source of truth shared by the `func.func` header and a `FuncConst`'s `func.constant`
/// signature, so a materialized function pointer's type matches its callee's header exactly. (#242)
fn ty_mlir(ty: &Type, ctx: &EmitCtx) -> Lowered<String> {
    let ty = peel_wrappers(ty); // Verified/Ref/Pinned spell as their runtime inner type
                                // A borrowed tensor is spelled as the tensor. A memref is already a reference, and a call
                                // site hands over the memref itself; `!llvm.ptr` gave the signature a type nothing passes
                                // (Vx#417).
    let ty = match ty {
        Type::Borrow { inner, .. } if crate::hir::flatten::tensor_gid_of(inner).is_some() => {
            peel_wrappers(inner)
        }
        other => other,
    };
    if let Some(e) = scalar_of(ty) {
        Ok(mlir_scalar(&e)
            .ok_or(Decline::TypeNotModelled {
                what: "a scalar with no MLIR spelling",
            })?
            .to_string())
    } else if let Some(et) = enum_scalar(ty, ctx) {
        Ok(et.to_string())
    } else if is_ptr_ty(ty) {
        Ok("!llvm.ptr".to_string())
    } else if let Some(gid) = tensor_gid_of(ty) {
        let (elem, shape) = ctx.tensors.get(&gid).ok_or(Decline::TypeNotModelled {
            what: "a tensor with no recorded shape",
        })?;
        tensor_memref_ty(elem, shape).ok_or(Decline::TypeNotModelled {
            what: "a tensor with no memref spelling",
        })
    } else if let Some(gid) = ctx.agg_gid(ty) {
        Ok(ctx
            .aggs
            .get(&gid)
            .ok_or(Decline::TypeNotModelled {
                what: "an aggregate with no struct type",
            })?
            .struct_ty
            .clone())
    } else {
        Err(Decline::TypeNotModelled {
            what: "a type with no MLIR spelling",
        })
    }
}

/// Emit a whole module — every function as a concatenated bare `func.func` — or `None` if *any*
/// function is outside the current subset (module-level keep-green atomicity: a partially lowered
/// module is never emitted, so the AST path stays the oracle for the whole program). Callees + struct
/// layouts resolve through the frozen registry; `tensor_types` is the concatenation of each function's
/// lowerer side table (`LocalWorkerState::local_tensor_types`). Wrap the result in `module { … }`.
#[allow(clippy::too_many_arguments)]
pub fn emit_module_mlir(
    funcs: &[(&Function, &[HirInstruction], &[TypeId])],
    registry: &ImmutableGlobalRegistry,
    tensor_types: &[(TypeId, ElementType, Vec<String>)],
    string_tables: &[&[String]],
    agg_layouts: &[(TypeId, Vec<u64>, Vec<String>)],
    alias_tables: &[&[(usize, usize, Vec<usize>)]],
    subspaces: &[SubspaceInfo],
    topo_archs: &[(i64, String)],
    sched: crate::config::Schedule,
) -> Lowered<String> {
    let setup = std::time::Instant::now();
    let mut ctx = EmitCtx::from_registry(registry, sched);
    for s in subspaces {
        ctx.subspaces.insert(s.dispatch_id, s.clone());
    }
    for (id, arch) in topo_archs {
        ctx.topo_archs.insert(*id, arch.clone());
    }
    for (gid, elem, shape) in tensor_types {
        ctx.tensors
            .entry(*gid)
            .or_insert_with(|| (elem.clone(), shape.clone()));
    }
    // Synthesized data-carrying enum-instance layouts (`Option<i32>` -> `{ i32, i32 }`): fold each into
    // the aggregate map so a `FieldStore`/`FieldLoad`/`Alloca` on its GID resolves like any struct.
    // Instance-dependent, so they aren't in the frozen registry (the lowerer synthesized them). (#242)
    for (gid, offsets, field_tys) in agg_layouts {
        ctx.aggs.entry(*gid).or_insert_with(|| AggLayout {
            struct_ty: format!("!llvm.struct<({})>", field_tys.join(", ")),
            offsets: offsets.clone(),
            field_tys: field_tys.clone(),
            // A synthesized enum instance carries no pointer-to-aggregate fields (its payload is a
            // scalar/pointer-to-scalar), so no chained field access resolves through it.
            field_pointee: vec![None; field_tys.len()],
            field_agg: vec![None; field_tys.len()],
        });
    }
    // Rebuild the callee map now that the synthesized enum-instance layouts are in `aggs`: a callee
    // returning `Option<i32>` (a data enum) resolves its `ret_agg` only once its layout is present,
    // which happens above — `EmitCtx::from_registry` built the callees before it (#242).
    if !agg_layouts.is_empty() {
        ctx.callees = build_callee_map(registry, &ctx.aggs, &ctx.agg_names, sched);
    }
    // Function GID → (param MLIR types, ret MLIR type), for a `FuncConst`'s `func.constant @name : sig`.
    // Built from the module's own functions (each `Function` carries its params); the target must be a
    // function whose whole signature is modelled (else the FuncConst declines at emit). (#242)
    // This is the bulk of the prologue and it was the largest serial section in the whole compile
    // (#314): `ty_mlir` over every slot of every function is ~11,000 calls on a 1,600-function
    // corpus, and it sat ahead of the parallel emit doing work proportional to the whole program.
    //
    // It parallelises because it is a pure map -- `ty_mlir` reads `aggs`/`agg_names`/`enums`/
    // `tensors` and never `func_sigs`, so no function's signature depends on another's. The
    // *insertion* stays serial and in `funcs` order: two functions can resolve to the same GID, and
    // then which one wins is decided by insertion order. Collecting in completion order would make
    // that a race.
    let compute_sig = |(func, _, _): &(&Function, &[HirInstruction], &[TypeId])| {
        let sig = registry.fn_sigs.get(func.name.as_ref())?;
        let params: Option<Vec<String>> = func
            .params
            .iter()
            .map(|(_, t)| ty_mlir(t, &ctx).ok())
            .collect();
        let ret = match &func.return_type {
            Type::Scalar(ElementType::Generic(_)) => None,
            t => Some(ty_mlir(t, &ctx).unwrap_or_else(|_| "()".to_string())),
        };
        match (params, ret) {
            (Some(params), Some(ret)) => Some((sig.gid, (params, ret))),
            _ => None,
        }
    };
    let sigs: Vec<(TypeId, (Vec<String>, String))> = if sched == crate::config::Schedule::Sequential
    {
        funcs.iter().filter_map(compute_sig).collect()
    } else {
        funcs.par_iter().filter_map(compute_sig).collect()
    };
    for (gid, s) in sigs {
        ctx.func_sigs.insert(gid, s);
    }
    // Two module-wide numbering schemes are threaded through the per-function emit:
    //
    //  - String literals. Each function's literals are numbered from a running module-wide base, so a
    //    `PrintStr`'s `@".str.<n>"` reference (emitted with the same `str_base`) resolves the global
    //    emitted here.
    //  - Alias scopes (M2b-2). 0 is reserved for the shared alias domain; each function reserves a
    //    contiguous block of `distinct[]` ids for its groups, so scopes stay distinct across functions
    //    even after inlining. (#275, §5.4)
    //
    // Both used to be running `&mut` counters carried through the loop, which forced the per-function
    // emit -- the actual code generation -- to run one function at a time. Neither *needs* to be
    // sequential: a function's literal count is `string_tables[fi].len()` and its alias-group count is
    // `max(group) + 1` over its own alias table, both known before anything is emitted. Turning them
    // into prefix sums makes the emit a `par_iter` and leaves the output byte-identical, because the
    // numbering is a function of position, never of arrival order (#311).
    crate::intern_mode::record("  codegen:setup", setup.elapsed());
    let emit_start = std::time::Instant::now();
    let mut str_bases: Vec<usize> = Vec::with_capacity(funcs.len());
    let mut distinct_bases: Vec<u32> = Vec::with_capacity(funcs.len());
    {
        let mut str_base = 0usize;
        let mut distinct_ctr: u32 = 1;
        for fi in 0..funcs.len() {
            str_bases.push(str_base);
            str_base += string_tables.get(fi).copied().unwrap_or(&[]).len();
            distinct_bases.push(distinct_ctr);
            distinct_ctr += alias_tables
                .get(fi)
                .copied()
                .unwrap_or(&[])
                .iter()
                .map(|(_, g, _)| *g as u32 + 1)
                .max()
                .unwrap_or(0);
        }
    }

    // Which runtime helpers each function calls, as a bitmask over `RUNTIME_HELPERS`.
    //
    // This used to be ten `out.contains(...)` scans over the *whole* concatenated module — on a
    // corpus emitting 5 MB, fifty megabytes of scanning, serial, after the parallel emit had
    // finished. `out` is exactly the concatenation of the per-function texts, so asking each
    // function about its own text gives the identical answer, in parallel, while that text is still
    // hot in cache. The patterns are built once rather than per function per helper.
    let helper_pats: Vec<String> = RUNTIME_HELPERS
        .iter()
        .map(|(name, _)| format!("@{name}("))
        .collect();

    type FnEmission = (String, Vec<(String, Vec<String>, String)>, u16);
    let emit_one =
        |(fi, (func, hir, types)): (usize, &(&Function, &[HirInstruction], &[TypeId]))| {
            let mut calls = Vec::new();
            let mut distinct_ctr = distinct_bases[fi];
            let text = emit_function_mlir(
                func,
                hir,
                types,
                &ctx,
                &mut calls,
                str_bases[fi],
                string_tables.get(fi).copied().unwrap_or(&[]),
                alias_tables.get(fi).copied().unwrap_or(&[]),
                &mut distinct_ctr,
            )?;
            let helpers = helper_pats.iter().enumerate().fold(0u16, |m, (i, pat)| {
                if text.contains(pat.as_str()) {
                    m | (1 << i)
                } else {
                    m
                }
            });
            Ok((text, calls, helpers))
        };
    let mut emitted: Vec<Lowered<FnEmission>> = if sched == crate::config::Schedule::Sequential {
        funcs.iter().enumerate().map(emit_one).collect()
    } else {
        funcs.par_iter().enumerate().map(emit_one).collect()
    };
    crate::intern_mode::record("  codegen:emit", emit_start.elapsed());

    // Reassembly is serial by necessity -- the output is ordered -- so it is the one part of codegen
    // that cannot be parallelised, and it is therefore the one part whose constant factor matters
    // most. Sizing the buffer up front is not a micro-optimisation here: a `String::new()` growing
    // to the several megabytes a real corpus emits reallocates ~20 times, and every reallocation
    // memcpys everything appended so far, so the append loop costs on the order of twice the output
    // size in pure copying. That cost lands entirely on the critical path and grows with the program
    // being compiled, which is exactly the shape that flattens a scaling curve.
    let out_len: usize = emitted
        .iter()
        .filter_map(|e| e.as_ref().ok().map(|(t, _, _)| t.len()))
        .sum();
    let mut globals = String::new();
    let mut calls: Vec<(String, Vec<String>, String)> = Vec::new();
    // First pass gathers only what the *header* needs -- the callee list, the runtime-helper union
    // and the string globals. The bodies are deliberately not touched yet.
    //
    // The declarations depend on `calls`, not on the assembled text, so they can be built before a
    // single byte of body is copied. That is what makes one copy enough: the header is finished
    // first, the final buffer is allocated at its exact size, and the bodies go straight into it.
    // Assembling bodies first forced the old order -- body, then header, then concatenate -- which
    // cost a full copy of a multi-megabyte module per concatenation.
    //
    // A decline is reported for the *first* declining function rather than whichever thread noticed
    // first, so `VX_FLAT_DBG` says the same thing it always did.
    let mut helper_mask = 0u16;
    for (fi, emission) in emitted.iter_mut().enumerate() {
        let (_, fn_calls, helpers) = match emission {
            Ok(e) => e,
            Err(why) => {
                if std::env::var("VX_FLAT_DBG").is_ok() {
                    eprintln!(
                        "[flat-dbg] emit declined for fn {}",
                        funcs[fi].0.name.as_ref()
                    );
                }
                return Err(why.clone());
            }
        };
        calls.append(fn_calls);
        helper_mask |= *helpers;
        let strs = string_tables.get(fi).copied().unwrap_or(&[]);
        for (li, s) in strs.iter().enumerate() {
            globals += &emit_string_global(str_bases[fi] + li, s);
        }
    }
    // Prepend `private` declarations for any runtime print helpers the bodies call (the JIT links
    // their implementations; the AST path declares them the same way). Which ones were determined
    // per function during the parallel emit; this is the union.
    let mut decls = String::new();
    for (i, (name, sig)) in RUNTIME_HELPERS.iter().enumerate() {
        if helper_mask & (1 << i) != 0 {
            decls += &format!("  func.func private @{name}{sig}\n");
        }
    }
    // Declare any *called-but-undefined* callee (an `extern`: no `func.func @name` body emitted in this
    // module) as `func.func private`. The signature comes from the emitted `func.call`, so they match;
    // the JIT links the symbol (libm via `-lm`, `libvx_std_core`, ...). Deduped, in first-seen order.
    let defined: std::collections::HashSet<&str> =
        funcs.iter().map(|(f, _, _)| f.name.as_ref()).collect();
    // Borrows from `calls` rather than cloning each name. The set only needs to answer "seen
    // already", and a corpus with cross-module calls produces one entry here per call site --
    // tens of thousands of `String` allocations to record facts about strings that are already in
    // memory and outlive the loop.
    let mut declared: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for (name, arg_types, ret) in &calls {
        if defined.contains(name.as_str()) || !declared.insert(name.as_str()) {
            continue;
        }
        let ret_sig = if ret.is_empty() {
            String::new()
        } else {
            format!(" -> {ret}")
        };
        decls += &format!(
            "  func.func private {}({}){ret_sig}\n",
            sym_ref(name),
            arg_types.join(", ")
        );
    }
    // One buffer, one copy of each body, exact capacity, wrapper included.
    //
    // The module text used to be copied three times before anything could parse it: bodies into
    // `out`, then `globals + decls + out` into a result, then a `format!` at each call site to wrap
    // it in `module { … }`. On a 1,000-module corpus that is a 51 MB buffer copied three times and
    // freed twice, and the cost is not the memcpy -- it is faulting in 150 MB of fresh pages, on the
    // critical path, at the end of an otherwise well-parallelised phase.
    //
    // Emitting the wrapper here rather than leaving it to callers is what removes the last of them:
    // a caller that has to wrap the return value cannot avoid copying it.
    const OPEN: &str = "module {\n";
    const CLOSE: &str = "}\n";
    let mut module =
        String::with_capacity(OPEN.len() + globals.len() + decls.len() + out_len + CLOSE.len());
    module.push_str(OPEN);
    module.push_str(&globals);
    module.push_str(&decls);
    for emission in emitted.iter().flatten() {
        module.push_str(&emission.0);
    }
    module.push_str(CLOSE);
    // The per-function texts were copied, not consumed, so their allocations are still live. Freeing
    // them here rather than during assembly keeps tens of thousands of frees off the critical path,
    // and every one of them is a *remote* free: built by a worker, released on this thread (#315).
    // `calls` is one entry per emitted call site -- three heap allocations each, all built by
    // workers -- so it is the same remote-free problem in miniature and gets the same treatment.
    // `declared` borrows from it, so this has to come after the declaration loop.
    if sched == crate::config::Schedule::Sequential {
        drop(emitted);
        drop(calls);
    } else {
        emitted.into_par_iter().for_each(drop);
        calls.into_par_iter().for_each(drop);
    }
    Ok(module)
}

/// Emit the module-level `llvm.mlir.global` for a string literal: an internal constant array holding
/// the null-terminated bytes, named `@".str.<n>"` to match the `llvm.mlir.addressof` a `PrintStr`
/// emits. The array length is the byte count *including* the terminator.
fn emit_string_global(n: usize, s: &str) -> String {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0); // C-string null terminator (matches the AST path's `format!("{}\0", value)`)
    let escaped = mlir_escape_bytes(&bytes);
    format!(
        "  llvm.mlir.global internal constant @\".str.{n}\"(\"{escaped}\") : !llvm.array<{} x i8>\n",
        bytes.len()
    )
}

/// A function symbol reference for emitted textual MLIR. A name that is a valid bare MLIR symbol
/// (`[A-Za-z_$.][A-Za-z0-9_$.]*` — covers `main`, `add`, `printMemrefF32`, and monomorph names like
/// `f32$sq`) is emitted bare (`@name`), matching the AST path's spelling so FileCheck stays stable.
/// A mangled method name carries `::` (`Vec::with_capacity$i32`), which is MLIR's
/// nested-symbol-reference separator — a bare `@Vec::with_capacity` fails to parse — so such a name
/// is quoted into a single flat symbol (`@"..."`, which links to the same underlying symbol). (#242)
fn sym_ref(name: &str) -> String {
    let bare = !name.is_empty()
        && !name.starts_with(|c: char| c.is_ascii_digit())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.'));
    if bare {
        format!("@{name}")
    } else {
        format!("@\"{}\"", name.replace('\\', "\\\\").replace('"', "\\\""))
    }
}

/// Escape raw bytes for an MLIR string literal: a printable ASCII byte other than `"` or `\` passes
/// through; everything else (including the null terminator and any non-ASCII byte) becomes a `\XX`
/// two-digit hex escape. Conservative but always valid MLIR.
fn mlir_escape_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        if b == b'"' || b == b'\\' || !(0x20..=0x7e).contains(&b) {
            out += &format!("\\{b:02X}");
        } else {
            out.push(b as char);
        }
    }
    out
}

/// The element type of a memref type string, e.g. `memref<2x4xf32> -> "f32"`,
/// `memref<4xf32, strided<…>> -> "f32"`.
fn memref_elem(memty: &str) -> Option<&str> {
    memref_lead_dims_and_elem(memty).map(|(_, e)| e)
}

/// The dims prefix (`2x4x`) and element (`f32`) of a memref type string. `memref<2x4xf32> ->
/// ("2x4x", "f32")`; a layout suffix (`, strided<…>`) is dropped.
fn memref_lead_dims_and_elem(memty: &str) -> Option<(&str, &str)> {
    let inner = memty.strip_prefix("memref<")?;
    let inner = inner.split(',').next()?.trim_end_matches('>');
    let elem_start = inner.rfind('x').map(|i| i + 1).unwrap_or(0);
    Some((&inner[..elem_start], &inner[elem_start..]))
}

/// The static length of a **rank-1** memref, e.g. `memref<4xf32, strided<…>> -> 4`. This is the
/// width of the `vector<Nx…>` a reduction loads the slice into.
///
/// `None` for any higher rank, matching the AST path's `slice_vec_len`. The load addresses a
/// single index, so a rank-2 operand would be read with one index too few and MLIR rejects the op.
/// Taking the leading dimension without checking the rank answered `memref<4x4xf32>` with 4 and
/// left the caller no way to tell.
fn memref_lead_dim(memty: &str) -> Option<i64> {
    let inner = memty.strip_prefix("memref<")?;
    // The shape is the text before any `, strided<…>` layout, e.g. `4xf32`; exactly one `x`
    // separating a dimension from the element type is rank 1.
    let shape = inner.split(',').next()?.trim_end_matches('>');
    let parts: Vec<&str> = shape.split('x').collect();
    if parts.len() != 2 {
        return None;
    }
    parts[0].parse::<i64>().ok()
}

/// Coerce a slice-elementwise operand register to a `vector<Nxf32>` value (matching the AST's
/// ` {alignment = 16 : i64}` when a whole-row vector access is provably 16-byte aligned, else
/// nothing. Without it the memref lowering stamps the element's alignment (4 for f32), NVPTX
/// cannot fuse the access, and a `vector.load` of a row arrives on the device as N scalar
/// `ld.global.b32` -- measured as the per-SM issue bound the vector path exists to fix (Vx#378 R1;
/// with the attribute the same row is four-element `v4` transactions).
///
/// The claim is sound for whole rows: every slice this emitter loads or stores is row `i` of an
/// allocation, at byte offset `i * rowbytes`. The base is 16-byte aligned everywhere rows come
/// from (`malloc` guarantees 16 on the host paths, `cudaMalloc` 256 on the device), so the row is
/// 16-byte aligned exactly when `rowbytes` is a multiple of 16 -- which is what this checks, from
/// the vector type's own width and element.
fn vector_align_attr(vecty: &str) -> &'static str {
    let Some(inner) = vecty
        .strip_prefix("vector<")
        .and_then(|s| s.strip_suffix('>'))
    else {
        return "";
    };
    let mut parts = inner.splitn(2, 'x');
    let (Some(n), Some(et)) = (parts.next(), parts.next()) else {
        return "";
    };
    let Ok(n) = n.parse::<u64>() else {
        return "";
    };
    let esize = match et {
        "f32" | "i32" => 4,
        "f64" | "i64" => 8,
        "f16" | "bf16" => 2,
        _ => return "",
    };
    if n * esize % 16 == 0 {
        " {alignment = 16 : i64}"
    } else {
        ""
    }
}

/// `to_vector`): an already-vector operand passes through, a rank-1 memref is `vector.load`ed, and a
/// scalar is `vector.broadcast`ed to the slice width. Emits into `body`; returns the vector SSA name.
/// `tag` disambiguates the emitted SSA names.
#[allow(clippy::too_many_arguments)]
fn coerce_vector(
    body: &mut String,
    tag: &str,
    op_reg: u32,
    vecty: &str,
    et: &str,
    names: &[String],
    mem_of: &[Option<String>],
    vec_of: &[Option<String>],
    etypes: &[Option<ElementType>],
) -> Lowered<String> {
    let name = names
        .get(op_reg as usize)
        .ok_or(Decline::TypeNotModelled {
            what: "an operand with no emitted name",
        })?
        .clone();
    if vec_of
        .get(op_reg as usize)
        .ok_or(Decline::TypeNotModelled {
            what: "an operand with no vector record",
        })?
        .is_some()
    {
        return Ok(name); // already a vector (a prior elementwise result)
    }
    if let Some(m) = mem_of
        .get(op_reg as usize)
        .ok_or(crate::emitter_gap!())?
        .clone()
    {
        let c0 = format!("%vc{tag}");
        let v = format!("%vl{tag}");
        body.push_str(&format!("  {c0} = arith.constant 0 : index\n"));
        // Half-precision STORAGE widens on load (Vx#320): when the row's element is
        // f16/bf16 and the op wants f32 lanes, load the narrow vector and `arith.extf`
        // it wide. The narrow row still earns the alignment attribute on its own terms
        // (64 halves are 128 bytes -- v8 packs).
        let row_elem = m
            .rsplit('x')
            .next()
            .ok_or(crate::emitter_gap!())?
            .trim_end_matches('>');
        let row_elem = row_elem
            .split(',')
            .next()
            .ok_or(crate::emitter_gap!())?
            .trim();
        if (row_elem == "f16" || row_elem == "bf16") && vecty.ends_with("xf32>") {
            let lanes = vecty
                .strip_prefix("vector<")
                .ok_or(crate::emitter_gap!())?
                .split('x')
                .next()
                .ok_or(crate::emitter_gap!())?
                .to_string();
            let nvec = format!("vector<{lanes}x{row_elem}>");
            let nal = vector_align_attr(&nvec);
            let nv = format!("%vn{tag}");
            body.push_str(&format!(
                "  {nv} = vector.load {name}[{c0}]{nal} : {m}, {nvec}\n"
            ));
            body.push_str(&format!("  {v} = arith.extf {nv} : {nvec} to {vecty}\n"));
            return Ok(v);
        }
        let al = vector_align_attr(vecty);
        body.push_str(&format!(
            "  {v} = vector.load {name}[{c0}]{al} : {m}, {vecty}\n"
        ));
        return Ok(v);
    }
    if let Some(se) = etypes
        .get(op_reg as usize)
        .ok_or(crate::emitter_gap!())?
        .clone()
    {
        // A half scalar widens before it is broadcast, the same way a half row does
        // (Vx#320). Broadcasting it as if it were already `et` names the wrong type for
        // the value and the module does not parse.
        let mut name = name;
        let se_mlir = crate::mlir_ty::mlir_scalar(&se).ok_or(crate::emitter_gap!())?;
        if se_mlir != et {
            if !se.is_float() {
                return Err(Decline::TypeNotModelled {
                    what: "a non-float scalar operand of an elementwise slice op",
                });
            }
            let w = format!("%vw{tag}");
            body.push_str(&format!("  {w} = arith.extf {name} : {se_mlir} to {et}\n"));
            name = w;
        }
        let v = format!("%vb{tag}");
        body.push_str(&format!(
            "  {v} = vector.broadcast {name} : {et} to {vecty}\n"
        ));
        return Ok(v);
    }
    Err(Decline::TypeNotModelled {
        what: "an operand that cannot be coerced to a vector",
    })
}

/// Comma-join integers (for `sizes: [..]` / `strides: [..]` lists).
fn join_i64(xs: &[i64]) -> String {
    xs.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The static MLIR memref type string for a tensor type, e.g. `(i32, ["4"]) -> "memref<4xi32>"`. A
/// non-numeric dim becomes `?` (dynamic). `None` for a non-scalar element.
fn tensor_memref_ty(elem: &ElementType, shape: &[String]) -> Option<String> {
    let et = mlir_scalar(elem)?;
    let dims: String = shape
        .iter()
        .map(|d| {
            let d = if d.parse::<i64>().is_ok() {
                d.as_str()
            } else {
                "?"
            };
            format!("{d}x")
        })
        .collect();
    Some(format!("memref<{dims}{et}>"))
}

/// Everything one function's instruction walk needs: the read-only context, the per-register
/// tables it fills in as it goes, and the MLIR text it appends to. The flat path has no AST to
/// consult, so an instruction that needs an *operand's* type recovers it from the table the
/// producing instruction wrote. Each opcode family lives in its own `impl` block under
/// `codegen/flat/emit/`.
pub(crate) struct FnEmit<'a> {
    pub(crate) func: &'a Function,
    pub(crate) types: &'a [TypeId],
    pub(crate) ctx: &'a EmitCtx,
    /// Base index of this function's slice of the module-level string globals.
    pub(crate) str_base: usize,
    /// This function's string side table, indexed by an instruction's `imm`. `PrintStr` and
    /// `StringConst` reach their bytes through a module-level global; `Abort` needs the text
    /// itself, because `cf.assert` carries its message as an inline attribute.
    pub(crate) strings: &'a [String],
    /// The function's scalar return element, when it has one — a `Ret` of a differently-typed
    /// value converts to this.
    pub(crate) ret_elem: Option<ElementType>,
    /// Alias-scope metadata for place-write field stores: each tagged store's stream position ->
    /// its own alias-scope `distinct[]` id and its disjoint-sibling ids. Numbered from the
    /// module-global counter, so scopes stay distinct across functions.
    pub(crate) alias_scope_of: HashMap<usize, (u32, Vec<u32>)>,
    /// Callees this function reaches, collected so the module can declare them.
    pub(crate) calls: &'a mut Vec<(String, Vec<String>, String)>,
    /// The SSA name each register carries.
    pub(crate) names: Vec<String>,
    /// The scalar element type each register carries, indexed by register (= instruction position
    /// in the stream) — the type-valued parallel to `names`. As the stream is walked, each
    /// value-producing instruction records its result type here, so any later instruction can
    /// recover the type of a register it *reads*. This is the flat-driven stand-in for reading an
    /// operand's type off the AST: there is no AST node to consult, so types are reconstructed on
    /// the way.
    ///
    /// Why not just read each instruction's own `type_idx`? Most values are self-describing that
    /// way, but two consumers need an *operand's* type, which their own `type_idx` doesn't give:
    ///   - `Cmp`: its own result type is `bool`, but choosing `cmpi`/`cmpf` + the signed/unsigned
    ///     predicate needs the *operands'* type -> `etypes[operand1]`.
    ///   - `Store`: an effect instruction (its `type_idx` is the no-type sentinel), but printing
    ///     `memref<T>` needs the slot's element type -> the `Alloca` records it, `Store` reads it
    ///     back.
    ///
    /// Calls use it too: a `func.call`'s argument types come from `etypes[arg_reg]`.
    ///
    /// Scalar-only today (`ElementType`); widening it to tensor/aggregate types is future work
    /// (#200).
    pub(crate) etypes: Vec<Option<ElementType>>,
    /// The aggregate GID for each register that is a struct-slot pointer (from an aggregate
    /// `Alloca`), so a `FieldLoad`/`FieldStore` on that slot recovers the struct's `!llvm.struct`
    /// type + field offsets. The aggregate analogue of `etypes` (kept separate: struct slots are
    /// pointers, not scalar values).
    pub(crate) agg_of: Vec<Option<TypeId>>,
    /// The aggregate GID for each register that holds a struct *value* (not a slot pointer): an
    /// aggregate `SlotLoad`, an aggregate-element `PtrIndex` read, or a struct-returning `Call`.
    /// Lets a consumer (a by-value call argument, a `PtrStore` of the value) print the
    /// `!llvm.struct` type. The value analogue of `agg_of` (which tracks struct *slots*).
    pub(crate) agg_val_of: Vec<Option<TypeId>>,
    /// The memref type string for each register that holds a tensor (from `TensorAlloc`), so a
    /// `TensorIndex`/`TensorStore` on it prints the right `memref<...>`.
    pub(crate) mem_of: Vec<Option<String>>,
    /// For a scalar-element *place* register (a `TensorIndex` with `imm = 1`): the base memref
    /// name, an `index`-typed index SSA name, and the base memref type — everything the following
    /// `TensorStore` needs to emit `memref.store %v, %base[%idx]`.
    pub(crate) place_of: Vec<Option<(String, String, String)>>,
    /// For a raw-pointer element *place* register (a `PtrIndex` with `imm = 1`): the pointee
    /// element's MLIR type, so the following `PtrStore` prints
    /// `llvm.store %v, %place : {elem}, !llvm.ptr`. The place register itself already holds the
    /// GEP'd element pointer (in `names`).
    pub(crate) pptr_elem: Vec<Option<String>>,
    /// The `vector<Nxf32>` type of each register holding an elementwise (slice) result, so a row
    /// `TensorStore` `vector.store`s it and a further elementwise passes it through.
    pub(crate) vec_of: Vec<Option<String>>,
    /// Whether each register holds an opaque `!llvm.ptr` *value* (a string const, a pointer param,
    /// a pointer-returning call, or a load from a pointer slot) — so a `Call` arg / `func.return`
    /// types it as `!llvm.ptr`. The pointer analogue of `etypes`.
    pub(crate) ptr_of: Vec<bool>,
    /// Whether each register is a pointer *slot* (an `llvm.alloca` of `!llvm.ptr`, a memory-mode
    /// pointer local), so a `Store`/`SlotLoad` on it uses `llvm.store`/`llvm.load` rather than
    /// `memref`.
    pub(crate) pslot_of: Vec<bool>,
    /// The element type of each register that is an *address-taken scalar slot* (an `llvm.alloca`
    /// of a scalar, `imm = 1` on the `Alloca`) — so `&x` yields a real `!llvm.ptr` (the slot
    /// register is also marked in `ptr_of`) and a `Store`/`SlotLoad` uses `llvm.store`/`llvm.load`
    /// of the element type rather than the rank-0 `memref` a never-borrowed scalar local uses. The
    /// scalar analogue of `pslot_of`, carrying the element so the load/store types match.
    pub(crate) sslot_of: Vec<Option<ElementType>>,
    /// The function body's MLIR text, appended to as the walk proceeds.
    /// The function's entry block, for allocations that must happen once.
    ///
    /// A stack slot emitted where its `let` appears sits inside whatever loop
    /// encloses it, and an `alloca` is only given back when the function
    /// returns -- so a slot in a loop body grows the stack once per iteration.
    /// A nest running a couple of million times then exhausts it and the
    /// program dies with SIGSEGV having compiled cleanly.
    ///
    /// These allocations take no operands, so hoisting them is always valid:
    /// the entry block dominates every use, and each iteration stores its own
    /// value into the slot before reading it.
    pub(crate) entry: String,
    pub(crate) body: String,
    /// Whether the block currently being emitted has a terminator yet (a block must end in one).
    pub(crate) terminated: bool,
    /// Argument value registers accumulated by the `Arg`s that immediately precede a `Call`; the
    /// `Call` consumes its `imm` trailing entries (a nested inner call sits between its own `Arg`s
    /// and the outer ones, so each call's args are exactly the tail — see `flatten::lower_call`).
    pub(crate) pending_args: Vec<u32>,
    /// The topology of the currently-open `vx.spawn` region (`Some` between `Spawn` and its
    /// matching `SpawnEnd`), remembered so `SpawnEnd` can emit the `topology` attribute. `None`
    /// outside a spawn; a nested spawn (already `Some`) is declined.
    pub(crate) spawn_topology: Option<i64>,
    /// Per-function sub-space bump allocator (space dispatch id -> next free byte), mirroring the
    /// AST path's `MeliorGenerator::subspace_offsets`: each `Transfer` into a granule'd space
    /// claims the next granule-rounded `offset` and advances the cursor, so both paths assign
    /// identical offsets. Reset per function, as in the AST codegen.
    pub(crate) subspace_offsets: HashMap<u64, u64>,
}

impl<'a> FnEmit<'a> {
    /// The scalar element type a type-stream index names, if it names one.
    pub(crate) fn ty_at(&self, ti: u32) -> Option<ElementType> {
        elem_of_gid(*self.types.get(ti as usize)?)
    }

    /// The scalar element type a register carries, if the producing instruction recorded one.
    pub(crate) fn elem_at(&self, r: u32) -> Option<ElementType> {
        self.etypes.get(r as usize)?.clone()
    }

    /// Emit one instruction into `body`. An opcode outside the flat subset declines, and
    /// the AST path stays the oracle for the whole function.
    /// Where a stack slot's allocation belongs.
    ///
    /// The function's entry block, so a slot inside a loop is taken once rather
    /// than once per iteration -- except inside a `vx.spawn`, where hoisting
    /// past the region boundary would change what the program means. A slot
    /// lifted out of a spawn stops being the region's own and becomes a value
    /// defined above it, which outlining then captures and passes in: a device
    /// kernel would receive a host stack pointer for what should be its own
    /// scratch. So a region keeps its slots, and a `let` in a loop *inside* a
    /// spawn still allocates per iteration.
    pub(crate) fn emit_slot(&mut self, text: &str) {
        match self.spawn_topology {
            // A device region is outlined into its own function, so a slot
            // lifted out of it stops being the kernel's own scratch and becomes
            // a value defined above it -- which outlining captures and passes
            // in, handing a device kernel a host stack pointer. Those stay.
            Some(t) if t != 0 => self.body += text,
            // A host region is inlined where it was written, so its slots
            // belong to the enclosing function like any other.
            _ => self.entry += text,
        }
    }

    pub(crate) fn step(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        match ins.opcode {
            // Parameter materialization: the register *is* the block argument, no op emitted. A
            // scalar param records its element type; a tensor param records its memref type (from the
            // side table) so later index/store ops address it.
            Opcode::Load => self.op_load(idx, ins),
            Opcode::Const => self.op_const(idx, ins),
            // Arithmetic: scalar (a scalar-GID result) or *elementwise* over a rank-1 float slice (a
            // tensor-GID result). The scalar form is `arith.{addi,mulf,…}`; the elementwise form
            // coerces each operand to a `vector<Nxf32>` (`vector.load`/`broadcast`), applies
            // `arith.{addf,subf,mulf,divf}`, and yields a vector that a row `TensorStore` writes back.
            Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div => self.op_binary(idx, ins),
            // Scalar comparison → `i1`; the relation is in `imm`, the operand type comes from the
            // first operand's tracked type (this instruction's own type is `bool`, the result).
            Opcode::Cmp => self.op_cmp(idx, ins),
            // Arithmetic negation `-x` (#214). `type_idx` is the result (= operand) scalar type. Float
            // → `arith.negf`; integers have no `negi`, so `0 - x` via `arith.subi`.
            Opcode::Neg => self.op_neg(idx, ins),
            // Logical / bitwise not `!x` (#214): `x ^ all-ones` (`1` for a bool `i1`, `-1` for ints).
            Opcode::Not => self.op_not(idx, ins),
            // Scalar `as` cast (#214): `type_idx` is the *target* type, `operand1` the source value
            // (whose type comes from its tracked `etypes`). The right `arith` conversion is chosen by
            // the source/target kinds + widths; a same-type cast is a no-op that just aliases.
            Opcode::Cast => self.op_cast(idx, ins),
            // A named local's stack slot. A scalar slot is a rank-0 memref (matching the AST codegen's
            // scalar locals); an aggregate (struct) slot is an `llvm.alloca` of the `!llvm.struct`
            // type, its pointer tracked in `agg_of` so field ops can address it.
            Opcode::Alloca => self.op_alloca(idx, ins),
            // Store a value into a slot (no result). A scalar slot is a rank-0 `memref`; an aggregate
            // slot (a struct value spilled from a struct-returning call) is an `llvm.store` (#215).
            Opcode::Store => self.op_store(idx, ins),
            // Load a value back from a slot; the result type is the slot's element (this
            // instruction's own `type_idx`).
            Opcode::SlotLoad => self.op_slot_load(idx, ins),
            // Block markers → MLIR blocks. Block 0 is the func's entry block (implicit; it carries the
            // params), so it gets no label; every other id opens `^bbN:`.
            Opcode::BlockStart => self.op_block_start(idx, ins),
            Opcode::Br => self.op_br(idx, ins),
            // `imm` packs the two targets as `then | (else << 32)` (see `flatten::pack_targets`).
            Opcode::CondBr => self.op_cond_br(idx, ins),
            Opcode::Ret => self.op_ret(idx, ins),
            // One argument of the following `Call`: record its value register (no op emitted).
            Opcode::Arg => self.op_arg(idx, ins),
            // A fixed-arity call. `type_idx` is the callee's GID (resolved to name + return type via
            // `ctx.callees`); `imm` is the arg count, taken from the tail of `pending_args`. Emit
            // `%r = func.call @name(%a, %b) : (Ta, Tb) -> Tret`.
            Opcode::Call => self.op_call(idx, ins),
            // Materialize a function pointer for a named function: `type_idx` is the target's GID
            // (name via `ctx.callees`, signature via `ctx.func_sigs`). Emit `func.constant @name : sig`
            // then cast the `FunctionType` value to an opaque `!llvm.ptr` (the ABI of a fn pointer),
            // tracked in `ptr_of`. (#242)
            Opcode::FuncConst => self.op_func_const(idx, ins),
            // An indirect call through a function pointer. `operand1` is the callee `!llvm.ptr`, `imm`
            // the arg count (the tail of `pending_args`, like `Call`), and this instruction's `type_idx`
            // the scalar return type. Reconstruct the function type `(arg types)->ret` from the actual
            // args, cast the pointer to it, and `func.call_indirect`. (#242)
            Opcode::CallIndirect => self.op_call_indirect(idx, ins),
            Opcode::AutoDiff => self.op_autodiff(idx, ins),
            Opcode::TensorLoad => self.op_tensor_load(idx, ins),
            Opcode::TensorDim => self.op_tensor_dim(idx, ins),
            // Store a scalar into a struct field (no result). `operand1` is the struct slot pointer,
            // `operand2` the value, `imm` the field's byte offset. GEP to the field, then `llvm.store`;
            // the field index comes from matching the offset against the layout, the value type from
            // the stored register's tracked type.
            Opcode::FieldStore => self.op_field_store(idx, ins),
            // Load a scalar struct field. `operand1` is the struct slot, `imm` the field's byte offset,
            // and this instruction's own `type_idx` the field's scalar type. GEP to the field, then
            // `llvm.load`.
            Opcode::FieldLoad => self.op_field_load(idx, ins),
            // The address of a by-value nested-aggregate field (`&outer.inner`): GEP to the field and
            // yield the pointer, tracked as an aggregate slot (its layout GID from `type_idx`), so a
            // chained field access or a method receiver addresses through it. (#242)
            Opcode::FieldAddr => self.op_field_addr(idx, ins),
            // Allocate a tensor buffer (`Tensor<T>([..])`): a static `memref` of the shape recovered
            // from the side table by GID. Its register is tracked in `mem_of` for later index/store.
            Opcode::TensorAlloc => self.op_tensor_alloc(idx, ins),
            // Zero the buffer a preceding `TensorAlloc` produced (`::new()` rather than
            // `::uninit()`): `linalg.fill`, the same fill a matmul emits before accumulating.
            Opcode::TensorZero => self.op_tensor_zero(idx, ins),
            // Fill the buffer a preceding `TensorAlloc` produced with a value (`::fill(v)`).
            Opcode::TensorFill => self.op_tensor_fill(idx, ins),
            // Index a tensor along its outermost dimension. `operand1` is the base tensor (memref),
            // `operand2` the index (`arith.index_cast` to `index`). A scalar-element result
            // (`type_idx` is a scalar GID) is a value read (`imm = 0` → `memref.load`) or an element
            // *place* (`imm = 1` → recorded for the following `TensorStore`). A sub-view result (a
            // tensor GID) rank-reduces the base to a row via `memref.reinterpret_cast` (contiguous
            // base only; a further sub-view of a strided row is deferred).
            Opcode::TensorIndex => self.op_tensor_index(idx, ins),
            // Reduce a rank-1 float slice to a scalar. `operand1` (and `operand2` for `dot`) are the
            // slices; `imm` the kind (0 = dot, 1 = sum, 2 = max, 3 = min). Each slice is `vector.load`ed
            // to a `vector<Nxf32>`; `dot` fuses the two with `arith.mulf`; then `vector.reduction`.
            // Float only, matching the AST oracle (`vector<Nxf32>` → `f32`).
            Opcode::Reduce => self.op_reduce(idx, ins),
            // `matmul_into(&mut dst, &a, &b)` (no result): `linalg.fill` + `linalg.matmul` on the
            // three whole-tensor memrefs, the same pair the AST path builds -- and the exact shape
            // `kernelKindOf` classifies, so a spawn whose whole job is this op still routes to
            // cuBLAS. The destination register rides the imm (see `Opcode::MatmulInto`).
            Opcode::Matmul => self.op_matmul(idx, ins),
            Opcode::MatmulInto => self.op_matmul_into(idx, ins),
            // `flash_attention_into(&mut o, &q, &k, &v, scale)` (no result): a serial
            // `o = softmax(q @ k^T * scale) @ v` nest, preceded by a `vx.attention_note` naming
            // which memref plays which role. The nest is the correctness contract — a runtime
            // that cannot (or will not) route runs it as written; the note is what
            // `kernelKindOf` classifies as `kind=attention` so a runtime that CAN route hands
            // the region to a fused vendor kernel instead. o, v and scale ride the imm (see
            // `Opcode::FlashAttnInto`). f16 storage with f32 arithmetic throughout, the same
            // split the widening contracts (Vx#320) give every half slice.
            Opcode::FlashAttnInto => self.op_flash_attn_into(idx, ins),
            // Store into a tensor place (no result). A scalar-element place (an `imm = 1`
            // `TensorIndex`) → `memref.store`; a row/sub-view place (an `imm = 0` `TensorIndex`, a row
            // memref in `mem_of`) takes an elementwise vector value → `vector.store`.
            Opcode::TensorStore => self.op_tensor_store(idx, ins),
            // Move a tensor to a memory space: `operand1` is the source, `imm` the target space's
            // dispatch id, `type_idx` the result tensor (same element + shape). Emits `vx.transfer`
            // (generic form) with `target_topology`; the vx→standard lowering turns it into an
            // alloc + `memref.copy` (it ignores the source's layout suffix, so the result is a plain
            // `memref<NxT>`). When the target space declares a sub-space descriptor, re-attach the
            // scheduling attrs (`space`/`within`/`granule`/`capacity`/`scope` + a bump-allocated
            // `offset`/`slots`) the AST path emits — a device backend needs them to place the tile
            // into VMEM/TMEM, and they are dropped otherwise (B1/P0-1).
            Opcode::Transfer => self.op_transfer(idx, ins),
            // Print a value (no result). A tensor is `memref.cast`'d to an unranked memref and passed
            // to the `printMemref*` runtime helper; a scalar goes to `print_*`. These are the same
            // helpers the AST path calls; `emit_module_mlir` prepends their `private` declarations.
            Opcode::Print => self.op_print(idx, ins),
            // Open a `vx.spawn` region (generic form). The op is inline in the enclosing block, which
            // continues after it; the instructions up to the matching `SpawnEnd` form the region body.
            // The ops immediately after `Spawn` (a control-flow body's setup, before its first explicit
            // block) go in the region's entry block, so open a label for it. `imm` is the topology
            // dispatch id (the same value the AST path emits as `vx.spawn`'s `topology` attribute).
            Opcode::Spawn => self.op_spawn(idx, ins),
            // Close the `vx.spawn` region: terminate its last block with `vx.yield` (unless a body
            // terminator already ended it), stamp the `topology` attribute, and resume emitting into
            // the enclosing block (which the spawn op did not terminate).
            Opcode::SpawnEnd => self.op_spawn_end(idx, ins),
            // Print a string literal (no result): take the address of the module-level global emitted
            // for this string (`@".str.<n>"`, `n = str_base + imm`) and call the `@print_str` runtime
            // helper. `emit_module_mlir` emits the global's bytes and the helper's `private` decl.
            // Conditional abort -- `assert`, and `abort()` with a constant-false condition.
            // `cf.assert` rather than a hand-written branch onto a call, because MLIR lowers it
            // for whichever target the code reaches: `puts` + `abort` + `unreachable` on the
            // host, `__assertfail` inside a kernel. It is also an ORDINARY op, not a
            // terminator, so it needs no block splitting here.
            // Block-level sync (Vx#379). A registered op with no results and no Pure trait, so
            // the greedy folder cannot erase it before the device clone rewrites it to
            // `gpu.barrier`; the host lowering erases it instead (a serial loop IS the barrier).
            Opcode::Barrier => self.op_barrier(idx, ins),
            Opcode::Abort => self.op_abort(idx, ins),
            Opcode::PrintStr => self.op_print_str(idx, ins),
            // A string literal in value position (#231): take the address of the module-level global
            // (`@".str.<n>"`, `n = str_base + imm`, the same numbering as `PrintStr`) as a first-class
            // `!llvm.ptr` value — what a `let s = "…"` binds or a string argument passes. The global's
            // bytes are emitted by `emit_module_mlir` from the string side table.
            Opcode::StringConst => self.op_string_const(idx, ins),
            // Index a raw pointer `p[i]` (`p : *mut T`): GEP the element, then either load it (a value
            // read) or hand back the element pointer as a store place. `operand1` is the base pointer,
            // `operand2` the (scalar) index, `type_idx` the pointee element type. The GEP's base
            // element type sets the stride, so `p[i]` addresses `base + i * sizeof(T)`. (#242)
            Opcode::PtrIndex => self.op_ptr_index(idx, ins),
            // Store into a raw-pointer place (no result): `operand1` is the `PtrIndex` place (the GEP'd
            // element pointer), `operand2` the value, and the pointee element type comes from the
            // place. (#242)
            Opcode::PtrStore => self.op_ptr_store(idx, ins),
            // Anything else (spawn, matmul, …) is outside this subset.
            _ => Err(Decline::Unsupported {
                what: "an opcode the emitter has no case for",
            }),
        }
    }
}

/// Emit a `func.func` for `func` from its flat HIR body, or a `Decline` naming the first construct
/// outside the current subset (scalar arithmetic + intra-function control flow + fixed-arity scalar
/// calls + all-scalar-field struct construction/field access; the AST path stays the oracle there).
/// `ctx` resolves the callee/struct GIDs the stream references. The returned text is a bare
/// `func.func` op; wrap it in a `module { … }` before parsing.
#[allow(clippy::too_many_arguments)]
pub fn emit_function_mlir(
    func: &Function,
    hir: &[HirInstruction],
    types: &[TypeId],
    ctx: &EmitCtx,
    calls: &mut Vec<(String, Vec<String>, String)>,
    str_base: usize,
    // This function's string side table, indexed by an instruction's `imm`. `PrintStr` and
    // `StringConst` reach their bytes through a module-level global; `Abort` needs the text
    // itself, because `cf.assert` carries its message as an inline attribute.
    strings: &[String],
    alias_stores: &[(usize, usize, Vec<usize>)],
    distinct_ctr: &mut u32,
) -> Lowered<String> {
    // Signature (taken from the resolved AST signature; the *body* is flat-driven). A scalar param is
    // its element type; a tensor param is a memref recovered by GID from the side table (`ctx.tensors`
    // holds it — the param's `Load` recorded it). Anything else declines.
    let mut params = Vec::new();
    for (i, (_, ty)) in func.params.iter().enumerate() {
        // Signature-position MLIR type: scalar, payload-free enum `i32`, `!llvm.ptr` (pointer /
        // fn-pointer), tensor memref, or by-value aggregate `!llvm.struct`; else the function declines.
        let pty = ty_mlir(ty, ctx)?;
        params.push(format!("%arg{i}: {pty}{}", param_alias_attrs(ty)));
    }
    let ret_elem = match &func.return_type {
        Type::Scalar(e) if !matches!(e, ElementType::Generic(_)) => Some(e.clone()),
        Type::Scalar(_) => {
            return Err(Decline::TypeNotModelled {
                what: "a generic scalar return type",
            })
        }
        _ => None, // non-scalar: a struct return is handled below; anything else is void
    };
    // The MLIR return type: a scalar, a payload-free enum's `i32` (#227), an `!llvm.struct` (a
    // by-value struct return, #215), or `None` for void. A struct return whose layout isn't modelled
    // declines the whole function.
    let ret_mlir: Option<String> = if let Some(e) = &ret_elem {
        Some(mlir_scalar(e).ok_or(crate::emitter_gap!())?.to_string())
    } else if let Some(et) = enum_scalar(&func.return_type, ctx) {
        Some(et.to_string())
    } else if is_ptr_ty(&func.return_type) {
        Some("!llvm.ptr".to_string()) // a pointer-returning function (#235)
    } else if let Some(gid) = ctx.agg_gid(&func.return_type) {
        Some(
            ctx.aggs
                .get(&gid)
                .ok_or(crate::emitter_gap!())?
                .struct_ty
                .clone(),
        )
    } else if let Some(mt) = tensor_memref_of_type(&func.return_type) {
        Some(mt) // a statically shaped tensor return, wrappers peeled (Vx#383)
    } else if crate::syntax::is_void_ty(&func.return_type) {
        None
    } else {
        // Anything else -- a dynamically shaped tensor return, an unmodelled nominal -- has no
        // spelling here. Treating it as void mis-signed the function: the body's Ret still carried the value,
        // and MLIR rejected the pair ("op has 1 operands, but enclosing function returns 0").
        return Err(Decline::TypeNotModelled {
            what: "a function return type with no MLIR spelling",
        });
    };

    // Alias-scope metadata for place-write field stores (M2b-2): map each tagged store's stream
    // position to its own alias-scope `distinct[]` id and its disjoint-sibling ids. Groups are numbered
    // from the module-global `distinct_ctr` (0 is the shared domain), so scopes stay distinct across
    // functions. The `FieldStore` arm attaches `alias_scopes`/`noalias_scopes` from this. (#275, §5.4)
    let alias_scope_of: HashMap<usize, (u32, Vec<u32>)> = {
        let n_groups = alias_stores
            .iter()
            .map(|(_, g, _)| *g as u32 + 1)
            .max()
            .unwrap_or(0);
        let base = *distinct_ctr;
        *distinct_ctr += n_groups;
        alias_stores
            .iter()
            .map(|(pos, own, sibs)| {
                (
                    *pos,
                    (
                        base + *own as u32,
                        sibs.iter().map(|s| base + *s as u32).collect(),
                    ),
                )
            })
            .collect()
    };

    let mut em = FnEmit {
        func,
        types,
        ctx,
        str_base,
        strings,
        ret_elem,
        alias_scope_of,
        calls,
        names: vec![String::new(); hir.len()],
        etypes: vec![None; hir.len()],
        agg_of: vec![None; hir.len()],
        agg_val_of: vec![None; hir.len()],
        mem_of: vec![None; hir.len()],
        place_of: vec![None; hir.len()],
        pptr_elem: vec![None; hir.len()],
        vec_of: vec![None; hir.len()],
        ptr_of: vec![false; hir.len()],
        pslot_of: vec![false; hir.len()],
        sslot_of: vec![None; hir.len()],
        entry: String::new(),
        body: String::new(),
        terminated: false,
        pending_args: Vec::new(),
        spawn_topology: None,
        subspace_offsets: HashMap::new(),
    };
    // `main` installs the runtime crash handler first, exactly as the AST codegen does (`is_main` ->
    // `func.call @vx_init_signals`), so a wild memory access is caught + backtraced rather than exiting
    // raw — otherwise a deliberately-crashing program (`tests/backend/fail/*_oob.vx`) diverges from the
    // oracle. Emitted into the entry block (block 0 has no label), before any local. (#242)
    if func.name.as_ref() == "main" {
        em.entry += "  func.call @vx_init_signals() : () -> ()\n";
    }

    for (idx, ins) in hir.iter().enumerate() {
        em.step(idx, ins)?;
    }

    // Every block must end in a terminator. A void function falls through to a bare `return`; a
    // scalar-returning function whose final block isn't em.terminated is either ill-typed or has an
    // unreachable trailing block (no value to return) — decline it, leaving the AST path the oracle.
    if !em.terminated {
        match &ret_mlir {
            None => em.body += "  func.return\n",
            Some(_) => return Err(crate::emitter_gap!()),
        }
    }

    let ret_sig = match &ret_mlir {
        Some(t) => format!(" -> {t}"),
        None => String::new(),
    };
    let mut out = format!(
        "func.func {}({}){} {{\n",
        sym_ref(&func.name),
        params.join(", "),
        ret_sig
    );
    out += &em.entry;
    out += &em.body;
    out += "}\n";
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::flatten::lower_function_to_hir;
    use crate::session::{GlobalSession, LocalWorkerState};
    use std::sync::Arc;

    #[test]
    fn memref_lead_dim_answers_only_for_rank_one() {
        // The reduction loads a slice with a single-index `vector.load`, so anything but rank 1
        // has to come back `None`. Taking the leading dimension without checking the rank
        // answered `memref<4x4xf32>` with 4, and the caller emitted an op MLIR rejects.
        assert_eq!(memref_lead_dim("memref<4xf32>"), Some(4));
        assert_eq!(
            memref_lead_dim("memref<4xf32, strided<[1], offset: ?>>"),
            Some(4)
        );
        assert_eq!(memref_lead_dim("memref<8xf16>"), Some(8));
        assert_eq!(memref_lead_dim("memref<4x4xf32>"), None);
        assert_eq!(memref_lead_dim("memref<2x3x4xf32>"), None);
        // A dynamic extent has no static width either.
        assert_eq!(memref_lead_dim("memref<?xf32>"), None);
    }

    fn parse_fn(src: &str) -> Function {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let prog = parser.parse().expect("parse failed");
        prog.functions.into_iter().next().expect("expected one fn")
    }

    /// Lower a function to flat HIR, emit MLIR, and check it parses + verifies in a real MLIR
    /// context — proving the flat stream produces valid MLIR end to end.
    fn emit_and_verify(src: &str) -> String {
        let f = parse_fn(src);
        let mut w = LocalWorkerState::new(Arc::new(GlobalSession::new(1)));
        assert!(lower_function_to_hir(&f, &mut w).is_ok(), "function lowers");
        let mlir = emit_function_mlir(
            &f,
            &w.local_hir_stream,
            &w.local_type_stream,
            &EmitCtx::default(),
            &mut Vec::new(),
            0,
            &w.local_string_table,
            &w.local_place_alias_stores,
            &mut 1,
        )
        .expect("emits flat MLIR");

        use melior::ir::operation::OperationLike;
        let registry = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&registry);
        let context = melior::Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        let module = melior::ir::Module::parse(&context, &mlir)
            .unwrap_or_else(|| panic!("emitted MLIR failed to parse:\n{mlir}"));
        assert!(
            module.as_operation().verify(),
            "emitted MLIR failed to verify:\n{mlir}"
        );
        mlir
    }

    /// Lower a whole program to flat HIR and emit the module, then check it parses + verifies in a
    /// real MLIR context — proving the module emitter (calls included) produces valid MLIR end to
    /// end. Mirrors the pipeline's registry build so callees resolve through `fn_sigs`.
    fn emit_module_and_verify(src: &str) -> String {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let mut prog = parser.parse().expect("parse failed");
        prog.module_path = "crate::t".into();
        let mut mods = vec![prog];
        let symbol_map = crate::resolver::build_symbol_map(&mods);
        mods[0].resolve_names(&symbol_map, &[]);
        let registry = crate::pipeline::build_frozen_registry(&mods).expect("registry builds");
        let session = Arc::new(GlobalSession::with_registry(1, registry));

        // Type-check so the type checker annotates each `StructInit` with its struct GID (a scratch
        // worker; the annotation lands on the AST, which the per-function lowering below then reads).
        let env_mods = mods.clone();
        let env = crate::hir::GlobalAstEnv::build(&env_mods);
        {
            let mut scratch = LocalWorkerState::new(session.clone());
            let mut checker = crate::hir::TypeChecker::new(&env, &mut scratch);
            for f in &mut mods[0].functions {
                checker.check_function(f);
            }
        }

        let mut lowered = Vec::new();
        for f in &mods[0].functions {
            let mut w = LocalWorkerState::new(session.clone());
            assert!(lower_function_to_hir(f, &mut w).is_ok(), "function lowers");
            lowered.push(w);
        }
        let funcs: Vec<(&Function, &[HirInstruction], &[TypeId])> = mods[0]
            .functions
            .iter()
            .zip(&lowered)
            .map(|(f, w)| {
                (
                    f,
                    w.local_hir_stream.as_slice(),
                    w.local_type_stream.as_slice(),
                )
            })
            .collect();
        let tensor_types: Vec<_> = lowered
            .iter()
            .flat_map(|w| w.local_tensor_types.iter().cloned())
            .collect();
        let string_tables: Vec<&[String]> = lowered
            .iter()
            .map(|w| w.local_string_table.as_slice())
            .collect();
        let agg_layouts: Vec<_> = lowered
            .iter()
            .flat_map(|w| w.local_agg_layouts.iter().cloned())
            .collect();
        let alias_tables: Vec<&[(usize, usize, Vec<usize>)]> = lowered
            .iter()
            .map(|w| w.local_place_alias_stores.as_slice())
            .collect();
        let mlir = emit_module_mlir(
            &funcs,
            &session.registry,
            &tensor_types,
            &string_tables,
            &agg_layouts,
            &alias_tables,
            &[],
            &[],
            crate::config::Schedule::Parallel,
        )
        .expect("emits flat module");

        use melior::ir::operation::OperationLike;
        let dialects = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&dialects);
        let context = melior::Context::new();
        context.append_dialect_registry(&dialects);
        context.load_all_available_dialects();
        crate::codegen::register_vx_dialect(&context); // for `vx.transfer`
        let module = melior::ir::Module::parse(&context, &mlir)
            .unwrap_or_else(|| panic!("emitted MLIR failed to parse:\n{mlir}"));
        assert!(
            module.as_operation().verify(),
            "emitted MLIR failed to verify:\n{mlir}"
        );
        mlir
    }

    #[test]
    fn emits_verifiable_integer_add() {
        let mlir = emit_and_verify("fn add(a: i32, b: i32) -> i32 { return a + b; }");
        assert!(mlir.contains("arith.addi %arg0, %arg1 : i32"), "{mlir}");
        assert!(mlir.contains("func.return"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_scalar_call() {
        // The module emitter emits both `func.func`s; the call site resolves `add` through the
        // registry's `fn_sigs` and prints a matching call signature.
        let mlir = emit_module_and_verify(
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n\
             fn main() -> i32 { return add(3, 4); }",
        );
        assert!(mlir.contains("func.call @add("), "{mlir}");
        assert!(mlir.contains("(i32, i32) -> i32"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_struct_construct_and_field_read() {
        // A struct built in place (`llvm.alloca` + field stores) then read back (GEP + `llvm.load`).
        let mlir = emit_module_and_verify(
            "struct Point { x: i32, y: i32 }\n\
             fn main() -> i32 { let p = Point { x: 3, y: 4 }; return p.x + p.y; }",
        );
        assert!(mlir.contains("llvm.alloca"), "{mlir}");
        assert!(mlir.contains("llvm.getelementptr"), "{mlir}");
        assert!(mlir.contains("llvm.store"), "{mlir}");
        assert!(mlir.contains("llvm.load"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_pointer_field_struct() {
        // A struct with a raw-pointer field (`Buf { data: *mut i32, len: i32 }`, the shape of `Vec`):
        // constructed with a pointer initializer (`llvm.store … : !llvm.ptr`), its pointer field read
        // back (`llvm.load … -> !llvm.ptr`) and its scalar field read — all through the `!llvm.struct`
        // whose first element is `!llvm.ptr`. (#242)
        let mlir = emit_module_and_verify(
            "struct Buf { data: *mut i32, len: i32 }\n\
             fn make(p: *mut i32) -> Buf { return Buf { data: p, len: 0 }; }\n\
             fn getlen(b: &Buf) -> i32 { return b.len; }\n\
             fn getdata(b: &Buf) -> *mut i32 { return b.data; }",
        );
        assert!(mlir.contains("!llvm.struct<(!llvm.ptr, i32)>"), "{mlir}");
        assert!(mlir.contains("llvm.store %arg0"), "{mlir}"); // the pointer field initializer
        assert!(
            mlir.contains("llvm.load") && mlir.contains("-> !llvm.ptr"),
            "{mlir}"
        );
    }

    #[test]
    fn emits_verifiable_raw_pointer_index() {
        // Raw-pointer indexing (`Vec`'s `self.data[i]`, #242): a read `p[i]` GEPs the element and
        // `llvm.load`s it; a write `p[i] = v` GEPs and `llvm.store`s. The GEP's base element type
        // (`i32`) sets the stride.
        let read = emit_module_and_verify(
            "fn idx(p: *mut i32, i: i32) -> i32 { return unsafe { p[i] }; }",
        );
        assert!(read.contains("llvm.getelementptr"), "{read}");
        assert!(read.contains("-> !llvm.ptr, i32"), "{read}");
        assert!(read.contains("llvm.load"), "{read}");
        let write = emit_module_and_verify(
            "fn set(p: *mut i32, i: i32, v: i32) -> i32 { unsafe { p[i] = v; } return 0; }",
        );
        assert!(write.contains("llvm.getelementptr"), "{write}");
        assert!(write.contains("llvm.store %arg2"), "{write}");
    }

    #[test]
    fn emits_verifiable_field_access_through_self_pointer() {
        // Field access through a `&mut self` pointer (`Vec::push`'s `self.len`): the pointer param is
        // GEP'd + loaded/stored directly (no aggregate slot), and a raw-pointer element store writes
        // `self.data[0]`. Exercises the emitter's `agg_of` tracking for a pointer-to-aggregate param.
        let mlir = emit_module_and_verify(
            "struct Buf { data: *mut i32, len: i32 }\n\
             fn bump(b: &mut Buf, v: i32) -> i32 { unsafe { b.data[0] = v; } b.len = b.len + 1; return 0; }",
        );
        assert!(mlir.contains("llvm.getelementptr %arg0"), "{mlir}"); // field GEP off the self pointer
        assert!(mlir.contains("llvm.store"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_aggregate_element_pointer_index() {
        // A raw pointer to an *aggregate* (`Vec<Vec<T>>`'s `*mut Vec`, #242): a by-value struct param
        // (`v: Inner` -> an `!llvm.struct` arg spilled to a slot), a whole-struct store through the
        // pointer (`dst[i] = v` -> GEP with the struct as the stride type + `llvm.store` of the struct
        // value), and a whole-struct load back (`src[i]` -> `llvm.load … -> !llvm.struct`, returned by
        // value).
        let store = emit_module_and_verify(
            "struct Inner { a: i32, b: i32 }\n\
             fn store_it(dst: *mut Inner, i: i32, v: Inner) -> i32 { unsafe { dst[i] = v; } return 0; }",
        );
        assert!(
            store.contains("(%arg0: !llvm.ptr, %arg1: i32, %arg2: !llvm.struct<(i32, i32)>)"),
            "{store}"
        );
        assert!(
            store.contains("-> !llvm.ptr, !llvm.struct<(i32, i32)>"),
            "{store}"
        ); // GEP stride = struct
        assert!(
            store.contains("llvm.store") && store.contains(": !llvm.struct<(i32, i32)>, !llvm.ptr"),
            "{store}"
        );
        let load = emit_module_and_verify(
            "struct Inner { a: i32, b: i32 }\n\
             fn load_it(src: *mut Inner, i: i32) -> Inner { return unsafe { src[i] }; }",
        );
        assert!(load.contains("-> !llvm.struct<(i32, i32)>"), "{load}"); // load of the struct
        assert!(
            load.contains("func.return") && load.contains(": !llvm.struct<(i32, i32)>"),
            "{load}"
        );
    }

    #[test]
    fn emits_verifiable_tensor_alloc_store_read() {
        // A tensor allocated as a static memref, filled by scalar-element stores, then read back:
        // `memref.alloc` + `arith.index_cast` + `memref.store`/`memref.load`.
        let mlir = emit_module_and_verify(
            "fn main() -> i32 { let mut q = Tensor<i32>([4]); q[0] = 5; q[1] = 6; q[2] = 7; \
             q[3] = 8; return q[2]; }",
        );
        assert!(mlir.contains("memref.alloc() : memref<4xi32>"), "{mlir}");
        assert!(mlir.contains("arith.index_cast"), "{mlir}");
        assert!(mlir.contains("memref.store"), "{mlir}");
        assert!(mlir.contains("memref.load"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_tensor_row_subview_read() {
        // A rank-2 tensor: `q[i]` rank-reduces to a strided row via `memref.reinterpret_cast`, and the
        // final scalar index loads through that row.
        let mlir = emit_module_and_verify(
            "fn main() -> i32 { let mut q = Tensor<i32>([2, 3]); q[0][0] = 1; q[0][1] = 2; \
             q[0][2] = 3; q[1][0] = 4; q[1][1] = 5; q[1][2] = 6; return q[1][2]; }",
        );
        assert!(mlir.contains("memref.reinterpret_cast"), "{mlir}");
        assert!(mlir.contains("strided<[1], offset: ?>"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_tensor_param_and_call() {
        // A helper taking a tensor param (a `memref` in the signature) called with a tensor argument.
        let mlir = emit_module_and_verify(
            "fn get(q: Tensor<i32, [4]>, i: i32) -> i32 { return q[i]; }\n\
             fn main() -> i32 { let mut q = Tensor<i32>([4]); q[0] = 5; q[1] = 6; q[2] = 7; \
             q[3] = 8; return get(q, 2); }",
        );
        assert!(mlir.contains("@get(%arg0: memref<4xi32>"), "{mlir}");
        assert!(mlir.contains("func.call @get("), "{mlir}");
        assert!(mlir.contains("(memref<4xi32>, i32) -> i32"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_tensor_sum_reduction() {
        // A float `sum` reduction lowers to `vector.load` + `vector.reduction<add>`; the scalar result
        // feeds a compare so the function still returns an i32.
        let mlir = emit_module_and_verify(
            "fn main() -> i32 { let mut q = Tensor<f32>([4]); q[0] = 1.0; q[1] = 2.0; q[2] = 3.0; \
             q[3] = 4.0; let mut r = 0; if sum(q) > 9.0 { r = 1; } return r; }",
        );
        assert!(mlir.contains("vector.load"), "{mlir}");
        assert!(mlir.contains("vector.reduction <add>"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_tensor_elementwise_and_row_store() {
        // `o[0] = q[0] * 2.0`: an elementwise scalar-broadcast multiply over a row, stored back into a
        // row via `vector.store`.
        let mlir = emit_module_and_verify(
            "fn main() -> i32 { let mut q = Tensor<f32>([2, 4]); q[0][0] = 1.0; q[0][1] = 2.0; \
             q[0][2] = 3.0; q[0][3] = 4.0; let mut o = Tensor<f32>([2, 4]); o[0] = q[0] * 2.0; \
             let mut r = 0; if o[0][1] > 3.5 { r = 1; } return r; }",
        );
        assert!(mlir.contains("vector.broadcast"), "{mlir}");
        assert!(mlir.contains("arith.mulf"), "{mlir}");
        assert!(mlir.contains("vector.store"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_tensor_transfer() {
        // `transfer(q, Memory::NPU_HBM)` -> a `vx.transfer` carrying the target topology dispatch id.
        let mlir = emit_module_and_verify(
            "fn main() -> i32 { let mut q = Tensor<f32>([2, 4]); q[0][0] = 1.0; q[0][1] = 2.0; \
             q[0][2] = 3.0; q[0][3] = 4.0; let o = transfer(q, Memory::NPU_HBM); let mut r = 0; \
             if o[0][2] > 2.5 { r = 1; } return r; }",
        );
        assert!(mlir.contains("\"vx.transfer\""), "{mlir}");
        assert!(mlir.contains("target_topology ="), "{mlir}");
    }

    #[test]
    fn emits_verifiable_spawn_region() {
        // `spawn on(<topology>) { <control-flow body> }` -> a `vx.spawn` op whose nested region holds
        // the body (here a `for` loop writing a tensor) and is terminated by `vx.yield` (#226). The
        // region's blocks are self-contained; the enclosing function continues after the op. This is
        // the structural bug class (an ill-formed region) that a parse + verify catches.
        let mlir = emit_module_and_verify(
            "fn main() -> i32 { let mut c = Tensor<f32>([4]); \
             spawn on(Topology::CPU) { for i in 0..4 { c[i] = 1.0; } } return 0; }",
        );
        assert!(mlir.contains("\"vx.spawn\""), "{mlir}");
        assert!(mlir.contains("topology ="), "{mlir}");
        assert!(mlir.contains("\"vx.yield\""), "{mlir}");
    }

    #[test]
    fn emits_verifiable_tensor_print() {
        // `print(q)` -> memref.cast to an unranked memref + a call to the printMemrefF32 runtime
        // helper, whose `private` declaration the module emitter prepends.
        let mlir = emit_module_and_verify(
            "fn main() -> i32 { let mut q = Tensor<f32>([2, 2]); q[0][0] = 1.0; q[0][1] = 2.0; \
             q[1][0] = 3.0; q[1][1] = 4.0; print(q); return 0; }",
        );
        assert!(
            mlir.contains("func.func private @printMemrefF32(memref<*xf32>)"),
            "{mlir}"
        );
        assert!(mlir.contains("memref.cast"), "{mlir}");
        assert!(mlir.contains("func.call @printMemrefF32("), "{mlir}");
    }

    #[test]
    fn emits_verifiable_float_arithmetic() {
        let mlir = emit_and_verify("fn f(a: f64, b: f64) -> f64 { return a * b + b; }");
        assert!(mlir.contains("arith.mulf"), "{mlir}");
        assert!(mlir.contains("arith.addf"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_unary_neg_and_not() {
        // Scalar unary ops now emit + verify (#214): float `-x` -> `arith.negf`, integer `-x` -> a
        // zeroed `arith.subi`, and `!b` -> `arith.xori` with all-ones.
        let fneg = emit_and_verify("fn f(a: f32) -> f32 { return -a; }");
        assert!(fneg.contains("arith.negf"), "{fneg}");
        let ineg = emit_and_verify("fn f(a: i32) -> i32 { return -a; }");
        assert!(ineg.contains("arith.subi"), "{ineg}");
        let lnot = emit_and_verify("fn f(a: bool) -> bool { return !a; }");
        assert!(lnot.contains("arith.xori"), "{lnot}");
    }

    #[test]
    fn emits_verifiable_constant_and_signed_div() {
        let mlir = emit_and_verify("fn g(a: i32) -> i32 { return a / 2; }");
        assert!(mlir.contains("arith.constant 2 : i32"), "{mlir}");
        assert!(mlir.contains("arith.divsi"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_if_else() {
        // Control flow: the memory model (alloca/store/load'd locals) + a `cf` diamond. The compare
        // drives the conditional branch; both branches reconverge at the merge block.
        let mlir = emit_and_verify(
            "fn c(a: i32) -> i32 { let mut x = a; if a < 0 { x = 0; } else { x = 1; } return x; }",
        );
        assert!(mlir.contains("memref.alloca() : memref<i32>"), "{mlir}");
        assert!(mlir.contains("arith.cmpi slt, "), "{mlir}");
        assert!(mlir.contains("cf.cond_br "), "{mlir}");
        assert!(mlir.contains("cf.br ^bb"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_for_loop() {
        // A `for` range loop: header (compare + cond_br), body, increment latch, exit — all wired
        // through slots for the induction variable and accumulator.
        let mlir = emit_and_verify(
            "fn sum(n: i32) -> i32 { let mut s = 0; for i in 0..n { s = s + i; } return s; }",
        );
        assert!(mlir.contains("arith.cmpi slt, "), "{mlir}");
        assert!(mlir.contains("memref.load "), "{mlir}");
        assert!(mlir.contains("memref.store "), "{mlir}");
        assert!(mlir.contains("cf.cond_br "), "{mlir}");
    }

    #[test]
    fn emits_verifiable_scalar_casts() {
        // Scalar `as` casts now emit + verify (#214): the right `arith` conversion per source/target
        // kind + width — widen/narrow ints, int<->float, float widen/narrow.
        let widen = emit_and_verify("fn c(a: i32) -> i64 { return a as i64; }");
        assert!(widen.contains("arith.extsi"), "{widen}");
        let narrow = emit_and_verify("fn c(a: i64) -> i32 { return a as i32; }");
        assert!(narrow.contains("arith.trunci"), "{narrow}");
        let i2f = emit_and_verify("fn c(a: i32) -> f32 { return a as f32; }");
        assert!(i2f.contains("arith.sitofp"), "{i2f}");
        let f2i = emit_and_verify("fn c(a: f32) -> i32 { return a as i32; }");
        assert!(f2i.contains("arith.fptosi"), "{f2i}");
        let f2f = emit_and_verify("fn c(a: f32) -> f64 { return a as f64; }");
        assert!(f2f.contains("arith.extf"), "{f2f}");
    }

    /// Every float the emitter writes has to survive a round trip through MLIR's parser.
    /// Rust's `{:?}` prints the shortest form that round-trips in Rust, which for small
    /// magnitudes drops the decimal point entirely: `0.00001` becomes `1e-5`. MLIR reads the
    /// `1`, then tries to parse `e-5` as the next operation and reports `custom op 'e' is
    /// unknown`, so the module the emitter just claimed will not parse (Vx#384).
    #[test]
    fn float_literals_keep_a_decimal_point_before_the_exponent() {
        // The exact case that broke `struct_codegen.vx`.
        assert_eq!(super::mlir_float_literal(1e-5), "1.0e-5");

        // The property, over a spread of magnitudes: whatever Rust chooses to print, the
        // mantissa always carries a point.
        for value in [1e-5, 1e-7, 2.5e-9, 288.0, 0.5, -0.25, 1.0, 3.4e38] {
            let text = super::mlir_float_literal(value);
            let mantissa = text.split(['e', 'E']).next().unwrap();
            assert!(
                mantissa.contains('.'),
                "{value} formatted as {text}, whose mantissa has no decimal point"
            );
        }
    }
}
