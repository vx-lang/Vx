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
use crate::gid::TypeId;
use crate::hir::bytecode::{HirInstruction, Opcode};
use crate::hir::flatten::{ptr_gid, scalar_gid, tensor_gid_of};
use crate::registry::ImmutableGlobalRegistry;
use crate::syntax::{ElementType, Function, Type};
use rayon::prelude::*;
use std::collections::HashMap;

/// The MLIR type string for a scalar element type. Integers are signless (signedness lives in the
/// op, e.g. `divsi`/`divui`); `bool` is `i1`.
fn mlir_scalar(elem: &ElementType) -> Option<&'static str> {
    use ElementType::*;
    Some(match elem {
        F16 => "f16",
        F32 => "f32",
        F64 => "f64",
        BF16 => "bf16",
        I4 | U4 => "i4",
        I8 | U8 => "i8",
        I16 | U16 => "i16",
        I32 | U32 => "i32",
        I64 | U64 => "i64",
        I128 | U128 => "i128",
        Bool => "i1",
        // fp8 is capacity/declaration-only for now: the JIT has no fp8 arithmetic,
        // so the flat path declines. Compute support is #249.
        F8E4M3 | F8E5M2 => return None,
        Generic(_) => return None,
    })
}

fn is_float(e: &ElementType) -> bool {
    e.is_float() // the single float-class predicate (`ElementType::is_float`); P1-4a
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
    match (is_float(src), is_float(tgt)) {
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

fn scalar_of(ty: &Type) -> Option<ElementType> {
    match ty {
        Type::Scalar(ElementType::Generic(_)) => None,
        Type::Scalar(e) => Some(e.clone()),
        _ => None,
    }
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
fn is_ptr_ty(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Pointer(..) | Type::Borrow { .. } | Type::Function(..) | Type::Closure(..)
    )
}

/// Whether a return type is `void` — spelled `Type::Struct("void", _)` (the AST codegen matches the
/// same). A void-returning call produces no result value; the flat emitter prints `-> ()`. (#230)
pub fn is_void_ty(ty: &Type) -> bool {
    matches!(ty, Type::Struct(n, _) if n.as_ref() == "void" || n.as_ref() == "none")
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
    let f = is_float(e);
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
    if is_float(e) {
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
    sched: crate::pipeline::Schedule,
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
            },
        )
    };
    if sched == crate::pipeline::Schedule::Sequential {
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
pub fn build_agg_map(
    registry: &ImmutableGlobalRegistry,
    sched: crate::pipeline::Schedule,
) -> AggMap {
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
                        Some(nested_ty) => {
                            field_tys.push(nested_ty);
                            field_pointee.push(None);
                            field_agg.push(Some(*nested_gid));
                        }
                        None => {
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
    if sched == crate::pipeline::Schedule::Sequential {
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
) -> Option<String> {
    use crate::layout::FieldTy;
    if visiting.contains(&gid) {
        return None;
    }
    let def = registry.layouts.get(&gid)?;
    if def.align_bytes == 0 || def.fields.is_empty() {
        return None;
    }
    visiting.push(gid);
    let mut field_tys = Vec::with_capacity(def.fields.len());
    for f in &def.fields {
        let ft = match &f.ty {
            FieldTy::Scalar(e) => mlir_scalar(e)?.to_string(),
            FieldTy::Opaque => "!llvm.ptr".to_string(),
            FieldTy::Nominal(n) => agg_struct_ty_of(*n, registry, visiting)?,
        };
        field_tys.push(ft);
    }
    visiting.pop();
    Some(format!("!llvm.struct<({})>", field_tys.join(", ")))
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
    /// Declared topology arch by dispatch id (`Topology Dev { arch: nvptx64, ... }` ->
    /// `1916 -> "nvptx64"`), so a `vx.spawn` can carry the arch its machine file declared and the
    /// device pipeline can gate on the DECLARATION rather than on the dispatch-id band -- which a
    /// custom topology can never enter (custom ids are 1000 + fnv % 1000 by construction, and the
    /// GPU band is [500, 600)). Built-in topologies are not in this map and keep riding the band
    /// (Vx#352, Vx#353 Track B).
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
const RUNTIME_HELPERS: [(&str, &str); 10] = [
    ("printMemrefF32", "(memref<*xf32>)"),
    ("printMemrefF64", "(memref<*xf64>)"),
    ("printMemrefI32", "(memref<*xi32>)"),
    ("printMemrefI64", "(memref<*xi64>)"),
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
    env.memories
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
        .collect()
}

impl EmitCtx {
    /// Build the callee + struct-layout maps from the registry. The tensor map is *not* in the
    /// registry (tensor types are structural); populate it separately from the lowerer's side table.
    pub fn from_registry(
        registry: &ImmutableGlobalRegistry,
        sched: crate::pipeline::Schedule,
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
fn ty_mlir(ty: &Type, ctx: &EmitCtx) -> Option<String> {
    if let Some(e) = scalar_of(ty) {
        Some(mlir_scalar(&e)?.to_string())
    } else if let Some(et) = enum_scalar(ty, ctx) {
        Some(et.to_string())
    } else if is_ptr_ty(ty) {
        Some("!llvm.ptr".to_string())
    } else if let Some(gid) = tensor_gid_of(ty) {
        let (elem, shape) = ctx.tensors.get(&gid)?;
        tensor_memref_ty(elem, shape)
    } else if let Some(gid) = ctx.agg_gid(ty) {
        Some(ctx.aggs.get(&gid)?.struct_ty.clone())
    } else {
        None
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
    sched: crate::pipeline::Schedule,
) -> Option<String> {
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
        let params: Option<Vec<String>> =
            func.params.iter().map(|(_, t)| ty_mlir(t, &ctx)).collect();
        let ret = match &func.return_type {
            Type::Scalar(ElementType::Generic(_)) => None,
            t => ty_mlir(t, &ctx).or(Some("()".to_string())),
        };
        match (params, ret) {
            (Some(params), Some(ret)) => Some((sig.gid, (params, ret))),
            _ => None,
        }
    };
    let sigs: Vec<(TypeId, (Vec<String>, String))> =
        if sched == crate::pipeline::Schedule::Sequential {
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
            Some((text, calls, helpers))
        };
    let mut emitted: Vec<Option<FnEmission>> = if sched == crate::pipeline::Schedule::Sequential {
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
        .filter_map(|e| e.as_ref().map(|(t, _, _)| t.len()))
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
        let Some((_, fn_calls, helpers)) = emission.as_mut() else {
            if std::env::var("VX_FLAT_DBG").is_ok() {
                eprintln!(
                    "[flat-dbg] emit declined for fn {}",
                    funcs[fi].0.name.as_ref()
                );
            }
            return None;
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
    if sched == crate::pipeline::Schedule::Sequential {
        drop(emitted);
        drop(calls);
    } else {
        emitted.into_par_iter().for_each(drop);
        calls.into_par_iter().for_each(drop);
    }
    Some(module)
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

/// The leading static dimension of a memref type string, e.g. `memref<4xf32, strided<…>> -> 4`. Used
/// as the `vector<Nx…>` width when loading a rank-1 slice for a reduction.
fn memref_lead_dim(memty: &str) -> Option<i64> {
    memty
        .strip_prefix("memref<")?
        .split('x')
        .next()?
        .parse::<i64>()
        .ok()
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
) -> Option<String> {
    let name = names.get(op_reg as usize)?.clone();
    if vec_of.get(op_reg as usize)?.is_some() {
        return Some(name); // already a vector (a prior elementwise result)
    }
    if let Some(m) = mem_of.get(op_reg as usize)?.clone() {
        let c0 = format!("%vc{tag}");
        let v = format!("%vl{tag}");
        let al = vector_align_attr(vecty);
        body.push_str(&format!("  {c0} = arith.constant 0 : index\n"));
        body.push_str(&format!(
            "  {v} = vector.load {name}[{c0}]{al} : {m}, {vecty}\n"
        ));
        return Some(v);
    }
    if etypes.get(op_reg as usize)?.is_some() {
        let v = format!("%vb{tag}");
        body.push_str(&format!(
            "  {v} = vector.broadcast {name} : {et} to {vecty}\n"
        ));
        return Some(v);
    }
    None
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

/// Emit a `func.func` for `func` from its flat HIR body, or `None` if the stream uses any construct
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
) -> Option<String> {
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
        Type::Scalar(_) => return None,
        _ => None, // non-scalar: a struct return is handled below; anything else is void
    };
    // The MLIR return type: a scalar, a payload-free enum's `i32` (#227), an `!llvm.struct` (a
    // by-value struct return, #215), or `None` for void. A struct return whose layout isn't modelled
    // declines the whole function.
    let ret_mlir: Option<String> = if let Some(e) = &ret_elem {
        Some(mlir_scalar(e)?.to_string())
    } else if let Some(et) = enum_scalar(&func.return_type, ctx) {
        Some(et.to_string())
    } else if is_ptr_ty(&func.return_type) {
        Some("!llvm.ptr".to_string()) // a pointer-returning function (#235)
    } else if let Some(gid) = ctx.agg_gid(&func.return_type) {
        Some(ctx.aggs.get(&gid)?.struct_ty.clone())
    } else {
        None
    };

    let ty_at = |ti: u32| -> Option<ElementType> { elem_of_gid(*types.get(ti as usize)?) };

    let mut names: Vec<String> = vec![String::new(); hir.len()];
    // The scalar element type each register carries, indexed by register (= instruction position in
    // the stream) — the type-valued parallel to `names` above. As the stream is walked, each value-
    // producing instruction records its result type here, so any later instruction can recover the
    // type of a register it *reads*. This is the flat-driven stand-in for reading an operand's type
    // off the AST: there is no AST node to consult, so we reconstruct types as we go.
    //
    // Why not just read each instruction's own `type_idx`? Most values are self-describing that way,
    // but two consumers need an *operand's* type, which their own `type_idx` doesn't give:
    //   - `Cmp`: its own result type is `bool`, but choosing `cmpi`/`cmpf` + the signed/unsigned
    //     predicate needs the *operands'* type -> `etypes[operand1]`.
    //   - `Store`: an effect instruction (its `type_idx` is the no-type sentinel), but printing
    //     `memref<T>` needs the slot's element type -> the `Alloca` records it, `Store` reads it back.
    // Calls use it too: a `func.call`'s argument types come from `etypes[arg_reg]`.
    //
    // Scalar-only today (`ElementType`); widening it to tensor/aggregate types is future work (#200).
    let mut etypes: Vec<Option<ElementType>> = vec![None; hir.len()];
    let elem_at = |etypes: &[Option<ElementType>], r: u32| -> Option<ElementType> {
        etypes.get(r as usize)?.clone()
    };
    // The aggregate GID for each register that is a struct-slot pointer (from an aggregate `Alloca`),
    // so a `FieldLoad`/`FieldStore` on that slot recovers the struct's `!llvm.struct` type + field
    // offsets. The aggregate analogue of `etypes` (kept separate: struct slots are pointers, not
    // scalar values).
    let mut agg_of: Vec<Option<TypeId>> = vec![None; hir.len()];
    // The aggregate GID for each register that holds a struct *value* (not a slot pointer): an
    // aggregate `SlotLoad`, an aggregate-element `PtrIndex` read, or a struct-returning `Call`. Lets a
    // consumer (a by-value call argument, a `PtrStore` of the value) print the `!llvm.struct` type.
    // The value analogue of `agg_of` (which tracks struct *slots*). (#242 Vec<Vec<T>>)
    let mut agg_val_of: Vec<Option<TypeId>> = vec![None; hir.len()];
    // The memref type string for each register that holds a tensor (from `TensorAlloc`), so a
    // `TensorIndex`/`TensorStore` on it prints the right `memref<...>`.
    let mut mem_of: Vec<Option<String>> = vec![None; hir.len()];
    // For a scalar-element *place* register (a `TensorIndex` with `imm = 1`): the base memref name, an
    // `index`-typed index SSA name, and the base memref type — everything the following `TensorStore`
    // needs to emit `memref.store %v, %base[%idx]`.
    let mut place_of: Vec<Option<(String, String, String)>> = vec![None; hir.len()];
    // For a raw-pointer element *place* register (a `PtrIndex` with `imm = 1`): the pointee element's
    // MLIR type, so the following `PtrStore` prints `llvm.store %v, %place : {elem}, !llvm.ptr`. The
    // place register itself already holds the GEP'd element pointer (in `names`). (#242)
    let mut pptr_elem: Vec<Option<String>> = vec![None; hir.len()];
    // The `vector<Nxf32>` type of each register holding an elementwise (slice) result, so a row
    // `TensorStore` `vector.store`s it and a further elementwise passes it through.
    let mut vec_of: Vec<Option<String>> = vec![None; hir.len()];
    // Whether each register holds an opaque `!llvm.ptr` *value* (a string const, a pointer param, a
    // pointer-returning call, or a load from a pointer slot) — so a `Call` arg / `func.return` types it
    // as `!llvm.ptr`. The pointer analogue of `etypes`. (#231/#235)
    let mut ptr_of: Vec<bool> = vec![false; hir.len()];
    // Whether each register is a pointer *slot* (an `llvm.alloca` of `!llvm.ptr`, a memory-mode pointer
    // local), so a `Store`/`SlotLoad` on it uses `llvm.store`/`llvm.load` rather than `memref`. (#235)
    let mut pslot_of: Vec<bool> = vec![false; hir.len()];
    // The element type of each register that is an *address-taken scalar slot* (an `llvm.alloca` of a
    // scalar, `imm = 1` on the `Alloca`) — so `&x` yields a real `!llvm.ptr` (the slot register is also
    // marked in `ptr_of`) and a `Store`/`SlotLoad` uses `llvm.store`/`llvm.load` of the element type
    // rather than the rank-0 `memref` a never-borrowed scalar local uses. The scalar analogue of
    // `pslot_of`, carrying the element so the load/store types match. (#230)
    let mut sslot_of: Vec<Option<ElementType>> = vec![None; hir.len()];
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
    let mut body = String::new();
    // `main` installs the runtime crash handler first, exactly as the AST codegen does (`is_main` ->
    // `func.call @vx_init_signals`), so a wild memory access is caught + backtraced rather than exiting
    // raw — otherwise a deliberately-crashing program (`tests/backend/fail/*_oob.vx`) diverges from the
    // oracle. Emitted into the entry block (block 0 has no label), before any local. (#242)
    if func.name.as_ref() == "main" {
        body += "  func.call @vx_init_signals() : () -> ()\n";
    }
    // Whether the block currently being emitted has a terminator yet (a block must end in one).
    let mut terminated = false;
    // Argument value registers accumulated by the `Arg`s that immediately precede a `Call`; the
    // `Call` consumes its `imm` trailing entries (a nested inner call sits between its own `Arg`s and
    // the outer ones, so each call's args are exactly the tail — see `flatten::lower_call`).
    let mut pending_args: Vec<u32> = Vec::new();
    // The topology of the currently-open `vx.spawn` region (`Some` between `Spawn` and its matching
    // `SpawnEnd`), remembered so `SpawnEnd` can emit the `topology` attribute. `None` outside a spawn;
    // a nested spawn (already `Some`) is declined.
    let mut spawn_topology: Option<i64> = None;
    // Per-function sub-space bump allocator (space dispatch id -> next free byte), mirroring the AST
    // path's `MeliorGenerator::subspace_offsets`: each `Transfer` into a granule'd space claims the
    // next granule-rounded `offset` and advances the cursor, so both paths assign identical offsets
    // (P0-1). Reset per function, as in the AST codegen.
    let mut subspace_offsets: HashMap<u64, u64> = HashMap::new();

    for (idx, ins) in hir.iter().enumerate() {
        match ins.opcode {
            // Parameter materialization: the register *is* the block argument, no op emitted. A
            // scalar param records its element type; a tensor param records its memref type (from the
            // side table) so later index/store ops address it.
            Opcode::Load => {
                names[idx] = format!("%arg{}", ins.imm);
                let gid = *types.get(ins.type_idx.0 as usize)?;
                if let Some(e) = elem_of_gid(gid) {
                    etypes[idx] = Some(e);
                } else if gid == ptr_gid() {
                    ptr_of[idx] = true; // a `!llvm.ptr` parameter (#235)
                                        // A pointer *to a modelled aggregate* (`self : &mut Vec<i32>`): also track its
                                        // pointee layout GID, so a `FieldLoad`/`FieldStore` through `self` GEPs the field
                                        // exactly as through an aggregate slot (#242).
                    if let Some(agg_gid) = func
                        .params
                        .get(ins.imm as usize)
                        .and_then(|(_, ty)| pointee_agg_gid(ty, ctx))
                    {
                        agg_of[idx] = Some(agg_gid);
                    }
                } else if let Some((elem, shape)) = ctx.tensors.get(&gid) {
                    mem_of[idx] = tensor_memref_ty(elem, shape);
                } else if ctx.aggs.contains_key(&gid) {
                    // A by-value aggregate parameter (`self : Option<i32>` in `Option::unwrap`, a
                    // by-value struct arg): an `!llvm.struct` value, tracked so it can be spilled to a
                    // slot / passed on by value. (#242)
                    agg_val_of[idx] = Some(gid);
                }
            }
            Opcode::Const => {
                let e = ty_at(ins.type_idx.0)?;
                let mt = mlir_scalar(&e)?;
                let lit = if is_float(&e) {
                    format!("{:?}", f64::from_bits(ins.imm))
                } else {
                    (ins.imm as i64).to_string()
                };
                let n = format!("%v{idx}");
                body += &format!("  {n} = arith.constant {lit} : {mt}\n");
                names[idx] = n;
                etypes[idx] = Some(e);
            }
            // Arithmetic: scalar (a scalar-GID result) or *elementwise* over a rank-1 float slice (a
            // tensor-GID result). The scalar form is `arith.{addi,mulf,…}`; the elementwise form
            // coerces each operand to a `vector<Nxf32>` (`vector.load`/`broadcast`), applies
            // `arith.{addf,subf,mulf,divf}`, and yields a vector that a row `TensorStore` writes back.
            Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div => {
                let result_gid = *types.get(ins.type_idx.0 as usize)?;
                if let Some(e) = elem_of_gid(result_gid) {
                    let mt = mlir_scalar(&e)?;
                    let op = arith_op(ins.opcode, &e)?;
                    let a = names.get(ins.operand1.0 as usize)?;
                    let b = names.get(ins.operand2.0 as usize)?;
                    let n = format!("%v{idx}");
                    // The latch increment of a stridable loop -- the marker the device clone
                    // widens to a stride (#251 flat; Vx#379 block/thread). Only an `Add` can
                    // carry it: `lower_for` is the sole tagger.
                    let attr = if ins.opcode == Opcode::Add {
                        match ins.imm {
                            crate::hir::bytecode::IMM_PARALLEL_STEP => " {vx.parallel_step}",
                            crate::hir::bytecode::IMM_BLOCK_STEP => " {vx.parallel_bstep}",
                            crate::hir::bytecode::IMM_THREAD_STEP => " {vx.parallel_tstep}",
                            _ => "",
                        }
                    } else {
                        ""
                    };
                    body += &format!("  {n} = {op} {a}, {b}{attr} : {mt}\n");
                    names[idx] = n;
                    etypes[idx] = Some(e);
                } else {
                    let (elem, shape) = ctx.tensors.get(&result_gid)?;
                    if !is_float(elem) {
                        return None; // the AST lowers only f32 elementwise
                    }
                    let et = mlir_scalar(elem)?;
                    let d: i64 = shape
                        .iter()
                        .map(|s| s.parse::<i64>().ok())
                        .collect::<Option<Vec<_>>>()?
                        .iter()
                        .product();
                    let vecty = format!("vector<{d}x{et}>");
                    let va = coerce_vector(
                        &mut body,
                        &format!("{idx}a"),
                        ins.operand1.0,
                        &vecty,
                        et,
                        &names,
                        &mem_of,
                        &vec_of,
                        &etypes,
                    )?;
                    let vb = coerce_vector(
                        &mut body,
                        &format!("{idx}b"),
                        ins.operand2.0,
                        &vecty,
                        et,
                        &names,
                        &mem_of,
                        &vec_of,
                        &etypes,
                    )?;
                    let op = match ins.opcode {
                        Opcode::Add => "arith.addf",
                        Opcode::Sub => "arith.subf",
                        Opcode::Mul => "arith.mulf",
                        Opcode::Div => "arith.divf",
                        _ => return None,
                    };
                    let n = format!("%v{idx}");
                    body += &format!("  {n} = {op} {va}, {vb} : {vecty}\n");
                    names[idx] = n;
                    vec_of[idx] = Some(vecty);
                }
            }
            // Scalar comparison → `i1`; the relation is in `imm`, the operand type comes from the
            // first operand's tracked type (this instruction's own type is `bool`, the result).
            Opcode::Cmp => {
                let e = elem_at(&etypes, ins.operand1.0)?;
                let mt = mlir_scalar(&e)?;
                let (op, pred) = cmp_op(ins.imm, &e)?;
                let a = names.get(ins.operand1.0 as usize)?;
                let b = names.get(ins.operand2.0 as usize)?;
                let n = format!("%v{idx}");
                body += &format!("  {n} = {op} {pred}, {a}, {b} : {mt}\n");
                names[idx] = n;
                etypes[idx] = Some(ElementType::Bool);
            }
            // Arithmetic negation `-x` (#214). `type_idx` is the result (= operand) scalar type. Float
            // → `arith.negf`; integers have no `negi`, so `0 - x` via `arith.subi`.
            Opcode::Neg => {
                let e = elem_of_gid(*types.get(ins.type_idx.0 as usize)?)?;
                let mt = mlir_scalar(&e)?;
                let a = names.get(ins.operand1.0 as usize)?.clone();
                let n = format!("%v{idx}");
                if is_float(&e) {
                    body += &format!("  {n} = arith.negf {a} : {mt}\n");
                } else {
                    let z = format!("%z{idx}");
                    body += &format!("  {z} = arith.constant 0 : {mt}\n");
                    body += &format!("  {n} = arith.subi {z}, {a} : {mt}\n");
                }
                names[idx] = n;
                etypes[idx] = Some(e);
            }
            // Logical / bitwise not `!x` (#214): `x ^ all-ones` (`1` for a bool `i1`, `-1` for ints).
            Opcode::Not => {
                let e = elem_of_gid(*types.get(ins.type_idx.0 as usize)?)?;
                let mt = mlir_scalar(&e)?;
                let a = names.get(ins.operand1.0 as usize)?.clone();
                let ones_val = if matches!(e, ElementType::Bool) {
                    "1"
                } else {
                    "-1"
                };
                let ones = format!("%ones{idx}");
                let n = format!("%v{idx}");
                body += &format!("  {ones} = arith.constant {ones_val} : {mt}\n");
                body += &format!("  {n} = arith.xori {a}, {ones} : {mt}\n");
                names[idx] = n;
                etypes[idx] = Some(e);
            }
            // Scalar `as` cast (#214): `type_idx` is the *target* type, `operand1` the source value
            // (whose type comes from its tracked `etypes`). The right `arith` conversion is chosen by
            // the source/target kinds + widths; a same-type cast is a no-op that just aliases.
            Opcode::Cast => {
                let src = elem_at(&etypes, ins.operand1.0)?;
                let tgt = elem_of_gid(*types.get(ins.type_idx.0 as usize)?)?;
                let a = names.get(ins.operand1.0 as usize)?.clone();
                let op = cast_op(&src, &tgt)?;
                if op.is_empty() {
                    names[idx] = a; // reinterpret (e.g. i32 as u32) -> alias
                } else {
                    let n = format!("%v{idx}");
                    body += &format!(
                        "  {n} = {op} {a} : {} to {}\n",
                        mlir_scalar(&src)?,
                        mlir_scalar(&tgt)?
                    );
                    names[idx] = n;
                }
                etypes[idx] = Some(tgt);
            }
            // A named local's stack slot. A scalar slot is a rank-0 memref (matching the AST codegen's
            // scalar locals); an aggregate (struct) slot is an `llvm.alloca` of the `!llvm.struct`
            // type, its pointer tracked in `agg_of` so field ops can address it.
            Opcode::Alloca => {
                let gid = *types.get(ins.type_idx.0 as usize)?;
                if let Some(e) = elem_of_gid(gid) {
                    let mt = mlir_scalar(&e)?;
                    let n = format!("%v{idx}");
                    if ins.imm == 1 {
                        // An address-taken scalar (`&x`): an `llvm.alloca` of the element type, yielding
                        // a real `!llvm.ptr` the borrow can hand out. Its `Store`/`SlotLoad` go through
                        // `sslot_of` (llvm.store/load); `ptr_of` marks it so `&x` types as a pointer arg
                        // / return. Do *not* set `etypes` — the slot register is a pointer, not a scalar
                        // value (that would mistype `&x` as its element at a call site). (#230)
                        let cnt = format!("%n{idx}");
                        body += &format!("  {cnt} = llvm.mlir.constant(1 : i32) : i32\n");
                        body += &format!("  {n} = llvm.alloca {cnt} x {mt} : (i32) -> !llvm.ptr\n");
                        names[idx] = n;
                        sslot_of[idx] = Some(e);
                        ptr_of[idx] = true;
                    } else {
                        body += &format!("  {n} = memref.alloca() : memref<{mt}>\n");
                        names[idx] = n;
                        etypes[idx] = Some(e);
                    }
                } else if gid == ptr_gid() {
                    // A pointer local (memory mode): an `llvm.alloca` of one `!llvm.ptr` cell, tracked
                    // in `pslot_of` so its `Store`/`SlotLoad` use `llvm.store`/`llvm.load`. (#235)
                    let cnt = format!("%n{idx}");
                    let n = format!("%v{idx}");
                    body += &format!("  {cnt} = llvm.mlir.constant(1 : i32) : i32\n");
                    body +=
                        &format!("  {n} = llvm.alloca {cnt} x !llvm.ptr : (i32) -> !llvm.ptr\n");
                    names[idx] = n;
                    pslot_of[idx] = true;
                } else {
                    let agg = ctx.aggs.get(&gid)?;
                    let cnt = format!("%n{idx}");
                    let n = format!("%v{idx}");
                    body += &format!("  {cnt} = llvm.mlir.constant(1 : i32) : i32\n");
                    body += &format!(
                        "  {n} = llvm.alloca {cnt} x {} : (i32) -> !llvm.ptr\n",
                        agg.struct_ty
                    );
                    names[idx] = n;
                    agg_of[idx] = Some(gid);
                }
            }
            // Store a value into a slot (no result). A scalar slot is a rank-0 `memref`; an aggregate
            // slot (a struct value spilled from a struct-returning call) is an `llvm.store` (#215).
            Opcode::Store => {
                let slot = names.get(ins.operand1.0 as usize)?.clone();
                let val = names.get(ins.operand2.0 as usize)?.clone();
                // The induction-variable init of a stridable loop carries its tag into the MLIR
                // text as a discardable attribute -- inert on the host path, the marker the
                // device clone offsets by an id (#251 flat grid-stride; Vx#379 block/thread).
                let attr = match ins.imm {
                    crate::hir::bytecode::IMM_PARALLEL_INIT => " {vx.parallel_init}",
                    crate::hir::bytecode::IMM_BLOCK_INIT => " {vx.parallel_binit}",
                    crate::hir::bytecode::IMM_THREAD_INIT => " {vx.parallel_tinit}",
                    _ => "",
                };
                if let Some(&Some(agg_gid)) = agg_of.get(ins.operand1.0 as usize) {
                    let agg = ctx.aggs.get(&agg_gid)?;
                    body += &format!(
                        "  llvm.store {val}, {slot} : {}, !llvm.ptr\n",
                        agg.struct_ty
                    );
                } else if *pslot_of.get(ins.operand1.0 as usize)? {
                    // A pointer local: store the `!llvm.ptr` value into its `llvm.alloca` cell. (#235)
                    body += &format!("  llvm.store {val}, {slot} : !llvm.ptr, !llvm.ptr\n");
                } else if let Some(e) = sslot_of.get(ins.operand1.0 as usize).cloned().flatten() {
                    // An address-taken scalar slot (an `llvm.alloca` of the element): `llvm.store`. (#230)
                    let mt = mlir_scalar(&e)?;
                    body += &format!("  llvm.store {val}, {slot}{attr} : {mt}, !llvm.ptr\n");
                } else {
                    let e = elem_at(&etypes, ins.operand1.0)?;
                    let mt = mlir_scalar(&e)?;
                    body += &format!("  memref.store {val}, {slot}[]{attr} : memref<{mt}>\n");
                }
            }
            // Load a value back from a slot; the result type is the slot's element (this
            // instruction's own `type_idx`).
            Opcode::SlotLoad => {
                // A pointer slot loads back an `!llvm.ptr` value (`llvm.load`); an aggregate slot loads
                // the whole `!llvm.struct` value (`llvm.load`, #242 Vec<Vec<T>>); a scalar slot loads
                // its element from the rank-0 memref (`memref.load`). (#235)
                if let Some(&Some(agg_gid)) = agg_of.get(ins.operand1.0 as usize) {
                    let agg = ctx.aggs.get(&agg_gid)?;
                    let slot = names.get(ins.operand1.0 as usize)?;
                    let n = format!("%v{idx}");
                    body += &format!(
                        "  {n} = llvm.load {slot} : !llvm.ptr -> {}\n",
                        agg.struct_ty
                    );
                    names[idx] = n;
                    agg_val_of[idx] = Some(agg_gid);
                } else if *pslot_of.get(ins.operand1.0 as usize)? {
                    let slot = names.get(ins.operand1.0 as usize)?;
                    let n = format!("%v{idx}");
                    body += &format!("  {n} = llvm.load {slot} : !llvm.ptr -> !llvm.ptr\n");
                    names[idx] = n;
                    ptr_of[idx] = true;
                } else if let Some(e) = sslot_of.get(ins.operand1.0 as usize).cloned().flatten() {
                    // An address-taken scalar slot: `llvm.load` the element back from the `!llvm.ptr`. (#230)
                    let mt = mlir_scalar(&e)?;
                    let slot = names.get(ins.operand1.0 as usize)?;
                    let n = format!("%v{idx}");
                    body += &format!("  {n} = llvm.load {slot} : !llvm.ptr -> {mt}\n");
                    names[idx] = n;
                    etypes[idx] = Some(e);
                } else {
                    let e = ty_at(ins.type_idx.0)?;
                    let mt = mlir_scalar(&e)?;
                    let slot = names.get(ins.operand1.0 as usize)?;
                    let n = format!("%v{idx}");
                    body += &format!("  {n} = memref.load {slot}[] : memref<{mt}>\n");
                    names[idx] = n;
                    etypes[idx] = Some(e);
                }
            }
            // Block markers → MLIR blocks. Block 0 is the func's entry block (implicit; it carries the
            // params), so it gets no label; every other id opens `^bbN:`.
            Opcode::BlockStart => {
                if ins.imm != 0 {
                    body += &format!("^bb{}:\n", ins.imm);
                }
                terminated = false;
            }
            Opcode::Br => {
                body += &format!("  cf.br ^bb{}\n", ins.imm);
                terminated = true;
            }
            // `imm` packs the two targets as `then | (else << 32)` (see `flatten::pack_targets`).
            Opcode::CondBr => {
                let cond = names.get(ins.operand1.0 as usize)?;
                let then_b = ins.imm & 0xffff_ffff;
                let else_b = ins.imm >> 32;
                body += &format!("  cf.cond_br {cond}, ^bb{then_b}, ^bb{else_b}\n");
                terminated = true;
            }
            Opcode::Ret => {
                let gid = *types.get(ins.type_idx.0 as usize)?;
                let a = names.get(ins.operand1.0 as usize)?.clone();
                if let Some(e) = elem_of_gid(gid) {
                    // Coerce the returned scalar to the function's declared return type if they differ
                    // (e.g. `return 10` — a default-`i32` literal — from an `-> i64` function ->
                    // `arith.extsi`), mirroring the AST's `coerce_type` at return. Otherwise the
                    // `func.return` type contradicts the signature.
                    let target = ret_elem.clone().unwrap_or_else(|| e.clone());
                    if mlir_scalar(&e)? != mlir_scalar(&target)? {
                        match cast_op(&e, &target)? {
                            "" => {
                                body += &format!("  func.return {a} : {}\n", mlir_scalar(&target)?)
                            }
                            op => {
                                let c = format!("%rc{idx}");
                                body += &format!(
                                    "  {c} = {op} {a} : {} to {}\n",
                                    mlir_scalar(&e)?,
                                    mlir_scalar(&target)?
                                );
                                body += &format!("  func.return {c} : {}\n", mlir_scalar(&target)?);
                            }
                        }
                    } else {
                        body += &format!("  func.return {a} : {}\n", mlir_scalar(&e)?);
                    }
                } else if gid == ptr_gid() {
                    body += &format!("  func.return {a} : !llvm.ptr\n"); // a pointer return (#235)
                } else if let Some(agg) = ctx.aggs.get(&gid) {
                    // A struct return (#215). The operand is either a slot pointer (a constructed
                    // struct) -> load the value; or already a struct value (a returned call result) ->
                    // return it directly.
                    if agg_of
                        .get(ins.operand1.0 as usize)
                        .copied()
                        .flatten()
                        .is_some()
                    {
                        let rv = format!("%rv{idx}");
                        body +=
                            &format!("  {rv} = llvm.load {a} : !llvm.ptr -> {}\n", agg.struct_ty);
                        body += &format!("  func.return {rv} : {}\n", agg.struct_ty);
                    } else {
                        body += &format!("  func.return {a} : {}\n", agg.struct_ty);
                    }
                } else {
                    return None;
                }
                terminated = true;
            }
            // One argument of the following `Call`: record its value register (no op emitted).
            Opcode::Arg => pending_args.push(ins.operand1.0),
            // A fixed-arity call. `type_idx` is the callee's GID (resolved to name + return type via
            // `ctx.callees`); `imm` is the arg count, taken from the tail of `pending_args`. Emit
            // `%r = func.call @name(%a, %b) : (Ta, Tb) -> Tret`.
            Opcode::Call => {
                let gid = *types.get(ins.type_idx.0 as usize)?;
                let callee = ctx.callees.get(&gid)?;
                // Return type: a scalar, an `!llvm.struct` by value for a struct-returning callee
                // (#215), a pointer, or `()` for a void callee (a `&mut` mutator called in statement
                // position, #230).
                let rt = if let Some(e) = &callee.ret {
                    mlir_scalar(e)?.to_string()
                } else if let Some(agg_gid) = callee.ret_agg {
                    ctx.aggs.get(&agg_gid)?.struct_ty.clone()
                } else if callee.ret_ptr {
                    "!llvm.ptr".to_string() // an FFI pointer-returning callee (#235)
                } else if callee.ret_void {
                    "()".to_string()
                } else {
                    return None;
                };
                let n = ins.imm as usize;
                if pending_args.len() < n {
                    return None;
                }
                let args = pending_args.split_off(pending_args.len() - n);
                let mut arg_names = Vec::with_capacity(n);
                let mut arg_types: Vec<String> = Vec::with_capacity(n);
                for a in &args {
                    arg_names.push(names.get(*a as usize)?.clone());
                    // A scalar arg is its element type; a pointer arg (a string value / FFI pointer, or
                    // an aggregate *slot* passed by reference — `&v` / a `self` pointer, #242) is
                    // `!llvm.ptr`; a tensor arg is its memref type.
                    let at = if let Some(e) = elem_at(&etypes, *a) {
                        mlir_scalar(&e)?.to_string()
                    } else if let Some(agg_gid) = agg_val_of.get(*a as usize).copied().flatten() {
                        // A by-value aggregate argument (`push(&outer, a)` passing `a : Vec<i32>` by
                        // value into a `Vec<Vec<T>>::push`) — an `!llvm.struct` value (#242).
                        ctx.aggs.get(&agg_gid)?.struct_ty.clone()
                    } else if *ptr_of.get(*a as usize)?
                        || agg_of.get(*a as usize).copied().flatten().is_some()
                    {
                        "!llvm.ptr".to_string()
                    } else {
                        mem_of.get(*a as usize)?.clone()?
                    };
                    arg_types.push(at);
                }
                if callee.ret_void {
                    // A void call binds no result register (MLIR forbids `%v = func.call ... -> ()`);
                    // the call is a pure effect (mutation through a `&mut` arg). The private extern decl
                    // records an empty return (no `->`) so a void `extern` declares as `(args)`. (#230)
                    body += &format!(
                        "  func.call {}({}) : ({}) -> ()\n",
                        sym_ref(&callee.name),
                        arg_names.join(", "),
                        arg_types.join(", "),
                    );
                    calls.push((callee.name.clone(), arg_types.clone(), String::new()));
                } else {
                    let nm = format!("%v{idx}");
                    body += &format!(
                        "  {nm} = func.call {}({}) : ({}) -> {rt}\n",
                        sym_ref(&callee.name),
                        arg_names.join(", "),
                        arg_types.join(", "),
                    );
                    // Record the callee's signature so the module emitter can declare it if it is a
                    // called-but-undefined symbol (an `extern`): the private decl's signature is taken
                    // from the emitted call, so they match by construction.
                    calls.push((callee.name.clone(), arg_types.clone(), rt.clone()));
                    names[idx] = nm;
                    if let Some(e) = &callee.ret {
                        etypes[idx] = Some(e.clone());
                    } else if callee.ret_ptr {
                        ptr_of[idx] = true; // the call result is a pointer value (#235)
                    } else if let Some(agg_gid) = callee.ret_agg {
                        // A struct-returning call result is a struct *value*; tracked so it can be
                        // spilled to a slot (`Store`), returned (`Ret`), or passed by value to another
                        // call (#242).
                        agg_val_of[idx] = Some(agg_gid);
                    }
                }
            }
            // Materialize a function pointer for a named function: `type_idx` is the target's GID
            // (name via `ctx.callees`, signature via `ctx.func_sigs`). Emit `func.constant @name : sig`
            // then cast the `FunctionType` value to an opaque `!llvm.ptr` (the ABI of a fn pointer),
            // tracked in `ptr_of`. (#242)
            Opcode::FuncConst => {
                let gid = *types.get(ins.type_idx.0 as usize)?;
                let callee = ctx.callees.get(&gid)?;
                let (params, ret) = ctx.func_sigs.get(&gid)?;
                let fnty = format!("({}) -> {}", params.join(", "), ret);
                let fc = format!("%fc{idx}");
                let nm = format!("%v{idx}");
                body += &format!(
                    "  {fc} = func.constant {} : {fnty}\n",
                    sym_ref(&callee.name)
                );
                body += &format!(
                    "  {nm} = builtin.unrealized_conversion_cast {fc} : {fnty} to !llvm.ptr\n"
                );
                names[idx] = nm;
                ptr_of[idx] = true;
            }
            // An indirect call through a function pointer. `operand1` is the callee `!llvm.ptr`, `imm`
            // the arg count (the tail of `pending_args`, like `Call`), and this instruction's `type_idx`
            // the scalar return type. Reconstruct the function type `(arg types)->ret` from the actual
            // args, cast the pointer to it, and `func.call_indirect`. (#242)
            Opcode::CallIndirect => {
                let ret_elem = ty_at(ins.type_idx.0)?;
                let rt = mlir_scalar(&ret_elem)?.to_string();
                let n = ins.imm as usize;
                if pending_args.len() < n {
                    return None;
                }
                let args = pending_args.split_off(pending_args.len() - n);
                let mut arg_names = Vec::with_capacity(n);
                let mut arg_types: Vec<String> = Vec::with_capacity(n);
                for a in &args {
                    arg_names.push(names.get(*a as usize)?.clone());
                    let at = if let Some(e) = elem_at(&etypes, *a) {
                        mlir_scalar(&e)?.to_string()
                    } else if let Some(agg_gid) = agg_val_of.get(*a as usize).copied().flatten() {
                        ctx.aggs.get(&agg_gid)?.struct_ty.clone()
                    } else if *ptr_of.get(*a as usize)?
                        || agg_of.get(*a as usize).copied().flatten().is_some()
                    {
                        "!llvm.ptr".to_string()
                    } else {
                        mem_of.get(*a as usize)?.clone()?
                    };
                    arg_types.push(at);
                }
                let fnty = format!("({}) -> {rt}", arg_types.join(", "));
                let fnptr = names.get(ins.operand1.0 as usize)?.clone();
                let fc = format!("%fc{idx}");
                let nm = format!("%v{idx}");
                body += &format!(
                    "  {fc} = builtin.unrealized_conversion_cast {fnptr} : !llvm.ptr to {fnty}\n"
                );
                body += &format!(
                    "  {nm} = func.call_indirect {fc}({}) : {fnty}\n",
                    arg_names.join(", ")
                );
                names[idx] = nm;
                etypes[idx] = Some(ret_elem);
            }
            // Store a scalar into a struct field (no result). `operand1` is the struct slot pointer,
            // `operand2` the value, `imm` the field's byte offset. GEP to the field, then `llvm.store`;
            // the field index comes from matching the offset against the layout, the value type from
            // the stored register's tracked type.
            Opcode::FieldStore => {
                let gid = (*agg_of.get(ins.operand1.0 as usize)?)?;
                let agg = ctx.aggs.get(&gid)?;
                let field_idx = agg.offsets.iter().position(|&o| o == ins.imm)?;
                // The field's declared MLIR type (a scalar element or `!llvm.ptr`), so a pointer field
                // (`Vec`'s `data`) stores an `!llvm.ptr` value and a scalar field its element (#242).
                let fty = agg.field_tys.get(field_idx)?.clone();
                let slot = names.get(ins.operand1.0 as usize)?;
                let val = names.get(ins.operand2.0 as usize)?;
                let p = format!("%p{idx}");
                body += &format!(
                    "  {p} = llvm.getelementptr {slot}[0, {field_idx}] : (!llvm.ptr) -> !llvm.ptr, {}\n",
                    agg.struct_ty
                );
                // A place-write store carries alias-scope metadata (M2b-2): it belongs to its own scope
                // and does not alias its disjoint siblings' scopes. Direct field stores are unscoped.
                let attrs = alias_scope_of
                    .get(&idx)
                    .map(|(own, sibs)| alias_store_attrs(*own, sibs))
                    .unwrap_or_default();
                body += &format!("  llvm.store {val}, {p}{attrs} : {fty}, !llvm.ptr\n");
            }
            // Load a scalar struct field. `operand1` is the struct slot, `imm` the field's byte offset,
            // and this instruction's own `type_idx` the field's scalar type. GEP to the field, then
            // `llvm.load`.
            Opcode::FieldLoad => {
                let gid = (*agg_of.get(ins.operand1.0 as usize)?)?;
                let agg = ctx.aggs.get(&gid)?;
                let field_idx = agg.offsets.iter().position(|&o| o == ins.imm)?;
                // The field's declared MLIR type drives the load: a scalar field yields its element
                // (tracked in `etypes`), a pointer field (`Vec`'s `data`) an `!llvm.ptr` value
                // (tracked in `ptr_of`) — the type is taken from the layout, not the read register's
                // `type_idx`, so a pointer field (whose `type_idx` is `ptr_gid`) resolves too (#242).
                let fty = agg.field_tys.get(field_idx)?.clone();
                // A pointer field pointing to a modelled aggregate (`VecIter`'s `vec : *const Vec<T>`)
                // tags its loaded value with the pointee GID, so a chained field access through it
                // (`(*self.vec).len`) GEPs the pointee. (#242)
                let pointee = agg.field_pointee.get(field_idx).copied().flatten();
                let slot = names.get(ins.operand1.0 as usize)?;
                let p = format!("%p{idx}");
                let n = format!("%v{idx}");
                body += &format!(
                    "  {p} = llvm.getelementptr {slot}[0, {field_idx}] : (!llvm.ptr) -> !llvm.ptr, {}\n",
                    agg.struct_ty
                );
                body += &format!("  {n} = llvm.load {p} : !llvm.ptr -> {fty}\n");
                names[idx] = n;
                if fty == "!llvm.ptr" {
                    ptr_of[idx] = true;
                    agg_of[idx] = pointee;
                } else if let Some(nested_gid) = agg.field_agg.get(field_idx).copied().flatten() {
                    // A by-value nested-aggregate field load yields the whole `!llvm.struct` value,
                    // tracked as an aggregate value so it can be re-stored / passed by value. (#242)
                    agg_val_of[idx] = Some(nested_gid);
                } else {
                    etypes[idx] = elem_from_mlir_scalar(&fty);
                }
            }
            // The address of a by-value nested-aggregate field (`&outer.inner`): GEP to the field and
            // yield the pointer, tracked as an aggregate slot (its layout GID from `type_idx`), so a
            // chained field access or a method receiver addresses through it. (#242)
            Opcode::FieldAddr => {
                let parent_gid = (*agg_of.get(ins.operand1.0 as usize)?)?;
                let agg = ctx.aggs.get(&parent_gid)?;
                let field_idx = agg.offsets.iter().position(|&o| o == ins.imm)?;
                let field_gid = *types.get(ins.type_idx.0 as usize)?;
                let slot = names.get(ins.operand1.0 as usize)?;
                let n = format!("%v{idx}");
                body += &format!(
                    "  {n} = llvm.getelementptr {slot}[0, {field_idx}] : (!llvm.ptr) -> !llvm.ptr, {}\n",
                    agg.struct_ty
                );
                names[idx] = n;
                // A nested-aggregate field's address is tracked as an aggregate slot, so a chained field
                // access GEPs through it (#242); a *scalar* field's address (`&param.scalar`, #275 M3b
                // reference returns) is a plain element pointer — the result GID is the field's scalar GID,
                // which is not in `aggs`, so track it as a pointer instead.
                if ctx.aggs.contains_key(&field_gid) {
                    agg_of[idx] = Some(field_gid);
                } else {
                    ptr_of[idx] = true;
                }
            }
            // Allocate a tensor buffer (`Tensor<T>([..])`): a static `memref` of the shape recovered
            // from the side table by GID. Its register is tracked in `mem_of` for later index/store.
            Opcode::TensorAlloc => {
                let gid = *types.get(ins.type_idx.0 as usize)?;
                let (elem, shape) = ctx.tensors.get(&gid)?;
                let memty = tensor_memref_ty(elem, shape)?;
                let n = format!("%v{idx}");
                // `operand2` may carry a memory-space dispatch id from `.with_memory` (Vx#379
                // stage B). A `scope: sm` space becomes a space-3 ALLOCA: on the host that is a
                // stack slot like any other, and in the device clone materializeGpuKernels turns
                // a static space-3 alloca into `.shared` storage -- the machinery #352 built.
                // Any other space stays a plain allocation; the annotation was advisory.
                let sm = ins.operand2.0 != 0
                    && ctx
                        .subspaces
                        .get(&(ins.operand2.0 as u64))
                        .and_then(|s| s.scope.as_deref())
                        == Some("sm");
                if sm {
                    let smty = format!("{}, 3>", memty.strip_suffix('>')?);
                    body += &format!("  {n} = memref.alloca() : {smty}\n");
                    names[idx] = n;
                    mem_of[idx] = Some(smty);
                } else {
                    body += &format!("  {n} = memref.alloc() : {memty}\n");
                    names[idx] = n;
                    mem_of[idx] = Some(memty);
                }
            }
            // Index a tensor along its outermost dimension. `operand1` is the base tensor (memref),
            // `operand2` the index (`arith.index_cast` to `index`). A scalar-element result
            // (`type_idx` is a scalar GID) is a value read (`imm = 0` → `memref.load`) or an element
            // *place* (`imm = 1` → recorded for the following `TensorStore`). A sub-view result (a
            // tensor GID) rank-reduces the base to a row via `memref.reinterpret_cast` (contiguous
            // base only; a further sub-view of a strided row is deferred).
            Opcode::TensorIndex => {
                let result_gid = *types.get(ins.type_idx.0 as usize)?;
                let base = names.get(ins.operand1.0 as usize)?.clone();
                let base_memty = mem_of.get(ins.operand1.0 as usize)?.clone()?;
                let imt = mlir_scalar(&elem_at(&etypes, ins.operand2.0)?)?;
                let iname = names.get(ins.operand2.0 as usize)?.clone();
                let ic = format!("%ic{idx}");
                body += &format!("  {ic} = arith.index_cast {iname} : {imt} to index\n");

                if let Some(e) = elem_of_gid(result_gid) {
                    // Scalar element: a value read or a store place (works on a contiguous or a
                    // strided-row base — `memref.load`/`store` handle both).
                    if ins.imm == 1 {
                        place_of[idx] = Some((base, ic, base_memty));
                    } else {
                        let n = format!("%v{idx}");
                        body += &format!("  {n} = memref.load {base}[{ic}] : {base_memty}\n");
                        names[idx] = n;
                        etypes[idx] = Some(e);
                    }
                } else {
                    // Row sub-view: reinterpret the contiguous base as the row at flat offset
                    // `index * product(row dims)`, with row-major strides over the remaining dims.
                    if base_memty.contains("strided") {
                        return None; // a sub-view of an already-strided row is deferred
                    }
                    let (elem, shape) = ctx.tensors.get(&result_gid)?;
                    let et = mlir_scalar(elem)?;
                    let dims: Vec<i64> = shape
                        .iter()
                        .map(|d| d.parse::<i64>().ok())
                        .collect::<Option<_>>()?; // symbolic dims not handled
                    let stride0: i64 = dims.iter().product();
                    let mut strides = vec![1i64; dims.len()];
                    for i in (0..dims.len().saturating_sub(1)).rev() {
                        strides[i] = strides[i + 1] * dims[i + 1];
                    }
                    let off = if stride0 == 1 {
                        ic.clone()
                    } else {
                        let cst = format!("%cs{idx}");
                        let o = format!("%off{idx}");
                        body += &format!("  {cst} = arith.constant {stride0} : index\n");
                        body += &format!("  {o} = arith.muli {ic}, {cst} : index\n");
                        o
                    };
                    let sizes_s = join_i64(&dims);
                    let strides_s = join_i64(&strides);
                    let dimx: String = dims.iter().map(|d| format!("{d}x")).collect();
                    // A sub-view of shared storage stays in its space: dropping the `, 3` here
                    // would make the row a generic pointer and the PTX would address `.shared`
                    // data with global loads.
                    let space_sfx = if base_memty.ends_with(", 3>") {
                        ", 3"
                    } else {
                        ""
                    };
                    let result_ty =
                        format!("memref<{dimx}{et}, strided<[{strides_s}], offset: ?>{space_sfx}>");
                    let n = format!("%v{idx}");
                    body += &format!(
                        "  {n} = memref.reinterpret_cast {base} to offset: [{off}], sizes: [{sizes_s}], strides: [{strides_s}] : {base_memty} to {result_ty}\n"
                    );
                    names[idx] = n;
                    mem_of[idx] = Some(result_ty);
                }
            }
            // Reduce a rank-1 float slice to a scalar. `operand1` (and `operand2` for `dot`) are the
            // slices; `imm` the kind (0 = dot, 1 = sum, 2 = max, 3 = min). Each slice is `vector.load`ed
            // to a `vector<Nxf32>`; `dot` fuses the two with `arith.mulf`; then `vector.reduction`.
            // Float only, matching the AST oracle (`vector<Nxf32>` → `f32`).
            Opcode::Reduce => {
                let e = ty_at(ins.type_idx.0)?;
                if !is_float(&e) {
                    return None; // the AST lowers only f32 reductions
                }
                let et = mlir_scalar(&e)?;
                let s0 = names.get(ins.operand1.0 as usize)?.clone();
                let m0 = mem_of.get(ins.operand1.0 as usize)?.clone()?;
                let d = memref_lead_dim(&m0)?;
                let vecty = format!("vector<{d}x{et}>");
                let c0 = format!("%rc{idx}");
                body += &format!("  {c0} = arith.constant 0 : index\n");
                let al = vector_align_attr(&vecty);
                let v0 = format!("%vl{idx}");
                body += &format!("  {v0} = vector.load {s0}[{c0}]{al} : {m0}, {vecty}\n");
                let (reduce_in, kind) = match ins.imm {
                    0 => {
                        let s1 = names.get(ins.operand2.0 as usize)?.clone();
                        let m1 = mem_of.get(ins.operand2.0 as usize)?.clone()?;
                        let v1 = format!("%vr{idx}");
                        body += &format!("  {v1} = vector.load {s1}[{c0}]{al} : {m1}, {vecty}\n");
                        let prod = format!("%vp{idx}");
                        body += &format!("  {prod} = arith.mulf {v0}, {v1} : {vecty}\n");
                        (prod, "add")
                    }
                    1 => (v0, "add"),
                    2 => (v0, "maximumf"),
                    3 => (v0, "minimumf"),
                    _ => return None,
                };
                let n = format!("%v{idx}");
                body += &format!(
                    "  {n} = vector.reduction <{kind}>, {reduce_in} : {vecty} into {et}\n"
                );
                names[idx] = n;
                etypes[idx] = Some(e);
            }
            // `matmul_into(&mut dst, &a, &b)` (no result): `linalg.fill` + `linalg.matmul` on the
            // three whole-tensor memrefs, the same pair the AST path builds -- and the exact shape
            // `kernelKindOf` classifies, so a spawn whose whole job is this op still routes to
            // cuBLAS. The destination register rides the imm (see `Opcode::MatmulInto`).
            Opcode::MatmulInto => {
                let a = names.get(ins.operand1.0 as usize)?.clone();
                let ma = mem_of.get(ins.operand1.0 as usize)?.clone()?;
                let b = names.get(ins.operand2.0 as usize)?.clone();
                let mb = mem_of.get(ins.operand2.0 as usize)?.clone()?;
                let dst = names.get(ins.imm as usize)?.clone();
                let md = mem_of.get(ins.imm as usize)?.clone()?;
                // "memref<8x16xf32>" -> "f32": the element is the segment after the last 'x',
                // shorn of the closing '>'. Floats only -- an int matmul declines the program to
                // the AST path rather than improvising linalg's integer semantics here.
                let et = md.rsplit('x').next()?.trim_end_matches('>').to_string();
                if et != "f32" && et != "f64" {
                    return None;
                }
                body += &format!("  %mz{idx} = arith.constant 0.0 : {et}\n");
                body += &format!("  linalg.fill ins(%mz{idx} : {et}) outs({dst} : {md})\n");
                body += &format!("  linalg.matmul ins({a}, {b} : {ma}, {mb}) outs({dst} : {md})\n");
            }
            // Store into a tensor place (no result). A scalar-element place (an `imm = 1`
            // `TensorIndex`) → `memref.store`; a row/sub-view place (an `imm = 0` `TensorIndex`, a row
            // memref in `mem_of`) takes an elementwise vector value → `vector.store`.
            Opcode::TensorStore => {
                if let Some((base, ic, memty)) = place_of.get(ins.operand1.0 as usize)?.clone() {
                    let vreg = ins.operand2.0;
                    let mut val = names.get(vreg as usize)?.clone();
                    // Coerce the stored scalar to the tensor's element type when they differ — a
                    // default-`f32` float literal `1.0` stored into a `bf16` tensor becomes
                    // `arith.truncf`, `f32 -> f64` becomes `arith.extf`, etc. The AST path does the
                    // same via `coerce_type` before its `memref.store`; without it the store is
                    // ill-typed (`f32` value into a `memref<..xbf16>`).
                    if let (Some(src_e), Some(tgt_s)) =
                        (elem_at(&etypes, vreg), memref_elem(&memty))
                    {
                        if let Some(tgt_e) = elem_from_mlir_scalar(tgt_s) {
                            match cast_op(&src_e, &tgt_e) {
                                Some("") => {} // same MLIR type: no conversion
                                Some(op) => {
                                    let c = format!("%tsc{idx}");
                                    body += &format!(
                                        "  {c} = {op} {val} : {} to {tgt_s}\n",
                                        mlir_scalar(&src_e)?
                                    );
                                    val = c;
                                }
                                None => return None, // unmodelled conversion -> decline (AST oracle)
                            }
                        }
                    }
                    body += &format!("  memref.store {val}, {base}[{ic}] : {memty}\n");
                } else if let Some(rowty) = mem_of.get(ins.operand1.0 as usize)?.clone() {
                    let dst = names.get(ins.operand1.0 as usize)?.clone();
                    let vecname = names.get(ins.operand2.0 as usize)?.clone();
                    let vecty = vec_of.get(ins.operand2.0 as usize)?.clone()?;
                    let c0 = format!("%sc{idx}");
                    let al = vector_align_attr(&vecty);
                    body += &format!("  {c0} = arith.constant 0 : index\n");
                    body +=
                        &format!("  vector.store {vecname}, {dst}[{c0}]{al} : {rowty}, {vecty}\n");
                } else {
                    return None;
                }
            }
            // Move a tensor to a memory space: `operand1` is the source, `imm` the target space's
            // dispatch id, `type_idx` the result tensor (same element + shape). Emits `vx.transfer`
            // (generic form) with `target_topology`; the vx→standard lowering turns it into an
            // alloc + `memref.copy` (it ignores the source's layout suffix, so the result is a plain
            // `memref<NxT>`). When the target space declares a sub-space descriptor, re-attach the
            // scheduling attrs (`space`/`within`/`granule`/`capacity`/`scope` + a bump-allocated
            // `offset`/`slots`) the AST path emits — a device backend needs them to place the tile
            // into VMEM/TMEM, and they are dropped otherwise (B1/P0-1).
            Opcode::Transfer => {
                let result_gid = *types.get(ins.type_idx.0 as usize)?;
                let (elem, shape) = ctx.tensors.get(&result_gid)?;
                let dstty = tensor_memref_ty(elem, shape)?;
                let src = names.get(ins.operand1.0 as usize)?.clone();
                let srcty = mem_of.get(ins.operand1.0 as usize)?.clone()?;
                let n = format!("%v{idx}");
                let mut attrs = format!("target_topology = {} : i32", ins.imm);
                if let Some(desc) = ctx.subspaces.get(&ins.imm) {
                    // Descriptor attrs, in the AST path's emission order (MLIR sorts on print, so the
                    // final parsed form is byte-identical regardless of the order emitted here).
                    attrs += &format!(", space = \"{}\"", desc.name);
                    if let Some(w) = &desc.within {
                        attrs += &format!(", within = \"{w}\"");
                    }
                    if let Some(g) = desc.granule {
                        attrs += &format!(", granule = {g} : i64");
                    }
                    if let Some(c) = desc.capacity {
                        attrs += &format!(", capacity = {c} : i64");
                    }
                    if let Some(s) = &desc.scope {
                        attrs += &format!(", scope = \"{s}\"");
                    }
                    if let Some(m) = &desc.managed {
                        attrs += &format!(", managed = \"{m}\"");
                    }
                    // SS2 bump allocation: a statically-shaped tile into a granule'd space claims the
                    // next granule-rounded `offset`; `slots` is the granule count it occupies.
                    let tile_bytes = static_tile_bytes(elem, shape);
                    if let (Some(granule), Some(bytes)) = (desc.granule, tile_bytes) {
                        if granule > 0 {
                            let rounded = bytes.div_ceil(granule) * granule;
                            let offset = *subspace_offsets.entry(ins.imm).or_insert(0);
                            subspace_offsets.insert(ins.imm, offset + rounded);
                            attrs += &format!(", offset = {offset} : i64");
                            attrs += &format!(", slots = {} : i64", rounded / granule);
                        }
                    }
                }
                body +=
                    &format!("  {n} = \"vx.transfer\"({src}) {{{attrs}}} : ({srcty}) -> {dstty}\n");
                names[idx] = n;
                mem_of[idx] = Some(dstty);
            }
            // Print a value (no result). A tensor is `memref.cast`'d to an unranked memref and passed
            // to the `printMemref*` runtime helper; a scalar goes to `print_*`. These are the same
            // helpers the AST path calls; `emit_module_mlir` prepends their `private` declarations.
            Opcode::Print => {
                let arg = names.get(ins.operand1.0 as usize)?.clone();
                if let Some(memty) = mem_of.get(ins.operand1.0 as usize)?.clone() {
                    let et = memref_elem(&memty)?;
                    let helper = match et {
                        "f32" => "printMemrefF32",
                        "f64" => "printMemrefF64",
                        "i32" => "printMemrefI32",
                        "i64" => "printMemrefI64",
                        _ => return None,
                    };
                    let c = format!("%pc{idx}");
                    body += &format!("  {c} = memref.cast {arg} : {memty} to memref<*x{et}>\n");
                    body += &format!("  func.call @{helper}({c}) : (memref<*x{et}>) -> ()\n");
                } else if let Some(e) = elem_at(&etypes, ins.operand1.0) {
                    let et = mlir_scalar(&e)?;
                    let helper = match et {
                        "f32" => "print_f32",
                        "f64" => "print_f64",
                        "i32" => "print_i32",
                        "i64" => "print_i64",
                        _ => return None,
                    };
                    let n = format!("%v{idx}");
                    body += &format!("  {n} = func.call @{helper}({arg}) : ({et}) -> i32\n");
                } else {
                    return None;
                }
            }
            // Open a `vx.spawn` region (generic form). The op is inline in the enclosing block, which
            // continues after it; the instructions up to the matching `SpawnEnd` form the region body.
            // The ops immediately after `Spawn` (a control-flow body's setup, before its first explicit
            // block) go in the region's entry block, so open a label for it. `imm` is the topology
            // dispatch id (the same value the AST path emits as `vx.spawn`'s `topology` attribute).
            Opcode::Spawn => {
                if spawn_topology.is_some() {
                    return None; // nested spawn is not modelled
                }
                spawn_topology = Some(ins.imm as i64);
                body += &format!("  \"vx.spawn\"() ({{\n^bbspawn{idx}:\n");
                terminated = false;
            }
            // Close the `vx.spawn` region: terminate its last block with `vx.yield` (unless a body
            // terminator already ended it), stamp the `topology` attribute, and resume emitting into
            // the enclosing block (which the spawn op did not terminate).
            Opcode::SpawnEnd => {
                let topo = spawn_topology.take()?;
                if !terminated {
                    body += "  \"vx.yield\"() : () -> ()\n";
                }
                // A topology that declared an `arch:` sends it along, so the device pipeline can
                // gate on the declaration instead of the dispatch-id band (Vx#352). Discardable
                // attribute on the generic form -- no dialect change involved.
                //
                // A nonzero `imm` is the trip count `parallel_outer_for` proved for the region's
                // outermost loop: `vx_parallel_trip` rides the same attribute dict, telling the
                // device pipeline the loop is safe to grid-stride and how wide the work is (#251).
                // The `SPAWN_TWO_LEVEL` bit (Vx#379) marks the trip as a BLOCK count instead, and
                // `vx_parallel_two_level` rides along so the launch is sized as blocks x threads.
                let two_level = ins.imm & crate::hir::flatten::SPAWN_TWO_LEVEL != 0;
                // The `SPAWN_COOP` bit (Vx#379 stage C): the barriers are INSIDE the thread
                // loop, so the region has no serial schedule and the host must refuse it --
                // `vx_parallel_coop` travels to the launch payload for exactly that refusal.
                let coop = ins.imm & crate::hir::flatten::SPAWN_COOP != 0;
                let trip_count = ins.imm & 0xffff_ffff;
                // Bits 32..47: the widest thread-loop trip, i.e. the block shape the launch
                // should use (Vx#379). Zero (a pre-two-level stream) falls back to 128.
                let threads = (ins.imm >> 32) & 0xffff;
                let trip = if trip_count > 0 {
                    format!(
                        ", vx_parallel_trip = {trip_count} : i64{}",
                        if two_level {
                            format!(
                                ", vx_parallel_two_level, vx_parallel_threads = {} : i64{}",
                                if threads > 0 { threads } else { 128 },
                                if coop { ", vx_parallel_coop" } else { "" }
                            )
                        } else {
                            String::new()
                        }
                    )
                } else {
                    String::new()
                };
                if let Some(arch) = ctx.topo_archs.get(&topo) {
                    body += &format!(
                        "  }}) {{arch = \"{arch}\", topology = {topo} : i32{trip}}} : () -> ()\n"
                    );
                } else {
                    body += &format!("  }}) {{topology = {topo} : i32{trip}}} : () -> ()\n");
                }
                terminated = false;
            }
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
            Opcode::Barrier => {
                body += "  \"vx.barrier\"() : () -> ()\n";
            }
            Opcode::Abort => {
                let cond = names.get(ins.operand1.0 as usize)?.clone();
                let msg = strings
                    .get(ins.imm as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("assertion failed");
                let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
                body += &format!("  cf.assert {cond}, \"{escaped}\"\n");
            }
            Opcode::PrintStr => {
                let n = str_base + ins.imm as usize;
                let p = format!("%pstrp{idx}");
                let r = format!("%pstr{idx}");
                body += &format!("  {p} = llvm.mlir.addressof @\".str.{n}\" : !llvm.ptr\n");
                body += &format!("  {r} = func.call @print_str({p}) : (!llvm.ptr) -> i32\n");
            }
            // A string literal in value position (#231): take the address of the module-level global
            // (`@".str.<n>"`, `n = str_base + imm`, the same numbering as `PrintStr`) as a first-class
            // `!llvm.ptr` value — what a `let s = "…"` binds or a string argument passes. The global's
            // bytes are emitted by `emit_module_mlir` from the string side table.
            Opcode::StringConst => {
                let n = str_base + ins.imm as usize;
                let p = format!("%v{idx}");
                body += &format!("  {p} = llvm.mlir.addressof @\".str.{n}\" : !llvm.ptr\n");
                names[idx] = p;
                ptr_of[idx] = true;
            }
            // Index a raw pointer `p[i]` (`p : *mut T`): GEP the element, then either load it (a value
            // read) or hand back the element pointer as a store place. `operand1` is the base pointer,
            // `operand2` the (scalar) index, `type_idx` the pointee element type. The GEP's base
            // element type sets the stride, so `p[i]` addresses `base + i * sizeof(T)`. (#242)
            Opcode::PtrIndex => {
                let base = names.get(ins.operand1.0 as usize)?.clone();
                // The pointee element is a scalar (`*mut i32`) or a by-value aggregate
                // (`*mut Vec<i32>`, #242 Vec<Vec<T>>). The GEP's base element type (`et`) sets the
                // stride either way; a struct element loads/stores the whole `!llvm.struct`.
                let gid = *types.get(ins.type_idx.0 as usize)?;
                // A pointer whose pointee is itself a pointer (`&&T`): the element is a bare `!llvm.ptr`
                // — the deref loads/stores an 8-byte pointer and the loaded value is itself a pointer
                // (`ptr_of`), so a further deref (`**rr`) chains. (#278)
                let elem_is_ptr = gid == ptr_gid();
                let (et, scalar_e, agg_gid) = if elem_is_ptr {
                    ("!llvm.ptr".to_string(), None, None)
                } else if let Some(e) = elem_of_gid(gid) {
                    (mlir_scalar(&e)?.to_string(), Some(e), None)
                } else if let Some(agg) = ctx.aggs.get(&gid) {
                    (agg.struct_ty.clone(), None, Some(gid))
                } else {
                    return None;
                };
                let imt = mlir_scalar(&elem_at(&etypes, ins.operand2.0)?)?;
                let iname = names.get(ins.operand2.0 as usize)?.clone();
                let p = format!("%pg{idx}");
                body += &format!(
                    "  {p} = llvm.getelementptr {base}[{iname}] : (!llvm.ptr, {imt}) -> !llvm.ptr, {et}\n"
                );
                if ins.imm == 1 {
                    // An element place: the following `PtrStore` writes through it.
                    names[idx] = p;
                    ptr_of[idx] = true;
                    pptr_elem[idx] = Some(et);
                } else {
                    let n = format!("%v{idx}");
                    body += &format!("  {n} = llvm.load {p} : !llvm.ptr -> {et}\n");
                    names[idx] = n;
                    etypes[idx] = scalar_e;
                    agg_val_of[idx] = agg_gid;
                    // The loaded value is a pointer (`*rr : &i32`): mark it so an outer deref treats it
                    // as a `!llvm.ptr` base rather than a scalar. (#278)
                    if elem_is_ptr {
                        ptr_of[idx] = true;
                    }
                }
            }
            // Store into a raw-pointer place (no result): `operand1` is the `PtrIndex` place (the GEP'd
            // element pointer), `operand2` the value, and the pointee element type comes from the
            // place. (#242)
            Opcode::PtrStore => {
                let place = names.get(ins.operand1.0 as usize)?.clone();
                let et = pptr_elem.get(ins.operand1.0 as usize)?.clone()?;
                let val = names.get(ins.operand2.0 as usize)?.clone();
                body += &format!("  llvm.store {val}, {place} : {et}, !llvm.ptr\n");
            }
            // Anything else (spawn, matmul, …) is outside this subset.
            _ => return None,
        }
    }

    // Every block must end in a terminator. A void function falls through to a bare `return`; a
    // scalar-returning function whose final block isn't terminated is either ill-typed or has an
    // unreachable trailing block (no value to return) — decline it, leaving the AST path the oracle.
    if !terminated {
        match &ret_mlir {
            None => body += "  func.return\n",
            Some(_) => return None,
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
    out += &body;
    out += "}\n";
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::flatten::lower_function_to_hir;
    use crate::session::{GlobalSession, LocalWorkerState};
    use std::sync::Arc;

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
        assert!(lower_function_to_hir(&f, &mut w), "function lowers");
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
        mods[0].resolve_names(&symbol_map);
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
            assert!(lower_function_to_hir(f, &mut w), "function lowers");
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
            crate::pipeline::Schedule::Parallel,
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
}
