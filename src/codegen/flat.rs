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
use crate::hir::flatten::{scalar_gid, tensor_gid_of};
use crate::registry::ImmutableGlobalRegistry;
use crate::syntax::{ElementType, Function, Type};
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
        Generic(_) => return None,
    })
}

fn is_float(e: &ElementType) -> bool {
    matches!(
        e,
        ElementType::F16 | ElementType::F32 | ElementType::F64 | ElementType::BF16
    )
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

/// The GID of a nominal (struct/enum) type, once name resolution has attached it -- for aggregate
/// return / value handling (#215). `None` for non-nominal or unresolved types.
fn nominal_gid_of(ty: &Type) -> Option<TypeId> {
    match ty {
        Type::Struct(_, Some(id)) | Type::Enum(_, Some(id)) => Some(*id),
        _ => None,
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
}

/// GID → callee: the reverse of the registry's name-keyed `fn_sigs`. A `Call`'s `type_idx` resolves
/// to a callee GID; this map recovers the symbol name (for `func.call @name`) and the return type
/// (for the call's result type) without a name→AST walk.
pub type CalleeMap = HashMap<TypeId, Callee>;

/// Build the GID→callee map from the frozen registry's function signatures.
pub fn build_callee_map(registry: &ImmutableGlobalRegistry) -> CalleeMap {
    registry
        .fn_sigs
        .iter()
        .map(|(name, sig)| {
            (
                sig.gid,
                Callee {
                    name: name.to_string(),
                    ret: scalar_of(&sig.ret_ty),
                    ret_agg: nominal_gid_of(&sig.ret_ty),
                },
            )
        })
        .collect()
}

/// The MLIR shape of an aggregate (struct) whose fields are all scalar: the `!llvm.struct<(...)>`
/// type (for the slot `llvm.alloca` and field `getelementptr`) and each field's byte offset in
/// declaration order — so a `FieldLoad`/`FieldStore`, which carries a byte offset, recovers the GEP
/// field index by `offsets.position(|o| o == offset)`.
pub struct AggLayout {
    pub struct_ty: String,
    pub offsets: Vec<u64>,
}

/// GID → aggregate layout, for the structs a stream constructs/reads. Only all-scalar-field structs
/// are modelled here (the flat HIR declines nested-aggregate/pointer fields anyway); an aggregate
/// absent from the map declines, keeping the AST path the oracle.
pub type AggMap = HashMap<TypeId, AggLayout>;

/// Build the GID→aggregate-layout map from the frozen registry's nominal layouts. Skips a struct
/// with any non-scalar field or an unmodelled (0-align stub) layout.
pub fn build_agg_map(registry: &ImmutableGlobalRegistry) -> AggMap {
    use crate::layout::FieldTy;
    let mut map = AggMap::new();
    for (gid, def) in &registry.layouts {
        if def.align_bytes == 0 || def.fields.is_empty() {
            continue; // unmodelled stub, or an enum/field-less type (no struct body to emit)
        }
        let mut field_tys = Vec::with_capacity(def.fields.len());
        let mut offsets = Vec::with_capacity(def.fields.len());
        let mut all_scalar = true;
        for f in &def.fields {
            match &f.ty {
                FieldTy::Scalar(e) => match mlir_scalar(e) {
                    Some(mt) => field_tys.push(mt.to_string()),
                    None => {
                        all_scalar = false;
                        break;
                    }
                },
                FieldTy::Nominal(_) | FieldTy::Opaque => {
                    all_scalar = false;
                    break;
                }
            }
            offsets.push(f.offset as u64);
        }
        if all_scalar {
            map.insert(
                *gid,
                AggLayout {
                    struct_ty: format!("!llvm.struct<({})>", field_tys.join(", ")),
                    offsets,
                },
            );
        }
    }
    map
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
    pub tensors: TensorMap,
    /// Names of payload-free (C-like) enums — an enum-typed value/param/return is a bare `i32`
    /// discriminant, not an aggregate (#227). Mirrors `ImmutableGlobalRegistry::enum_variants`.
    pub enums: std::collections::HashSet<String>,
}

impl EmitCtx {
    /// Build the callee + struct-layout maps from the registry. The tensor map is *not* in the
    /// registry (tensor types are structural); populate it separately from the lowerer's side table.
    pub fn from_registry(registry: &ImmutableGlobalRegistry) -> Self {
        Self {
            callees: build_callee_map(registry),
            aggs: build_agg_map(registry),
            tensors: TensorMap::new(),
            enums: registry
                .enum_variants
                .keys()
                .map(|s| s.as_ref().to_string())
                .collect(),
        }
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

/// Emit a whole module — every function as a concatenated bare `func.func` — or `None` if *any*
/// function is outside the current subset (module-level keep-green atomicity: a partially lowered
/// module is never emitted, so the AST path stays the oracle for the whole program). Callees + struct
/// layouts resolve through the frozen registry; `tensor_types` is the concatenation of each function's
/// lowerer side table (`LocalWorkerState::local_tensor_types`). Wrap the result in `module { … }`.
pub fn emit_module_mlir(
    funcs: &[(&Function, &[HirInstruction], &[TypeId])],
    registry: &ImmutableGlobalRegistry,
    tensor_types: &[(TypeId, ElementType, Vec<String>)],
    string_tables: &[&[String]],
) -> Option<String> {
    let mut ctx = EmitCtx::from_registry(registry);
    for (gid, elem, shape) in tensor_types {
        ctx.tensors
            .entry(*gid)
            .or_insert_with(|| (elem.clone(), shape.clone()));
    }
    let mut out = String::new();
    let mut globals = String::new();
    let mut calls: Vec<(String, Vec<String>, String)> = Vec::new();
    // Each function's string literals are numbered from a running module-wide base, so a `PrintStr`'s
    // `@".str.<n>"` reference (emitted with the same `str_base`) resolves the global emitted here.
    let mut str_base = 0usize;
    for (fi, (func, hir, types)) in funcs.iter().enumerate() {
        out += &emit_function_mlir(func, hir, types, &ctx, &mut calls, str_base)?;
        let strs = string_tables.get(fi).copied().unwrap_or(&[]);
        for (li, s) in strs.iter().enumerate() {
            globals += &emit_string_global(str_base + li, s);
        }
        str_base += strs.len();
    }
    // Prepend `private` declarations for any runtime print helpers the bodies call (the JIT links
    // their implementations; the AST path declares them the same way).
    let mut decls = String::new();
    for (name, sig) in [
        ("printMemrefF32", "(memref<*xf32>)"),
        ("printMemrefF64", "(memref<*xf64>)"),
        ("printMemrefI32", "(memref<*xi32>)"),
        ("printMemrefI64", "(memref<*xi64>)"),
        ("print_f32", "(f32) -> i32"),
        ("print_f64", "(f64) -> i32"),
        ("print_i32", "(i32) -> i32"),
        ("print_i64", "(i64) -> i32"),
        ("print_str", "(!llvm.ptr) -> i32"),
    ] {
        if out.contains(&format!("@{name}(")) {
            decls += &format!("  func.func private @{name}{sig}\n");
        }
    }
    // Declare any *called-but-undefined* callee (an `extern`: no `func.func @name` body emitted in this
    // module) as `func.func private`. The signature comes from the emitted `func.call`, so they match;
    // the JIT links the symbol (libm via `-lm`, `libvx_std_core`, ...). Deduped, in first-seen order.
    let defined: std::collections::HashSet<&str> =
        funcs.iter().map(|(f, _, _)| f.name.as_ref()).collect();
    let mut declared: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (name, arg_types, ret) in &calls {
        if defined.contains(name.as_str()) || !declared.insert(name.clone()) {
            continue;
        }
        let ret_sig = if ret.is_empty() {
            String::new()
        } else {
            format!(" -> {ret}")
        };
        decls += &format!(
            "  func.func private @{name}({}){ret_sig}\n",
            arg_types.join(", ")
        );
    }
    Some(globals + &decls + &out)
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
        body.push_str(&format!("  {c0} = arith.constant 0 : index\n"));
        body.push_str(&format!(
            "  {v} = vector.load {name}[{c0}] : {m}, {vecty}\n"
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
pub fn emit_function_mlir(
    func: &Function,
    hir: &[HirInstruction],
    types: &[TypeId],
    ctx: &EmitCtx,
    calls: &mut Vec<(String, Vec<String>, String)>,
    str_base: usize,
) -> Option<String> {
    // Signature (taken from the resolved AST signature; the *body* is flat-driven). A scalar param is
    // its element type; a tensor param is a memref recovered by GID from the side table (`ctx.tensors`
    // holds it — the param's `Load` recorded it). Anything else declines.
    let mut params = Vec::new();
    for (i, (_, ty)) in func.params.iter().enumerate() {
        let pty = if let Some(e) = scalar_of(ty) {
            mlir_scalar(&e)?.to_string()
        } else if let Some(et) = enum_scalar(ty, ctx) {
            et.to_string() // a payload-free enum param -> its i32 discriminant (#227)
        } else if let Some(gid) = tensor_gid_of(ty) {
            let (elem, shape) = ctx.tensors.get(&gid)?;
            tensor_memref_ty(elem, shape)?
        } else {
            return None;
        };
        params.push(format!("%arg{i}: {pty}"));
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
    } else if let Some(gid) = nominal_gid_of(&func.return_type) {
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
    // The memref type string for each register that holds a tensor (from `TensorAlloc`), so a
    // `TensorIndex`/`TensorStore` on it prints the right `memref<...>`.
    let mut mem_of: Vec<Option<String>> = vec![None; hir.len()];
    // For a scalar-element *place* register (a `TensorIndex` with `imm = 1`): the base memref name, an
    // `index`-typed index SSA name, and the base memref type — everything the following `TensorStore`
    // needs to emit `memref.store %v, %base[%idx]`.
    let mut place_of: Vec<Option<(String, String, String)>> = vec![None; hir.len()];
    // The `vector<Nxf32>` type of each register holding an elementwise (slice) result, so a row
    // `TensorStore` `vector.store`s it and a further elementwise passes it through.
    let mut vec_of: Vec<Option<String>> = vec![None; hir.len()];
    let mut body = String::new();
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
                } else if let Some((elem, shape)) = ctx.tensors.get(&gid) {
                    mem_of[idx] = tensor_memref_ty(elem, shape);
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
                    body += &format!("  {n} = {op} {a}, {b} : {mt}\n");
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
                    body += &format!("  {n} = memref.alloca() : memref<{mt}>\n");
                    names[idx] = n;
                    etypes[idx] = Some(e);
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
                if let Some(&Some(agg_gid)) = agg_of.get(ins.operand1.0 as usize) {
                    let agg = ctx.aggs.get(&agg_gid)?;
                    body += &format!(
                        "  llvm.store {val}, {slot} : {}, !llvm.ptr\n",
                        agg.struct_ty
                    );
                } else {
                    let e = elem_at(&etypes, ins.operand1.0)?;
                    let mt = mlir_scalar(&e)?;
                    body += &format!("  memref.store {val}, {slot}[] : memref<{mt}>\n");
                }
            }
            // Load a value back from a slot; the result type is the slot's element (this
            // instruction's own `type_idx`).
            Opcode::SlotLoad => {
                let e = ty_at(ins.type_idx.0)?;
                let mt = mlir_scalar(&e)?;
                let slot = names.get(ins.operand1.0 as usize)?;
                let n = format!("%v{idx}");
                body += &format!("  {n} = memref.load {slot}[] : memref<{mt}>\n");
                names[idx] = n;
                etypes[idx] = Some(e);
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
                // Return type: a scalar, or an `!llvm.struct` by value for a struct-returning callee
                // (#215). A void return isn't in this subset yet.
                let rt = if let Some(e) = &callee.ret {
                    mlir_scalar(e)?.to_string()
                } else if let Some(agg_gid) = callee.ret_agg {
                    ctx.aggs.get(&agg_gid)?.struct_ty.clone()
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
                    // A scalar arg is its element type; a tensor arg is its memref type.
                    let at = if let Some(e) = elem_at(&etypes, *a) {
                        mlir_scalar(&e)?.to_string()
                    } else {
                        mem_of.get(*a as usize)?.clone()?
                    };
                    arg_types.push(at);
                }
                let nm = format!("%v{idx}");
                body += &format!(
                    "  {nm} = func.call @{}({}) : ({}) -> {rt}\n",
                    callee.name,
                    arg_names.join(", "),
                    arg_types.join(", "),
                );
                // Record the callee's signature so the module emitter can declare it if it is a
                // called-but-undefined symbol (an `extern`): the private decl's signature is taken from
                // the emitted call, so they match by construction.
                calls.push((callee.name.clone(), arg_types.clone(), rt.clone()));
                names[idx] = nm;
                if let Some(e) = &callee.ret {
                    etypes[idx] = Some(e.clone());
                }
                // else: a struct value tracked by `names[idx]`; a following `Store` spills it to a slot
                // and a `Ret` returns it directly (#215).
            }
            // Store a scalar into a struct field (no result). `operand1` is the struct slot pointer,
            // `operand2` the value, `imm` the field's byte offset. GEP to the field, then `llvm.store`;
            // the field index comes from matching the offset against the layout, the value type from
            // the stored register's tracked type.
            Opcode::FieldStore => {
                let gid = (*agg_of.get(ins.operand1.0 as usize)?)?;
                let agg = ctx.aggs.get(&gid)?;
                let field_idx = agg.offsets.iter().position(|&o| o == ins.imm)?;
                let fty = mlir_scalar(&elem_at(&etypes, ins.operand2.0)?)?;
                let slot = names.get(ins.operand1.0 as usize)?;
                let val = names.get(ins.operand2.0 as usize)?;
                let p = format!("%p{idx}");
                body += &format!(
                    "  {p} = llvm.getelementptr {slot}[0, {field_idx}] : (!llvm.ptr) -> !llvm.ptr, {}\n",
                    agg.struct_ty
                );
                body += &format!("  llvm.store {val}, {p} : {fty}, !llvm.ptr\n");
            }
            // Load a scalar struct field. `operand1` is the struct slot, `imm` the field's byte offset,
            // and this instruction's own `type_idx` the field's scalar type. GEP to the field, then
            // `llvm.load`.
            Opcode::FieldLoad => {
                let gid = (*agg_of.get(ins.operand1.0 as usize)?)?;
                let agg = ctx.aggs.get(&gid)?;
                let field_idx = agg.offsets.iter().position(|&o| o == ins.imm)?;
                let e = ty_at(ins.type_idx.0)?;
                let mt = mlir_scalar(&e)?;
                let slot = names.get(ins.operand1.0 as usize)?;
                let p = format!("%p{idx}");
                let n = format!("%v{idx}");
                body += &format!(
                    "  {p} = llvm.getelementptr {slot}[0, {field_idx}] : (!llvm.ptr) -> !llvm.ptr, {}\n",
                    agg.struct_ty
                );
                body += &format!("  {n} = llvm.load {p} : !llvm.ptr -> {mt}\n");
                names[idx] = n;
                etypes[idx] = Some(e);
            }
            // Allocate a tensor buffer (`Tensor<T>([..])`): a static `memref` of the shape recovered
            // from the side table by GID. Its register is tracked in `mem_of` for later index/store.
            Opcode::TensorAlloc => {
                let gid = *types.get(ins.type_idx.0 as usize)?;
                let (elem, shape) = ctx.tensors.get(&gid)?;
                let memty = tensor_memref_ty(elem, shape)?;
                let n = format!("%v{idx}");
                body += &format!("  {n} = memref.alloc() : {memty}\n");
                names[idx] = n;
                mem_of[idx] = Some(memty);
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
                    let result_ty =
                        format!("memref<{dimx}{et}, strided<[{strides_s}], offset: ?>>");
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
                let v0 = format!("%vl{idx}");
                body += &format!("  {v0} = vector.load {s0}[{c0}] : {m0}, {vecty}\n");
                let (reduce_in, kind) = match ins.imm {
                    0 => {
                        let s1 = names.get(ins.operand2.0 as usize)?.clone();
                        let m1 = mem_of.get(ins.operand2.0 as usize)?.clone()?;
                        let v1 = format!("%vr{idx}");
                        body += &format!("  {v1} = vector.load {s1}[{c0}] : {m1}, {vecty}\n");
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
                    body += &format!("  {c0} = arith.constant 0 : index\n");
                    body += &format!("  vector.store {vecname}, {dst}[{c0}] : {rowty}, {vecty}\n");
                } else {
                    return None;
                }
            }
            // Move a tensor to a memory space: `operand1` is the source, `imm` the target space's
            // dispatch id, `type_idx` the result tensor (same element + shape). Emits `vx.transfer`
            // (generic form) with `target_topology`; the vx→standard lowering turns it into an
            // alloc + `memref.copy` (it ignores the source's layout suffix, so the result is a plain
            // `memref<NxT>`). The extra scheduling attrs the AST adds (`space`, `granule`, …) are
            // discardable metadata and don't affect lowering.
            Opcode::Transfer => {
                let result_gid = *types.get(ins.type_idx.0 as usize)?;
                let (elem, shape) = ctx.tensors.get(&result_gid)?;
                let dstty = tensor_memref_ty(elem, shape)?;
                let src = names.get(ins.operand1.0 as usize)?.clone();
                let srcty = mem_of.get(ins.operand1.0 as usize)?.clone()?;
                let n = format!("%v{idx}");
                body += &format!(
                    "  {n} = \"vx.transfer\"({src}) {{target_topology = {} : i32}} : ({srcty}) -> {dstty}\n",
                    ins.imm
                );
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
                body += &format!("  }}) {{topology = {topo} : i32}} : () -> ()\n");
                terminated = false;
            }
            // Print a string literal (no result): take the address of the module-level global emitted
            // for this string (`@".str.<n>"`, `n = str_base + imm`) and call the `@print_str` runtime
            // helper. `emit_module_mlir` emits the global's bytes and the helper's `private` decl.
            Opcode::PrintStr => {
                let n = str_base + ins.imm as usize;
                let p = format!("%pstrp{idx}");
                let r = format!("%pstr{idx}");
                body += &format!("  {p} = llvm.mlir.addressof @\".str.{n}\" : !llvm.ptr\n");
                body += &format!("  {r} = func.call @print_str({p}) : (!llvm.ptr) -> i32\n");
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
        "func.func @{}({}){} {{\n",
        func.name,
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
        )
        .expect("emits flat MLIR");

        use melior::ir::operation::OperationLike;
        let registry = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&registry);
        let context = melior::Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        let module = melior::ir::Module::parse(&context, &format!("module {{\n{mlir}}}\n"))
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
        let mlir = emit_module_mlir(&funcs, &session.registry, &tensor_types, &string_tables)
            .expect("emits flat module");

        use melior::ir::operation::OperationLike;
        let dialects = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&dialects);
        let context = melior::Context::new();
        context.append_dialect_registry(&dialects);
        context.load_all_available_dialects();
        crate::codegen::register_vx_dialect(&context); // for `vx.transfer`
        let module = melior::ir::Module::parse(&context, &format!("module {{\n{mlir}}}\n"))
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
