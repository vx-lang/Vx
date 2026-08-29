use super::*;

//===- ast.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file defines the Abstract Syntax Tree (AST) structures for the Vx language.
// It contains the enums and structs representing expressions, statements, types,
// and declarations, serving as the foundational data model for the entire frontend
// of the compiler.
//
//===----------------------------------------------------------------------===//
use crate::symbol::Symbol;
use crate::syntax;

/// The region field of a lifetime signature: how a `&T`'s scope depth is encoded.
///
/// Here rather than in the borrow checker because `region_id` is a field on
/// [`Type::Borrow`], and the parser has to write the "unset" sentinel into it long before
/// any checking happens. The rules that *interpret* these values stay in `crate::borrow`.
pub const REGION_MASK: u64 = 0x0FFF;

/// The reserved "region not yet assigned" sentinel: the maximum value the 12-bit region field can
/// hold. A parsed reference type carries it until a scope depth is assigned (see
/// `src/parser/types.rs`), and it surfaces in generic-deduction diagnostics as `region_id: 4095`.
///
/// It is **not** a scope depth — `verify_subtyping_bounds` treats it as a wildcard, never comparing
/// it numerically, so an unset region neither satisfies nor fails subtyping by accident (#267). It
/// is reserved: real depths are clamped to [`REGION_MAX`] so none ever equals the sentinel. Anyone
/// narrowing this field (e.g. #265 shrinking slot 0 to 9 bits) must keep a reserved sentinel at the
/// new field's maximum and clamp real depths below it — the numeric value must never be trusted as a
/// region. See `docs/discussions/borrow_checker_architecture.md` §2.
pub const REGION_UNSET: u64 = REGION_MASK;

/// The largest assignable real region (scope depth): one below the [`REGION_UNSET`] sentinel, so a
/// genuine depth can never be mistaken for "unset". Deeper nesting is clamped to this (shortest-lived
/// valid region) rather than overflowing into the sentinel.
pub const REGION_MAX: u64 = REGION_MASK - 1;

/// Slot 0 (the return slot) reserves its top 3 region bits for the return-provenance code (#265), so
/// its region field is only **9 bits** — slots 1-3 keep the full 12. Any rule that reads slot 0's
/// region must mask with this, not [`REGION_MASK`]: reading the wide field would fold the provenance
/// bits into the lifetime and corrupt the comparison (exactly the truncation #267 warned of).
pub const REGION_MASK_0: u64 = 0x01FF;

/// The slot-0 counterpart of [`REGION_UNSET`]: the maximum of the narrowed 9-bit return-slot region
/// is its reserved "unset" sentinel (#265/#267). Recognised as a wildcard exactly like [`REGION_UNSET`].
pub const REGION_UNSET_0: u64 = REGION_MASK_0;

/// The largest assignable real region in slot 0 — one below [`REGION_UNSET_0`]. A return lifetime is
/// by construction a parameter's or `'static`, so it never needs deep nesting; anything deeper clamps
/// here rather than colliding with the sentinel or the provenance bits.
pub const REGION_MAX_0: u64 = REGION_MASK_0 - 1;

/// The element type of a scalar, or `None` for a generic or a non-scalar.
///
/// Here rather than in either backend because it asks a question about the AST, and both the
/// flattener and the flat emitter carried an identical copy of it.
pub fn scalar_of(ty: &Type) -> Option<ElementType> {
    match ty {
        Type::Scalar(ElementType::Generic(_)) => None,
        Type::Scalar(e) => Some(e.clone()),
        _ => None,
    }
}

/// Whether a return type is `void` — spelled `Type::Struct("void", _)`. A void-returning call
/// produces no result value; the flat emitter prints `-> ()`.
pub fn is_void_ty(ty: &Type) -> bool {
    matches!(ty, Type::Struct(n, _) if n.as_ref() == "void" || n.as_ref() == "none")
}

/// A module's top-level names mapped to their GIDs, and every module's table by module path.
///
/// Here rather than in `crate::resolver` because the AST's own `resolve_names` walk threads them,
/// and the AST cannot depend on the resolver that consumes it.
pub type SymbolTable = std::collections::HashMap<Symbol, crate::gid::TypeId>;
pub type SymbolMap = std::collections::HashMap<Symbol, SymbolTable>;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Default, Hash)]
pub struct Span {
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

/// Do two topology indices denote the same device?
///
/// By VALUE, not by AST node. `PartialEq` on `Topology` used to be derived, so two spellings of
/// device 0 were unequal whenever their index literals differed in any field -- and they do,
/// routinely: `Topology::gpu()` below and `arch.rs` build the literal with `ty: None`, while
/// `check_transfer_expr` writes `ty: Some(I32)` inline, and a written-down `Topology::GPU[0]`
/// picks up `Some(I32)` when the checker infers its literal's type.
///
/// The visible cost was that a declared `Pinned<_, Topology::GPU[0]>` never matched what
/// `transfer` produces, so a placed tensor could not be passed to a function or stored in a
/// struct field at all (Vx#355). Annotating the one constructor would have fixed that program
/// and left the class open, since the two conventions are still spread across the tree.
///
/// This is also what `docs/discussions/brainstorming/hardware_monad_topology.md` asks for:
/// "topology identity = (registered-kind, index-term), with index equality decided by the
/// const-evaluator". Literals are decided here; anything else falls back to structural
/// equality, which is what the derive did for every case.
fn topology_index_eq(a: &Expr, b: &Expr) -> bool {
    if let (Expr::Number(x), Expr::Number(y)) = (a, b) {
        return match (
            x.value.as_ref().parse::<i128>(),
            y.value.as_ref().parse::<i128>(),
        ) {
            (Ok(xi), Ok(yi)) => xi == yi,
            // Not integers after all: compare how they were spelled rather than guessing.
            _ => x.value == y.value,
        };
    }
    a == b
}

/// Two topologies are equal when they name the SAME DEVICE.
///
/// Hand-written rather than derived, and that is a semantic choice, not a style one. The
/// derive compared the index EXPRESSION as an AST node, so device 0 was not device 0 whenever
/// the two literals had been built differently — which happens constantly, because the tree
/// contains both conventions (`ty: None` from `Topology::gpu()` and `arch.rs`, `ty: Some(I32)`
/// from `check_transfer_expr` and from any annotation the checker has inferred). Spans counted
/// too, so the same device written in two places was two devices (Vx#355).
///
/// GUARANTEES
///
/// - It is a full equivalence relation: reflexive, symmetric, transitive. Reflexivity is the
///   one that needs a test rather than an argument — see below.
/// - Same kind, same index value ⇒ equal, however each side's literal was spelled or
///   annotated. `gpu(0)` from a constructor equals `Topology::GPU[0]` read from source.
/// - Different kind ⇒ not equal, even at the same index. `GPU[0] != NPU[0]`.
/// - Different index value ⇒ not equal. The index is what makes two devices nameable at all
///   ("prefill here, decode there"), so collapsing it would be worse than the bug this fixes.
///
/// REQUIRES
///
/// - **Any `Hash` on `Topology` must agree with this.** Hashing the index AST would put two
///   equal topologies in different buckets, which is a broken `HashMap`, not a slow one.
///   `Topology` deliberately has no `Hash`; the hashable identity is [`TopologyKind`], and
///   note that kind DROPS the index, so a kind-keyed map conflates every device of a kind.
/// - **A new variant needs an arm here.** The match is on a PAIR, which Rust cannot
///   exhaustiveness-check, so a forgotten variant silently compares unequal to itself. The
///   `_` arm asserts against exactly that and `every_topology_variant_is_equal_to_itself`
///   fires it.
/// - **`Eq` is not implemented.** The relation would justify it, but the non-literal fallback
///   defers to `Expr`'s derived `PartialEq`, so the claim would only be as good as that.
///
/// DOES NOT GUARANTEE
///
/// - A non-literal index is still compared structurally, so `GPU[i]` and `GPU[j]` are unequal
///   even when `i == j` at runtime. Deciding those needs the const-evaluator, which is what
///   `docs/discussions/brainstorming/hardware_monad_topology.md` means by "index equality
///   decided by the const-evaluator / prover". Literals are the part that is decided today.
/// - Two literals that are not both parseable integers fall back to comparing how they were
///   spelled, rather than guessing what they denote.
impl PartialEq for Topology {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Topology::CPU, Topology::CPU)
            | (Topology::AMX, Topology::AMX)
            | (Topology::ANE, Topology::ANE)
            | (Topology::CpuAvx512, Topology::CpuAvx512)
            | (Topology::CpuNeon, Topology::CpuNeon)
            | (Topology::Current, Topology::Current) => true,
            (Topology::NPU(a), Topology::NPU(b))
            | (Topology::AccCore(a), Topology::AccCore(b))
            | (Topology::GPU(a), Topology::GPU(b)) => topology_index_eq(a, b),
            (Topology::Custom(a), Topology::Custom(b)) => a == b,
            (Topology::Slice(ba, sa, ea), Topology::Slice(bb, sb, eb)) => {
                ba == bb && topology_index_eq(sa, sb) && topology_index_eq(ea, eb)
            }
            // Different variants are different topologies. The assert is what a hand-written
            // `PartialEq` costs: the derive covered every variant automatically, and a match on
            // a PAIR cannot be exhaustiveness-checked without writing all N^2 cross arms. So a
            // variant added above and forgotten here would land in this arm against ITSELF and
            // report not-equal -- a topology that is not itself, silently, which would take a
            // very confusing bug report to find. Same discriminant reaching here is always a
            // missing arm, never a real answer.
            _ => {
                debug_assert!(
                    std::mem::discriminant(self) != std::mem::discriminant(other),
                    "Topology::eq has no arm for this variant, so it compares unequal to itself; \
                     add it above"
                );
                false
            }
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub enum Topology {
    CPU,
    NPU(Box<Expr>),
    AccCore(Box<Expr>),
    AMX,
    ANE,
    /// A discrete GPU, by device index. Bare `Topology::gpu(0)` in source means
    /// device 0; `Topology::gpu(0)[1]` names the second.
    ///
    /// The index is what makes more than one of them nameable. Until it existed
    /// a program could not say "prefill here, decode there" at all -- every
    /// `spawn on(Topology::gpu(0))` denoted the same anonymous device, and
    /// `Topology::gpu(0)[1]` was a parse error.
    GPU(Box<Expr>),
    CpuAvx512,
    CpuNeon,
    Slice(Box<Topology>, Box<Expr>, Box<Expr>), // For NPU[0..4] etc.
    /// A user-defined topology, identified by name and described in the topology
    /// registry (`crate::arch::topology_descriptor`). This is the open identity that
    /// lets users add their own topologies without editing the language.
    Custom(Symbol),
    Current,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum MemorySpace {
    CPUDRAM,
    NPUHBM,
    GpuHbm,
    LocalSRAM,
    NicRam,
    RemoteHbm,
    /// A user-defined memory space, identified by name. Lets a custom topology declare a
    /// novel memory (not one of the built-ins). See `Topology::Custom`.
    Custom(Symbol),
}

impl MemorySpace {
    /// Map a surface name (as written after `Memory::`) to a space: the built-in names map to
    /// their variants; any other identifier is a user-defined `Custom` space. Single source of
    /// truth shared by the parser and the memory hierarchy.
    pub fn from_name(name: &str) -> MemorySpace {
        match name {
            "CPU_DRAM" => MemorySpace::CPUDRAM,
            "NPU_HBM" => MemorySpace::NPUHBM,
            "GPU_HBM" => MemorySpace::GpuHbm,
            "Local_SRAM" => MemorySpace::LocalSRAM,
            "NIC_RAM" => MemorySpace::NicRam,
            "Remote_HBM" => MemorySpace::RemoteHbm,
            other => MemorySpace::Custom(Symbol::from(other)),
        }
    }

    /// The surface name of a space (the inverse of `from_name`), for diagnostics.
    pub fn name(&self) -> String {
        match self {
            MemorySpace::CPUDRAM => "CPU_DRAM".to_string(),
            MemorySpace::NPUHBM => "NPU_HBM".to_string(),
            MemorySpace::GpuHbm => "GPU_HBM".to_string(),
            MemorySpace::LocalSRAM => "Local_SRAM".to_string(),
            MemorySpace::NicRam => "NIC_RAM".to_string(),
            MemorySpace::RemoteHbm => "Remote_HBM".to_string(),
            MemorySpace::Custom(s) => s.as_ref().to_string(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum TopologyKind {
    CPU,
    NPU,
    AccCore,
    AMX,
    ANE,
    GPU,
    CpuAvx512,
    CpuNeon,
    Slice,
    Custom(Symbol),
    Current,
}

impl Topology {
    /// `Topology::gpu(0)[index]`.
    ///
    /// Most callers want device 0 -- a site that has no device to name, or a
    /// test that does not care which one. Spelling it out keeps those honest:
    /// picking 0 is a choice, and one that is wrong on a machine where the
    /// program meant a particular device.
    pub fn gpu(index: i64) -> Topology {
        Topology::GPU(Box::new(Expr::Number(crate::syntax::NumberExpr::new(
            index.to_string(),
            None,
            crate::syntax::Span::default(),
        ))))
    }

    pub fn kind(&self) -> TopologyKind {
        match self {
            Topology::CPU => TopologyKind::CPU,
            Topology::NPU(_) => TopologyKind::NPU,
            Topology::AccCore(_) => TopologyKind::AccCore,
            Topology::AMX => TopologyKind::AMX,
            Topology::ANE => TopologyKind::ANE,
            Topology::GPU(_) => TopologyKind::GPU,
            Topology::CpuAvx512 => TopologyKind::CpuAvx512,
            Topology::CpuNeon => TopologyKind::CpuNeon,
            Topology::Slice(..) => TopologyKind::Slice,
            Topology::Custom(name) => TopologyKind::Custom(name.clone()),
            Topology::Current => TopologyKind::Current,
        }
    }

    pub fn is_same_kind(&self, other: &Self) -> bool {
        self.kind() == other.kind()
    }

    /// The literal device index, when the topology has one and it is a literal.
    pub fn device_index(&self) -> Option<i64> {
        let literal = |e: &Expr| match e {
            Expr::Number(n) => n.value.parse::<i64>().ok(),
            _ => None,
        };
        match self {
            Topology::NPU(e) | Topology::AccCore(e) | Topology::GPU(e) => literal(e),
            _ => None,
        }
    }

    /// Whether two topologies denote the same device.
    ///
    /// Not the derived `PartialEq`, which compares the index *expression* --
    /// span and inferred type included -- so the same topology written in two
    /// places can compare unequal while naming one device. `Topology::GPU` in a
    /// function signature and at its call site produced exactly that: identical
    /// text, one index stamped `i32` and one not.
    ///
    /// Kind first, then index by value when both are literals. A non-literal
    /// index is not resolvable here, so those fall back to structural equality
    /// rather than being assumed equal.
    pub fn same_device(&self, other: &Self) -> bool {
        if self.kind() != other.kind() {
            return false;
        }
        match (self.device_index(), other.device_index()) {
            (Some(a), Some(b)) => a == b,
            _ => self == other,
        }
    }

    /// Substitute const generics into this topology's *index* expressions, so `NPU[R]` in a
    /// `shard<const R : i32>` body becomes `NPU[5]` at the `shard<5>` instantiation. Without this
    /// a monomorphized index stayed a bare identifier, `topology_dispatch_id` fell back to device
    /// 0, and every shard of a parallel program silently targeted the same device (#284).
    ///
    /// Substitutes only the indices; a topology *variable* (`<D: Topology>`) is bound separately
    /// during monomorphization, since it maps to a `Topology` rather than a `Type`.
    pub fn substitute(&self, mapping: &std::collections::HashMap<Symbol, Type>) -> Topology {
        match self {
            Topology::NPU(e) => Topology::NPU(Box::new(e.substitute(mapping))),
            Topology::GPU(e) => Topology::GPU(Box::new(e.substitute(mapping))),
            Topology::AccCore(e) => Topology::AccCore(Box::new(e.substitute(mapping))),
            Topology::Slice(base, start, end) => Topology::Slice(
                Box::new(base.substitute(mapping)),
                Box::new(start.substitute(mapping)),
                Box::new(end.substitute(mapping)),
            ),
            other => other.clone(),
        }
    }

    /// The surface spelling of a topology, for diagnostics — `NPU[0]`, `NPU[0..144]`, `RubinCPX`.
    /// Diagnostics previously interpolated the `Debug` form, which rendered an indexed topology as
    /// `NPU(Number(NumberExpr { value: "0", ty: Some(I32), span: .. }))` — unreadable in a headline
    /// error. An index that is not a literal falls back to its own display (`NPU[i]`). (#253)
    pub fn display_name(&self) -> String {
        let idx = |e: &Expr| -> String {
            match e {
                Expr::Number(n) => n.value.to_string(),
                Expr::Identifier(id) => id.name.to_string(),
                _ => "?".to_string(),
            }
        };
        match self {
            Topology::CPU => "CPU".to_string(),
            Topology::NPU(e) => format!("NPU[{}]", idx(e)),
            Topology::AccCore(e) => format!("AccCore[{}]", idx(e)),
            Topology::AMX => "AMX".to_string(),
            Topology::ANE => "ANE".to_string(),
            Topology::GPU(e) => format!("GPU[{}]", idx(e)),
            Topology::CpuAvx512 => "CpuAvx512".to_string(),
            Topology::CpuNeon => "CpuNeon".to_string(),
            Topology::Slice(base, start, end) => {
                format!("{}[{}..{}]", base.display_name(), idx(start), idx(end))
            }
            Topology::Custom(name) => name.to_string(),
            Topology::Current => "Current".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ElementType {
    F16,
    F32,
    F64,
    BF16,
    F8E4M3,
    F8E5M2,
    I4,
    U4,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    I128,
    U128,
    Bool,
    Generic(Symbol),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    /// A statically shaped tensor. Its extents are part of the type, so `[2, 3]` and `[4, 5]`
    /// are different types and get different monomorphs (Vx#401). A dims-less spelling still
    /// parses to this and still means "shape unknown"; Vx#399 moves that reading to `DynTensor`
    /// and makes the dims-less `Tensor` unspellable.
    Tensor(ElementType, Vec<Expr>, Option<Topology>),
    /// A tensor whose shape is not known until run time — a model config's dimensions, a
    /// shape-polymorphic library function. Carries no extents by construction, so nothing can
    /// read a shape off it that a static check would then trust (Vx#399). The verified variant
    /// (a declared upper bound, Vx#245) is a later addition to this variant.
    DynTensor(ElementType, Option<Topology>),
    Matrix,
    Ref(Box<Type>, MemorySpace),
    Borrow {
        inner: Box<Type>,
        mem_space: Option<MemorySpace>,
        is_mut: bool,
        region_id: usize,
    }, // (type, mem_space, is_mut, region_id)
    Pointer(Box<Type>, Option<MemorySpace>, bool), // (type, mem_space, is_mut)
    Scalar(ElementType),
    Struct(Symbol, Option<crate::gid::TypeId>),
    Enum(Symbol, Option<crate::gid::TypeId>),
    Verified(Box<Type>),
    Pinned(Box<Type>, Topology),
    Generic(Symbol, Option<crate::gid::TypeId>), // e.g. T
    GenericInstance(Box<Type>, Vec<Type>),       // e.g. Config<f32>
    Module(Symbol, std::collections::HashMap<Symbol, Type>), // (path, exported_symbols)
    Simd(ElementType, usize),                    // e.g. <4 x f32>
    Function(Vec<Type>, Box<Type>),              // e.g. fn(i32, f32) -> f32
    Closure(Vec<Type>, Box<Type>),               // Fat pointer closure type
    Const(Box<syntax::expr::Expr>),              // E.g., generic const argument like `10`
    Unknown,
}

pub trait Mangle {
    fn mangle_to(&self, w: &mut dyn std::fmt::Write) -> std::fmt::Result;
    fn mangle(&self) -> String {
        let mut s = String::new();
        let _ = self.mangle_to(&mut s);
        s
    }
}

impl Type {
    pub fn is_linear(&self) -> bool {
        matches!(
            self,
            Type::Ref(_, _)
                | Type::Tensor(_, _, _)
                | Type::Matrix
                | Type::Verified(_)
                | Type::Pinned(_, _)
                | Type::Struct(_, _)
                | Type::Enum(_, _)
        )
    }

    pub fn substitute(&self, mapping: &std::collections::HashMap<Symbol, Type>) -> Type {
        match self {
            Type::Generic(name, _) => {
                if let Some(concrete) = mapping.get(name) {
                    concrete.clone()
                } else {
                    self.clone()
                }
            }
            Type::Const(expr) => Type::Const(Box::new(expr.substitute(mapping))),
            Type::GenericInstance(base, args) => {
                let new_base = base.substitute(mapping);
                let new_args = args.iter().map(|a| a.substitute(mapping)).collect();
                Type::GenericInstance(Box::new(new_base), new_args)
            }
            Type::Borrow {
                inner,
                mem_space,
                is_mut,
                region_id,
            } => Type::Borrow {
                inner: Box::new(inner.substitute(mapping)),
                mem_space: mem_space.clone(),
                is_mut: *is_mut,
                region_id: *region_id,
            },
            Type::Pointer(inner, mem, is_mut) => {
                Type::Pointer(Box::new(inner.substitute(mapping)), mem.clone(), *is_mut)
            }
            Type::Tensor(el_ty, dims, top) => {
                let new_el_ty = if let ElementType::Generic(ref name) = el_ty {
                    if let Some(Type::Scalar(concrete_el)) = mapping.get(name) {
                        concrete_el.clone()
                    } else {
                        el_ty.clone()
                    }
                } else {
                    el_ty.clone()
                };
                let new_dims = dims.iter().map(|d| d.substitute(mapping)).collect();
                Type::Tensor(new_el_ty, new_dims, top.clone())
            }
            Type::Ref(inner, mem) => Type::Ref(Box::new(inner.substitute(mapping)), mem.clone()),
            Type::Verified(inner) => Type::Verified(Box::new(inner.substitute(mapping))),
            Type::Pinned(inner, top) => {
                Type::Pinned(Box::new(inner.substitute(mapping)), top.clone())
            }
            Type::Simd(el_ty, n) => {
                let new_el_ty = if let ElementType::Generic(ref name) = el_ty {
                    if let Some(Type::Scalar(concrete_el)) = mapping.get(name) {
                        concrete_el.clone()
                    } else {
                        el_ty.clone()
                    }
                } else {
                    el_ty.clone()
                };
                Type::Simd(new_el_ty, *n)
            }
            Type::Function(params, ret) => {
                let new_params = params.iter().map(|p| p.substitute(mapping)).collect();
                let new_ret = Box::new(ret.substitute(mapping));
                Type::Function(new_params, new_ret)
            }
            Type::Closure(params, ret) => {
                let new_params = params.iter().map(|p| p.substitute(mapping)).collect();
                let new_ret = Box::new(ret.substitute(mapping));
                Type::Closure(new_params, new_ret)
            }
            Type::Unknown => Type::Unknown,
            _ => self.clone(),
        }
    }

    // Mangle trait is implemented below

    pub fn topology(&self) -> Option<Topology> {
        match self {
            Type::Tensor(_, _, top) => top.clone(),
            _ => None,
        }
    }
}

impl ElementType {
    /// Whether this is a floating-point element type (`f16`/`f32`/`f64`/`bf16`/`f8e4m3`/`f8e5m2`).
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            ElementType::F16
                | ElementType::F32
                | ElementType::F64
                | ElementType::BF16
                | ElementType::F8E4M3
                | ElementType::F8E5M2
        )
    }

    /// Storage width in bits — the single source of truth for element widths, from which
    /// `hir::memory::element_bits` (dense bits) and `layout::scalar_size_align` (padded bytes =
    /// `ceil(bits/8)`) both derive, so the two can no longer drift (they disagreed on `I4` before:
    /// 4 dense bits vs 1 padded byte, both from independent tables). `None` for an un-instantiated
    /// generic. Adding a numeric format sets its width here, in one place. (P1-4a; see the
    /// heterogeneous gap analysis §9.8.1.)
    pub fn bits(&self) -> Option<u32> {
        Some(match self {
            ElementType::Bool => 1,
            ElementType::I4 | ElementType::U4 => 4,
            ElementType::I8 | ElementType::U8 | ElementType::F8E4M3 | ElementType::F8E5M2 => 8,
            ElementType::F16 | ElementType::BF16 | ElementType::I16 | ElementType::U16 => 16,
            ElementType::F32 | ElementType::I32 | ElementType::U32 => 32,
            ElementType::F64 | ElementType::I64 | ElementType::U64 => 64,
            ElementType::I128 | ElementType::U128 => 128,
            ElementType::Generic(_) => return None,
        })
    }
}

impl std::fmt::Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElementType::F16 => write!(f, "f16"),
            ElementType::F32 => write!(f, "f32"),
            ElementType::F64 => write!(f, "f64"),
            ElementType::BF16 => write!(f, "bf16"),
            ElementType::F8E4M3 => write!(f, "f8e4m3"),
            ElementType::F8E5M2 => write!(f, "f8e5m2"),
            ElementType::I4 => write!(f, "i4"),
            ElementType::U4 => write!(f, "u4"),
            ElementType::I8 => write!(f, "i8"),
            ElementType::U8 => write!(f, "u8"),
            ElementType::I16 => write!(f, "i16"),
            ElementType::U16 => write!(f, "u16"),
            ElementType::I32 => write!(f, "i32"),
            ElementType::U32 => write!(f, "u32"),
            ElementType::I64 => write!(f, "i64"),
            ElementType::U64 => write!(f, "u64"),
            ElementType::I128 => write!(f, "i128"),
            ElementType::U128 => write!(f, "u128"),
            ElementType::Bool => write!(f, "Bool"),
            ElementType::Generic(g) => write!(f, "{}", g),
        }
    }
}

impl std::str::FromStr for ElementType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f16" => Ok(ElementType::F16),
            "f32" => Ok(ElementType::F32),
            "f64" => Ok(ElementType::F64),
            "bf16" => Ok(ElementType::BF16),
            "f8e4m3" => Ok(ElementType::F8E4M3),
            "f8e5m2" => Ok(ElementType::F8E5M2),
            "i4" => Ok(ElementType::I4),
            "u4" => Ok(ElementType::U4),
            "i8" => Ok(ElementType::I8),
            "u8" => Ok(ElementType::U8),
            "i16" => Ok(ElementType::I16),
            "u16" => Ok(ElementType::U16),
            "i32" => Ok(ElementType::I32),
            "u32" => Ok(ElementType::U32),
            "i64" => Ok(ElementType::I64),
            "u64" => Ok(ElementType::U64),
            "i128" => Ok(ElementType::I128),
            "u128" => Ok(ElementType::U128),
            "bool" | "Bool" => Ok(ElementType::Bool),
            _ => Err(format!("Unknown element type '{}'", s)),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Scalar(el) => write!(f, "{}", el),
            Type::Struct(name, _) => write!(f, "{}", name),
            Type::Enum(name, _) => write!(f, "{}", name),
            Type::Pointer(inner, _, is_mut) => {
                if *is_mut {
                    write!(f, "*mut {}", inner)
                } else {
                    write!(f, "*const {}", inner)
                }
            }
            Type::Borrow { inner, is_mut, .. } => {
                if *is_mut {
                    write!(f, "&mut {}", inner)
                } else {
                    write!(f, "&{}", inner)
                }
            }
            Type::GenericInstance(base, args) => {
                write!(f, "{}<", base)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ">")
            }
            Type::Generic(name, _) => write!(f, "{}", name),
            Type::Const(expr) => {
                if let syntax::expr::Expr::Number(n) = &**expr {
                    write!(f, "{}", n.value)
                } else if let syntax::expr::Expr::StringLiteral(s) = &**expr {
                    write!(f, "\"{}\"", s.value)
                } else {
                    write!(f, "{{{:?}}}", expr)
                }
            }
            Type::Unknown => write!(f, "?"),
            Type::Function(params, ret) => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Type::Closure(params, ret) => {
                write!(f, "|")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, "| -> {}", ret)
            }
            _ => write!(f, "{:?}", self), // Fallback for complex types
        }
    }
}

impl Mangle for Type {
    fn mangle_to(&self, w: &mut dyn std::fmt::Write) -> std::fmt::Result {
        match self {
            Type::Scalar(el) => el.mangle_to(w),
            Type::Simd(el, n) => {
                write!(w, "Simd$")?;
                el.mangle_to(w)?;
                write!(w, "${}", n)
            }
            Type::Pointer(inner, _, is_mut) => {
                write!(w, "ptr${}$", if *is_mut { "mut" } else { "const" })?;
                inner.mangle_to(w)
            }
            Type::Borrow { inner, is_mut, .. } => {
                write!(w, "ref${}$", if *is_mut { "mut" } else { "const" })?;
                inner.mangle_to(w)
            }
            Type::Ref(inner, _) => {
                write!(w, "ref$")?;
                inner.mangle_to(w)
            }
            // Extents are part of the identity: `[2, 3]` and `[4, 5]` are different types that
            // lower to different memrefs, so a generic instantiated at both needs two symbols
            // rather than one body serving both (Vx#401). Rank alone was too coarse to separate
            // them. Topology stays excluded. A dimension the canonical forms below cannot spell
            // contributes `d`, which is why two such dimensions can still share a name.
            Type::Tensor(el, dims, _) => {
                write!(w, "Tensor$")?;
                el.mangle_to(w)?;
                for (i, d) in dims.iter().enumerate() {
                    write!(w, "{}", if i == 0 { "$" } else { "x" })?;
                    match d {
                        Expr::Number(n) => write!(w, "{}", n.value)?,
                        Expr::Identifier(id) => write!(w, "{}", id.name)?,
                        _ => write!(w, "d")?,
                    }
                }
                Ok(())
            }
            // No extents to carry: a dynamic tensor's shape is not part of its identity because
            // it has none until run time (Vx#399).
            Type::DynTensor(el, _) => {
                write!(w, "DynTensor$")?;
                el.mangle_to(w)
            }
            Type::Matrix => write!(w, "Matrix"),
            Type::Struct(name, _) => write!(w, "{}", name),
            Type::Enum(name, _) => write!(w, "{}", name),
            Type::Generic(name, _) => write!(w, "{}", name),
            Type::GenericInstance(base, args) => {
                base.mangle_to(w)?;
                for arg in args {
                    write!(w, "$")?;
                    arg.mangle_to(w)?;
                }
                Ok(())
            }
            Type::Function(_, _) => write!(w, "fn"),
            Type::Closure(_, _) => write!(w, "closure"),
            Type::Verified(inner) => {
                write!(w, "Verified$")?;
                inner.mangle_to(w)
            }
            Type::Pinned(inner, _) => {
                write!(w, "Pinned$")?;
                inner.mangle_to(w)
            }
            Type::Const(expr) => {
                let debug_str = format!("{:?}", expr);
                let sanitized: String = debug_str
                    .chars()
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect();
                write!(w, "const${}", sanitized)
            }
            Type::Module(name, _) => write!(w, "Module${}", name),
            Type::Unknown => write!(w, "Unknown"),
        }
    }
}

impl Mangle for ElementType {
    fn mangle_to(&self, w: &mut dyn std::fmt::Write) -> std::fmt::Result {
        match self {
            ElementType::Generic(g) => write!(w, "{}", g),
            _ => write!(w, "{}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_type_substitute_generic_simple() {
        let ty = Type::Generic("T".into(), None);
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::I32));
        let result = ty.substitute(&mapping);
        assert_eq!(result, Type::Scalar(ElementType::I32));
    }

    #[test]
    fn test_type_substitute_no_match_passthrough() {
        let ty = Type::Generic("U".into(), None);
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::I32));
        let result = ty.substitute(&mapping);
        // U is not in the mapping, so it stays Generic("U")
        assert_eq!(result, Type::Generic("U".into(), None));
    }

    #[test]
    fn test_type_substitute_borrow_nested() {
        let ty = Type::Borrow {
            inner: Box::new(Type::Generic("T".into(), None)),
            mem_space: None,
            is_mut: true,
            region_id: 0,
        };
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::F32));
        let result = ty.substitute(&mapping);
        assert_eq!(
            result,
            Type::Borrow {
                inner: Box::new(Type::Scalar(ElementType::F32)),
                mem_space: None,
                is_mut: true,
                region_id: 0,
            }
        );
    }

    #[test]
    fn test_type_substitute_tensor_element() {
        let ty = Type::Tensor(ElementType::Generic("T".into()), vec![], None);
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::F32));
        let result = ty.substitute(&mapping);
        assert_eq!(result, Type::Tensor(ElementType::F32, vec![], None));
    }

    #[test]
    fn test_type_substitute_generic_instance() {
        let ty = Type::GenericInstance(
            Box::new(Type::Struct("Vec".into(), None)),
            vec![Type::Generic("T".into(), None)],
        );
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::I32));
        let result = ty.substitute(&mapping);
        assert_eq!(
            result,
            Type::GenericInstance(
                Box::new(Type::Struct("Vec".into(), None)),
                vec![Type::Scalar(ElementType::I32)],
            )
        );
    }

    #[test]
    fn test_type_substitute_function() {
        let ty = Type::Function(
            vec![Type::Generic("T".into(), None)],
            Box::new(Type::Generic("T".into(), None)),
        );
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::F64));
        let result = ty.substitute(&mapping);
        assert_eq!(
            result,
            Type::Function(
                vec![Type::Scalar(ElementType::F64)],
                Box::new(Type::Scalar(ElementType::F64)),
            )
        );
    }

    #[test]
    fn test_type_substitute_unknown_passthrough() {
        let ty = Type::Unknown;
        let mapping = HashMap::new();
        assert_eq!(ty.substitute(&mapping), Type::Unknown);
    }

    #[test]
    fn test_type_is_linear_true_cases() {
        assert!(Type::Tensor(ElementType::F32, vec![], None).is_linear());
        assert!(Type::Matrix.is_linear());
        assert!(Type::Ref(
            Box::new(Type::Scalar(ElementType::I32)),
            MemorySpace::CPUDRAM
        )
        .is_linear());
        assert!(Type::Verified(Box::new(Type::Scalar(ElementType::I32))).is_linear());
        assert!(Type::Pinned(Box::new(Type::Scalar(ElementType::I32)), Topology::CPU).is_linear());
        assert!(Type::Struct("Foo".into(), None).is_linear());
        assert!(Type::Enum("Bar".into(), None).is_linear());
    }

    #[test]
    fn test_type_is_linear_false_cases() {
        assert!(!Type::Scalar(ElementType::I32).is_linear());
        assert!(!Type::Unknown.is_linear());
        assert!(!Type::Function(vec![], Box::new(Type::Unknown)).is_linear());
        assert!(!Type::Generic("T".into(), None).is_linear());
    }

    #[test]
    fn test_type_topology_tensor_with_topology() {
        let ty = Type::Tensor(ElementType::F32, vec![], Some(Topology::gpu(0)));
        assert_eq!(ty.topology(), Some(Topology::gpu(0)));
    }

    #[test]
    fn test_type_topology_tensor_without_topology() {
        let ty = Type::Tensor(ElementType::F32, vec![], None);
        assert_eq!(ty.topology(), None);
    }

    #[test]
    fn test_type_topology_non_tensor_returns_none() {
        assert_eq!(Type::Scalar(ElementType::I32).topology(), None);
        assert_eq!(Type::Struct("X".into(), None).topology(), None);
    }

    #[test]
    fn test_topology_is_same_kind() {
        // CPU, CpuAvx512, CpuNeon are all distinct TopologyKinds
        assert!(Topology::CPU.is_same_kind(&Topology::CPU));
        assert!(!Topology::CPU.is_same_kind(&Topology::CpuAvx512));
        assert!(!Topology::CPU.is_same_kind(&Topology::gpu(0)));
        assert!(Topology::gpu(0).is_same_kind(&Topology::gpu(0)));
    }

    fn make_npu_expr(idx: &str) -> Box<Expr> {
        Box::new(Expr::Number(super::super::expr::NumberExpr::new(
            idx.to_string(),
            None,
            Span::default(),
        )))
    }

    #[test]
    fn test_topology_kind_all_variants() {
        assert_eq!(Topology::CPU.kind(), TopologyKind::CPU);
        assert_eq!(Topology::NPU(make_npu_expr("0")).kind(), TopologyKind::NPU);
        assert_eq!(
            Topology::AccCore(make_npu_expr("0")).kind(),
            TopologyKind::AccCore
        );
        assert_eq!(Topology::AMX.kind(), TopologyKind::AMX);
        assert_eq!(Topology::ANE.kind(), TopologyKind::ANE);
        assert_eq!(Topology::gpu(0).kind(), TopologyKind::GPU);
        assert_eq!(Topology::CpuAvx512.kind(), TopologyKind::CpuAvx512);
        assert_eq!(Topology::CpuNeon.kind(), TopologyKind::CpuNeon);
        assert_eq!(Topology::Current.kind(), TopologyKind::Current);
        let slice = Topology::Slice(
            Box::new(Topology::NPU(make_npu_expr("0"))),
            make_npu_expr("0"),
            make_npu_expr("4"),
        );
        assert_eq!(slice.kind(), TopologyKind::Slice);
    }

    #[test]
    fn test_topology_is_same_kind_npu_different_indices() {
        let npu0 = Topology::NPU(make_npu_expr("0"));
        let npu1 = Topology::NPU(make_npu_expr("1"));
        // Same kind (both NPU) even though different indices
        assert!(npu0.is_same_kind(&npu1));
        // But they are not PartialEq-equal
        assert_ne!(npu0, npu1);
    }

    #[test]
    fn test_topology_is_same_kind_acccore_different_indices() {
        let ac0 = Topology::AccCore(make_npu_expr("0"));
        let ac3 = Topology::AccCore(make_npu_expr("3"));
        assert!(ac0.is_same_kind(&ac3));
        assert_ne!(ac0, ac3);
    }

    #[test]
    fn test_type_substitute_preserves_topology() {
        let ty = Type::Tensor(
            ElementType::Generic("T".into()),
            vec![],
            Some(Topology::gpu(0)),
        );
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::F32));
        let result = ty.substitute(&mapping);
        assert_eq!(
            result,
            Type::Tensor(ElementType::F32, vec![], Some(Topology::gpu(0)))
        );
    }

    #[test]
    fn test_type_substitute_pinned_preserves_topology() {
        let ty = Type::Pinned(Box::new(Type::Generic("T".into(), None)), Topology::ANE);
        let mut mapping = HashMap::new();
        mapping.insert("T".into(), Type::Scalar(ElementType::I32));
        let result = ty.substitute(&mapping);
        assert_eq!(
            result,
            Type::Pinned(Box::new(Type::Scalar(ElementType::I32)), Topology::ANE)
        );
    }

    #[test]
    fn test_mangle_tensor_ignores_topology_and_shape() {
        let ty_with = Type::Tensor(ElementType::F32, vec![], Some(Topology::gpu(0)));
        let ty_without = Type::Tensor(ElementType::F32, vec![], None);
        // Topology is intentionally not included in mangling
        assert_eq!(ty_with.mangle(), ty_without.mangle());
        assert_eq!(ty_with.mangle(), "Tensor$f32");
    }

    #[test]
    fn test_mangle_tensor_separates_extents() {
        // Two shapes are two types and lower to two memrefs, so they must not share a
        // monomorph's symbol (Vx#401). Rank alone merged them.
        let num = |v: &str| {
            crate::syntax::Expr::Number(crate::syntax::NumberExpr::new(
                v.to_string(),
                None,
                Span::default(),
            ))
        };
        let a = Type::Tensor(ElementType::F32, vec![num("2"), num("3")], None);
        let b = Type::Tensor(ElementType::F32, vec![num("4"), num("5")], None);
        assert_eq!(a.mangle(), "Tensor$f32$2x3");
        assert_ne!(a.mangle(), b.mangle());
        // A rank-1 [23] must not collide with a rank-2 [2, 3].
        let c = Type::Tensor(ElementType::F32, vec![num("23")], None);
        assert_ne!(a.mangle(), c.mangle());
    }

    #[test]
    fn test_mangle_pinned_type() {
        let ty = Type::Pinned(Box::new(Type::Scalar(ElementType::I32)), Topology::ANE);
        assert_eq!(ty.mangle(), "Pinned$i32");
    }

    /// Two spellings of device 0 are the same device, however their index literal was built
    /// (Vx#355). This is the unit-level lock on the defect: `Topology::gpu()` builds the index
    /// with `ty: None`, while `check_transfer_expr` and a checked source annotation produce
    /// `ty: Some(I32)`, and the derived `PartialEq` called those two different devices.
    ///
    /// The visible consequence was that a written `Pinned<_, Topology::GPU[0]>` never matched
    /// what `transfer` produces, so a placed tensor could not be passed to a function or held
    /// in a struct field at all.
    fn indexed(annotated: bool, value: &str) -> Topology {
        Topology::GPU(Box::new(Expr::Number(NumberExpr::new(
            value.to_string(),
            if annotated {
                Some(ElementType::I32)
            } else {
                None
            },
            Span::default(),
        ))))
    }

    #[test]
    fn topology_index_identity_ignores_how_the_literal_was_built() {
        // The exact pair the bug turned on: same device, different literal annotation.
        assert_eq!(indexed(true, "0"), indexed(false, "0"));
        assert_eq!(indexed(false, "0"), Topology::gpu(0));
        // A span difference must not split a device either -- same reasoning, and the derive
        // compared spans too.
        let mut spanned = NumberExpr::new("0".to_string(), Some(ElementType::I32), Span::default());
        spanned.span = Span {
            line: 7,
            column: 3,
            length: 1,
        };
        assert_eq!(
            Topology::GPU(Box::new(Expr::Number(spanned))),
            Topology::gpu(0)
        );

        // Different devices stay different: the fix must not collapse the index, which is the
        // whole reason the index exists ("prefill here, decode there").
        assert_ne!(indexed(true, "0"), indexed(true, "1"));
        assert_ne!(Topology::gpu(0), Topology::gpu(1));
        // And a different KIND with the same index is still a different topology.
        assert_ne!(
            Topology::gpu(0),
            Topology::NPU(Box::new(Expr::Number(NumberExpr::new(
                "0".to_string(),
                None,
                Span::default(),
            ))))
        );
    }

    /// Every variant is equal to itself. Trivially true of a derived `PartialEq`, and NOT
    /// trivially true of a hand-written one: the impl matches on a PAIR of topologies, which
    /// Rust cannot exhaustiveness-check, so a variant added to the enum and forgotten in the
    /// impl falls into the `_` arm against itself and reports not-equal.
    ///
    /// The impl carries a `debug_assert!` for exactly that, and this test is what fires it.
    /// Listing the variants by hand is the point -- add one to `Topology` and this fails until
    /// it is listed here too, which is the reminder to go and add its arm.
    #[test]
    fn every_topology_variant_is_equal_to_itself() {
        let idx = || {
            Box::new(Expr::Number(NumberExpr::new(
                "3".to_string(),
                None,
                Span::default(),
            )))
        };
        let all = [
            Topology::CPU,
            Topology::NPU(idx()),
            Topology::AccCore(idx()),
            Topology::AMX,
            Topology::ANE,
            Topology::GPU(idx()),
            Topology::CpuAvx512,
            Topology::CpuNeon,
            Topology::Slice(Box::new(Topology::CPU), idx(), idx()),
            Topology::Custom("Dev".into()),
            Topology::Current,
        ];
        for t in &all {
            assert_eq!(t, t, "a topology must be equal to itself: {t:?}");
        }
        // And distinct variants stay distinct, so the reflexivity above is not coming from an
        // arm that says yes to everything.
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "distinct topologies compared equal: {a:?} vs {b:?}");
                }
            }
        }
    }
}
