//===- decl.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Abstract Syntax Tree nodes for declarations, including traits, structs, and functions.
//
//===----------------------------------------------------------------------===//

use super::*;
use crate::symbol::Symbol;

#[derive(Debug, PartialEq, Clone)]
pub enum GenericParam {
    Type { name: Symbol, bound: Option<Symbol> },
    Const { name: Symbol, ty: Type },
}

impl GenericParam {
    pub fn name(&self) -> &str {
        match self {
            GenericParam::Type { name, .. } => name,
            GenericParam::Const { name, .. } => name,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    pub name: Symbol,
    pub generics: Vec<GenericParam>,
    pub params: Vec<(Symbol, Type)>,
    pub topology: Topology,
    pub return_type: Type,
    pub requires: Vec<Expr>,
    pub ensures: Vec<Expr>,
    /// `where Transfer<S, D>` constraints: pairs of topology names (generic topology
    /// variables or concrete topologies) that must have a transfer path in the cost
    /// graph. Discharged at each generic call once the variables are bound.
    pub where_transfers: Vec<(Symbol, Symbol)>,
    pub body: Vec<Statement>,
    pub doc_comment: Option<String>,
}

impl Function {
    pub fn clone_signature(&self, preserve_body: bool) -> Self {
        Self {
            name: self.name.clone(),
            generics: self.generics.clone(),
            params: self.params.clone(),
            topology: self.topology.clone(),
            return_type: self.return_type.clone(),
            requires: self.requires.clone(),
            ensures: self.ensures.clone(),
            where_transfers: self.where_transfers.clone(),
            body: if preserve_body || !self.generics.is_empty() {
                self.body.clone()
            } else {
                Vec::new()
            },
            doc_comment: self.doc_comment.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct StructDecl {
    pub name: Symbol,
    pub generics: Vec<GenericParam>,
    pub fields: Vec<(Symbol, Type)>,
    pub doc_comment: Option<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct EnumDecl {
    pub name: Symbol,
    pub generics: Vec<GenericParam>,
    pub variants: Vec<(Symbol, Option<Vec<Type>>)>,
    pub doc_comment: Option<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ExternDecl {
    pub name: Symbol,
    pub is_safe: bool,
    pub params: Vec<(Symbol, Type)>,
    pub return_type: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MethodSignature {
    pub name: Symbol,
    pub params: Vec<(Symbol, Type)>,
    pub return_type: Type,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TraitDecl {
    pub name: Symbol,
    pub generics: Vec<GenericParam>,
    pub methods: Vec<MethodSignature>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ImplBlock {
    pub generics: Vec<GenericParam>,
    pub trait_name: Option<Symbol>,
    pub target_type: Type,
    pub methods: Vec<Function>,
}

impl ImplBlock {
    pub fn clone_signature(&self) -> Self {
        Self {
            generics: self.generics.clone(),
            trait_name: self.trait_name.clone(),
            target_type: self.target_type.clone(),
            // Always preserve method bodies: methods only exist in impl blocks, so
            // method-call monomorphization clones the body from the type-check env
            // (env.impls). Dropping it (as the signature clone does for free
            // functions, whose bodies come from `program.functions`) yields an
            // empty method body and invalid IR. See GitHub #146.
            methods: self
                .methods
                .iter()
                .map(|m| m.clone_signature(true))
                .collect(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ImportDecl {
    pub path: Vec<Symbol>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MacroDefDecl {
    pub name: Symbol,
    pub rules: Vec<MacroRule>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone)]
pub struct MacroRule {
    pub matcher: Vec<TokenTree>,
    pub transcriber: Vec<TokenTree>,
}

/// A byte quantity, e.g. `256 KB`, normalized to bytes (binary multipliers: `KB` = 1024).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ByteSize(pub u64);

/// The denominator of a bandwidth rate.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RatePer {
    /// `/s`
    Second,
    /// `/cyc`
    Cycle,
}

/// A bandwidth, e.g. `8 TB/s` or `128 B/cyc`; the numerator is normalized to bytes.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Bandwidth {
    pub bytes: u64,
    pub per: RatePer,
}

/// How a memory space is managed (M5 uses this to decide the transfer obligation).
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum Management {
    /// Programmer-managed: movement in/out requires an explicit `transfer`.
    Explicit,
    /// Hardware-cached: movement may be implicit. The default.
    #[default]
    Cached,
}

/// How data physically crosses into a space from its parent, which decides how a multi-hop walk
/// composes (vx-review#26).
///
/// Measured on two architectures, and the law tracks this property rather than the machine:
///
///   H100  HBM->L2->SMEM   streamed (cp.async)     bottleneck fits  -13.9%
///   H100  L1->REG->SMEM   sequenced               sum fits          -9.3%
///   M4    L2->REG->SMEM   sequenced               sum fits         +13.2%
///   M4    HBM->REG->SMEM  sequenced               sum fits          +2.7%
///
/// `within:` says one space contains another; it does not say whether crossing that boundary is a
/// hardware path or a pair of instructions. Without this distinction any single global law is
/// wrong on half the routes -- summing overstates a streamed route by ~67%, and taking the
/// bottleneck understates a sequenced one by ~49%. Same mistake, opposite directions.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum Crossing {
    /// A load into a register followed by a store back out. The instructions genuinely happen one
    /// after the other, so the legs genuinely add.
    ///
    /// The default, deliberately: it is what the algebra has always done, so introducing this
    /// property changes no prediction anywhere and the frozen cells stay byte-identical. Declaring
    /// `streamed` is then an explicit, dated machine-file edit with a localized effect, rather than
    /// a silent model change that moves 240 cells at once.
    #[default]
    Sequenced,
    /// A hardware engine fills the destination without passing through registers (Hopper's
    /// `cp.async`, a DMA). Nothing stages, so the narrowest leg alone sets the rate.
    Streamed,
}

/// The execution level at which a memory space is private / replicated. Ordered broadest to
/// narrowest — locality *narrows* going down a `within:` hierarchy (a per-SM space sits inside
/// a per-device space, never the reverse). Lets `capacity` be read as a per-scope budget:
/// a `Sm`-scoped 256 KB is one SM's TMEM, not a global pool.
#[derive(Debug, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub enum Scope {
    /// The whole device (HBM, L2).
    Device,
    /// One streaming multiprocessor (SMEM, TMEM).
    Sm,
    /// One cooperative thread array / thread block.
    Cta,
    /// One thread (registers).
    Thread,
}

/// A first-class memory-space declaration:
/// `Memory <Name> { within:, capacity:, bandwidth:, managed:, granule: }`.
///
/// Carried on the AST (`Program.memories`) and indexed by `GlobalAstEnv` for semantic
/// analysis; codegen reads it from the `Program`. Deliberately *not* registered in a
/// process-global registry (unlike topologies), so declarations cannot leak between
/// compilations. See `docs/discussions/implementation_plans/first_class_memory_spaces.md`.
#[derive(Debug, PartialEq, Clone)]
pub struct MemoryDecl {
    pub name: Symbol,
    /// `within: Memory::X` — the parent space in the hierarchy tree; `None` for a root.
    pub parent: Option<MemorySpace>,
    pub capacity: Option<ByteSize>,
    pub bandwidth: Option<Bandwidth>,
    /// `clock: 1.98 GHz` — the clock a `B/cyc` bandwidth on this space is denominated in, in Hz.
    ///
    /// Required to convert this space's rate against a `B/s` one. Without it a path that mixes
    /// `B/cyc` and `B/s` is not derivable, because inventing a clock is exactly the silent
    /// conversion PREDICTIONS.md decision 5 forbids.
    pub clock_hz: Option<u64>,
    pub managed: Management,
    pub granule: Option<ByteSize>,
    /// The execution level this space is private to (`scope: sm` etc.); `None` = unscoped.
    pub scope: Option<Scope>,
    /// `replicas: 132` — how many instances of this space the device contains.
    ///
    /// A scoped space's `bandwidth:` is the rate of ONE instance, while an enclosing device-scoped
    /// space quotes an aggregate. Composing the two without this count silently adds a per-SM rate
    /// to a device-wide one, which is how `L2->SMEM` stayed at -87% even after the containment fix:
    /// L2's 12 TB/s is 6060 B/cyc device-wide against SMEM's 128 per SM, so the larger term simply
    /// vanished.
    pub replicas: Option<u64>,
    /// `overcommit`: opt out of the *cumulative* budget error — the working set placed here may
    /// exceed `capacity` (downgraded to a warning). The programmer asserts the tiles do not all
    /// coexist, so the conservative sum should not block them.
    pub overcommit: bool,
    /// `crossing: streamed` -- how data enters this space from its parent, which decides whether a
    /// walk through it sums its legs or takes the slowest (vx-review#26). Defaults to `sequenced`,
    /// which is the pre-existing behaviour.
    pub crossing: Crossing,
    pub doc_comment: Option<String>,
}

/// `impl transfer Memory::<From> -> Memory::<To> { fn ... }` — a transfer lowering: the code a
/// movement across this edge emits, written in Vx and compiled by our own front end
/// (docs/custom_transfer_contract.md).
///
/// Distinct from the two existing `transfer` forms on purpose. The topology clause
/// (`transfer Memory::A -> Memory::B : cost`) declares that an edge EXISTS and what it costs; the
/// expression (`transfer(x, Memory::B)`) asks for a movement. This declares HOW the movement is
/// performed — and its `relaxed|sync` grade, its aliasing and its overhead are eventually read off
/// the body rather than declared, which is what supersedes the edge clause's declared marker.
#[derive(Debug, PartialEq, Clone)]
pub struct TransferImplDecl {
    /// The edge this lowering implements, `from -> to`. Matched against the topology's declared
    /// edges by sema (not yet wired); a lowering for an edge no topology declares is meaningless.
    pub from: MemorySpace,
    pub to: MemorySpace,
    /// The lowering's functions, `fn move(...)` by convention. Parsed as ordinary Vx functions —
    /// the design's point being that every front-end check CAN apply to them. Today macro
    /// expansion and the structural checks (E6015) visit these bodies; full type-checking is
    /// not yet wired, so do not read this field as verified code. Which shapes are legal (copy
    /// fills a `dst`; an alias returns a view) is sema's question, not the parser's, so the
    /// parser accepts any `fn` items here.
    pub methods: Vec<Function>,
    pub doc_comment: Option<String>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub module_path: Symbol,
    pub imports: Vec<ImportDecl>,
    pub macros: Vec<MacroDefDecl>,
    pub externs: Vec<ExternDecl>,
    pub structs: Vec<StructDecl>,
    pub enums: Vec<EnumDecl>,
    pub traits: Vec<TraitDecl>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<Function>,
    /// User-defined topologies declared in this program (`Topology <Name> { ... }`). Like
    /// `memories`, the full descriptors live here on the AST — *not* a process-global registry —
    /// so declarations never leak between compilations and the parallel pipeline needs no lock.
    /// Sema indexes them via `GlobalAstEnv` and seeds the per-compilation transfer cost graph.
    pub topologies: Vec<crate::arch::TopologyDecl>,
    /// User-defined memory spaces declared in this program (`Memory <Name> { ... }`). Unlike
    /// topologies, the full descriptors live here on the AST (not a global registry); sema
    /// indexes them via `GlobalAstEnv` and codegen reads them from the `Program`.
    pub memories: Vec<MemoryDecl>,
    /// Transfer lowerings declared in this program (`impl transfer A -> B { ... }`). Parsed and
    /// carried; nothing consumes them yet — the checks in docs/custom_transfer_contract.md land
    /// against this field.
    pub transfer_impls: Vec<TransferImplDecl>,
}

pub type VxModule = Program;
pub type VxFunction = Function;

impl Program {
    pub fn add(&mut self, func: VxFunction) {
        self.functions.push(func);
    }

    pub fn clone_signature(&self) -> Self {
        Self {
            module_path: self.module_path.clone(),
            imports: self.imports.clone(),
            macros: self.macros.clone(),
            externs: self.externs.clone(),
            structs: self.structs.clone(),
            enums: self.enums.clone(),
            traits: self.traits.clone(),
            impls: self.impls.iter().map(|i| i.clone_signature()).collect(),
            functions: self
                .functions
                .iter()
                .map(|f| f.clone_signature(false))
                .collect(),
            topologies: self.topologies.clone(),
            memories: self.memories.clone(),
            // Bodies stripped like `functions`/`impls`: the signature clone exists so the parallel
            // pipeline can share a light Program, and a lowering body is as heavy as any other.
            transfer_impls: self
                .transfer_impls
                .iter()
                .map(|t| TransferImplDecl {
                    from: t.from.clone(),
                    to: t.to.clone(),
                    methods: t.methods.iter().map(|f| f.clone_signature(false)).collect(),
                    doc_comment: t.doc_comment.clone(),
                })
                .collect(),
        }
    }
}
