//===- arch.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TransferCostGraph, formalizing the memory and topology
// algebra for cross-device accesses and transfers.
//
//===----------------------------------------------------------------------===//

use crate::syntax::{MemorySpace, Topology, Type};
use std::collections::HashMap;

use crate::syntax;
pub struct TransferCostGraph {
    /// Adjacency list for MemorySpace data transfers.
    /// Directed edge from A -> B means memory can be transferred from A to B.
    transfer_edges: HashMap<MemorySpace, Vec<(MemorySpace, u32)>>,

    /// Cached all-pairs shortest paths for data transfers.
    cost_matrix: HashMap<(MemorySpace, MemorySpace), u32>,

    /// Declared *link* bandwidths, for edges whose cost the containment tree cannot derive.
    ///
    /// A host↔device hop is the motivating case: `CPU_DRAM` and `HBM` do not nest, so they have no
    /// nearest common ancestor and the roofline walk returns nothing — which is why that seam, the
    /// most important one in the model, predicted `null` on every fleet file. Its bandwidth belongs
    /// to the link (PCIe, C2C), not to either memory, so it is declared on the edge and looked up
    /// here.
    edge_rates: HashMap<(MemorySpace, MemorySpace), syntax::Bandwidth>,

    /// Per-byte time, in attoseconds, for every edge whose cost the model can actually predict.
    ///
    /// This is what route selection minimises. An edge absent from this map has no
    /// predictable cost -- a `Fixed` weight, or a containment hop denominated in cycles that
    /// nothing can compare against a link's seconds -- and routing falls back to preferring routes
    /// that avoid it, then to hop count, rather than inventing a number for it.
    ///
    /// Attoseconds because the spread is large: 12 TB/s is 83,333 as/B and 63 GB/s is 15,873,015,
    /// so picoseconds would round the fast on-die hops to zero and make them free.
    route_cost: HashMap<(MemorySpace, MemorySpace), u64>,

    /// `Derived` edges awaiting `resolve_derived_route_costs`.
    ///
    /// Their cost is the containment roofline between their endpoints, which lives in
    /// `MemoryHierarchy` and is not reachable from here -- the graph is built from topology
    /// declarations and the hierarchy from memory declarations. Recorded at seed time and priced
    /// once the caller supplies the memories.
    derived_edges: Vec<(MemorySpace, MemorySpace)>,

    /// The unitless `transfer A -> B : 300` figures, kept apart from the routing weights.
    ///
    /// Since edges may now decline to declare a cost, the weight the router uses is not always a
    /// figure anyone wrote down — a `Derived` edge weighs 1 so hop count breaks ties. Reporting
    /// that 1 as "the declared cost" would be inventing data, so the two are stored separately and
    /// the diagnostics record only ever quotes this one.
    edge_declared: HashMap<(MemorySpace, MemorySpace), u32>,

    /// Every topology's descriptor (built-ins plus the user-declared `Topology { ... }` of *this*
    /// compilation), held per-instance rather than in a process-global registry. This is the
    /// data-oriented, lock-free home the parallel pipeline needs (see
    /// docs/parallel_compiler_architecture.md): the graph is built once per compilation and shared
    /// across worker threads by `&`, so a topology declared in one program cannot leak into
    /// another and nothing is synchronized.
    descriptors: HashMap<crate::syntax::TopologyKind, TopologyDescriptor>,
}

/// Verdict of the type-level USE rule: can a value on `var_topology` be read while
/// running on `active_topology`?
///
/// - `Visible`: readable in place — a visibility edge / unified memory (cost 0, no
///   data movement). The value stays where it is.
/// - `NeedsSeam`: a transfer path exists but the location is not directly visible,
///   so under the explicit-seam policy the programmer must write `transfer(...)`.
/// - `Unreachable`: no transfer path exists at all.
///
/// The USE-DIRECT / USE-NEEDS-SEAM rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reachability {
    Visible,
    NeedsSeam { cost: u32 },
    Unreachable,
}

/// Declarative description of a topology's memory behaviour — the data that used to be
/// hardcoded across `default_memory_for` and the special-cases in `is_type_accessible`.
///
/// A topology is described by where its values live (`default_space`) and which memory
/// spaces it can directly address (`visibility`, i.e. unified-memory reach; this always
/// includes `default_space`). The built-ins come from `builtin_descriptors`; a user
/// `Topology { ... }` declaration is carried on the AST (`TopologyDecl` on `Program.topologies`)
/// and seeded into the per-compilation `TransferCostGraph` — the metadata is data, not `match`
/// arms, and not a global registry -- a registry behind the enum.
/// A declared transfer edge (morphism): a hop `from -> to` with a `cost` grade and a
/// consistency grade. `sync` = a synchronizing transfer (release/acquire) that preserves a
/// boundary contract; `!sync` = a relaxed escape hatch whose visibility the seam engine
/// cannot guarantee (see coherence checking in `hir`).
///
/// `copy_engine` declares that a hardware engine (a DMA, Ampere's `cp.async`) can drive this
/// hop without passing through registers. It is a *capability*, not a choice: declaring it
/// makes `raw::async_copy` legal in an `impl transfer` lowering for this edge (Vx#353 A2),
/// and nothing more. Distinct from `crossing:` on the memory space, which answers how legs
/// *compose in cost*, not whether an engine exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferEdge {
    pub from: MemorySpace,
    pub to: MemorySpace,
    pub cost: EdgeCost,
    pub sync: bool,
    pub copy_engine: bool,
}

/// Where a declared edge's cost comes from. **Exactly one source per edge** — an edge that both
/// declares a cost and can have one derived from its endpoints' bandwidths is an error (E6013),
/// because two answers to "what does this hop cost" is not a model, it is a coin flip whose
/// outcome depends on which consumer asks.
///
/// The rule was not free: every fleet file declared `transfer HBM -> L2 : 40` *and* gave `L2` a
/// bandwidth, so both costs existed for that edge, and the compiler routed by one and reported the
/// other. Nothing detected it.
/// Attoseconds per second. The denomination route selection works in -- see `route_cost`.
const ATTOS_PER_SEC: u128 = 1_000_000_000_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeCost {
    /// `transfer A -> B` — the edge asserts reachability only. Its cost is the containment-derived
    /// roofline from the endpoints' `bandwidth:` figures. This is the normal case for a hop *within*
    /// a device, where the spaces nest and their bandwidths describe the move.
    Derived,
    /// `transfer A -> B : 64 GB/s` — the *link's* own bandwidth. Needed where containment cannot
    /// derive a cost because the endpoints do not nest: a host↔device link is a property of PCIe or
    /// C2C, not of either memory it connects.
    Rate(crate::syntax::Bandwidth),
    /// `transfer A -> B : 300` — a unitless relative cost. Not a physical prediction, and it does
    /// not scale with transfer size; kept for edges no bandwidth figure describes, and reported as
    /// such rather than quoted as a time.
    Fixed(u32),
}

impl EdgeCost {
    /// The unitless weight this edge contributes, kept for reporting and as a routing tie-break.
    ///
    /// No longer the primary routing key. It used to be, on the reasoning that "route choice
    /// happens before a byte count is known, so a size-dependent cost cannot decide it" -- which
    /// M5 showed to be a false premise. Every edge costs `bytes / bandwidth`, a
    /// line through the origin, so the ratio between two routes does not depend on bytes and the
    /// cheapest route can be chosen once, for all sizes, from the per-byte rate alone. Minimising
    /// hops instead picked a route 3.57x slower than one the same graph already contained.
    pub fn routing_weight(self) -> u32 {
        match self {
            EdgeCost::Derived | EdgeCost::Rate(_) => 1,
            EdgeCost::Fixed(c) => c,
        }
    }

    /// This edge's own cost per byte in attoseconds, when the edge declares a link rate.
    ///
    /// `None` for `Derived` (the containment tree prices it -- see `resolve_derived_route_costs`),
    /// for `Fixed` (a relative latency is not a time), and for a link declared in `B/cyc`, which
    /// cannot be compared against a link declared in `B/s` without a clock the edge does not
    /// carry. All three mean "not predictable here", which routing treats as a last resort rather
    /// than as free.
    pub fn per_byte_attos(self) -> Option<u64> {
        match self {
            EdgeCost::Rate(bw) if bw.per == crate::syntax::RatePer::Second && bw.bytes > 0 => {
                u64::try_from(ATTOS_PER_SEC / bw.bytes as u128).ok()
            }
            _ => None,
        }
    }

    /// The declared figure to report as this edge's cost, if it declared one at all.
    pub fn declared(self) -> Option<u32> {
        match self {
            EdgeCost::Fixed(c) => Some(c),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyDescriptor {
    pub default_space: MemorySpace,
    pub visibility: Vec<MemorySpace>,
    /// The instruction set this topology executes -- `arch: x86_64` in the declaration.
    ///
    /// A machine description that does not say this cannot answer what code to emit for it,
    /// and nothing else in the file implies it: a filename and a comment are not readable by
    /// the compiler. It is what a target triple is derived from (#342), which is why there is
    /// no separate `--target` to disagree with it.
    ///
    /// `None` where a declaration predates the field; required of a host, which is the
    /// machine the program itself runs on.
    pub arch: Option<crate::symbol::Symbol>,
    /// Transfer edges (morphisms) this topology contributes to the cost graph. Seeded into
    /// a `TransferCostGraph` via `seed_from_topologies`.
    pub transfers: Vec<TransferEdge>,
    /// The element types this hardware can represent -- `dtypes: [f32, f16]` in the declaration.
    ///
    /// `None` is *undeclared*, not empty, and undeclared is permissive: a machine file that says
    /// nothing about element types constrains nothing, so every file written before this field
    /// existed keeps its meaning. Silence cannot mean "supports nothing" without turning every
    /// older declaration into a machine that can hold no data at all.
    ///
    /// The distinction this exists to draw is a real one in silicon and a sharp one: an H100 has
    /// no fp4, and a program that places an fp4 tensor on one is asking for hardware that is not
    /// there. That is answerable here, before anything is emitted, because the placement is in the
    /// type and the machine is a declaration the compiler already reads.
    pub dtypes: Option<Vec<crate::syntax::ElementType>>,
}

/// A user-declared topology: its name plus its descriptor. Carried on the AST
/// (`Program.topologies`) and indexed per-compilation by `GlobalAstEnv`, exactly like
/// `MemoryDecl` — *not* a process-global registry. Parsed by `parser::decl::parse_topology_decl`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyDecl {
    pub name: crate::symbol::Symbol,
    pub descriptor: TopologyDescriptor,
}

/// A way a declared topology fails its coherence obligations. Graph-decidable here; the
/// consistency obligation (a relaxed edge losing visibility) is discharged separately via
/// `hir::seam`. See `descriptor_coherence`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoherenceIssue {
    /// The topology cannot see its own default memory space (`default_space ∉ visibility`).
    DefaultNotVisible,
    /// No transfer path reaches the topology's memory from the host, so data can never be
    /// moved there.
    MemoryUnreachableFromHost,
}

/// The graph-decidable coherence obligations for one descriptor, checked against `graph`
/// (which must already be seeded with the topology's edges). Pure; the consistency
/// obligation is handled by the caller via the seam engine.
pub fn descriptor_coherence(
    desc: &TopologyDescriptor,
    graph: &TransferCostGraph,
) -> Vec<CoherenceIssue> {
    let mut issues = Vec::new();
    if !desc.visibility.contains(&desc.default_space) {
        issues.push(CoherenceIssue::DefaultNotVisible);
    }
    let reachable = desc.visibility.contains(&MemorySpace::CPUDRAM)
        || desc.default_space == MemorySpace::CPUDRAM
        || graph
            .transfer_path(&MemorySpace::CPUDRAM, &desc.default_space)
            .is_some();
    if !reachable {
        issues.push(CoherenceIssue::MemoryUnreachableFromHost);
    }
    issues
}

// ---- Runtime dispatch ids ----------------------------------------------------------------
// The device a `vx.spawn topology(N)` / `vx.transfer target_topology = N` targets. Kept here
// (next to the topology registry) so the topology-based and memory-space-based mappings stay
// adjacent and cannot silently diverge: a topology and its canonical memory space share an id
// (e.g. GPU and GpuHbm are both 500), and `Custom` uses the same FNV scheme in both.

/// FNV-1a over `s` (not `DefaultHasher`, whose algorithm may change between Rust releases), so
/// the derived dispatch ids are reproducible across toolchains -- they are part of the runtime
/// dispatch contract. Shared by the `Custom` and `Slice` banded id schemes.
fn fnv32(s: &str) -> u32 {
    let mut hash: u32 = 2166136261;
    for b in s.as_bytes() {
        hash ^= *b as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}

/// The lowest id a declared name can be given. Everything below is spoken for: the built-in
/// topologies and their canonical memory spaces occupy 0..999 in hundred-wide bands, and slices
/// occupy 2000..2999.
pub const CUSTOM_DISPATCH_ID_BASE: i32 = 3000;

/// How many ids a declared name can be given: `CUSTOM_DISPATCH_ID_BASE..=i32::MAX`.
///
/// Kept as wide as the wire type allows, because the width is what decides whether two declared
/// names collide. In the old 1000..1999 band the answer was "constantly": 20 names collided 18% of
/// the time, 40 names 55%, and the 250-space corpus the scaling benchmark runs on lost 42 of its
/// descriptors outright. Two billion slots put 1000 names at 0.025% and the corpus at zero.
const CUSTOM_DISPATCH_ID_SPAN: u32 = (i32::MAX as u32) - (CUSTOM_DISPATCH_ID_BASE as u32) + 1;

/// A stable per-name dispatch id at or above [`CUSTOM_DISPATCH_ID_BASE`].
///
/// Still one-way and still capable of colliding -- E6016 refuses a collision rather than resolving
/// it, and `runtime/vx_manifest.h` mirrors this function byte for byte, so the two must change
/// together.
fn fnv_dispatch_id(name: &str) -> i32 {
    CUSTOM_DISPATCH_ID_BASE + (fnv32(name) % CUSTOM_DISPATCH_ID_SPAN) as i32
}

/// A topology index's compile-time value: a literal, or exact integer arithmetic over literals
/// (`GPU[TP * 2]` once monomorphization has substituted its const generics). `None` when the index
/// is a genuine runtime value.
pub fn const_topology_index(expr: &crate::syntax::Expr) -> Option<i32> {
    match expr {
        crate::syntax::Expr::Number(n) => n.value.parse::<i32>().ok(),
        crate::syntax::Expr::BinaryOp(b) => {
            let l = const_topology_index(&b.lhs)?;
            let r = const_topology_index(&b.rhs)?;
            match b.op {
                crate::syntax::BinaryOp::Add => l.checked_add(r),
                crate::syntax::BinaryOp::Sub => l.checked_sub(r),
                crate::syntax::BinaryOp::Mul => l.checked_mul(r),
                crate::syntax::BinaryOp::Div => (r != 0).then(|| l / r),
                crate::syntax::BinaryOp::MatMul => None,
            }
        }
        _ => None,
    }
}

/// The first index expression in `top` that is *not* a compile-time constant, if any. A
/// non-constant index cannot be resolved to a device instance at compile time, so
/// `topology_dispatch_id` falls back to 0 -- silently placing `GPU[i]` on device 0. Callers use
/// this to warn loudly instead (#284); the fallback itself stays, since declining would reject
/// programs that compile today.
pub fn non_constant_index(top: &Topology) -> Option<&crate::syntax::Expr> {
    match top {
        // Every indexed kind, and the list must stay that way: an omission here is not a missing
        // warning but a wrong placement reported as a right one. GPU was omitted while the comment
        // above used `GPU[i]` as its example (#345).
        Topology::NPU(e) | Topology::AccCore(e) | Topology::GPU(e) => {
            const_topology_index(e).is_none().then_some(&**e)
        }
        Topology::Slice(base, start, end) => non_constant_index(base)
            .or_else(|| const_topology_index(start).is_none().then_some(&**start))
            .or_else(|| const_topology_index(end).is_none().then_some(&**end)),
        _ => None,
    }
}

/// A topology's device index, folding constant arithmetic. Falls back to 0 for a non-constant
/// index -- see `non_constant_index`, which callers consult to warn about exactly that case.
fn topology_index(expr: &crate::syntax::Expr) -> i32 {
    const_topology_index(expr).unwrap_or(0)
}

/// Dispatch id for a topology (`vx.spawn topology(N)`). Single source of truth.
pub fn topology_dispatch_id(top: &Topology) -> i32 {
    match top {
        Topology::CPU | Topology::Current => 0,
        Topology::NPU(e) => 100 + topology_index(e),
        Topology::AccCore(e) => 200 + topology_index(e),
        Topology::AMX => 300,
        Topology::ANE => 400,
        Topology::GPU(e) => 500 + topology_index(e),
        Topology::CpuAvx512 => 600,
        Topology::CpuNeon => 700,
        // A slice is identified by (base topology, extent): `NPU[0..144]` and `NPU[0..72]` are
        // distinct devices for dispatch and seam identity, so they carry distinct stable ids
        // (B4, #253) — every slice used to collapse onto one constant (900), which made a slice
        // unable to name an NVL domain. FNV over the canonical triple, banded to 2000..2999
        // (declared names start above at `CUSTOM_DISPATCH_ID_BASE`).
        Topology::Slice(base, start, end) => {
            let key = format!(
                "{}:{}:{}",
                topology_dispatch_id(base),
                topology_index(start),
                topology_index(end)
            );
            2000 + (fnv32(&key) % 1000) as i32
        }
        Topology::Custom(name) => fnv_dispatch_id(name),
    }
}

/// Dispatch id for a memory-space transfer target (`vx.transfer target_topology = N`). Each
/// space maps to its canonical owning topology's id, so it agrees with `topology_dispatch_id`.
pub fn memory_space_dispatch_id(mem: &MemorySpace) -> i32 {
    match mem {
        MemorySpace::CPUDRAM => 0,     // CPU
        MemorySpace::NPUHBM => 100,    // NPU
        MemorySpace::LocalSRAM => 200, // AccCore
        MemorySpace::GpuHbm => 500,    // GPU
        // Network memory, in bands of its own (#348).
        //
        // Both used to answer 300, which is `Topology::AMX`. A `transfer` into remote memory
        // therefore emitted `target_topology = 300` and a plugin reading it saw a request for
        // Apple's matrix coprocessor -- not a missing feature but an actively wrong answer, and one
        // no diagnostic could catch because the number was valid. They also could not be told apart
        // from each other, so staging into a NIC's buffer and landing in a peer's HBM were one
        // event.
        //
        // These are the last two free 100-bands below the slice band (2000..2999) and the range
        // declared names occupy (`CUSTOM_DISPATCH_ID_BASE` and up). Nothing decodes them yet;
        // giving them distinct identities is what lets a plugin start to.
        MemorySpace::NicRam => 800,
        MemorySpace::RemoteHbm => 900,
        MemorySpace::Custom(name) => fnv_dispatch_id(name),
    }
}

/// A target-independent address space: what the memory *is*, not what number a particular GPU
/// ISA gives it. Codegen maps this to a concrete annotation per target (`nvptx_addrspace` today;
/// the `#gpu.address_space<..>` attribute spelling when a GPU backend lands, #251), so the
/// numbering lives in exactly one place instead of being hardcoded at every use.
///
/// Previously codegen emitted raw integers keyed only on the built-in `MemorySpace` variant, which
/// was wrong twice over: on-chip scratchpad was given NVPTX 2 (reserved, not shared), and *every*
/// user-declared space collapsed onto 4 -- NVPTX **constant** memory, which is read-only. A B200
/// model declaring `Memory SMEM`/`Memory TMEM`/`Memory L2` therefore made all three the same
/// read-only space. See #258.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressSpace {
    /// Host memory (also NVPTX generic/flat).
    Host,
    /// Device-global memory: HBM, GDDR, L2.
    Global,
    /// Shared/on-chip memory private to one SM or thread block: SMEM, LDS.
    Workgroup,
    /// Per-thread private memory: registers, local spill.
    Private,
}

impl AddressSpace {
    /// The NVPTX numbering (0 generic, 1 global, 3 shared, 5 local). Only this function knows the
    /// target's numbers; everything upstream reasons in terms of the enum.
    pub fn nvptx_addrspace(self) -> i32 {
        match self {
            AddressSpace::Host => 0,
            AddressSpace::Global => 1,
            AddressSpace::Workgroup => 3,
            AddressSpace::Private => 5,
        }
    }

    /// The MLIR `#gpu.address_space<..>` attribute this corresponds to, for the GPU lowering path
    /// that consumes `--convert-gpu-to-nvvm` (#251). `None` for host memory, which carries no
    /// attribute.
    pub fn gpu_attr(self) -> Option<&'static str> {
        match self {
            AddressSpace::Host => None,
            AddressSpace::Global => Some("#gpu.address_space<global>"),
            AddressSpace::Workgroup => Some("#gpu.address_space<workgroup>"),
            AddressSpace::Private => Some("#gpu.address_space<private>"),
        }
    }
}

/// The address space of a *built-in* memory space. A user-declared (`Custom`) space has no
/// intrinsic address space -- it is whatever its declaration's `scope:` says -- so this returns
/// `None` for it and callers use [`declared_address_space`], which has the declaration in hand.
/// `None` also for the network spaces, which have no GPU analogue at all (#258).
pub fn builtin_address_space(mem: &MemorySpace) -> Option<AddressSpace> {
    match mem {
        MemorySpace::CPUDRAM => Some(AddressSpace::Host),
        // Device global memory.
        MemorySpace::NPUHBM | MemorySpace::GpuHbm => Some(AddressSpace::Global),
        // On-chip scratchpad is *shared* memory, not the reserved space it used to be given.
        MemorySpace::LocalSRAM => Some(AddressSpace::Workgroup),
        // Network-attached memory is not addressable from a GPU kernel; there is no honest
        // mapping, so callers must diagnose rather than silently pick one.
        MemorySpace::NicRam | MemorySpace::RemoteHbm => None,
        MemorySpace::Custom(_) => None,
    }
}

/// The address space of a memory space, consulting its declaration when it is user-declared: the
/// `scope:` facet is the source of truth (`device` => global, `sm`/`cta` => workgroup, `thread` =>
/// private), so `Memory SMEM { scope: sm }` lands in shared without the programmer ever naming an
/// address space, and two declared spaces at different scopes stay distinguishable.
///
/// `None` when no honest mapping exists -- an undeclared custom space, a declaration with no
/// `scope:`, or a space with no GPU analogue. The caller reports that rather than defaulting,
/// which is how every declared space used to become read-only constant memory (#258).
pub fn declared_address_space(
    mem: &MemorySpace,
    decl: Option<&crate::syntax::MemoryDecl>,
) -> Option<AddressSpace> {
    if let Some(built_in) = builtin_address_space(mem) {
        return Some(built_in);
    }
    match decl?.scope? {
        crate::syntax::Scope::Device => Some(AddressSpace::Global),
        crate::syntax::Scope::Sm | crate::syntax::Scope::Cta => Some(AddressSpace::Workgroup),
        crate::syntax::Scope::Thread => Some(AddressSpace::Private),
    }
}

/// The address space for a topology, derived from its default memory space so that a value
/// expressed as `Pinned<T, GPU>` and one as `Ref<T, GpuHbm>` land in the same address space
/// (previously two separate maps disagreed -- GPU was 5 but GpuHbm was 1). `decls` supplies the
/// compilation's memory declarations so a custom topology's space is scope-mapped (#258); `None`
/// when the space has no honest mapping, which the caller diagnoses.
pub fn topology_address_space(
    top: &Topology,
    decls: &std::collections::HashMap<MemorySpace, crate::syntax::MemoryDecl>,
    topologies: &std::collections::HashMap<crate::symbol::Symbol, TopologyDescriptor>,
) -> Option<AddressSpace> {
    if matches!(top, Topology::Current) {
        return Some(AddressSpace::Host);
    }
    // The topology's default memory space determines its address space. Built-ins are known
    // statically; a *declared* topology names its memory (`Topology SmemDev { memory:
    // Memory::SMEM }`), so consult its descriptor before falling back to the like-named
    // convention (`Memory::Foo <-> Topology::Foo`) for an undeclared one.
    let space = builtin_descriptors()
        .get(&top.kind())
        .map(|d| d.default_space.clone())
        .or_else(|| match top {
            Topology::Custom(name) => topologies.get(name).map(|d| d.default_space.clone()),
            _ => None,
        })
        .unwrap_or_else(|| match top {
            Topology::Custom(name) => MemorySpace::from_name(name.as_ref()),
            _ => MemorySpace::CPUDRAM,
        });
    declared_address_space(&space, decls.get(&space))
}

/// The space a topology kind holds, from the built-in table alone.
///
/// The parser has no declaration table, so a placement written as a device starts here and
/// `resolve_names` corrects it where a declaration says otherwise. A custom kind falls back to
/// the like-named space, which is the convention `default_memory_for` already applies.
pub fn builtin_default_space(kind: &crate::syntax::TopologyKind) -> MemorySpace {
    use crate::syntax::TopologyKind as K;
    match kind {
        K::CPU | K::AMX | K::CpuAvx512 | K::CpuNeon => MemorySpace::CPUDRAM,
        K::GPU => MemorySpace::GpuHbm,
        K::NPU | K::ANE | K::Slice => MemorySpace::NPUHBM,
        K::AccCore => MemorySpace::LocalSRAM,
        K::Custom(name) => MemorySpace::from_name(name.as_ref()),
        K::Current => MemorySpace::CPUDRAM,
    }
}

/// The topology a built-in memory space belongs to.
///
/// Stated rather than derived. The obvious derivation -- invert `default_space` over the
/// descriptor table -- is not a function: `CPU_DRAM` is the default of CPU, AMX, CpuAvx512 and
/// CpuNeon, and `NPU_HBM` of NPU, ANE and Slice. Nor does `within:` answer it, since that is a
/// containment relation whose root is always host memory, which would make GPU memory
/// host-owned. Ownership is a fact about the space, so it is written here once.
///
/// `NIC_RAM` and `Remote_HBM` are the host's provisionally. A NIC is host-attached, so that one
/// is close to right; remote HBM is memory on another machine and its owner is a topology the
/// type system cannot name yet. Neither is added to any topology's visibility, so this settles
/// where they are without granting local access to them.
pub fn builtin_space_owner(space: &MemorySpace) -> Option<crate::syntax::TopologyKind> {
    use crate::syntax::TopologyKind as K;
    Some(match space {
        MemorySpace::CPUDRAM => K::CPU,
        MemorySpace::GpuHbm => K::GPU,
        MemorySpace::NPUHBM => K::NPU,
        MemorySpace::LocalSRAM => K::AccCore,
        MemorySpace::NicRam | MemorySpace::RemoteHbm => K::CPU,
        MemorySpace::Custom(_) => return None,
    })
}

/// The lowest device index, for a topology kind that carries one. A space names a kind, not an
/// instance -- `Memory::GPU_HBM` says which memory, never which GPU.
fn first_device_of(kind: &crate::syntax::TopologyKind) -> Topology {
    use crate::syntax::TopologyKind as K;
    let zero = || {
        Box::new(crate::syntax::Expr::Number(crate::syntax::NumberExpr {
            value: "0".into(),
            ty: None,
            span: crate::syntax::Span::default(),
        }))
    };
    match kind {
        K::CPU => Topology::CPU,
        K::AMX => Topology::AMX,
        K::ANE => Topology::ANE,
        K::CpuAvx512 => Topology::CpuAvx512,
        K::CpuNeon => Topology::CpuNeon,
        K::GPU => Topology::GPU(zero()),
        K::NPU | K::Slice => Topology::NPU(zero()),
        K::AccCore => Topology::AccCore(zero()),
        K::Custom(name) => Topology::Custom(name.clone()),
        K::Current => Topology::Current,
    }
}

/// The space a topology holds, over a table of topology descriptors.
///
/// The declaration wins where there is one -- `Topology SmemDev { memory: Memory::SMEM }` holds
/// `SMEM`, not the like-named `Memory::SmemDev` the built-in fallback would invent. Declarations
/// are always `TopologyKind::Custom`, so consulting them first cannot shadow a built-in.
pub fn default_space_in(
    top: &Topology,
    descriptors: &HashMap<crate::syntax::TopologyKind, TopologyDescriptor>,
) -> MemorySpace {
    let kind = top.kind();
    descriptors
        .get(&kind)
        .map(|d| d.default_space.clone())
        .unwrap_or_else(|| builtin_default_space(&kind))
}

/// The topology a memory space belongs to, over a table of topology descriptors, or why it
/// cannot be told.
///
/// This is the direction `default_space_in` does not run, and the reason a placement can be
/// written either way round: `Topology::GPU` gives the space, `Memory::GPU_HBM` gives the device.
/// A built-in space has a stated owner. A declared one is owned by the topology whose `memory:`
/// names it -- exactly one, since a space held by two devices is ambiguous in a way a default
/// would silently pick a side on.
///
/// Taking the table rather than a `TransferCostGraph` is what lets name resolution answer this
/// without building one: the graph is a per-compilation object with an all-pairs sweep in it, and
/// resolution runs before there is one.
pub fn owning_topology_in(
    space: &MemorySpace,
    descriptors: &HashMap<crate::syntax::TopologyKind, TopologyDescriptor>,
) -> Result<Topology, String> {
    if let Some(kind) = builtin_space_owner(space) {
        return Ok(first_device_of(&kind));
    }
    let mut owners: Vec<&crate::syntax::TopologyKind> = descriptors
        .iter()
        .filter(|(_, d)| d.default_space == *space)
        .map(|(k, _)| k)
        .collect();
    owners.sort_by_key(|k| format!("{k:?}"));
    match owners.as_slice() {
        [one] => Ok(first_device_of(one)),
        [] => Err(format!(
            "no topology declares `Memory::{}` as its memory, so there is no device it \
             names; give a topology `memory: Memory::{}`",
            space.name(),
            space.name()
        )),
        many => Err(format!(
            "`Memory::{}` is declared as the memory of {} topologies ({}), so which device \
             it names is ambiguous; write the topology instead",
            space.name(),
            many.len(),
            many.iter()
                .map(|k| format!("{k:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// The built-in topology descriptions, encoding what `arch.rs` previously hardcoded.
/// `visibility` includes each topology's own `default_space`, which subsumes the old
/// "a topology sees its own memory" special-cases (NPU→NPUHBM, AccCore→LocalSRAM,
/// GPU→GpuHbm).
fn builtin_descriptors() -> HashMap<crate::syntax::TopologyKind, TopologyDescriptor> {
    use crate::syntax::TopologyKind as K;
    use MemorySpace::*;
    let d = |default_space: MemorySpace, visibility: &[MemorySpace]| TopologyDescriptor {
        arch: None,
        default_space,
        visibility: visibility.to_vec(),
        transfers: Vec::new(), // built-in transfer edges live in TransferCostGraph::default
        // The built-ins describe a class of device rather than a part, and which element types a
        // GPU has is a property of the part. Undeclared, so they constrain nothing.
        dtypes: None,
    };
    let mut m = HashMap::new();
    m.insert(K::CPU, d(CPUDRAM, &[CPUDRAM, NPUHBM]));
    m.insert(K::GPU, d(GpuHbm, &[GpuHbm, CPUDRAM]));
    m.insert(K::NPU, d(NPUHBM, &[NPUHBM]));
    m.insert(K::ANE, d(NPUHBM, &[NPUHBM, CPUDRAM]));
    m.insert(K::AMX, d(CPUDRAM, &[CPUDRAM]));
    m.insert(K::AccCore, d(LocalSRAM, &[LocalSRAM]));
    m.insert(K::CpuAvx512, d(CPUDRAM, &[CPUDRAM]));
    m.insert(K::CpuNeon, d(CPUDRAM, &[CPUDRAM]));
    m.insert(K::Slice, d(NPUHBM, &[NPUHBM]));
    m
}

// Topology descriptors are no longer a process-global registry. A `TransferCostGraph` carries
// them per-compilation (`descriptors` field, seeded from the built-ins plus `Program.topologies`
// via `seed_from_topologies`); see `descriptor` / `default_memory_for` methods below. This is the
// data-oriented, lock-free model of docs/parallel_compiler_architecture.md.

impl Default for TransferCostGraph {
    fn default() -> Self {
        let mut graph = Self {
            transfer_edges: HashMap::new(),
            cost_matrix: HashMap::new(),
            edge_rates: HashMap::new(),
            route_cost: HashMap::new(),
            derived_edges: Vec::new(),
            edge_declared: HashMap::new(),
            descriptors: builtin_descriptors(),
        };

        // Standard Transfer Paths
        // Host <-> HBM
        graph.add_transfer_edge(MemorySpace::CPUDRAM, MemorySpace::NPUHBM, 50);
        graph.add_transfer_edge(MemorySpace::NPUHBM, MemorySpace::CPUDRAM, 50);

        // Host <-> discrete-GPU HBM (x86-TSO host <-> NVPTX scoped-RC11 device).
        // The seam at this boundary is checked by the per-seam obligation; the
        // device kernel is lowered through the NVPTX backend (see eval G1).
        graph.add_transfer_edge(MemorySpace::CPUDRAM, MemorySpace::GpuHbm, 50);
        graph.add_transfer_edge(MemorySpace::GpuHbm, MemorySpace::CPUDRAM, 50);

        // HBM <-> SRAM
        graph.add_transfer_edge(MemorySpace::NPUHBM, MemorySpace::LocalSRAM, 10);
        graph.add_transfer_edge(MemorySpace::LocalSRAM, MemorySpace::NPUHBM, 10);

        // --- NEW: Complex Topology Simulation Paths ---

        // Host -> RemoteGPU (Cost 300)
        graph.add_transfer_edge(MemorySpace::CPUDRAM, MemorySpace::RemoteHbm, 300);

        // NPU -> NIC (Cost 5)
        graph.add_transfer_edge(MemorySpace::NPUHBM, MemorySpace::NicRam, 5);

        // NIC -> RemoteGPU (Cost 20)
        graph.add_transfer_edge(MemorySpace::NicRam, MemorySpace::RemoteHbm, 20);

        // Topology→MemorySpace visibility is described by the topology registry
        // (`builtin_descriptors`), not built here.

        graph.precompute_costs();
        graph
    }
}

// How many times the shortest-path sweep has run on this thread, and over how many spaces each
// time. Test-only, and a `thread_local` rather than a global counter: per-thread state is not
// shared mutable state, so it neither violates the isolation the compiler guarantees nor trips
// the lint that enforces it.
//
// This exists because the sweep is `O(spaces^2)` searches and it is the whole of `env_build`,
// which is serial. Running it once more than needed does not change any answer -- the second
// sweep overwrites the first -- so nothing observable goes wrong and the only symptom is that
// every compile of a program declaring machines gets slower. That is exactly the kind of
// regression that is invisible until someone measures, and it has now happened twice: once per
// function before the graph was hoisted to one per compilation, and once per compile after
// (Vx#380).
//
// A plain comment rather than a doc comment: rustdoc generates nothing for a macro
// invocation, so `///` here is rejected by the `unused_doc_comments` lint under -D warnings.
#[cfg(test)]
thread_local! {
    static SWEEP_SIZES: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Forget the sweeps recorded so far on this thread.
#[cfg(test)]
pub(crate) fn reset_sweep_sizes() {
    SWEEP_SIZES.with(|v| v.borrow_mut().clear());
}

/// The space count of every sweep run on this thread since the last reset, in order.
#[cfg(test)]
pub(crate) fn sweep_sizes() -> Vec<usize> {
    SWEEP_SIZES.with(|v| v.borrow().clone())
}

impl TransferCostGraph {
    pub fn add_transfer_edge(&mut self, src: MemorySpace, dst: MemorySpace, cost: u32) {
        self.transfer_edges
            .entry(src)
            .or_default()
            .push((dst, cost));
    }

    /// Add a descriptor's declared transfer edges to the graph (does not recompute costs —
    /// call `precompute_costs` after, or use `seed_from_topology_registry`).
    pub fn apply_descriptor_edges(&mut self, desc: &TopologyDescriptor) {
        for e in &desc.transfers {
            self.add_transfer_edge(e.from.clone(), e.to.clone(), e.cost.routing_weight());
            if let EdgeCost::Rate(bw) = e.cost {
                self.edge_rates.insert((e.from.clone(), e.to.clone()), bw);
            }
            if let Some(c) = e.cost.declared() {
                self.edge_declared.insert((e.from.clone(), e.to.clone()), c);
            }
            // The routing key. A declared link rate prices itself; a `Derived` edge is deferred to
            // `resolve_derived_route_costs`, which needs the memory declarations; a `Fixed` weight
            // prices nothing and is deliberately left absent rather than mapped to zero.
            //
            // `min` on collision, matching `add_transfer_edge`'s accumulate-and-take-the-cheapest:
            // several descriptors can contribute the same edge, and taking the last would make the
            // result depend on descriptor order.
            if let Some(attos) = e.cost.per_byte_attos() {
                let key = (e.from.clone(), e.to.clone());
                let best = self.route_cost.get(&key).map_or(attos, |&c| c.min(attos));
                self.route_cost.insert(key, best);
            } else if matches!(e.cost, EdgeCost::Derived) {
                self.derived_edges.push((e.from.clone(), e.to.clone()));
            }
        }
    }

    /// Price the `Derived` edges from the containment tree, then rebuild the shortest-path matrix.
    ///
    /// Separate from seeding because the two halves of a machine file land in different places: the
    /// graph is built from `Topology { transfer ... }` and the roofline from `Memory { bandwidth:
    /// ... }`. Until this runs, a containment hop is routable but unpriced, and routing will avoid
    /// it in favour of anything that declares a rate -- which is wrong on every fleet SKU, where
    /// the on-die hops are containment edges. Callers that build a graph from a machine file must
    /// call this; `TypeChecker`'s environment does.
    ///
    /// The probe is 1 GiB rather than one byte: `hop_cost` rounds each hop up to a whole unit, so a
    /// one-byte probe would report every edge at the same 1-unit floor and rank them all equal.
    pub fn resolve_derived_route_costs<'a, I>(&mut self, memories: I)
    where
        I: IntoIterator<Item = &'a crate::syntax::MemoryDecl>,
    {
        const PROBE_BYTES: u64 = 1 << 30;
        let hierarchy = crate::hir::memory::MemoryHierarchy::build(memories);
        let edges = std::mem::take(&mut self.derived_edges);
        for (from, to) in &edges {
            // Only a seconds-denominated roofline can share a scale with a link rate. A cycle
            // count would need a clock to convert, and a machine file may decline to declare one;
            // leaving it unpriced keeps such an edge routable but never preferred on a false
            // comparison.
            if let Some(d) = hierarchy.derived_transfer_cost(from, to, PROBE_BYTES) {
                if d.per == crate::syntax::RatePer::Second {
                    // `hop_cost` yields picoseconds; 1 ps = 10^6 as.
                    let attos = (d.value as u128 * 1_000_000) / PROBE_BYTES as u128;
                    if let Ok(a) = u64::try_from(attos) {
                        self.route_cost.insert((from.clone(), to.clone()), a);
                    }
                }
            }
        }
        self.derived_edges = edges;
        self.precompute_costs();
    }

    /// Add the transfer edges declared by every registered topology (built-in + user-defined)
    /// and recompute shortest paths. `TransferCostGraph::default()` deliberately does *not* do
    /// this so it stays hermetic; the real compiler path (`TypeChecker::new`) calls this so
    /// user-declared morphisms take effect.
    /// Fold in the topologies declared by *this* compilation (`Program.topologies`): record each
    /// descriptor and add its transfer edges, then recompute the shortest-path matrix. The
    /// per-compilation replacement for the old global-registry seed.
    pub fn seed_from_topologies(&mut self, topologies: &[TopologyDecl]) {
        self.add_topology_edges(topologies);
        self.precompute_costs();
    }

    /// The edge-and-descriptor half of `seed_from_topologies`, without the shortest-path sweep.
    ///
    /// Separate because `precompute_costs` is O(spaces^2) shortest-path searches, and a caller that
    /// is about to change edge costs anyway -- `resolve_derived_route_costs` does, and ends with its
    /// own sweep -- would have the first sweep's whole result overwritten. On a program declaring
    /// one machine that waste is invisible. On one declaring 800 it was HALF of `env_build`, which
    /// is itself serial, so it came straight off the critical path of every compile.
    ///
    /// This is the same defect this graph had once before, one level up: the sweep used to run
    /// twice per *function*, in `default()` and again in `seed_from_topologies`. Hoisting the graph
    /// to one per compilation (see `GlobalAstEnv::build`) fixed the per-function part and left the
    /// double sweep in place.
    pub fn add_topology_edges(&mut self, topologies: &[TopologyDecl]) {
        for decl in topologies {
            self.apply_descriptor_edges(&decl.descriptor);
            self.descriptors.insert(
                crate::syntax::TopologyKind::Custom(decl.name.clone()),
                decl.descriptor.clone(),
            );
        }
    }

    /// Returns the default memory space for a given topology, from its registered
    /// descriptor (see `topology_descriptor`).
    /// The descriptor for a topology kind (built-in or a topology declared in this compilation),
    /// from the graph's per-compilation `descriptors` — no global registry.
    pub fn descriptor(&self, kind: &crate::syntax::TopologyKind) -> Option<&TopologyDescriptor> {
        self.descriptors.get(kind)
    }

    /// The topology a memory space belongs to, or why it cannot be told.
    ///
    /// A built-in space has a stated owner. A declared one is owned by the topology whose
    /// `memory:` names it -- exactly one, since a space held by two devices is ambiguous in a
    /// way a default would silently pick a side on.
    ///
    /// This is the direction `default_memory_for` does not run, and the reason a placement can
    /// be written either way round: `Topology::GPU` gives the space, `Memory::GPU_HBM` gives
    /// the device.
    pub fn owning_topology(&self, space: &MemorySpace) -> Result<Topology, String> {
        owning_topology_in(space, &self.descriptors)
    }

    /// Is this space one some topology in this compilation holds?
    ///
    /// Held is wider than owned: `owning_topology` asks which single device's `memory:` names the
    /// space, and a space that several devices can see, or that only appears in a `visible:` list,
    /// has no single owner while still being a real place. A space no topology names either way is
    /// nowhere, and a value cannot be put there.
    pub fn is_space_held(&self, space: &MemorySpace) -> bool {
        self.descriptors
            .values()
            .any(|d| d.default_space == *space || d.visibility.contains(space))
    }

    /// The default memory space a topology's values live in.
    pub fn default_memory_for(&self, topology: &Topology) -> MemorySpace {
        if let Topology::Current = topology {
            unreachable!("Must specify a concrete topology other than Current")
        }
        self.descriptor(&topology.kind())
            .map(|d| d.default_space.clone())
            .unwrap_or_else(|| match topology {
                // The `Memory::Foo <-> Topology::Foo` naming convention, in the direction
                // `topology_address_space` already applies: a transfer into a custom space
                // records residence as `Pinned<_, Topology::Foo>`, so an *undeclared* custom
                // topology's memory is the like-named space — not host DRAM, which made every
                // custom-space value look CPU-resident to the visibility check (#253).
                Topology::Custom(name) => MemorySpace::from_name(name.as_ref()),
                _ => MemorySpace::CPUDRAM,
            })
    }

    /// Precomputes the all-pairs shortest path transfer costs. Covers the built-in spaces
    /// plus any user-defined (`Custom`) spaces that appear in declared transfer edges, so
    /// `transfer_cost`/`can_transfer` work for custom memory too.
    pub fn precompute_costs(&mut self) {
        let mut spaces: std::collections::HashSet<MemorySpace> = [
            MemorySpace::CPUDRAM,
            MemorySpace::NPUHBM,
            MemorySpace::GpuHbm,
            MemorySpace::LocalSRAM,
            MemorySpace::NicRam,
            MemorySpace::RemoteHbm,
        ]
        .into_iter()
        .collect();
        for (src, neighbors) in &self.transfer_edges {
            spaces.insert(src.clone());
            for (dst, _) in neighbors {
                spaces.insert(dst.clone());
            }
        }
        let mut spaces: Vec<MemorySpace> = spaces.into_iter().collect();
        #[cfg(test)]
        SWEEP_SIZES.with(|v| v.borrow_mut().push(spaces.len()));
        // A HashSet has no order, and this is the node set every pair below is drawn from. Sorting
        // it costs nothing next to the searches and makes the work split the same way on every run.
        spaces.sort_by_key(|s| format!("{s:?}"));

        // Every pair is an independent pair of Dijkstra passes reading `&self`, and none of them
        // reads `cost_matrix` -- the matrix being filled is not an input to filling it. So this
        // loop is embarrassingly parallel, and on a program declaring many machines it is the
        // whole of a serial phase (Vx#363). It stays sequential here because `precompute_costs`
        // cannot see the compile's `Schedule`, and a bare `par_iter` would make the rayon-free
        // baseline parallel -- measured at 800 machines, the "sequential" column came out at
        // 140 ms against the 1-thread column's 393 ms, which makes every ratio taken against it
        // meaningless.
        for src in &spaces {
            for dst in &spaces {
                if let Some((cost, _)) = self.transfer_path(src, dst) {
                    self.cost_matrix.insert((src.clone(), dst.clone()), cost);
                }
            }
        }
    }

    /// Verifies if a variable belonging to `var_topology` with type `ty`
    /// is accessible from the `active_topology`.
    pub fn is_type_accessible(
        &self,
        active_topology: &Topology,
        var_topology: &Topology,
        ty: &Type,
    ) -> bool {
        if let Type::Pinned(_, pinned_top) = ty {
            if pinned_top == active_topology {
                return true;
            }
            let mem = self.default_memory_for(pinned_top);
            let mock_ty = Type::Ref(Box::new(Type::Scalar(syntax::ElementType::F32)), mem);
            return self.is_type_accessible(active_topology, pinned_top, &mock_ty);
        }

        // Where the value *is*, which after a transfer is not where it was declared.
        // The binding's topology is the scope that owns the name; a placed tensor
        // carries its own device, and letting the binding speak for it made the
        // host-resident fallback below fire for a value sitting in device memory.
        let placed_top = ty.placement().map(|p| self.placement_topology(p));
        let var_topology = placed_top.as_ref().unwrap_or(var_topology);

        // Determine the memory space of the variable. A placed tensor says where it lives in its
        // own type, so it is read there rather than reconstructed from the owning device -- which
        // is the point of carrying a placement at all (Vx#429).
        let target_mem = match (ty.placement(), ty) {
            (Some(p), _) => self.placement_space(p),
            (None, Type::Ref(_, mem)) => mem.clone(),
            _ => {
                if var_topology == active_topology {
                    return true;
                }
                self.default_memory_for(var_topology)
            }
        };

        // Visibility is now data: consult the active topology's descriptor. Its
        // `visibility` set includes its own default space, subsuming the old
        // NPU→NPUHBM / AccCore→LocalSRAM / GPU→GpuHbm special-cases.
        let active_kind = active_topology.kind();
        if let Some(desc) = self.descriptor(&active_kind) {
            if desc.visibility.contains(&target_mem) {
                return true;
            }
            // Host-resident data is visible to any topology that can see host DRAM
            // (unified-memory fallback).
            if *var_topology == Topology::CPU && desc.visibility.contains(&MemorySpace::CPUDRAM) {
                return true;
            }
        }

        false
    }

    /// The memory space where a value of type `ty` owned by `var_topology` lives.
    /// The space a placement occupies, derived from the half the source wrote.
    ///
    /// A placement's derived half is filled in by name resolution. One frontend runs that after
    /// the type checker and the other before it, and a placement the checker mints mid-check is
    /// never resolved at all -- so a rule that reads the derived half sees whatever the parser
    /// guessed. For `Topology::Dev` the guess is a like-named `Memory::Dev` that no declaration
    /// mentions, which made a tensor on a declared topology unreachable from that same topology.
    pub fn placement_space(&self, p: &crate::syntax::Placement) -> MemorySpace {
        match p.written() {
            crate::syntax::Written::Space => p.space.clone(),
            crate::syntax::Written::Device => self.default_memory_for(&p.topology),
        }
    }

    /// The device holding a placement, derived from the half the source wrote. The mirror of
    /// [`Self::placement_space`], for the same reason.
    pub fn placement_topology(&self, p: &crate::syntax::Placement) -> Topology {
        match p.written() {
            crate::syntax::Written::Device => p.topology.clone(),
            crate::syntax::Written::Space => owning_topology_in(&p.space, &self.descriptors)
                .unwrap_or_else(|_| p.topology.clone()),
        }
    }

    fn memory_of(&self, var_topology: &Topology, ty: &Type) -> MemorySpace {
        if let Some(p) = ty.placement() {
            return self.placement_space(p);
        }
        match ty {
            Type::Pinned(_, top) => self.default_memory_for(top),
            Type::Ref(_, mem) => mem.clone(),
            _ => self.default_memory_for(var_topology),
        }
    }

    /// The type-level USE verdict for reading a `var_topology` value (type `ty`) from
    /// `active_topology`. `Visible` is the existing accessibility relation (unified
    /// memory / same location, no move); otherwise we report whether a transfer path
    /// exists (`NeedsSeam`, carrying its cost) or not (`Unreachable`). This is purely
    /// a refinement of `is_type_accessible` used to produce precise diagnostics; it
    /// does not change what is accepted. See `Reachability`.
    pub fn reachable(
        &self,
        active_topology: &Topology,
        var_topology: &Topology,
        ty: &Type,
    ) -> Reachability {
        if self.is_type_accessible(active_topology, var_topology, ty) {
            return Reachability::Visible;
        }
        let var_mem = self.memory_of(var_topology, ty);
        let active_mem = self.default_memory_for(active_topology);
        match self.transfer_path(&var_mem, &active_mem) {
            Some((cost, _)) => Reachability::NeedsSeam { cost },
            None => Reachability::Unreachable,
        }
    }

    /// Determines the minimum data movement cost and path between two memory spaces using Dijkstra's algorithm.
    /// Whether any declared topology contributes a *relaxed* (`!sync`) transfer edge for the hop
    /// `from -> to`: a declared escape hatch whose visibility the seam engine cannot guarantee. A
    /// transfer's use site consults this so a hop over a declared relaxed edge is routed through the
    /// same seam obligation the `*_relaxed` intrinsics take — yielding a per-buffer E6004 at the use
    /// site, not only the coarse declaration-time W1027 (P0-3).
    pub fn is_relaxed_edge(&self, from: &MemorySpace, to: &MemorySpace) -> bool {
        self.descriptors.values().any(|d| {
            d.transfers
                .iter()
                .any(|e| &e.from == from && &e.to == to && !e.sync)
        })
    }

    /// The cheapest route from `source` to `target`, and the declared-weight sum along it.
    ///
    /// Route selection minimises PREDICTED COST, not hop count. The old objective
    /// was hop count, which is unrelated to time: on a partially-meshed box it took a one-hop SYS
    /// crawl over a two-hop NVLink relay the same graph already contained, 3.57x slower.
    ///
    /// Two passes, because the graph mixes two incomparable currencies. An edge that declares a
    /// bandwidth has a per-byte time; an edge that declares only a unitless `: N` has a relative
    /// latency that is not a time and cannot be added to one.
    ///
    ///   1. **Priced edges only, minimising attoseconds per byte.** If the target is reachable
    ///      this way, that answer wins: it is a route the model can actually predict, and a cost
    ///      the compiler reports should be one it computed.
    ///   2. **All edges, minimising the declared weights.** The pre-existing behaviour, used only
    ///      when no fully-priced route exists.
    ///
    /// Two passes rather than one lexicographic key, and that is not a style choice. A single key
    /// of (any-unpriced-hop, attoseconds, weight) is what this function had first, and it is
    /// wrong: an unpriced hop contributes zero attoseconds, so a route made *entirely* of unpriced
    /// hops scores 0 on that component and beats a route that has one unpriced hop plus a real
    /// priced one. It rewards a route for being unpriceable. `fleet_routes_exist_iff_reachable_
    /// and_are_cost_minimal` caught exactly that on `node-8gpu.vx` once its inter-device edges
    /// were given bandwidths.
    ///
    /// Counting unpriced hops instead of flagging them is also wrong, in the other direction: when
    /// every edge is unpriced the count degenerates into hop count and discards the declared
    /// weights, which are then the only information there is.
    ///
    /// Per-byte rather than for a concrete transfer, so the result is size-independent and a
    /// precomputed all-pairs matrix stays valid: every edge costs `bytes/bandwidth`, a line
    /// through the origin, so the ratio between two routes does not depend on bytes. Adding a
    /// fixed per-transfer term would break that.
    pub fn transfer_path(
        &self,
        source: &MemorySpace,
        target: &MemorySpace,
    ) -> Option<(u32, Vec<MemorySpace>)> {
        if source == target {
            return Some((0, vec![source.clone()]));
        }
        self.search(source, target, true)
            .or_else(|| self.search(source, target, false))
    }

    /// One Dijkstra pass. `priced_only` restricts the search to edges that declare a per-byte cost
    /// and minimises that cost; otherwise every edge is admissible and the declared weights are
    /// minimised.
    ///
    /// Returns the declared-weight sum along the chosen route either way. Every existing consumer
    /// -- the cost matrix, the `total_cost` field of the diagnostics record -- reads that as "the
    /// unitless figures along this route", and changing which route is chosen must not quietly
    /// change what the number means.
    fn search(
        &self,
        source: &MemorySpace,
        target: &MemorySpace,
        priced_only: bool,
    ) -> Option<(u32, Vec<MemorySpace>)> {
        use std::collections::BinaryHeap;

        #[derive(Eq, PartialEq, PartialOrd, Ord, Clone, Copy, Default)]
        struct Key {
            /// Attoseconds per byte on a priced pass; zero and unused otherwise.
            cost: u64,
            /// Declared weights, the objective on the unpriced pass and a tie-break on the priced
            /// one.
            weight: u32,
        }

        #[derive(Eq, PartialEq)]
        struct State {
            key: Key,
            mem: MemorySpace,
        }

        impl Ord for State {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other.key.cmp(&self.key) // Reverse for min-heap
            }
        }

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::new();
        let mut dists: HashMap<MemorySpace, Key> = HashMap::new();
        let mut predecessors: HashMap<MemorySpace, MemorySpace> = HashMap::new();

        heap.push(State {
            key: Key::default(),
            mem: source.clone(),
        });
        dists.insert(source.clone(), Key::default());

        while let Some(State { key, mem }) = heap.pop() {
            if mem == *target {
                let mut path = Vec::new();
                let mut curr = mem.clone();
                while curr != *source {
                    path.push(curr.clone());
                    curr = predecessors.get(&curr).unwrap().clone();
                }
                path.push(source.clone());
                path.reverse();
                return Some((key.weight, path));
            }

            if let Some(current_dist) = dists.get(&mem) {
                if key > *current_dist {
                    continue;
                }
            }

            if let Some(neighbors) = self.transfer_edges.get(&mem) {
                for (next, edge_weight) in neighbors {
                    let priced = self.route_cost.get(&(mem.clone(), next.clone())).copied();
                    if priced_only && priced.is_none() {
                        continue;
                    }
                    let next_key = Key {
                        cost: key.cost.saturating_add(if priced_only {
                            priced.unwrap_or(0)
                        } else {
                            0
                        }),
                        weight: key.weight.saturating_add(*edge_weight),
                    };
                    let is_better = dists.get(next).is_none_or(|&c| next_key < c);

                    if is_better {
                        dists.insert(next.clone(), next_key);
                        predecessors.insert(next.clone(), mem.clone());
                        heap.push(State {
                            key: next_key,
                            mem: next.clone(),
                        });
                    }
                }
            }
        }

        None
    }

    /// Determines the minimum data movement cost between two memory spaces using Dijkstra's algorithm.
    pub fn transfer_cost(&self, source: &MemorySpace, target: &MemorySpace) -> Option<u32> {
        self.cost_matrix
            .get(&(source.clone(), target.clone()))
            .copied()
    }

    /// The bandwidth declared on the `source -> target` link itself, if any.
    ///
    /// Takes precedence over the containment-derived roofline: a link rate is a statement about
    /// *this* hop, while the roofline infers one from the spaces at its ends. The two cannot both
    /// apply — E6013 rejects an edge that has a declared cost and a derivable one — so this is a
    /// lookup, not a tie-break.
    pub fn link_rate(
        &self,
        source: &MemorySpace,
        target: &MemorySpace,
    ) -> Option<syntax::Bandwidth> {
        self.edge_rates
            .get(&(source.clone(), target.clone()))
            .copied()
    }

    /// Every space that appears as an endpoint of any declared edge, in a deterministic order.
    ///
    /// The graph's node set is wider than any one machine file's `Memory` declarations, because
    /// the built-in topology seeds edges of its own. A reachability oracle that enumerates only
    /// the file's spaces is searching a subgraph and will "prove" routes non-minimal that are in
    /// fact minimal through a space it never looked at (S1).
    pub fn nodes(&self) -> Vec<MemorySpace> {
        let mut out: Vec<MemorySpace> = Vec::new();
        let mut keys: Vec<&MemorySpace> = self.transfer_edges.keys().collect();
        keys.sort_by_key(|k| format!("{k:?}"));
        for k in keys {
            if !out.contains(k) {
                out.push(k.clone());
            }
            let mut ns: Vec<&MemorySpace> = self.transfer_edges[k].iter().map(|(n, _)| n).collect();
            ns.sort_by_key(|n| format!("{n:?}"));
            for n in ns {
                if !out.contains(n) {
                    out.push(n.clone());
                }
            }
        }
        out
    }

    /// The unitless cost the file declared for this edge, if it declared one.
    ///
    /// `None` for an edge that leaves its cost to be derived — which is not the same as a cost of
    /// zero or of one, and the diagnostics record says `null` rather than quoting a routing weight
    /// nobody wrote.
    pub fn declared_edge_cost(&self, from: &MemorySpace, to: &MemorySpace) -> Option<u32> {
        self.edge_declared.get(&(from.clone(), to.clone())).copied()
    }

    /// Whether `from -> to` is a **directly declared** edge, as opposed to merely reachable.
    ///
    /// `can_transfer` consults the all-pairs matrix and so is true for multi-hop routes too;
    /// this is the single-hop question, which is what "every hop in a synthesized path is a
    /// declared edge" needs in order to mean anything (S1).
    pub fn has_direct_edge(&self, from: &MemorySpace, to: &MemorySpace) -> bool {
        self.transfer_edges
            .get(from)
            .is_some_and(|ns| ns.iter().any(|(n, _)| n == to))
    }

    /// The weight of a directly declared `from -> to` edge, if there is one. When several
    /// declarations contribute the same edge, the cheapest wins -- which is what the router uses.
    pub fn direct_edge_weight(&self, from: &MemorySpace, to: &MemorySpace) -> Option<u32> {
        self.transfer_edges
            .get(from)?
            .iter()
            .filter(|(n, _)| n == to)
            .map(|(_, w)| *w)
            .min()
    }

    /// Determines if a data transfer between two memory spaces is physically supported.
    pub fn can_transfer(&self, source: &MemorySpace, target: &MemorySpace) -> bool {
        self.cost_matrix
            .contains_key(&(source.clone(), target.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syntax::{ElementType, Expr, NumberExpr, Span};

    // The topology registry is `thread_local` (see `TOPOLOGY_REGISTRY`), and libtest runs each
    // test on its own thread, so registry-mutating tests are isolated by construction -- no
    // serialization guard is needed.

    fn make_tensor() -> Type {
        Type::Tensor(ElementType::F32, vec![], None)
    }

    fn make_npu() -> Topology {
        Topology::NPU(Box::new(Expr::Number(NumberExpr::new(
            "0".to_string(),
            None,
            Span::default(),
        ))))
    }

    fn make_acc_core() -> Topology {
        Topology::AccCore(Box::new(Expr::Number(NumberExpr::new(
            "0".to_string(),
            None,
            Span::default(),
        ))))
    }

    fn mem_decl(name: &str, scope: Option<crate::syntax::Scope>) -> crate::syntax::MemoryDecl {
        crate::syntax::MemoryDecl {
            name: name.into(),
            parent: None,
            capacity: None,
            bandwidth: None,
            clock_hz: None,
            replicas: None,
            managed: Default::default(),
            granule: None,
            scope,
            overcommit: false,
            crossing: crate::syntax::Crossing::default(),
            doc_comment: None,
        }
    }

    /// #348: a transfer target names one space, and network memory names its own.
    ///
    /// `NicRam` and `RemoteHbm` both answered 300, which is `Topology::AMX`'s dispatch id. That is
    /// not a missing feature but a wrong answer that reads as a valid one: a `transfer` into a
    /// peer's memory arrived at a plugin as a request for Apple's matrix coprocessor, and no
    /// diagnostic could object because the number was in range. The two network spaces were also
    /// indistinguishable from each other, so staging into a NIC buffer and landing in a peer's HBM
    /// were the same event.
    ///
    /// The property under test is uniqueness, not the particular constants: any two spaces that a
    /// program can name separately must dispatch separately, or the runtime cannot honour a
    /// distinction the type system spent effort maintaining.
    #[test]
    fn every_memory_space_dispatches_distinctly() {
        let spaces = [
            MemorySpace::CPUDRAM,
            MemorySpace::NPUHBM,
            MemorySpace::LocalSRAM,
            MemorySpace::GpuHbm,
            MemorySpace::NicRam,
            MemorySpace::RemoteHbm,
        ];
        for (i, a) in spaces.iter().enumerate() {
            for b in &spaces[i + 1..] {
                assert_ne!(
                    memory_space_dispatch_id(a),
                    memory_space_dispatch_id(b),
                    "{a:?} and {b:?} share a dispatch id"
                );
            }
        }

        // And specifically off AMX, which is the collision that existed.
        assert_ne!(
            memory_space_dispatch_id(&MemorySpace::NicRam),
            topology_dispatch_id(&Topology::AMX)
        );
        assert_ne!(
            memory_space_dispatch_id(&MemorySpace::RemoteHbm),
            topology_dispatch_id(&Topology::AMX)
        );

        // The bands the header mirrors (VX_TOPO_NIC_BASE / VX_TOPO_REMOTE_BASE). Pinned because a
        // plugin decodes against those constants, so the two files must not drift.
        assert_eq!(memory_space_dispatch_id(&MemorySpace::NicRam), 800);
        assert_eq!(memory_space_dispatch_id(&MemorySpace::RemoteHbm), 900);
    }

    /// #258: the built-in spaces map to what they *are*, not to the numbers the old table
    /// hardcoded — on-chip scratchpad is shared memory (NVPTX 3), never the reserved space 2.
    #[test]
    fn builtin_spaces_map_to_correct_address_spaces() {
        assert_eq!(
            builtin_address_space(&MemorySpace::CPUDRAM),
            Some(AddressSpace::Host)
        );
        assert_eq!(
            builtin_address_space(&MemorySpace::GpuHbm),
            Some(AddressSpace::Global)
        );
        assert_eq!(
            builtin_address_space(&MemorySpace::NPUHBM),
            Some(AddressSpace::Global)
        );
        // The headline correction: scratchpad is shared (3), not the reserved NVPTX space 2.
        assert_eq!(
            builtin_address_space(&MemorySpace::LocalSRAM),
            Some(AddressSpace::Workgroup)
        );
        assert_eq!(AddressSpace::Workgroup.nvptx_addrspace(), 3);
        assert_eq!(AddressSpace::Global.nvptx_addrspace(), 1);
        assert_eq!(AddressSpace::Private.nvptx_addrspace(), 5);
        assert_eq!(AddressSpace::Host.nvptx_addrspace(), 0);
        // No NVPTX number is ever 4 (constant/read-only) — the space every declared memory
        // used to collapse onto.
        for a in [
            AddressSpace::Host,
            AddressSpace::Global,
            AddressSpace::Workgroup,
            AddressSpace::Private,
        ] {
            assert_ne!(a.nvptx_addrspace(), 4, "{a:?} must not be constant memory");
        }
        // Network memory has no GPU analogue: no silent answer.
        assert_eq!(builtin_address_space(&MemorySpace::NicRam), None);
        assert_eq!(builtin_address_space(&MemorySpace::RemoteHbm), None);
    }

    /// #258's more serious defect: every user-declared space used to become address space 4.
    /// The `scope:` facet now distinguishes them, and an unscoped/undeclared one declines.
    #[test]
    fn declared_spaces_map_by_scope_and_decline_when_unmappable() {
        use crate::syntax::Scope;
        let smem = MemorySpace::Custom("SMEM".into());
        let l2 = MemorySpace::Custom("L2".into());
        let regs = MemorySpace::Custom("REGS".into());
        let tmem = MemorySpace::Custom("TMEM".into());

        assert_eq!(
            declared_address_space(&smem, Some(&mem_decl("SMEM", Some(Scope::Sm)))),
            Some(AddressSpace::Workgroup)
        );
        assert_eq!(
            declared_address_space(&l2, Some(&mem_decl("L2", Some(Scope::Device)))),
            Some(AddressSpace::Global)
        );
        assert_eq!(
            declared_address_space(&regs, Some(&mem_decl("REGS", Some(Scope::Thread)))),
            Some(AddressSpace::Private)
        );
        // A CTA-private space shares the workgroup space with an SM-private one.
        assert_eq!(
            declared_address_space(&smem, Some(&mem_decl("SMEM", Some(Scope::Cta)))),
            Some(AddressSpace::Workgroup)
        );
        // SMEM (sm) and L2 (device) are distinguishable — they used to be identical.
        assert_ne!(
            declared_address_space(&smem, Some(&mem_decl("SMEM", Some(Scope::Sm)))),
            declared_address_space(&l2, Some(&mem_decl("L2", Some(Scope::Device)))),
        );
        // No `scope:` and no declaration at all: decline, so the caller diagnoses rather than
        // lowering stores into read-only constant memory.
        assert_eq!(
            declared_address_space(&tmem, Some(&mem_decl("TMEM", None))),
            None
        );
        assert_eq!(declared_address_space(&tmem, None), None);
    }

    #[test]
    fn test_default_memory_mappings() {
        let g = TransferCostGraph::default();
        assert_eq!(g.default_memory_for(&Topology::CPU), MemorySpace::CPUDRAM);
        // A discrete GPU's home memory is its own device HBM (not host DRAM):
        // the host<->device boundary is a real seam, checked by the per-seam obligation.
        assert_eq!(g.default_memory_for(&Topology::gpu(0)), MemorySpace::GpuHbm);
        assert_eq!(g.default_memory_for(&Topology::AMX), MemorySpace::CPUDRAM);
        assert_eq!(g.default_memory_for(&Topology::ANE), MemorySpace::NPUHBM);
        assert_eq!(g.default_memory_for(&make_npu()), MemorySpace::NPUHBM);
        assert_eq!(
            g.default_memory_for(&make_acc_core()),
            MemorySpace::LocalSRAM
        );
    }

    #[test]
    fn test_accessibility_same_topology() {
        let graph = TransferCostGraph::default();
        let ty = make_tensor();
        // Exact same topology is always accessible
        assert!(graph.is_type_accessible(&Topology::CPU, &Topology::CPU, &ty));
        assert!(graph.is_type_accessible(&Topology::ANE, &Topology::ANE, &ty));
        assert!(graph.is_type_accessible(&make_npu(), &make_npu(), &ty));
    }

    #[test]
    fn test_accessibility_host_unified_memory() {
        let graph = TransferCostGraph::default();
        let ty = make_tensor();
        // AMX, ANE, GPU can read variables stored in Host topology (unified-memory
        // fallback: each has a visibility edge to CPUDRAM).
        assert!(graph.is_type_accessible(&Topology::AMX, &Topology::CPU, &ty));
        assert!(graph.is_type_accessible(&Topology::ANE, &Topology::CPU, &ty));
        assert!(graph.is_type_accessible(&Topology::gpu(0), &Topology::CPU, &ty));

        // ANE defaults to NPUHBM, which Host can see, so Host can read an ANE var.
        assert!(graph.is_type_accessible(&Topology::CPU, &Topology::ANE, &ty));
        // A discrete GPU's var lives in GPU HBM, which the host CANNOT see directly:
        // the host<->device boundary is exactly the seam the obligation guards.
        assert!(!graph.is_type_accessible(&Topology::CPU, &Topology::gpu(0), &ty));
    }

    #[test]
    fn test_accessibility_pinned_memory() {
        let graph = TransferCostGraph::default();
        let pinned_ane = Type::Pinned(Box::new(make_tensor()), Topology::ANE);
        let pinned_host = Type::Pinned(Box::new(make_tensor()), Topology::CPU);

        // ANE can access ANE pinned
        assert!(graph.is_type_accessible(&Topology::ANE, &Topology::CPU, &pinned_ane));
        // Host CAN access ANE pinned (as a handle)
        assert!(graph.is_type_accessible(&Topology::CPU, &Topology::CPU, &pinned_ane));
        // Host can access Host pinned
        assert!(graph.is_type_accessible(&Topology::CPU, &Topology::ANE, &pinned_host));
    }

    #[test]
    fn test_accessibility_memory_space_refs() {
        let graph = TransferCostGraph::default();
        let ref_hbm = Type::Ref(Box::new(make_tensor()), MemorySpace::NPUHBM);
        let ref_dram = Type::Ref(Box::new(make_tensor()), MemorySpace::CPUDRAM);

        // HBM reachable by NPU and ANE and Host
        assert!(graph.is_type_accessible(&make_npu(), &Topology::CPU, &ref_hbm));
        assert!(graph.is_type_accessible(&Topology::ANE, &Topology::CPU, &ref_hbm));
        assert!(graph.is_type_accessible(&Topology::CPU, &Topology::CPU, &ref_hbm));
        assert!(!graph.is_type_accessible(&make_acc_core(), &Topology::CPU, &ref_hbm)); // AccCore has LocalSRAM

        // DRAM reachable by Host, AMX, GPU, ANE
        assert!(graph.is_type_accessible(&Topology::CPU, &Topology::ANE, &ref_dram));
        assert!(graph.is_type_accessible(&Topology::AMX, &Topology::ANE, &ref_dram));
        assert!(graph.is_type_accessible(&Topology::gpu(0), &Topology::ANE, &ref_dram));
        assert!(graph.is_type_accessible(&Topology::ANE, &Topology::CPU, &ref_dram));
        // NPU doesn't directly reach DRAM in this default unified memory model
        assert!(!graph.is_type_accessible(&make_npu(), &Topology::CPU, &ref_dram));
    }

    #[test]
    fn test_transfer_legal_paths() {
        let graph = TransferCostGraph::default();
        // Identity
        assert!(graph.can_transfer(&MemorySpace::CPUDRAM, &MemorySpace::CPUDRAM));

        // Host <-> NPU HBM
        assert!(graph.can_transfer(&MemorySpace::CPUDRAM, &MemorySpace::NPUHBM));
        assert!(graph.can_transfer(&MemorySpace::NPUHBM, &MemorySpace::CPUDRAM));

        // NPU HBM <-> Local SRAM
        assert!(graph.can_transfer(&MemorySpace::NPUHBM, &MemorySpace::LocalSRAM));
        assert!(graph.can_transfer(&MemorySpace::LocalSRAM, &MemorySpace::NPUHBM));
    }

    #[test]
    fn test_transfer_multi_hop_paths() {
        let graph = TransferCostGraph::default();
        // Local SRAM <-> Host DRAM (BFS multi-hop routing makes this valid)
        assert!(graph.can_transfer(&MemorySpace::LocalSRAM, &MemorySpace::CPUDRAM));
        assert!(graph.can_transfer(&MemorySpace::CPUDRAM, &MemorySpace::LocalSRAM));

        // Let's actually verify the multi-hop path array generated
        let path = graph.transfer_path(&MemorySpace::CPUDRAM, &MemorySpace::LocalSRAM);
        assert!(path.is_some());
        let (cost, hops) = path.unwrap();
        // Default graph: CPUDRAM -> NPUHBM -> LocalSRAM
        // cost = 50 + 10 = 60
        assert_eq!(cost, 60);
        assert_eq!(
            hops,
            vec![
                MemorySpace::CPUDRAM,
                MemorySpace::NPUHBM,
                MemorySpace::LocalSRAM
            ]
        );
    }
    #[test]
    fn test_transfer_cost_same_space_is_zero() {
        let graph = TransferCostGraph::default();
        assert_eq!(
            graph.transfer_cost(&MemorySpace::CPUDRAM, &MemorySpace::CPUDRAM),
            Some(0)
        );
        assert_eq!(
            graph.transfer_cost(&MemorySpace::NPUHBM, &MemorySpace::NPUHBM),
            Some(0)
        );
        assert_eq!(
            graph.transfer_cost(&MemorySpace::LocalSRAM, &MemorySpace::LocalSRAM),
            Some(0)
        );
    }

    #[test]
    fn test_transfer_cost_direct_hop() {
        let graph = TransferCostGraph::default();
        assert_eq!(
            graph.transfer_cost(&MemorySpace::CPUDRAM, &MemorySpace::NPUHBM),
            Some(50)
        );
        assert_eq!(
            graph.transfer_cost(&MemorySpace::NPUHBM, &MemorySpace::CPUDRAM),
            Some(50)
        );
        assert_eq!(
            graph.transfer_cost(&MemorySpace::NPUHBM, &MemorySpace::LocalSRAM),
            Some(10)
        );
    }

    #[test]
    fn test_transfer_cost_multi_hop() {
        let graph = TransferCostGraph::default();
        // CPUDRAM -> NPUHBM -> LocalSRAM = 50 + 10 = 60
        assert_eq!(
            graph.transfer_cost(&MemorySpace::CPUDRAM, &MemorySpace::LocalSRAM),
            Some(60)
        );
    }

    #[test]
    fn test_transfer_cost_nic_to_remote_cheaper_than_direct() {
        let graph = TransferCostGraph::default();
        // Direct: CPUDRAM -> RemoteHbm = 300
        // Via NIC: CPUDRAM -> NPUHBM -> NicRam -> RemoteHbm = 50 + 5 + 20 = 75
        // Dijkstra should find the cheaper path
        let cost = graph.transfer_cost(&MemorySpace::CPUDRAM, &MemorySpace::RemoteHbm);
        assert_eq!(cost, Some(75));

        // Verify the path goes via NIC
        let (_, hops) = graph
            .transfer_path(&MemorySpace::CPUDRAM, &MemorySpace::RemoteHbm)
            .unwrap();
        assert_eq!(
            hops,
            vec![
                MemorySpace::CPUDRAM,
                MemorySpace::NPUHBM,
                MemorySpace::NicRam,
                MemorySpace::RemoteHbm,
            ]
        );
    }

    // --- Dijkstra shortest-path over a controlled custom graph ---------------------------------
    // `transfer_path` runs Dijkstra live over the declared edges; `precompute_costs` fills the
    // cost matrix by Dijkstra over all pairs. These build a small graph of custom spaces (which do
    // not collide with the built-in edges) to exercise the search in isolation.

    /// A custom memory space named `name`.
    fn cs(name: &str) -> MemorySpace {
        MemorySpace::Custom(crate::symbol::Symbol::from(name))
    }

    /// A `TopologyDescriptor` carrying one edge, for the routing tests below.
    fn rate_edge(from: MemorySpace, to: MemorySpace, gb_per_s: u64) -> TransferEdge {
        TransferEdge {
            from,
            to,
            cost: EdgeCost::Rate(syntax::Bandwidth {
                bytes: gb_per_s * 1_000_000_000,
                per: syntax::RatePer::Second,
            }),
            sync: true,
            copy_engine: false,
        }
    }

    #[test]
    fn routing_takes_the_faster_route_not_the_shorter_one() {
        // M5. The shape of a partially-meshed box: A reaches C directly over a slow
        // link, or in two hops over fast ones. Hop count picks the slow direct edge; cost picks the
        // relay, which is 3.57x faster end to end.
        let desc = TopologyDescriptor {
            default_space: cs("A"),
            visibility: vec![cs("A"), cs("B"), cs("C")],
            transfers: vec![
                rate_edge(cs("A"), cs("C"), 63),  // SYS: one hop, slow
                rate_edge(cs("A"), cs("B"), 450), // NVLink relay: two hops, fast
                rate_edge(cs("B"), cs("C"), 450),
            ],
            arch: None,
            dtypes: None,
        };
        let mut g = TransferCostGraph::default();
        g.apply_descriptor_edges(&desc);
        g.precompute_costs();
        let (_, path) = g.transfer_path(&cs("A"), &cs("C")).unwrap();
        assert_eq!(
            path,
            vec![cs("A"), cs("B"), cs("C")],
            "route selection must minimise predicted cost, not hop count"
        );
    }

    #[test]
    fn routing_prefers_a_route_it_can_price() {
        // A priced two-hop route against an unpriced direct edge whose declared weight (1) is as
        // low as it gets. The priced route wins: an unpriced edge is a gap in the machine file,
        // and a cost the compiler reports has to be a cost it actually computed.
        let desc = TopologyDescriptor {
            default_space: cs("A"),
            visibility: vec![cs("A"), cs("B"), cs("C")],
            transfers: vec![
                TransferEdge {
                    from: cs("A"),
                    to: cs("C"),
                    cost: EdgeCost::Fixed(1),
                    sync: true,
                    copy_engine: false,
                },
                rate_edge(cs("A"), cs("B"), 1),
                rate_edge(cs("B"), cs("C"), 1),
            ],
            arch: None,
            dtypes: None,
        };
        let mut g = TransferCostGraph::default();
        g.apply_descriptor_edges(&desc);
        g.precompute_costs();
        let (_, path) = g.transfer_path(&cs("A"), &cs("C")).unwrap();
        assert_eq!(path, vec![cs("A"), cs("B"), cs("C")]);
    }

    #[test]
    fn unpriced_routes_still_order_by_declared_weight() {
        // The guard on the `unpriced` component being a flag rather than a count. When nothing on
        // either route can be priced, the declared weights are the only ordering there is, and a
        // count would silently replace them with hop count -- picking the direct edge of weight 10
        // over a three-hop chain of weight 3.
        let mut g = TransferCostGraph::default();
        g.add_transfer_edge(cs("P"), cs("S"), 10);
        g.add_transfer_edge(cs("P"), cs("Q"), 1);
        g.add_transfer_edge(cs("Q"), cs("R"), 1);
        g.add_transfer_edge(cs("R"), cs("S"), 1);
        let (cost, path) = g.transfer_path(&cs("P"), &cs("S")).unwrap();
        assert_eq!(cost, 3);
        assert_eq!(path, vec![cs("P"), cs("Q"), cs("R"), cs("S")]);
    }

    #[test]
    fn reported_cost_stays_the_declared_weight_sum() {
        // Route selection changed; what `transfer_path` *reports* did not. The diagnostics record
        // quotes this as the unitless figures along the chosen path, so it must not quietly become
        // a time.
        let desc = TopologyDescriptor {
            default_space: cs("A"),
            visibility: vec![cs("A"), cs("B"), cs("C")],
            transfers: vec![
                rate_edge(cs("A"), cs("B"), 450),
                rate_edge(cs("B"), cs("C"), 450),
            ],
            arch: None,
            dtypes: None,
        };
        let mut g = TransferCostGraph::default();
        g.apply_descriptor_edges(&desc);
        g.precompute_costs();
        let (cost, _) = g.transfer_path(&cs("A"), &cs("C")).unwrap();
        assert_eq!(cost, 2, "two `Rate` edges weigh 1 each");
    }

    #[test]
    fn dijkstra_prefers_cheaper_multihop_over_direct() {
        // A direct but expensive edge (10) vs a longer but cheaper chain (1+1+1 = 3).
        let mut g = TransferCostGraph::default();
        g.add_transfer_edge(cs("A"), cs("D"), 10);
        g.add_transfer_edge(cs("A"), cs("B"), 1);
        g.add_transfer_edge(cs("B"), cs("C"), 1);
        g.add_transfer_edge(cs("C"), cs("D"), 1);
        let (cost, path) = g.transfer_path(&cs("A"), &cs("D")).unwrap();
        assert_eq!(cost, 3);
        assert_eq!(path, vec![cs("A"), cs("B"), cs("C"), cs("D")]);
    }

    #[test]
    fn dijkstra_relaxes_to_a_shorter_path() {
        // C is first reachable directly at cost 4, then relaxed to 2 once B is settled.
        let mut g = TransferCostGraph::default();
        g.add_transfer_edge(cs("A"), cs("C"), 4);
        g.add_transfer_edge(cs("A"), cs("B"), 1);
        g.add_transfer_edge(cs("B"), cs("C"), 1);
        let (cost, path) = g.transfer_path(&cs("A"), &cs("C")).unwrap();
        assert_eq!(cost, 2);
        assert_eq!(path, vec![cs("A"), cs("B"), cs("C")]);
    }

    #[test]
    fn dijkstra_diamond_picks_cheaper_branch() {
        // Two disjoint branches to D: A->B->D = 6 vs A->C->D = 3.
        let mut g = TransferCostGraph::default();
        g.add_transfer_edge(cs("A"), cs("B"), 1);
        g.add_transfer_edge(cs("B"), cs("D"), 5);
        g.add_transfer_edge(cs("A"), cs("C"), 2);
        g.add_transfer_edge(cs("C"), cs("D"), 1);
        let (cost, path) = g.transfer_path(&cs("A"), &cs("D")).unwrap();
        assert_eq!(cost, 3);
        assert_eq!(path, vec![cs("A"), cs("C"), cs("D")]);
    }

    #[test]
    fn dijkstra_edges_are_directed() {
        // An edge A->B does not imply B->A.
        let mut g = TransferCostGraph::default();
        g.add_transfer_edge(cs("A"), cs("B"), 1);
        assert!(g.transfer_path(&cs("A"), &cs("B")).is_some());
        assert!(g.transfer_path(&cs("B"), &cs("A")).is_none());
    }

    #[test]
    fn dijkstra_unreachable_target_is_none() {
        // `Island` only has an outgoing edge; nothing reaches it, so there is no path.
        let mut g = TransferCostGraph::default();
        g.add_transfer_edge(cs("A"), cs("B"), 1);
        g.add_transfer_edge(cs("Island"), cs("B"), 1);
        assert!(g.transfer_path(&cs("A"), &cs("Island")).is_none());
    }

    #[test]
    fn precompute_costs_fills_matrix_via_dijkstra() {
        // The cached cost matrix is the Dijkstra all-pairs shortest path: the A->C entry is the
        // cheaper 2-hop (A->B->C), not the direct edge (5), and the reverse direction is absent.
        let mut g = TransferCostGraph::default();
        g.add_transfer_edge(cs("A"), cs("B"), 1);
        g.add_transfer_edge(cs("B"), cs("C"), 1);
        g.add_transfer_edge(cs("A"), cs("C"), 5);
        g.precompute_costs();
        assert_eq!(g.transfer_cost(&cs("A"), &cs("C")), Some(2));
        assert!(g.can_transfer(&cs("A"), &cs("C")));
        assert_eq!(g.transfer_cost(&cs("C"), &cs("A")), None);
    }

    #[test]
    fn test_can_transfer_reflexive_all_spaces() {
        let graph = TransferCostGraph::default();
        for space in &[
            MemorySpace::CPUDRAM,
            MemorySpace::NPUHBM,
            MemorySpace::LocalSRAM,
            MemorySpace::NicRam,
            MemorySpace::RemoteHbm,
        ] {
            assert!(
                graph.can_transfer(space, space),
                "Reflexive transfer should always work for {:?}",
                space
            );
        }
    }

    #[test]
    fn test_accessibility_npu_cannot_reach_dram_directly() {
        let graph = TransferCostGraph::default();
        let ref_dram = Type::Ref(Box::new(make_tensor()), MemorySpace::CPUDRAM);
        // NPU can only see NPUHBM, not CPUDRAM directly
        assert!(!graph.is_type_accessible(&make_npu(), &Topology::CPU, &ref_dram));
    }

    #[test]
    fn test_accessibility_acccore_cannot_reach_npuhbm() {
        let graph = TransferCostGraph::default();
        let ref_hbm = Type::Ref(Box::new(make_tensor()), MemorySpace::NPUHBM);
        assert!(!graph.is_type_accessible(&make_acc_core(), &Topology::CPU, &ref_hbm));
    }

    #[test]
    fn test_accessibility_gpu_cannot_reach_local_sram() {
        let graph = TransferCostGraph::default();
        let ref_sram = Type::Ref(Box::new(make_tensor()), MemorySpace::LocalSRAM);
        // GPU only sees CPUDRAM; LocalSRAM is only accessible by AccCore
        // Use AccCore as var_topology to avoid the host unified memory fallback
        assert!(!graph.is_type_accessible(&Topology::gpu(0), &make_acc_core(), &ref_sram));
    }

    #[test]
    fn test_accessibility_cpuavx512_sees_dram() {
        let graph = TransferCostGraph::default();
        let ty = make_tensor();
        // CpuAvx512 defaults to CPUDRAM, same topology means accessible
        assert!(graph.is_type_accessible(&Topology::CpuAvx512, &Topology::CpuAvx512, &ty));
    }

    #[test]
    fn test_accessibility_cpuneon_sees_dram() {
        let graph = TransferCostGraph::default();
        let ty = make_tensor();
        assert!(graph.is_type_accessible(&Topology::CpuNeon, &Topology::CpuNeon, &ty));
    }

    #[test]
    fn test_default_memory_cpuavx512_and_cpuneon() {
        let g = TransferCostGraph::default();
        assert_eq!(
            g.default_memory_for(&Topology::CpuAvx512),
            MemorySpace::CPUDRAM
        );
        assert_eq!(
            g.default_memory_for(&Topology::CpuNeon),
            MemorySpace::CPUDRAM
        );
    }

    #[test]
    fn test_default_memory_slice() {
        let slice = Topology::Slice(
            Box::new(make_npu()),
            Box::new(Expr::Number(NumberExpr::new(
                "0".to_string(),
                None,
                Span::default(),
            ))),
            Box::new(Expr::Number(NumberExpr::new(
                "4".to_string(),
                None,
                Span::default(),
            ))),
        );
        assert_eq!(
            TransferCostGraph::default().default_memory_for(&slice),
            MemorySpace::NPUHBM
        );
    }

    #[test]
    #[should_panic(expected = "Must specify a concrete topology")]
    fn test_default_memory_for_current_panics() {
        let _ = TransferCostGraph::default().default_memory_for(&Topology::Current);
    }

    #[test]
    fn test_accessibility_gpu_cannot_reach_npuhbm() {
        let graph = TransferCostGraph::default();
        let ref_hbm = Type::Ref(Box::new(make_tensor()), MemorySpace::NPUHBM);
        // GPU can see CPUDRAM only, not NPU_HBM
        assert!(!graph.is_type_accessible(&Topology::gpu(0), &make_npu(), &ref_hbm));
    }

    #[test]
    fn test_accessibility_npu_cannot_reach_local_sram() {
        let graph = TransferCostGraph::default();
        let ref_sram = Type::Ref(Box::new(make_tensor()), MemorySpace::LocalSRAM);
        // NPU sees NPUHBM and CPUDRAM, not LocalSRAM
        assert!(!graph.is_type_accessible(&make_npu(), &make_acc_core(), &ref_sram));
    }

    #[test]
    fn test_reachable_same_topology_is_visible() {
        let graph = TransferCostGraph::default();
        assert_eq!(
            graph.reachable(&Topology::gpu(0), &Topology::gpu(0), &make_tensor()),
            Reachability::Visible
        );
    }

    #[test]
    fn test_reachable_host_to_gpu_needs_seam() {
        let graph = TransferCostGraph::default();
        // A discrete-GPU value lives in GPU HBM, not visible from the host: a transfer
        // path exists (GpuHbm -> CPUDRAM, cost 50), so the verdict is NeedsSeam.
        assert_eq!(
            graph.reachable(&Topology::CPU, &Topology::gpu(0), &make_tensor()),
            Reachability::NeedsSeam { cost: 50 }
        );
    }

    #[test]
    fn test_reachable_unreachable_when_no_path() {
        let graph = TransferCostGraph::default();
        // RemoteHbm has no outgoing edges, so a value pinned there is unreachable from
        // the host. Use a non-CPU var_topology to avoid the host-unified-memory shortcut.
        let ref_remote = Type::Ref(Box::new(make_tensor()), MemorySpace::RemoteHbm);
        assert_eq!(
            graph.reachable(&Topology::CPU, &Topology::gpu(0), &ref_remote),
            Reachability::Unreachable
        );
    }

    #[test]
    fn test_registry_seeded_with_builtins() {
        // Topology metadata is data (builtin_descriptors), not `match` arms. A fresh graph
        // carries the built-in descriptors.
        let g = TransferCostGraph::default();
        let gpu = g.descriptor(&syntax::TopologyKind::GPU).unwrap();
        assert_eq!(gpu.default_space, MemorySpace::GpuHbm);
        assert!(gpu.visibility.contains(&MemorySpace::GpuHbm)); // its own device memory
        assert!(gpu.visibility.contains(&MemorySpace::CPUDRAM)); // unified-memory reach

        let npu = g.descriptor(&syntax::TopologyKind::NPU).unwrap();
        assert_eq!(npu.default_space, MemorySpace::NPUHBM);
        // NPU cannot directly address host DRAM (matches is_type_accessible expectations).
        assert!(!npu.visibility.contains(&MemorySpace::CPUDRAM));
    }

    /// A `TopologyDecl` for tests, standing in for a parsed `Topology <name> { ... }`.
    fn topo_decl(name: &str, default_space: MemorySpace) -> TopologyDecl {
        TopologyDecl {
            name: crate::symbol::Symbol::from(name),
            descriptor: TopologyDescriptor {
                arch: None,
                default_space: default_space.clone(),
                visibility: vec![default_space],
                transfers: Vec::new(),
                dtypes: None,
            },
        }
    }

    #[test]
    fn seed_from_topologies_makes_descriptor_queryable() {
        // The AST-carried model: seeding a declared topology into a graph makes it queryable
        // on *that* graph (no global registry).
        let decl = topo_decl("MyTPU", MemorySpace::LocalSRAM);
        let mut g = TransferCostGraph::default();
        assert!(g
            .descriptor(&syntax::TopologyKind::Custom("MyTPU".into()))
            .is_none());
        g.seed_from_topologies(std::slice::from_ref(&decl));
        let d = g
            .descriptor(&syntax::TopologyKind::Custom("MyTPU".into()))
            .unwrap();
        assert_eq!(d.default_space, MemorySpace::LocalSRAM);
    }

    #[test]
    fn test_custom_topology_end_to_end() {
        // A user-defined topology, seeded into a graph: the enum identity `Topology::Custom(name)`
        // flows through default_memory_for + is_type_accessible with no hardcoded arm.
        let decl = topo_decl("MyTPU", MemorySpace::LocalSRAM);
        let top = Topology::Custom("MyTPU".into());
        let mut graph = TransferCostGraph::default();
        graph.seed_from_topologies(std::slice::from_ref(&decl));

        // Placement comes from the seeded descriptor.
        assert_eq!(graph.default_memory_for(&top), MemorySpace::LocalSRAM);
        // It can read its own memory space...
        let ref_sram = Type::Ref(Box::new(make_tensor()), MemorySpace::LocalSRAM);
        assert!(graph.is_type_accessible(&top, &Topology::CPU, &ref_sram));
        // ...but not a space it does not list.
        let ref_gpu = Type::Ref(Box::new(make_tensor()), MemorySpace::GpuHbm);
        assert!(!graph.is_type_accessible(&top, &Topology::gpu(0), &ref_gpu));
    }

    #[test]
    fn topologies_are_per_graph_not_global() {
        // The whole point of the AST-carried model: a topology seeded into one compilation's
        // graph does not exist in another's -- isolation by construction, no reset needed.
        let decl = topo_decl("AcmeCore", MemorySpace::LocalSRAM);
        let mut a = TransferCostGraph::default();
        a.seed_from_topologies(std::slice::from_ref(&decl));
        let b = TransferCostGraph::default(); // a separate compilation, not seeded

        let kind = syntax::TopologyKind::Custom("AcmeCore".into());
        assert!(
            a.descriptor(&kind).is_some(),
            "seeded graph has the topology"
        );
        assert!(
            b.descriptor(&kind).is_none(),
            "unseeded graph must not see another compilation's topology"
        );
        // Built-ins are present in both.
        assert!(a.descriptor(&syntax::TopologyKind::GPU).is_some());
        assert!(b.descriptor(&syntax::TopologyKind::GPU).is_some());
    }

    #[test]
    fn test_descriptor_transfer_edges_applied() {
        // A declared topology contributes transfer edges (morphisms) to the graph.
        // Hermetic: applies a local descriptor's edges to a fresh graph, no global state.
        let mut graph = TransferCostGraph::default();
        // No direct GpuHbm -> LocalSRAM edge in the built-in graph.
        let before = graph.transfer_path(&MemorySpace::GpuHbm, &MemorySpace::LocalSRAM);
        let desc = TopologyDescriptor {
            arch: None,
            default_space: MemorySpace::LocalSRAM,
            visibility: vec![MemorySpace::LocalSRAM],
            transfers: vec![TransferEdge {
                from: MemorySpace::GpuHbm,
                to: MemorySpace::LocalSRAM,
                cost: crate::arch::EdgeCost::Fixed(7),
                sync: true,
                copy_engine: false,
            }],
            dtypes: None,
        };
        graph.apply_descriptor_edges(&desc);
        graph.precompute_costs();
        assert_eq!(
            graph
                .transfer_path(&MemorySpace::GpuHbm, &MemorySpace::LocalSRAM)
                .map(|(c, _)| c),
            Some(7),
            "declared edge should give a direct cost-7 morphism (was {:?})",
            before.map(|(c, _)| c)
        );
    }

    #[test]
    fn test_custom_memory_space_edge() {
        // A user-defined memory space flows through the cost graph via a declared edge.
        // Hermetic: local graph + local descriptor, no global registry.
        let acme = MemorySpace::Custom(crate::symbol::Symbol::from("AcmeSRAM"));
        let mut graph = TransferCostGraph::default();
        let desc = TopologyDescriptor {
            arch: None,
            default_space: acme.clone(),
            visibility: vec![acme.clone()],
            transfers: vec![TransferEdge {
                from: MemorySpace::CPUDRAM,
                to: acme.clone(),
                cost: crate::arch::EdgeCost::Fixed(25),
                sync: true,
                copy_engine: false,
            }],
            dtypes: None,
        };
        graph.apply_descriptor_edges(&desc);
        graph.precompute_costs();
        // precompute_costs now covers custom spaces, so the cached cost is available.
        assert_eq!(graph.transfer_cost(&MemorySpace::CPUDRAM, &acme), Some(25));
        assert!(graph.can_transfer(&MemorySpace::CPUDRAM, &acme));
    }

    #[test]
    fn test_descriptor_coherence() {
        let graph = TransferCostGraph::default();
        // An island: a custom memory space with no edge from the host is unreachable.
        let island = MemorySpace::Custom(crate::symbol::Symbol::from("IslandRAM"));
        let bad = TopologyDescriptor {
            arch: None,
            default_space: island.clone(),
            visibility: vec![island.clone()],
            transfers: Vec::new(),
            dtypes: None,
        };
        assert!(
            descriptor_coherence(&bad, &graph).contains(&CoherenceIssue::MemoryUnreachableFromHost)
        );

        // default_space not in visibility -> DefaultNotVisible.
        let bad2 = TopologyDescriptor {
            arch: None,
            default_space: MemorySpace::LocalSRAM,
            visibility: vec![MemorySpace::CPUDRAM],
            transfers: Vec::new(),
            dtypes: None,
        };
        assert!(descriptor_coherence(&bad2, &graph).contains(&CoherenceIssue::DefaultNotVisible));

        // A built-in-backed topology reachable from host is coherent.
        let good = TopologyDescriptor {
            arch: None,
            default_space: MemorySpace::NPUHBM,
            visibility: vec![MemorySpace::NPUHBM],
            transfers: Vec::new(),
            dtypes: None,
        };
        assert!(descriptor_coherence(&good, &graph).is_empty());
    }

    #[test]
    fn test_transfer_path_npu_to_remote_via_nic() {
        let graph = TransferCostGraph::default();
        let result = graph.transfer_path(&MemorySpace::NPUHBM, &MemorySpace::RemoteHbm);
        assert!(result.is_some());
        let (cost, path) = result.unwrap();
        // NPUHBM -> NicRam (5) -> RemoteHbm (20) = 25
        assert_eq!(cost, 25);
        assert_eq!(
            path,
            vec![
                MemorySpace::NPUHBM,
                MemorySpace::NicRam,
                MemorySpace::RemoteHbm
            ]
        );
    }

    /// The two statements of "what space does this built-in device hold" have to agree.
    ///
    /// `builtin_default_space` exists because the parser needs the answer without a descriptor
    /// table to allocate, and `builtin_descriptors` is what everything downstream reads. Two
    /// tables of one fact drift silently -- a device added to one and forgotten in the other
    /// would place values in a different space depending on which side of name resolution asked.
    #[test]
    fn the_built_in_default_space_agrees_with_the_descriptor_table() {
        for (kind, desc) in builtin_descriptors() {
            assert_eq!(
                builtin_default_space(&kind),
                desc.default_space,
                "{kind:?} holds two different spaces depending on who asks"
            );
        }
    }

    #[test]
    fn a_built_in_space_names_the_device_that_holds_it() {
        let g = TransferCostGraph::default();
        assert_eq!(
            g.owning_topology(&MemorySpace::CPUDRAM).unwrap(),
            Topology::CPU
        );
        assert_eq!(
            g.owning_topology(&MemorySpace::GpuHbm).unwrap(),
            Topology::gpu(0)
        );
        assert!(matches!(
            g.owning_topology(&MemorySpace::NPUHBM).unwrap(),
            Topology::NPU(_)
        ));
        assert!(matches!(
            g.owning_topology(&MemorySpace::LocalSRAM).unwrap(),
            Topology::AccCore(_)
        ));
    }

    #[test]
    fn the_owner_is_stated_because_inverting_the_descriptors_is_not_a_function() {
        // `CPU_DRAM` is the default space of CPU, AMX, CpuAvx512 and CpuNeon, and `NPU_HBM` of
        // NPU, ANE and Slice. Deriving the owner by inverting that table would make the two
        // most-used spaces ambiguous, so the answer comes from `builtin_space_owner` instead.
        let g = TransferCostGraph::default();
        let sharing_cpu_dram = g
            .descriptors
            .values()
            .filter(|d| d.default_space == MemorySpace::CPUDRAM)
            .count();
        assert!(
            sharing_cpu_dram > 1,
            "the ambiguity this test exists for is gone"
        );
        assert_eq!(
            g.owning_topology(&MemorySpace::CPUDRAM).unwrap(),
            Topology::CPU
        );
    }

    #[test]
    fn a_space_no_topology_claims_has_no_owner_rather_than_a_default() {
        let g = TransferCostGraph::default();
        let orphan = MemorySpace::Custom("Scratchpad".into());
        let err = g.owning_topology(&orphan).unwrap_err();
        assert!(err.contains("no topology declares"), "{err}");
        assert!(err.contains("Scratchpad"), "{err}");
    }

    #[test]
    fn a_space_two_topologies_claim_is_ambiguous_rather_than_first_wins() {
        let mut g = TransferCostGraph::default();
        let shared = MemorySpace::Custom("Shared".into());
        for name in ["DevA", "DevB"] {
            g.descriptors.insert(
                crate::syntax::TopologyKind::Custom(name.into()),
                TopologyDescriptor {
                    arch: None,
                    default_space: shared.clone(),
                    visibility: vec![shared.clone()],
                    transfers: Vec::new(),
                    dtypes: None,
                },
            );
        }
        let err = g.owning_topology(&shared).unwrap_err();
        assert!(err.contains("ambiguous"), "{err}");
        assert!(err.contains("DevA") && err.contains("DevB"), "{err}");
    }
}
