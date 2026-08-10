//===- arch.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
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
/// See `docs/discussions/brainstorming/hardware_monad_topology.md` (the USE-DIRECT /
/// USE-NEEDS-SEAM rules).
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
/// arms, and not a global registry (see `docs/discussions/brainstorming/hardware_monad_topology.md`,
/// "registry behind the enum").
/// A declared transfer edge (morphism): a hop `from -> to` with a `cost` grade and a
/// consistency grade. `sync` = a synchronizing transfer (release/acquire) that preserves a
/// boundary contract; `!sync` = a relaxed escape hatch whose visibility the seam engine
/// cannot guarantee (see coherence checking in `hir`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferEdge {
    pub from: MemorySpace,
    pub to: MemorySpace,
    pub cost: EdgeCost,
    pub sync: bool,
}

/// Where a declared edge's cost comes from. **Exactly one source per edge** — an edge that both
/// declares a cost and can have one derived from its endpoints' bandwidths is an error (E6013),
/// because two answers to "what does this hop cost" is not a model, it is a coin flip whose
/// outcome depends on which consumer asks.
///
/// The rule was not free: every fleet file declared `transfer HBM -> L2 : 40` *and* gave `L2` a
/// bandwidth, so both costs existed for that edge, and the compiler routed by one and reported the
/// other. Nothing detected it.
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
    /// The weight this edge contributes to route *selection*.
    ///
    /// Deliberately not the predicted cost. Route choice happens before a byte count is known, so a
    /// size-dependent cost cannot decide it; `Derived` and `Rate` edges therefore weigh 1 and the
    /// route is chosen by hop count among them. That the model's route choice is size-independent
    /// while the real fastest route is not is a known limit, pre-registered as experiment M5.
    pub fn routing_weight(self) -> u32 {
        match self {
            EdgeCost::Derived | EdgeCost::Rate(_) => 1,
            EdgeCost::Fixed(c) => c,
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

/// A stable per-name dispatch id in the 1000..1999 band.
fn fnv_dispatch_id(name: &str) -> i32 {
    1000 + (fnv32(name) % 1000) as i32
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
        // (`Custom` names own 1000..1999).
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
        // Network memory has no dedicated topology; kept at 300 (overlaps AMX) for now.
        MemorySpace::NicRam | MemorySpace::RemoteHbm => 300,
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
        }
    }

    /// Add the transfer edges declared by every registered topology (built-in + user-defined)
    /// and recompute shortest paths. `TransferCostGraph::default()` deliberately does *not* do
    /// this so it stays hermetic; the real compiler path (`TypeChecker::new`) calls this so
    /// user-declared morphisms take effect.
    /// Fold in the topologies declared by *this* compilation (`Program.topologies`): record each
    /// descriptor and add its transfer edges, then recompute the shortest-path matrix. The
    /// per-compilation replacement for the old global-registry seed.
    pub fn seed_from_topologies(&mut self, topologies: &[TopologyDecl]) {
        for decl in topologies {
            self.apply_descriptor_edges(&decl.descriptor);
            self.descriptors.insert(
                crate::syntax::TopologyKind::Custom(decl.name.clone()),
                decl.descriptor.clone(),
            );
        }
        self.precompute_costs();
    }

    /// Returns the default memory space for a given topology, from its registered
    /// descriptor (see `topology_descriptor`).
    /// The descriptor for a topology kind (built-in or a topology declared in this compilation),
    /// from the graph's per-compilation `descriptors` — no global registry.
    pub fn descriptor(&self, kind: &crate::syntax::TopologyKind) -> Option<&TopologyDescriptor> {
        self.descriptors.get(kind)
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
        let spaces: Vec<MemorySpace> = spaces.into_iter().collect();
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

        // Determine the memory space of the variable
        let target_mem = match ty {
            Type::Ref(_, mem) => mem.clone(),
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
    fn memory_of(&self, var_topology: &Topology, ty: &Type) -> MemorySpace {
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

    pub fn transfer_path(
        &self,
        source: &MemorySpace,
        target: &MemorySpace,
    ) -> Option<(u32, Vec<MemorySpace>)> {
        if source == target {
            return Some((0, vec![source.clone()]));
        }

        use std::collections::BinaryHeap;

        #[derive(Eq, PartialEq)]
        struct State {
            cost: u32,
            mem: MemorySpace,
        }

        impl Ord for State {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other.cost.cmp(&self.cost) // Reverse for min-heap
            }
        }

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::new();
        let mut dists = HashMap::new();
        let mut predecessors: HashMap<MemorySpace, MemorySpace> = HashMap::new();

        heap.push(State {
            cost: 0,
            mem: source.clone(),
        });
        dists.insert(source.clone(), 0);

        while let Some(State { cost, mem }) = heap.pop() {
            if mem == *target {
                let mut path = Vec::new();
                let mut curr = mem.clone();
                while curr != *source {
                    path.push(curr.clone());
                    curr = predecessors.get(&curr).unwrap().clone();
                }
                path.push(source.clone());
                path.reverse();
                return Some((cost, path));
            }

            if let Some(current_dist) = dists.get(&mem) {
                if cost > *current_dist {
                    continue;
                }
            }

            if let Some(neighbors) = self.transfer_edges.get(&mem) {
                for (next, edge_cost) in neighbors {
                    let next_cost = cost + edge_cost;
                    let is_better = dists.get(next).is_none_or(|&c| next_cost < c);

                    if is_better {
                        dists.insert(next.clone(), next_cost);
                        predecessors.insert(next.clone(), mem.clone());
                        heap.push(State {
                            cost: next_cost,
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
    /// fact minimal through a space it never looked at (S1, vx-review#9).
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
    /// declared edge" needs in order to mean anything (S1, vx-review#9).
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
            managed: Default::default(),
            granule: None,
            scope,
            overcommit: false,
            doc_comment: None,
        }
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
            }],
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
            }],
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
        };
        assert!(descriptor_coherence(&bad2, &graph).contains(&CoherenceIssue::DefaultNotVisible));

        // A built-in-backed topology reachable from host is coherent.
        let good = TopologyDescriptor {
            arch: None,
            default_space: MemorySpace::NPUHBM,
            visibility: vec![MemorySpace::NPUHBM],
            transfers: Vec::new(),
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
}
