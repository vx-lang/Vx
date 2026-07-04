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
    // Topology→MemorySpace visibility now lives in the global topology registry
    // (`topology_descriptor`), not on the graph.
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
/// includes `default_space`). Seeded with the built-in topologies below; `register_topology`
/// lets a plugin add or override one. This is the first substrate step toward user-definable
/// topologies (see `docs/discussions/brainstorming/hardware_monad_topology.md`, "registry
/// behind the enum"): the metadata is now data, not `match` arms. (Introducing a *new*
/// topology *identity* from source still needs a `Topology::Custom`-style variant + parser
/// support; this step opens the description, not yet the name.)
#[derive(Debug, Clone)]
pub struct TopologyDescriptor {
    pub default_space: MemorySpace,
    pub visibility: Vec<MemorySpace>,
}

/// The built-in topology descriptions, encoding what `arch.rs` previously hardcoded.
/// `visibility` includes each topology's own `default_space`, which subsumes the old
/// "a topology sees its own memory" special-cases (NPU→NPUHBM, AccCore→LocalSRAM,
/// GPU→GpuHbm).
fn builtin_descriptors() -> HashMap<crate::syntax::TopologyKind, TopologyDescriptor> {
    use crate::syntax::TopologyKind as K;
    use MemorySpace::*;
    let d = |default_space: MemorySpace, visibility: &[MemorySpace]| TopologyDescriptor {
        default_space,
        visibility: visibility.to_vec(),
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

static TOPOLOGY_REGISTRY: std::sync::LazyLock<
    std::sync::RwLock<HashMap<crate::syntax::TopologyKind, TopologyDescriptor>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(builtin_descriptors()));

/// The description registered for a topology kind, if any.
pub fn topology_descriptor(kind: &crate::syntax::TopologyKind) -> Option<TopologyDescriptor> {
    TOPOLOGY_REGISTRY.read().unwrap().get(kind).cloned()
}

/// Register (or override) the description for a topology kind. The extension hook a hardware
/// plugin uses to describe its memory model to the compiler.
pub fn register_topology(kind: crate::syntax::TopologyKind, desc: TopologyDescriptor) {
    TOPOLOGY_REGISTRY.write().unwrap().insert(kind, desc);
}

impl Default for TransferCostGraph {
    fn default() -> Self {
        let mut graph = Self {
            transfer_edges: HashMap::new(),
            cost_matrix: HashMap::new(),
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

    /// Returns the default memory space for a given topology, from its registered
    /// descriptor (see `topology_descriptor`).
    pub fn default_memory_for(topology: &Topology) -> MemorySpace {
        if let Topology::Current = topology {
            unreachable!("Must specify a concrete topology other than Current")
        }
        topology_descriptor(&topology.kind())
            .map(|d| d.default_space)
            .unwrap_or(MemorySpace::CPUDRAM)
    }

    /// Precomputes the all-pairs shortest path transfer costs.
    pub fn precompute_costs(&mut self) {
        let spaces = [
            MemorySpace::CPUDRAM,
            MemorySpace::NPUHBM,
            MemorySpace::GpuHbm,
            MemorySpace::LocalSRAM,
            MemorySpace::NicRam,
            MemorySpace::RemoteHbm,
        ];
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
            let mem = Self::default_memory_for(pinned_top);
            let mock_ty = Type::Ref(Box::new(Type::Scalar(syntax::ElementType::F32)), mem);
            return Self::is_type_accessible(self, active_topology, pinned_top, &mock_ty);
        }

        // Determine the memory space of the variable
        let target_mem = match ty {
            Type::Ref(_, mem) => mem.clone(),
            _ => {
                if var_topology == active_topology {
                    return true;
                }
                Self::default_memory_for(var_topology)
            }
        };

        // Visibility is now data: consult the active topology's descriptor. Its
        // `visibility` set includes its own default space, subsuming the old
        // NPU→NPUHBM / AccCore→LocalSRAM / GPU→GpuHbm special-cases.
        let active_kind = active_topology.kind();
        if let Some(desc) = topology_descriptor(&active_kind) {
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
    fn memory_of(var_topology: &Topology, ty: &Type) -> MemorySpace {
        match ty {
            Type::Pinned(_, top) => Self::default_memory_for(top),
            Type::Ref(_, mem) => mem.clone(),
            _ => Self::default_memory_for(var_topology),
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
        let var_mem = Self::memory_of(var_topology, ty);
        let active_mem = Self::default_memory_for(active_topology);
        match self.transfer_path(&var_mem, &active_mem) {
            Some((cost, _)) => Reachability::NeedsSeam { cost },
            None => Reachability::Unreachable,
        }
    }

    /// Determines the minimum data movement cost and path between two memory spaces using Dijkstra's algorithm.
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

    #[test]
    fn test_default_memory_mappings() {
        assert_eq!(
            TransferCostGraph::default_memory_for(&Topology::CPU),
            MemorySpace::CPUDRAM
        );
        // A discrete GPU's home memory is its own device HBM (not host DRAM):
        // the host<->device boundary is a real seam, checked by the per-seam obligation.
        assert_eq!(
            TransferCostGraph::default_memory_for(&Topology::GPU),
            MemorySpace::GpuHbm
        );
        assert_eq!(
            TransferCostGraph::default_memory_for(&Topology::AMX),
            MemorySpace::CPUDRAM
        );

        assert_eq!(
            TransferCostGraph::default_memory_for(&Topology::ANE),
            MemorySpace::NPUHBM
        );
        assert_eq!(
            TransferCostGraph::default_memory_for(&make_npu()),
            MemorySpace::NPUHBM
        );

        assert_eq!(
            TransferCostGraph::default_memory_for(&make_acc_core()),
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
        assert!(graph.is_type_accessible(&Topology::GPU, &Topology::CPU, &ty));

        // ANE defaults to NPUHBM, which Host can see, so Host can read an ANE var.
        assert!(graph.is_type_accessible(&Topology::CPU, &Topology::ANE, &ty));
        // A discrete GPU's var lives in GPU HBM, which the host CANNOT see directly:
        // the host<->device boundary is exactly the seam the obligation guards.
        assert!(!graph.is_type_accessible(&Topology::CPU, &Topology::GPU, &ty));
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
        assert!(graph.is_type_accessible(&Topology::GPU, &Topology::ANE, &ref_dram));
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
        assert!(!graph.is_type_accessible(&Topology::GPU, &make_acc_core(), &ref_sram));
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
        assert_eq!(
            TransferCostGraph::default_memory_for(&Topology::CpuAvx512),
            MemorySpace::CPUDRAM
        );
        assert_eq!(
            TransferCostGraph::default_memory_for(&Topology::CpuNeon),
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
            TransferCostGraph::default_memory_for(&slice),
            MemorySpace::NPUHBM
        );
    }

    #[test]
    #[should_panic(expected = "Must specify a concrete topology")]
    fn test_default_memory_for_current_panics() {
        let _ = TransferCostGraph::default_memory_for(&Topology::Current);
    }

    #[test]
    fn test_accessibility_gpu_cannot_reach_npuhbm() {
        let graph = TransferCostGraph::default();
        let ref_hbm = Type::Ref(Box::new(make_tensor()), MemorySpace::NPUHBM);
        // GPU can see CPUDRAM only, not NPU_HBM
        assert!(!graph.is_type_accessible(&Topology::GPU, &make_npu(), &ref_hbm));
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
            graph.reachable(&Topology::GPU, &Topology::GPU, &make_tensor()),
            Reachability::Visible
        );
    }

    #[test]
    fn test_reachable_host_to_gpu_needs_seam() {
        let graph = TransferCostGraph::default();
        // A discrete-GPU value lives in GPU HBM, not visible from the host: a transfer
        // path exists (GpuHbm -> CPUDRAM, cost 50), so the verdict is NeedsSeam.
        assert_eq!(
            graph.reachable(&Topology::CPU, &Topology::GPU, &make_tensor()),
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
            graph.reachable(&Topology::CPU, &Topology::GPU, &ref_remote),
            Reachability::Unreachable
        );
    }

    #[test]
    fn test_registry_seeded_with_builtins() {
        // Topology metadata is data (builtin_descriptors), not `match` arms.
        let gpu = topology_descriptor(&syntax::TopologyKind::GPU).unwrap();
        assert_eq!(gpu.default_space, MemorySpace::GpuHbm);
        assert!(gpu.visibility.contains(&MemorySpace::GpuHbm)); // its own device memory
        assert!(gpu.visibility.contains(&MemorySpace::CPUDRAM)); // unified-memory reach

        let npu = topology_descriptor(&syntax::TopologyKind::NPU).unwrap();
        assert_eq!(npu.default_space, MemorySpace::NPUHBM);
        // NPU cannot directly address host DRAM (matches is_type_accessible expectations).
        assert!(!npu.visibility.contains(&MemorySpace::CPUDRAM));
    }

    #[test]
    fn test_register_topology_extends_registry() {
        // The extension hook: registering a descriptor makes it queryable. Uses the
        // otherwise-undescribed `Current` kind so this cannot perturb other tests.
        assert!(topology_descriptor(&syntax::TopologyKind::Current).is_none());
        register_topology(
            syntax::TopologyKind::Current,
            TopologyDescriptor {
                default_space: MemorySpace::LocalSRAM,
                visibility: vec![MemorySpace::LocalSRAM],
            },
        );
        let d = topology_descriptor(&syntax::TopologyKind::Current).unwrap();
        assert_eq!(d.default_space, MemorySpace::LocalSRAM);
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
