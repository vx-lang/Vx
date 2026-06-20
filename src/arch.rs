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

use crate::ast::{MemorySpace, Topology, Type};
use std::collections::HashMap;

use crate::ast;
pub struct TransferCostGraph {
    /// Adjacency list for MemorySpace data transfers.
    /// Directed edge from A -> B means memory can be transferred from A to B.
    transfer_edges: HashMap<MemorySpace, Vec<(MemorySpace, u32)>>,

    /// Cached all-pairs shortest paths for data transfers.
    cost_matrix: HashMap<(MemorySpace, MemorySpace), u32>,

    /// Adjacency list for Topology to MemorySpace visibility.
    /// Directed edge from Top -> Mem means Top can directly read/write Mem.
    visibility_edges: HashMap<ast::TopologyKind, Vec<MemorySpace>>,
}

impl Default for TransferCostGraph {
    fn default() -> Self {
        let mut graph = Self {
            transfer_edges: HashMap::new(),
            cost_matrix: HashMap::new(),
            visibility_edges: HashMap::new(),
        };

        // Standard Transfer Paths
        // Host <-> HBM
        graph.add_transfer_edge(MemorySpace::CPUDRAM, MemorySpace::NPUHBM, 50);
        graph.add_transfer_edge(MemorySpace::NPUHBM, MemorySpace::CPUDRAM, 50);

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

        // Standard Visibility Paths
        // Host can access DRAM and HBM
        graph.add_visibility_edge(Topology::CPU, MemorySpace::CPUDRAM);
        graph.add_visibility_edge(Topology::CPU, MemorySpace::NPUHBM);
        // GPUs, AMX, ANE can access DRAM (Unified Memory Fallback)
        graph.add_visibility_edge(Topology::AMX, MemorySpace::CPUDRAM);
        graph.add_visibility_edge(Topology::ANE, MemorySpace::CPUDRAM);
        graph.add_visibility_edge(Topology::GPU, MemorySpace::CPUDRAM);

        // ANE also accesses HBM
        graph.add_visibility_edge(Topology::ANE, MemorySpace::NPUHBM);

        // NPU and Slice reach HBM (we handle dynamic NPU IDs in the accessor method)
        // AccCore reaches SRAM (handled dynamically as well)

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

    pub fn add_visibility_edge(&mut self, top: Topology, mem: MemorySpace) {
        self.visibility_edges
            .entry(top.kind())
            .or_default()
            .push(mem);
    }

    /// Returns the default memory space for a given topology.
    pub fn default_memory_for(topology: &Topology) -> MemorySpace {
        match topology {
            Topology::CPU | Topology::CpuAvx512 | Topology::CpuNeon => MemorySpace::CPUDRAM,
            Topology::NPU(_) => MemorySpace::NPUHBM,
            Topology::AccCore(_) => MemorySpace::LocalSRAM,
            Topology::AMX => MemorySpace::CPUDRAM,
            Topology::ANE => MemorySpace::NPUHBM,
            Topology::GPU => MemorySpace::CPUDRAM,
            Topology::Slice(_, _, _) => MemorySpace::NPUHBM,
            Topology::Current => {
                unreachable!("Must specify a concrete topology other than Current")
            }
        }
    }

    /// Precomputes the all-pairs shortest path transfer costs.
    pub fn precompute_costs(&mut self) {
        let spaces = [
            MemorySpace::CPUDRAM,
            MemorySpace::NPUHBM,
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
            let mock_ty = Type::Ref(Box::new(Type::Scalar(ast::ElementType::F32)), mem);
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

        // If the variable lives in its own default space, check visibility graph
        // Handle dynamic topologies
        let active_kind = active_topology.kind();

        // Hardcode the dynamic matching rules that aren't easily static HashMap entries
        if active_kind == ast::TopologyKind::NPU && target_mem == MemorySpace::NPUHBM {
            return true;
        }
        if active_kind == ast::TopologyKind::AccCore && target_mem == MemorySpace::LocalSRAM {
            return true;
        }

        // Check formal visibility edges
        if let Some(visible_mems) = self.visibility_edges.get(&active_kind) {
            if visible_mems.contains(&target_mem) {
                return true;
            }
        }

        // Host unified memory fallback (handled by graph edges but we can explicitly check if needed)
        // Check if var_topology is Host, and active_topology has visibility to CPUDRAM
        if *var_topology == Topology::CPU {
            if let Some(visible_mems) = self.visibility_edges.get(&active_kind) {
                if visible_mems.contains(&MemorySpace::CPUDRAM) {
                    return true;
                }
            }
        }

        false
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
    use ast::{ElementType, Expr, NumberExpr, Span};

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
        assert_eq!(
            TransferCostGraph::default_memory_for(&Topology::GPU),
            MemorySpace::CPUDRAM
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
        // AMX, ANE, GPU can read variables stored in Host topology
        assert!(graph.is_type_accessible(&Topology::AMX, &Topology::CPU, &ty));
        assert!(graph.is_type_accessible(&Topology::ANE, &Topology::CPU, &ty));
        assert!(graph.is_type_accessible(&Topology::GPU, &Topology::CPU, &ty));

        // But under formal graph memory, since ANE/GPU default to CPUDRAM (for GPU) and NPUHBM (for ANE),
        // and Host can see both CPUDRAM and NPUHBM, Host can technically read those memory spaces.
        // The formal graph makes memory spaces the single source of truth!
        assert!(graph.is_type_accessible(&Topology::CPU, &Topology::ANE, &ty));
        assert!(graph.is_type_accessible(&Topology::CPU, &Topology::GPU, &ty));
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
}
