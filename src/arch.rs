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

    /// Adjacency list for Topology to MemorySpace visibility.
    /// Directed edge from Top -> Mem means Top can directly read/write Mem.
    visibility_edges: Vec<(Topology, Vec<MemorySpace>)>,
}

impl Default for TransferCostGraph {
    fn default() -> Self {
        let mut graph = Self {
            transfer_edges: HashMap::new(),
            visibility_edges: Vec::new(),
        };

        // Standard Transfer Paths
        // Host <-> HBM
        graph.add_transfer_edge(MemorySpace::HostDRAM, MemorySpace::NPUHBM, 50);
        graph.add_transfer_edge(MemorySpace::NPUHBM, MemorySpace::HostDRAM, 50);

        // HBM <-> SRAM
        graph.add_transfer_edge(MemorySpace::NPUHBM, MemorySpace::LocalSRAM, 10);
        graph.add_transfer_edge(MemorySpace::LocalSRAM, MemorySpace::NPUHBM, 10);

        // Standard Visibility Paths
        // Host can access DRAM and HBM
        graph.add_visibility_edge(Topology::Host, MemorySpace::HostDRAM);
        graph.add_visibility_edge(Topology::Host, MemorySpace::NPUHBM);
        // GPUs, AMX, ANE can access DRAM (Unified Memory Fallback)
        graph.add_visibility_edge(Topology::AMX, MemorySpace::HostDRAM);
        graph.add_visibility_edge(Topology::ANE, MemorySpace::HostDRAM);
        graph.add_visibility_edge(Topology::GPU, MemorySpace::HostDRAM);

        // ANE also accesses HBM
        graph.add_visibility_edge(Topology::ANE, MemorySpace::NPUHBM);

        // NPU and Slice reach HBM (we handle dynamic NPU IDs in the accessor method)
        // AccCore reaches SRAM (handled dynamically as well)

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
        if let Some(entry) = self.visibility_edges.iter_mut().find(|(t, _)| *t == top) {
            entry.1.push(mem);
        } else {
            self.visibility_edges.push((top, vec![mem]));
        }
    }

    /// Returns the default memory space for a given topology.
    pub fn default_memory_for(topology: &Topology) -> MemorySpace {
        match topology {
            Topology::Host | Topology::Host_AVX512 | Topology::Host_Neon => MemorySpace::HostDRAM,
            Topology::NPU(_) => MemorySpace::NPUHBM,
            Topology::AccCore(_) => MemorySpace::LocalSRAM,
            Topology::AMX => MemorySpace::HostDRAM,
            Topology::ANE => MemorySpace::NPUHBM,
            Topology::GPU => MemorySpace::HostDRAM,
            Topology::Slice(_, _, _) => MemorySpace::NPUHBM,
            Topology::Current => {
                unreachable!("Must specify a concrete topology other than Current")
            }
        }
    }

    /// Helper to generalize topologies with dynamic indices (e.g. NPU(0) -> NPU(any)).
    fn generalize_topology(top: &Topology) -> Topology {
        match top {
            Topology::NPU(_) | Topology::Slice(_, _, _) => {
                Topology::NPU(Box::new(ast::Expr::Number(ast::NumberExpr::new(
                    "0".to_string(),
                    None,
                    ast::Span::default(),
                ))))
            }
            Topology::AccCore(_) => Topology::AccCore(Box::new(ast::Expr::Number(
                ast::NumberExpr::new("0".to_string(), None, ast::Span::default()),
            ))),
            _ => top.clone(),
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
        let active_gen = Self::generalize_topology(active_topology);

        // Hardcode the dynamic matching rules that aren't easily static HashMap entries
        if matches!(active_gen, Topology::NPU(_)) && target_mem == MemorySpace::NPUHBM {
            return true;
        }
        if matches!(active_gen, Topology::AccCore(_)) && target_mem == MemorySpace::LocalSRAM {
            return true;
        }

        // Check formal visibility edges
        if let Some((_, visible_mems)) =
            self.visibility_edges.iter().find(|(t, _)| *t == active_gen)
        {
            if visible_mems.contains(&target_mem) {
                return true;
            }
        }

        // Host unified memory fallback (handled by graph edges but we can explicitly check if needed)
        // Check if var_topology is Host, and active_topology has visibility to HostDRAM
        if *var_topology == Topology::Host {
            if let Some((_, visible_mems)) =
                self.visibility_edges.iter().find(|(t, _)| *t == active_gen)
            {
                if visible_mems.contains(&MemorySpace::HostDRAM) {
                    return true;
                }
            }
        }

        false
    }

    /// Determines the minimum data movement cost between two memory spaces using Dijkstra's algorithm.
    pub fn transfer_cost(&self, source: &MemorySpace, target: &MemorySpace) -> Option<u32> {
        if source == target {
            return Some(0);
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

        heap.push(State {
            cost: 0,
            mem: source.clone(),
        });
        dists.insert(source.clone(), 0);

        while let Some(State { cost, mem }) = heap.pop() {
            if mem == *target {
                return Some(cost);
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

    /// Determines if a data transfer between two memory spaces is physically supported.
    pub fn can_transfer(&self, source: &MemorySpace, target: &MemorySpace) -> bool {
        self.transfer_cost(source, target).is_some()
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
            TransferCostGraph::default_memory_for(&Topology::Host),
            MemorySpace::HostDRAM
        );
        assert_eq!(
            TransferCostGraph::default_memory_for(&Topology::GPU),
            MemorySpace::HostDRAM
        );
        assert_eq!(
            TransferCostGraph::default_memory_for(&Topology::AMX),
            MemorySpace::HostDRAM
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
        assert!(graph.is_type_accessible(&Topology::Host, &Topology::Host, &ty));
        assert!(graph.is_type_accessible(&Topology::ANE, &Topology::ANE, &ty));
        assert!(graph.is_type_accessible(&make_npu(), &make_npu(), &ty));
    }

    #[test]
    fn test_accessibility_host_unified_memory() {
        let graph = TransferCostGraph::default();
        let ty = make_tensor();
        // AMX, ANE, GPU can read variables stored in Host topology
        assert!(graph.is_type_accessible(&Topology::AMX, &Topology::Host, &ty));
        assert!(graph.is_type_accessible(&Topology::ANE, &Topology::Host, &ty));
        assert!(graph.is_type_accessible(&Topology::GPU, &Topology::Host, &ty));

        // But under formal graph memory, since ANE/GPU default to HostDRAM (for GPU) and NPUHBM (for ANE),
        // and Host can see both HostDRAM and NPUHBM, Host can technically read those memory spaces.
        // The formal graph makes memory spaces the single source of truth!
        assert!(graph.is_type_accessible(&Topology::Host, &Topology::ANE, &ty));
        assert!(graph.is_type_accessible(&Topology::Host, &Topology::GPU, &ty));
    }

    #[test]
    fn test_accessibility_pinned_memory() {
        let graph = TransferCostGraph::default();
        let pinned_ane = Type::Pinned(Box::new(make_tensor()), Topology::ANE);
        let pinned_host = Type::Pinned(Box::new(make_tensor()), Topology::Host);

        // ANE can access ANE pinned
        assert!(graph.is_type_accessible(&Topology::ANE, &Topology::Host, &pinned_ane));
        // Host CAN access ANE pinned (as a handle)
        assert!(graph.is_type_accessible(&Topology::Host, &Topology::Host, &pinned_ane));
        // Host can access Host pinned
        assert!(graph.is_type_accessible(&Topology::Host, &Topology::ANE, &pinned_host));
    }

    #[test]
    fn test_accessibility_memory_space_refs() {
        let graph = TransferCostGraph::default();
        let ref_hbm = Type::Ref(Box::new(make_tensor()), MemorySpace::NPUHBM);
        let ref_dram = Type::Ref(Box::new(make_tensor()), MemorySpace::HostDRAM);

        // HBM reachable by NPU and ANE and Host
        assert!(graph.is_type_accessible(&make_npu(), &Topology::Host, &ref_hbm));
        assert!(graph.is_type_accessible(&Topology::ANE, &Topology::Host, &ref_hbm));
        assert!(graph.is_type_accessible(&Topology::Host, &Topology::Host, &ref_hbm));
        assert!(!graph.is_type_accessible(&make_acc_core(), &Topology::Host, &ref_hbm)); // AccCore has LocalSRAM

        // DRAM reachable by Host, AMX, GPU, ANE
        assert!(graph.is_type_accessible(&Topology::Host, &Topology::ANE, &ref_dram));
        assert!(graph.is_type_accessible(&Topology::AMX, &Topology::ANE, &ref_dram));
        assert!(graph.is_type_accessible(&Topology::GPU, &Topology::ANE, &ref_dram));
        assert!(graph.is_type_accessible(&Topology::ANE, &Topology::Host, &ref_dram));
        // NPU doesn't directly reach DRAM in this default unified memory model
        assert!(!graph.is_type_accessible(&make_npu(), &Topology::Host, &ref_dram));
    }

    #[test]
    fn test_transfer_legal_paths() {
        let graph = TransferCostGraph::default();
        // Identity
        assert!(graph.can_transfer(&MemorySpace::HostDRAM, &MemorySpace::HostDRAM));

        // Host <-> NPU HBM
        assert!(graph.can_transfer(&MemorySpace::HostDRAM, &MemorySpace::NPUHBM));
        assert!(graph.can_transfer(&MemorySpace::NPUHBM, &MemorySpace::HostDRAM));

        // NPU HBM <-> Local SRAM
        assert!(graph.can_transfer(&MemorySpace::NPUHBM, &MemorySpace::LocalSRAM));
        assert!(graph.can_transfer(&MemorySpace::LocalSRAM, &MemorySpace::NPUHBM));
    }

    #[test]
    fn test_transfer_multi_hop_paths() {
        let graph = TransferCostGraph::default();
        // Local SRAM <-> Host DRAM (BFS multi-hop routing makes this valid)
        assert!(graph.can_transfer(&MemorySpace::LocalSRAM, &MemorySpace::HostDRAM));
        assert!(graph.can_transfer(&MemorySpace::HostDRAM, &MemorySpace::LocalSRAM));
    }
}
