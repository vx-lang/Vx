//===- arch.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file defines the HardwareGraph, formalizing the memory and topology
// algebra for cross-device accesses and transfers.
//
//===----------------------------------------------------------------------===//

use crate::ast::{MemorySpace, Topology, Type};

pub struct HardwareGraph;

impl HardwareGraph {
    /// Returns the default memory space for a given topology.
    pub fn default_memory_for(topology: &Topology) -> MemorySpace {
        match topology {
            Topology::NPU(_) => MemorySpace::NPUHBM,
            Topology::AccCore(_) => MemorySpace::LocalSRAM,
            Topology::Host => MemorySpace::HostDRAM,
            Topology::AMX => MemorySpace::HostDRAM,
            Topology::ANE => MemorySpace::NPUHBM,
            Topology::GPU => MemorySpace::HostDRAM,
            Topology::Slice(_, _, _) => MemorySpace::NPUHBM,
        }
    }

    /// Verifies if a variable belonging to `var_topology` with type `ty`
    /// is accessible from the `active_topology`.
    pub fn is_type_accessible(
        active_topology: &Topology,
        var_topology: &Topology,
        ty: &Type,
    ) -> bool {
        if let Type::Pinned(_, pinned_top) = ty {
            if pinned_top == active_topology {
                return true;
            }
            let mem = Self::default_memory_for(pinned_top);
            let mock_ty = Type::Ref(Box::new(Type::Scalar(crate::ast::ElementType::F32)), mem);
            return Self::is_type_accessible(active_topology, pinned_top, &mock_ty);
        }

        // If it's a specific memory reference, check reachability
        if let Type::Ref(_, MemorySpace::NPUHBM) = ty {
            return matches!(
                active_topology,
                Topology::NPU(_) | Topology::Slice(_, _, _) | Topology::ANE
            );
        }

        if let Type::Ref(_, MemorySpace::HostDRAM) = ty {
            return matches!(
                active_topology,
                Topology::Host | Topology::AMX | Topology::GPU | Topology::ANE
            );
        }

        // If no explicit memory qualifier, it defaults to the variable's topology
        if var_topology == active_topology {
            return true;
        }

        // Host unified memory fallbacks
        if *var_topology == Topology::Host
            && matches!(
                active_topology,
                Topology::AMX | Topology::ANE | Topology::GPU
            )
        {
            return true;
        }

        false
    }

    /// Determines if a data transfer between two memory spaces is physically supported.
    pub fn can_transfer(source: &MemorySpace, target: &MemorySpace) -> bool {
        match (source, target) {
            (a, b) if a == b => true,
            (MemorySpace::HostDRAM, MemorySpace::NPUHBM) => true,
            (MemorySpace::NPUHBM, MemorySpace::HostDRAM) => true,

            (MemorySpace::LocalSRAM, MemorySpace::NPUHBM) => true,
            (MemorySpace::NPUHBM, MemorySpace::LocalSRAM) => true,

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{ElementType, Expr, NumberExpr, Span};

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
            HardwareGraph::default_memory_for(&Topology::Host),
            MemorySpace::HostDRAM
        );
        assert_eq!(
            HardwareGraph::default_memory_for(&Topology::GPU),
            MemorySpace::HostDRAM
        );
        assert_eq!(
            HardwareGraph::default_memory_for(&Topology::AMX),
            MemorySpace::HostDRAM
        );

        assert_eq!(
            HardwareGraph::default_memory_for(&Topology::ANE),
            MemorySpace::NPUHBM
        );
        assert_eq!(
            HardwareGraph::default_memory_for(&make_npu()),
            MemorySpace::NPUHBM
        );

        assert_eq!(
            HardwareGraph::default_memory_for(&make_acc_core()),
            MemorySpace::LocalSRAM
        );
    }

    #[test]
    fn test_accessibility_same_topology() {
        let ty = make_tensor();
        // Exact same topology is always accessible
        assert!(HardwareGraph::is_type_accessible(
            &Topology::Host,
            &Topology::Host,
            &ty
        ));
        assert!(HardwareGraph::is_type_accessible(
            &Topology::ANE,
            &Topology::ANE,
            &ty
        ));
        assert!(HardwareGraph::is_type_accessible(
            &make_npu(),
            &make_npu(),
            &ty
        ));
    }

    #[test]
    fn test_accessibility_host_unified_memory() {
        let ty = make_tensor();
        // AMX, ANE, GPU can read variables stored in Host topology
        assert!(HardwareGraph::is_type_accessible(
            &Topology::AMX,
            &Topology::Host,
            &ty
        ));
        assert!(HardwareGraph::is_type_accessible(
            &Topology::ANE,
            &Topology::Host,
            &ty
        ));
        assert!(HardwareGraph::is_type_accessible(
            &Topology::GPU,
            &Topology::Host,
            &ty
        ));

        // But Host cannot read ANE or GPU specific topology variables directly
        assert!(!HardwareGraph::is_type_accessible(
            &Topology::Host,
            &Topology::ANE,
            &ty
        ));
        assert!(!HardwareGraph::is_type_accessible(
            &Topology::Host,
            &Topology::GPU,
            &ty
        ));
    }

    #[test]
    fn test_accessibility_pinned_memory() {
        let pinned_ane = Type::Pinned(Box::new(make_tensor()), Topology::ANE);
        let pinned_host = Type::Pinned(Box::new(make_tensor()), Topology::Host);

        // ANE can access ANE pinned
        assert!(HardwareGraph::is_type_accessible(
            &Topology::ANE,
            &Topology::Host,
            &pinned_ane
        ));
        // Host cannot access ANE pinned
        assert!(!HardwareGraph::is_type_accessible(
            &Topology::Host,
            &Topology::Host,
            &pinned_ane
        ));
        // Host can access Host pinned
        assert!(HardwareGraph::is_type_accessible(
            &Topology::Host,
            &Topology::ANE,
            &pinned_host
        ));
    }

    #[test]
    fn test_accessibility_memory_space_refs() {
        let ref_hbm = Type::Ref(Box::new(make_tensor()), MemorySpace::NPUHBM);
        let ref_dram = Type::Ref(Box::new(make_tensor()), MemorySpace::HostDRAM);

        // HBM reachable by NPU and ANE
        assert!(HardwareGraph::is_type_accessible(
            &make_npu(),
            &Topology::Host,
            &ref_hbm
        ));
        assert!(HardwareGraph::is_type_accessible(
            &Topology::ANE,
            &Topology::Host,
            &ref_hbm
        ));
        assert!(!HardwareGraph::is_type_accessible(
            &Topology::Host,
            &Topology::Host,
            &ref_hbm
        ));
        assert!(!HardwareGraph::is_type_accessible(
            &make_acc_core(),
            &Topology::Host,
            &ref_hbm
        )); // AccCore has LocalSRAM

        // DRAM reachable by Host, AMX, GPU, ANE
        assert!(HardwareGraph::is_type_accessible(
            &Topology::Host,
            &Topology::ANE,
            &ref_dram
        ));
        assert!(HardwareGraph::is_type_accessible(
            &Topology::AMX,
            &Topology::ANE,
            &ref_dram
        ));
        assert!(HardwareGraph::is_type_accessible(
            &Topology::GPU,
            &Topology::ANE,
            &ref_dram
        ));
        assert!(HardwareGraph::is_type_accessible(
            &Topology::ANE,
            &Topology::Host,
            &ref_dram
        ));
        // NPU doesn't directly reach DRAM in this default unified memory model
        assert!(!HardwareGraph::is_type_accessible(
            &make_npu(),
            &Topology::Host,
            &ref_dram
        ));
    }

    #[test]
    fn test_transfer_legal_paths() {
        // Identity
        assert!(HardwareGraph::can_transfer(
            &MemorySpace::HostDRAM,
            &MemorySpace::HostDRAM
        ));

        // Host <-> NPU HBM
        assert!(HardwareGraph::can_transfer(
            &MemorySpace::HostDRAM,
            &MemorySpace::NPUHBM
        ));
        assert!(HardwareGraph::can_transfer(
            &MemorySpace::NPUHBM,
            &MemorySpace::HostDRAM
        ));

        // NPU HBM <-> Local SRAM
        assert!(HardwareGraph::can_transfer(
            &MemorySpace::NPUHBM,
            &MemorySpace::LocalSRAM
        ));
        assert!(HardwareGraph::can_transfer(
            &MemorySpace::LocalSRAM,
            &MemorySpace::NPUHBM
        ));
    }

    #[test]
    fn test_transfer_illegal_paths() {
        // Local SRAM <-> Host DRAM (must go through HBM)
        assert!(!HardwareGraph::can_transfer(
            &MemorySpace::LocalSRAM,
            &MemorySpace::HostDRAM
        ));
        assert!(!HardwareGraph::can_transfer(
            &MemorySpace::HostDRAM,
            &MemorySpace::LocalSRAM
        ));
    }
}
