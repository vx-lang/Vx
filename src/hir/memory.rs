//===- memory.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The declared memory-space hierarchy (`Memory <Name> { within: ... }`). A per-compilation
// view built from `Program.memories` (not a process-global registry). Provides the containment
// queries later milestones need (capacity checks, derived transfer costs) and the M2 coherence
// laws: `within:` is acyclic, a sub-space's capacity does not exceed its parent's, and declared
// properties are positive.
//
//===----------------------------------------------------------------------===//

use crate::syntax::{Bandwidth, ByteSize, ElementType, Expr, MemoryDecl, MemorySpace};
use std::collections::{HashMap, HashSet};

/// Bit width of a tensor element; `None` for an un-instantiated generic element.
pub fn element_bits(elem: &ElementType) -> Option<u64> {
    Some(match elem {
        ElementType::Bool => 1,
        ElementType::I4 | ElementType::U4 => 4,
        ElementType::I8 | ElementType::U8 => 8,
        ElementType::F16 | ElementType::BF16 | ElementType::I16 | ElementType::U16 => 16,
        ElementType::F32 | ElementType::I32 | ElementType::U32 => 32,
        ElementType::F64 | ElementType::I64 | ElementType::U64 => 64,
        ElementType::I128 | ElementType::U128 => 128,
        ElementType::Generic(_) => return None,
    })
}

/// Byte size of a statically-shaped tensor: `ceil(element_bits × Π(dims) / 8)`. `None` when the
/// shape is empty or any dimension is not a compile-time integer literal (so the size — and thus
/// any capacity check — is unknown).
pub fn static_tensor_bytes(elem: &ElementType, dims: &[Expr]) -> Option<u64> {
    if dims.is_empty() {
        return None;
    }
    let mut count: u64 = 1;
    for d in dims {
        let Expr::Number(num) = d else {
            return None;
        };
        count = count.checked_mul(num.value.as_ref().parse::<u64>().ok()?)?;
    }
    Some(element_bits(elem)?.checked_mul(count)?.div_ceil(8))
}

/// A per-compilation view of the declared memory hierarchy. Nodes are `MemorySpace`s (a
/// declared space's identity is `MemorySpace::from_name(name)`, so `Memory GPU_HBM { ... }`
/// describes the built-in `GPU_HBM`); edges are `within:` parent links.
pub struct MemoryHierarchy<'a> {
    spaces: HashMap<MemorySpace, &'a MemoryDecl>,
}

/// A coherence violation in the declared memory hierarchy.
#[derive(Debug, PartialEq, Eq)]
pub enum MemoryCoherenceIssue {
    /// `within:` reaches `space` again — a containment cycle.
    Cycle { space: MemorySpace },
    /// `child`'s capacity exceeds its `parent`'s — a sub-space larger than what contains it.
    CapacityExceedsParent {
        child: MemorySpace,
        parent: MemorySpace,
        child_bytes: u64,
        parent_bytes: u64,
    },
    /// A declared property (`capacity` / `bandwidth` / `granule`) is non-positive (zero).
    NonPositiveProperty {
        space: MemorySpace,
        property: &'static str,
    },
}

impl<'a> MemoryHierarchy<'a> {
    /// Build the hierarchy from a program's memory declarations.
    pub fn build<I: IntoIterator<Item = &'a MemoryDecl>>(decls: I) -> Self {
        let mut spaces = HashMap::new();
        for d in decls {
            spaces.insert(MemorySpace::from_name(d.name.as_ref()), d);
        }
        Self { spaces }
    }

    /// The descriptor declared for a space, if any.
    pub fn descriptor(&self, space: &MemorySpace) -> Option<&'a MemoryDecl> {
        self.spaces.get(space).copied()
    }

    /// The declared `within:` parent of a space, if any.
    pub fn parent(&self, space: &MemorySpace) -> Option<MemorySpace> {
        self.spaces.get(space).and_then(|d| d.parent.clone())
    }

    /// Ancestors of `space`, nearest first, excluding `space`. Cycle-safe: stops on the first
    /// repeated node so a malformed (cyclic) declaration cannot loop forever.
    pub fn ancestors(&self, space: &MemorySpace) -> Vec<MemorySpace> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        seen.insert(space.clone());
        let mut cur = self.parent(space);
        while let Some(p) = cur {
            if !seen.insert(p.clone()) {
                break; // cycle — stop
            }
            cur = self.parent(&p);
            out.push(p);
        }
        out
    }

    /// True if `inner` is contained (transitively) within `outer`.
    pub fn contains(&self, outer: &MemorySpace, inner: &MemorySpace) -> bool {
        outer != inner && self.ancestors(inner).iter().any(|a| a == outer)
    }

    /// The nearest common ancestor of `a` and `b` in the containment tree, if any. A space is
    /// treated as its own ancestor, so `nearest_common_ancestor(x, child_of_x) == Some(x)`.
    pub fn nearest_common_ancestor(&self, a: &MemorySpace, b: &MemorySpace) -> Option<MemorySpace> {
        let mut a_chain: HashSet<MemorySpace> = HashSet::new();
        a_chain.insert(a.clone());
        a_chain.extend(self.ancestors(a));
        if a_chain.contains(b) {
            return Some(b.clone());
        }
        self.ancestors(b)
            .into_iter()
            .find(|anc| a_chain.contains(anc))
    }

    /// True if `space` is part of a `within:` cycle (reachable from itself).
    fn in_cycle(&self, space: &MemorySpace) -> bool {
        let mut seen = HashSet::new();
        let mut cur = self.parent(space);
        while let Some(p) = cur {
            if &p == space {
                return true;
            }
            if !seen.insert(p.clone()) {
                return false; // a cycle exists upstream, but `space` is not in it
            }
            cur = self.parent(&p);
        }
        false
    }

    /// Every coherence violation in the declared hierarchy, in a deterministic order.
    pub fn coherence_issues(&self) -> Vec<MemoryCoherenceIssue> {
        let mut issues = Vec::new();
        for (space, decl) in &self.spaces {
            let cyclic = self.in_cycle(space);
            if cyclic {
                issues.push(MemoryCoherenceIssue::Cycle {
                    space: space.clone(),
                });
            }
            if let Some(ByteSize(0)) = decl.capacity {
                issues.push(MemoryCoherenceIssue::NonPositiveProperty {
                    space: space.clone(),
                    property: "capacity",
                });
            }
            if let Some(ByteSize(0)) = decl.granule {
                issues.push(MemoryCoherenceIssue::NonPositiveProperty {
                    space: space.clone(),
                    property: "granule",
                });
            }
            if let Some(Bandwidth { bytes: 0, .. }) = decl.bandwidth {
                issues.push(MemoryCoherenceIssue::NonPositiveProperty {
                    space: space.clone(),
                    property: "bandwidth",
                });
            }
            // Capacity monotonicity: child ≤ parent, when both are known and the parent is a
            // declared space with a capacity. Skipped for cyclic nodes (already an error).
            if !cyclic {
                if let (Some(ByteSize(child_bytes)), Some(parent)) =
                    (decl.capacity, decl.parent.clone())
                {
                    if let Some(ByteSize(parent_bytes)) =
                        self.spaces.get(&parent).and_then(|p| p.capacity)
                    {
                        if child_bytes > parent_bytes {
                            issues.push(MemoryCoherenceIssue::CapacityExceedsParent {
                                child: space.clone(),
                                parent,
                                child_bytes,
                                parent_bytes,
                            });
                        }
                    }
                }
            }
        }
        // `HashMap` iteration is unordered; sort for stable diagnostics.
        issues.sort_by_key(|i| format!("{:?}", i));
        issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::Management;

    fn mem(name: &str, parent: Option<&str>, capacity: Option<u64>) -> MemoryDecl {
        MemoryDecl {
            name: name.into(),
            parent: parent.map(MemorySpace::from_name),
            capacity: capacity.map(ByteSize),
            bandwidth: None,
            managed: Management::default(),
            granule: None,
            doc_comment: None,
        }
    }

    fn space(name: &str) -> MemorySpace {
        MemorySpace::from_name(name)
    }

    #[test]
    fn containment_and_ancestors() {
        // TMEM within SMEM within HBM.
        let decls = vec![
            mem("HBM", None, Some(1024)),
            mem("SMEM", Some("HBM"), Some(256)),
            mem("TMEM", Some("SMEM"), Some(64)),
        ];
        let h = MemoryHierarchy::build(&decls);

        assert_eq!(
            h.ancestors(&space("TMEM")),
            vec![space("SMEM"), space("HBM")]
        );
        assert!(h.contains(&space("HBM"), &space("TMEM")));
        assert!(h.contains(&space("SMEM"), &space("TMEM")));
        assert!(!h.contains(&space("TMEM"), &space("HBM")));
        assert!(!h.contains(&space("HBM"), &space("HBM"))); // not self
    }

    #[test]
    fn nearest_common_ancestor_of_siblings_and_lineage() {
        // HBM { SMEM { TMEM }, L2 } — SMEM and L2 are siblings under HBM.
        let decls = vec![
            mem("HBM", None, None),
            mem("SMEM", Some("HBM"), None),
            mem("TMEM", Some("SMEM"), None),
            mem("L2", Some("HBM"), None),
        ];
        let h = MemoryHierarchy::build(&decls);
        assert_eq!(
            h.nearest_common_ancestor(&space("TMEM"), &space("L2")),
            Some(space("HBM"))
        );
        // A space is its own ancestor: nca(SMEM, TMEM) = SMEM.
        assert_eq!(
            h.nearest_common_ancestor(&space("SMEM"), &space("TMEM")),
            Some(space("SMEM"))
        );
    }

    #[test]
    fn detects_within_cycle() {
        let decls = vec![mem("A", Some("B"), None), mem("B", Some("A"), None)];
        let h = MemoryHierarchy::build(&decls);
        let issues = h.coherence_issues();
        assert!(issues
            .iter()
            .any(|i| matches!(i, MemoryCoherenceIssue::Cycle { space } if *space == space_a())));
        // ancestors is still cycle-safe (terminates).
        assert!(h.ancestors(&space("A")).len() <= 1);
    }

    fn space_a() -> MemorySpace {
        space("A")
    }

    #[test]
    fn detects_self_loop() {
        let decls = vec![mem("Loop", Some("Loop"), None)];
        let h = MemoryHierarchy::build(&decls);
        assert!(h
            .coherence_issues()
            .iter()
            .any(|i| matches!(i, MemoryCoherenceIssue::Cycle { .. })));
    }

    #[test]
    fn detects_capacity_exceeding_parent() {
        // Child (512) larger than parent SMEM (256): a violation.
        let decls = vec![
            mem("SMEM", None, Some(256)),
            mem("Big", Some("SMEM"), Some(512)),
        ];
        let issues = MemoryHierarchy::build(&decls).coherence_issues();
        assert_eq!(
            issues,
            vec![MemoryCoherenceIssue::CapacityExceedsParent {
                child: space("Big"),
                parent: space("SMEM"),
                child_bytes: 512,
                parent_bytes: 256,
            }]
        );
    }

    #[test]
    fn well_formed_hierarchy_has_no_issues() {
        let decls = vec![
            mem("HBM", None, Some(1_000_000)),
            mem("SMEM", Some("HBM"), Some(256)),
            mem("TMEM", Some("SMEM"), Some(256)), // equal to parent is allowed
        ];
        assert!(MemoryHierarchy::build(&decls).coherence_issues().is_empty());
    }

    #[test]
    fn detects_non_positive_property() {
        let mut d = mem("Z", None, Some(0));
        d.granule = Some(ByteSize(0));
        let issues = MemoryHierarchy::build(std::slice::from_ref(&d)).coherence_issues();
        assert_eq!(
            issues
                .iter()
                .filter(|i| matches!(i, MemoryCoherenceIssue::NonPositiveProperty { .. }))
                .count(),
            2
        );
    }

    fn dim(n: &str) -> Expr {
        Expr::Number(crate::syntax::NumberExpr::new(
            n.to_string(),
            None,
            crate::syntax::Span::default(),
        ))
    }

    #[test]
    fn tensor_bytes_dense_and_subbyte() {
        // 256x256 f32 = 262144 bytes (256 KiB).
        assert_eq!(
            static_tensor_bytes(&ElementType::F32, &[dim("256"), dim("256")]),
            Some(256 * 1024)
        );
        // 8x8 i4 = 64 elems * 4 bits = 256 bits = 32 bytes (sub-byte packs).
        assert_eq!(
            static_tensor_bytes(&ElementType::I4, &[dim("8"), dim("8")]),
            Some(32)
        );
        // 3 bools = 3 bits -> ceil to 1 byte.
        assert_eq!(
            static_tensor_bytes(&ElementType::Bool, &[dim("3")]),
            Some(1)
        );
    }

    #[test]
    fn tensor_bytes_dynamic_shape_is_unknown() {
        // A non-literal dimension (an identifier) => size unknown => no check possible.
        let dyn_dim = Expr::Identifier(crate::syntax::IdentifierExpr {
            name: "N".into(),
            span: crate::syntax::Span::default(),
        });
        assert_eq!(
            static_tensor_bytes(&ElementType::F32, &[dyn_dim, dim("4")]),
            None
        );
        // Empty shape (scalar-broadcast tensor) is also unknown.
        assert_eq!(static_tensor_bytes(&ElementType::F32, &[]), None);
        // Un-instantiated generic element is unknown.
        assert_eq!(
            static_tensor_bytes(&ElementType::Generic("T".into()), &[dim("4")]),
            None
        );
    }
}
