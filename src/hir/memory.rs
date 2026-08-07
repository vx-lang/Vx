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

use crate::syntax::{
    Bandwidth, ByteSize, ElementType, Expr, MemoryDecl, MemorySpace, RatePer, Scope,
};
use std::collections::{HashMap, HashSet};

/// A bandwidth-derived transfer cost, in the bandwidth's rate unit: **cycles** for `B/cyc`,
/// **picoseconds** for `B/s`. This is the paper's roofline: `T = bytes / bandwidth`.
///
/// Time-based costs are picoseconds, not seconds, because the cost is an integer and the
/// quantities are sub-nanosecond. At whole-second resolution `ceil(bytes / (B/s))` is 1 for every
/// transfer below one second, which is every transfer any of these machines performs: a 4 KiB tile
/// and a 64 MiB tile over the same 12 TB/s link both came out as `1`, and a 2.4x spread in declared
/// L2 bandwidth across the fleet produced no difference at all. Picoseconds keep a 4 KiB tile over
/// a 16 TB/s link (256 ps) distinguishable from the next size up, and `u64` still spans ~213 days.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct DerivedCost {
    pub value: u64,
    pub per: RatePer,
}

/// Picoseconds per second — the scale factor that gives a `B/s` roofline usable integer resolution.
const PICOS_PER_SEC: u64 = 1_000_000_000_000;

/// One hop's roofline cost: `bytes / bandwidth`, in the rate's unit (cycles, or picoseconds).
///
/// The multiply happens before the divide so the scaling does not lose the precision it exists to
/// buy, and it is checked rather than wrapping: `bytes * 1e12` overflows `u64` above ~18 PiB, which
/// is beyond any declared capacity but is reachable by a nonsense program, and a wrapped cost would
/// read as a fast transfer rather than an error.
pub fn hop_cost(bytes: u64, bw: crate::syntax::Bandwidth) -> Option<u64> {
    match bw.per {
        RatePer::Cycle => Some(bytes.div_ceil(bw.bytes)),
        RatePer::Second => Some(bytes.checked_mul(PICOS_PER_SEC)?.div_ceil(bw.bytes)),
    }
}

/// Bit width of a tensor element (dense packing, e.g. `I4` = 4 bits) — a thin wrapper over the single
/// width source `ElementType::bits`. `None` for an un-instantiated generic element. (P1-4a)
pub fn element_bits(elem: &ElementType) -> Option<u64> {
    elem.bits().map(|b| b as u64)
}

/// A tensor dimension's compile-time value: an integer literal, or exact integer arithmetic over
/// literals (`2 * LAYERS * CTX` once monomorphization has substituted its const generics). `None`
/// for anything not constant — a runtime identifier, a call — which keeps the capacity check
/// fail-closed (the caller then reports W1029 "unverified" rather than inventing a size).
///
/// Deliberately exact `u64` rather than the checker's `f64` `eval_expr`: these become byte counts
/// compared against a declared capacity, and a model-shape product (layers x context x heads x
/// head_dim x batch) reaches magnitudes where float rounding would silently shift a verdict.
/// Overflow, division by zero, and a negative intermediate all yield `None` rather than wrapping.
fn const_dim(e: &Expr) -> Option<u64> {
    match e {
        Expr::Number(n) => n.value.as_ref().parse::<u64>().ok(),
        Expr::BinaryOp(b) => {
            let l = const_dim(&b.lhs)?;
            let r = const_dim(&b.rhs)?;
            match b.op {
                crate::syntax::BinaryOp::Add => l.checked_add(r),
                crate::syntax::BinaryOp::Sub => l.checked_sub(r),
                crate::syntax::BinaryOp::Mul => l.checked_mul(r),
                crate::syntax::BinaryOp::Div => (r != 0).then_some(l / r),
                crate::syntax::BinaryOp::MatMul => None,
            }
        }
        _ => None,
    }
}

/// Byte size of a statically-shaped tensor: `ceil(element_bits × Π(dims) / 8)`. `None` when the
/// shape is empty or any dimension is not a compile-time constant (so the size — and thus any
/// capacity check — is unknown).
///
/// Dimensions are const-folded, not merely pattern-matched against a literal. A real model shape
/// is arithmetic — a KV cache is `2 * layers * context` by `heads * head_dim` — and requiring a
/// bare literal meant every such placement was silently reported "unverified" (W1029) instead of
/// admitted or rejected, which would have made an admission matrix over realistic configs vacuous.
pub fn static_tensor_bytes(elem: &ElementType, dims: &[Expr]) -> Option<u64> {
    if dims.is_empty() {
        return None;
    }
    let mut count: u64 = 1;
    for d in dims {
        count = count.checked_mul(const_dim(d)?)?;
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
    /// A sub-space's `scope` is *broader* than its parent's — locality must narrow (never
    /// widen) going down the hierarchy (e.g. a `device`-scoped space inside an `sm` one).
    ScopeWidensInChild {
        child: MemorySpace,
        parent: MemorySpace,
        child_scope: Scope,
        parent_scope: Scope,
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

    /// The bandwidth-derived cost of moving `bytes` from `src` to `dst` along the containment
    /// tree: the sum, over each space on the path *excluding* their nearest common ancestor, of
    /// `ceil(bytes / bandwidth)`. This is the paper's roofline (`T = bytes / (B/cyc)`); e.g. a
    /// tile read into SMEM at 128 B/cyc costs `bytes/128` cycles.
    ///
    /// Returns `None` when the cost is not derivable: `src == dst`, no common ancestor, a path
    /// space lacks a `bandwidth`, or the path's bandwidths mix rate units (cycles vs seconds).
    /// (Explicit topology-declared `transfer … : C` edges remain the fixed reachability cost;
    /// this is the additive roofline estimate, not a replacement for that graph.)
    pub fn derived_transfer_cost(
        &self,
        src: &MemorySpace,
        dst: &MemorySpace,
        bytes: u64,
    ) -> Option<DerivedCost> {
        if src == dst {
            return None;
        }
        let nca = self.nearest_common_ancestor(src, dst)?;
        // Spaces on the path, excluding the NCA: each endpoint and its ancestors up to (but not
        // including) the NCA. The NCA is the shared reservoir and adds no bandwidth term.
        let mut path: Vec<MemorySpace> = Vec::new();
        for endpoint in [src, dst] {
            for s in std::iter::once(endpoint.clone()).chain(self.ancestors(endpoint)) {
                if s == nca {
                    break;
                }
                path.push(s);
            }
        }
        let mut total: u64 = 0;
        let mut unit: Option<RatePer> = None;
        for s in &path {
            let bw = self.descriptor(s)?.bandwidth?;
            if bw.bytes == 0 {
                return None;
            }
            match unit {
                None => unit = Some(bw.per),
                Some(u) if u == bw.per => {}
                _ => return None, // mixed rate units cannot be summed
            }
            total = total.saturating_add(hop_cost(bytes, bw)?);
        }
        Some(DerivedCost {
            value: total,
            per: unit?,
        })
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
                // Locality must narrow (or stay equal) down the hierarchy.
                if let (Some(child_scope), Some(parent)) = (decl.scope, decl.parent.clone()) {
                    if let Some(parent_scope) = self.spaces.get(&parent).and_then(|p| p.scope) {
                        if child_scope < parent_scope {
                            issues.push(MemoryCoherenceIssue::ScopeWidensInChild {
                                child: space.clone(),
                                parent,
                                child_scope,
                                parent_scope,
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
            scope: None,
            overcommit: false,
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
    fn detects_scope_widening_and_accepts_narrowing() {
        // A device-scoped space nested in an sm-scoped one is incoherent (locality widens).
        let mut sm = mem("Sm0", None, None);
        sm.scope = Some(Scope::Sm);
        let mut wide = mem("Wide", Some("Sm0"), None);
        wide.scope = Some(Scope::Device);
        let issues = MemoryHierarchy::build(&[sm.clone(), wide]).coherence_issues();
        assert!(issues
            .iter()
            .any(|i| matches!(i, MemoryCoherenceIssue::ScopeWidensInChild { .. })));

        // The faithful B200 direction (sm inside device) is fine.
        let mut dev = mem("Dev", None, None);
        dev.scope = Some(Scope::Device);
        let mut tmem = mem("Tmem", Some("Dev"), None);
        tmem.scope = Some(Scope::Sm);
        assert!(!MemoryHierarchy::build(&[dev, tmem])
            .coherence_issues()
            .iter()
            .any(|i| matches!(i, MemoryCoherenceIssue::ScopeWidensInChild { .. })));
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

    fn mul(l: Expr, r: Expr) -> Expr {
        Expr::BinaryOp(crate::syntax::BinaryOpExpr {
            lhs: Box::new(l),
            op: crate::syntax::BinaryOp::Mul,
            rhs: Box::new(r),
            span: crate::syntax::Span::default(),
        })
    }

    /// A real model shape is arithmetic, not a bare literal: a KV cache is `2 * layers * context`
    /// by `heads * head_dim`. Requiring a literal per dimension made every such placement report
    /// W1029 "unverified" instead of an admission verdict, which would make a matrix over
    /// realistic configs vacuous. Dimensions are const-folded.
    #[test]
    fn tensor_bytes_folds_arithmetic_dimensions() {
        // 2 * 512 x 512 f32 = 2 MiB. Previously `None` (not a bare literal).
        assert_eq!(
            static_tensor_bytes(&ElementType::F32, &[mul(dim("2"), dim("512")), dim("512")]),
            Some(2 * 1024 * 1024)
        );
        // The KV-cache shape: (2 * layers * ctx) x (heads * head_dim), f16.
        let rows = mul(mul(dim("2"), dim("2")), dim("128")); // 512
        let cols = mul(dim("8"), dim("64")); // 512
        assert_eq!(
            static_tensor_bytes(&ElementType::F16, &[rows, cols]),
            Some(512 * 512 * 2)
        );
        // Folding is exact integer arithmetic, so a product far past f64's exact-integer range
        // is either right or `None` -- never silently rounded into a different verdict.
        assert_eq!(
            static_tensor_bytes(
                &ElementType::F32,
                &[mul(dim("4294967296"), dim("4294967296"))]
            ),
            None,
            "overflow declines rather than wrapping"
        );
    }

    /// A genuinely dynamic dimension must still decline, so the capacity check stays fail-closed
    /// and the caller reports W1029 rather than inventing a size.
    #[test]
    fn tensor_bytes_declines_non_constant_dimensions() {
        let ident = Expr::Identifier(crate::syntax::IdentifierExpr {
            name: "n".into(),
            span: crate::syntax::Span::default(),
        });
        assert_eq!(
            static_tensor_bytes(&ElementType::F32, &[ident.clone(), dim("4")]),
            None
        );
        // Arithmetic *containing* a runtime value is equally unknown.
        assert_eq!(
            static_tensor_bytes(&ElementType::F32, &[mul(dim("2"), ident)]),
            None
        );
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
        // 64x64 f8e4m3 = 4096 elems * 8 bits = 4096 bytes (fp8 stores like i8).
        assert_eq!(
            static_tensor_bytes(&ElementType::F8E4M3, &[dim("64"), dim("64")]),
            Some(4096)
        );
        // f8e5m2 has the same storage width; the variants differ only in exponent/mantissa split.
        assert_eq!(
            static_tensor_bytes(&ElementType::F8E5M2, &[dim("50"), dim("50")]),
            Some(2500)
        );
    }

    fn mem_bw(name: &str, parent: Option<&str>, bw_bytes: u64, per: RatePer) -> MemoryDecl {
        let mut d = mem(name, parent, None);
        d.bandwidth = Some(Bandwidth {
            bytes: bw_bytes,
            per,
        });
        d
    }

    #[test]
    fn derived_cost_single_hop_roofline() {
        // Leaf at 128 B/cyc within Root. A 16384-byte tile: 16384/128 = 128 cycles.
        let decls = vec![
            mem("Root", None, None),
            mem_bw("Leaf", Some("Root"), 128, RatePer::Cycle),
        ];
        let h = MemoryHierarchy::build(&decls);
        assert_eq!(
            h.derived_transfer_cost(&space("Root"), &space("Leaf"), 16384),
            Some(DerivedCost {
                value: 128,
                per: RatePer::Cycle
            })
        );
    }

    #[test]
    fn derived_cost_two_hop_sums_bandwidths() {
        // A (128 B/cyc) and B (256 B/cyc) are siblings under Root; A->B goes via Root, touching
        // both: 16384/128 + 16384/256 = 128 + 64 = 192 cycles.
        let decls = vec![
            mem("Root", None, None),
            mem_bw("A", Some("Root"), 128, RatePer::Cycle),
            mem_bw("B", Some("Root"), 256, RatePer::Cycle),
        ];
        let h = MemoryHierarchy::build(&decls);
        assert_eq!(
            h.derived_transfer_cost(&space("A"), &space("B"), 16384),
            Some(DerivedCost {
                value: 192,
                per: RatePer::Cycle
            })
        );
    }

    #[test]
    fn derived_cost_none_when_not_derivable() {
        // Missing bandwidth on a path space.
        let no_bw = vec![mem("Root", None, None), mem("NoBw", Some("Root"), None)];
        assert_eq!(
            MemoryHierarchy::build(&no_bw).derived_transfer_cost(
                &space("Root"),
                &space("NoBw"),
                16384
            ),
            None
        );
        // Same space (a no-op move).
        let one = vec![mem_bw("X", None, 128, RatePer::Cycle)];
        assert_eq!(
            MemoryHierarchy::build(&one).derived_transfer_cost(&space("X"), &space("X"), 100),
            None
        );
        // Unrelated spaces (no common ancestor).
        let two_roots = vec![
            mem_bw("P", None, 128, RatePer::Cycle),
            mem_bw("Q", None, 128, RatePer::Cycle),
        ];
        assert_eq!(
            MemoryHierarchy::build(&two_roots).derived_transfer_cost(&space("P"), &space("Q"), 100),
            None
        );
        // Mixed rate units (cyc + s) cannot be summed.
        let mixed = vec![
            mem("Root", None, None),
            mem_bw("Cyc", Some("Root"), 10, RatePer::Cycle),
            mem_bw("Sec", Some("Root"), 10, RatePer::Second),
        ];
        assert_eq!(
            MemoryHierarchy::build(&mixed).derived_transfer_cost(&space("Cyc"), &space("Sec"), 100),
            None
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
