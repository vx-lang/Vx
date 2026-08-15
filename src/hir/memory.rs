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

/// One transfer, as an element of a set that is in flight together (vx-review#17).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Flow {
    pub src: MemorySpace,
    pub dst: MemorySpace,
    pub bytes: u64,
}

/// A space on a flow's route that another flow in the same set also traverses.
///
/// `flows` counts every flow through the space including this one, so it is always >= 2. It is the
/// "number of users" a `bandwidth / users` model would divide by -- recorded, not applied: whether
/// that is the right law is what M3 measures, and the issue notes it may be a cliff rather than a
/// smooth falloff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sharing {
    pub space: MemorySpace,
    pub flows: usize,
}

/// What the model can say about one flow, given the others it is in flight with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlowCost {
    /// The roofline cost, priced as though this flow had the hardware to itself.
    ///
    /// Still the isolated figure even when `sharing` is non-empty. That gap is deliberate and is
    /// the point of the type: the model has no contention term, so the honest thing is to report
    /// the cost it can compute alongside the sharing it cannot yet price, rather than to quietly
    /// report the isolated number as though it were the answer.
    pub cost: Option<DerivedCost>,
    /// The spaces this flow is charged for, in the order `route_spaces` yields them.
    pub route: Vec<MemorySpace>,
    /// Spaces on `route` that at least one other flow in the set also traverses.
    pub sharing: Vec<Sharing>,
}

impl FlowCost {
    /// Whether any space on this flow's route carries another flow.
    pub fn is_contended(&self) -> bool {
        !self.sharing.is_empty()
    }

    /// The most-shared space on this route and its user count -- the bottleneck a contention law
    /// would price first. `None` when the flow is exclusive.
    pub fn worst_sharing(&self) -> Option<&Sharing> {
        self.sharing.iter().max_by_key(|s| s.flows)
    }
}

/// Picoseconds per second — the scale factor that gives a `B/s` roofline usable integer resolution.
const PICOS_PER_SEC: u64 = 1_000_000_000_000;

/// Round `bytes` up to a whole number of `granule`s — what a tile actually *occupies* in a space
/// with an allocation granule, as opposed to what it contains. A 1-byte tile in a 1 KiB-granule
/// space consumes 1 KiB.
///
/// One definition, because there were two: the per-tile capacity check and the working-set sum
/// each spelled `raw.div_ceil(g) * g` inline, so a change to the rounding rule could have been
/// applied to one and not the other, and the two answers would have disagreed about the same tile.
/// `None` or a zero granule means no rounding. Idempotent by construction, which S3 asserts
/// (vx-review#11).
pub fn granule_round(bytes: u64, granule: Option<u64>) -> u64 {
    match granule {
        Some(g) if g > 0 => bytes.div_ceil(g).saturating_mul(g),
        _ => bytes,
    }
}

/// One hop's roofline cost: `bytes / bandwidth`, in the rate's unit (cycles, or picoseconds).
///
/// The multiply happens before the divide so the picosecond scaling does not lose the precision it
/// exists to buy, and the intermediate is `u128`. `u64` is not enough: `bytes * 1e12` overflows it
/// at only ~18 MB, so a 64 MiB tile -- an ordinary transfer, and smaller than a single attention
/// block on the fleet's own corpus -- produced no cost at all. The *result* still fits `u64`
/// comfortably (64 GiB over a 100 GB/s link is 6.9e10 ps), so only the intermediate needs the
/// width; the final narrowing is checked rather than truncating.
pub fn hop_cost(bytes: u64, bw: crate::syntax::Bandwidth) -> Option<u64> {
    match bw.per {
        RatePer::Cycle => Some(bytes.div_ceil(bw.bytes)),
        RatePer::Second => {
            let scaled = (bytes as u128).checked_mul(PICOS_PER_SEC as u128)?;
            u64::try_from(scaled.div_ceil(bw.bytes as u128)).ok()
        }
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

    /// The spaces a `src -> dst` move is charged for: each endpoint and its ancestors up to (but
    /// not including) the nearest common ancestor.
    ///
    /// For siblings the NCA is a reservoir the transfer passes through, not a rate it is charged,
    /// so it contributes no bandwidth term. But when the NCA *is* one of the endpoints -- a
    /// containment hop like `SMEM within L2` -- it is not passive: it is where the data physically
    /// comes from or goes to, and its delivery rate is squarely on the critical path. Excluding it
    /// prices only the destination's side of the edge and charges the source nothing.
    ///
    /// Measured on an H100 (vx-review#15): `L2->SMEM` priced at SMEM's declared 128 B/cyc alone
    /// scored -91.3% against hardware. Charging both halves -- L2's measured read rate of
    /// 23.7 B/cyc per SM plus SMEM's measured write rate of 77.8 -- predicts 18.2 B/cyc against a
    /// measured 16.6, which is within 10%. The missing term was the whole error.
    ///
    /// Split out from `derived_transfer_cost` because the route is also what says which flows
    /// contend: two transfers collide exactly where their routes share a space (vx-review#17).
    pub fn route_spaces(&self, src: &MemorySpace, dst: &MemorySpace) -> Option<Vec<MemorySpace>> {
        if src == dst {
            return None;
        }
        let nca = self.nearest_common_ancestor(src, dst)?;
        let mut path: Vec<MemorySpace> = Vec::new();
        for endpoint in [src, dst] {
            for s in std::iter::once(endpoint.clone()).chain(self.ancestors(endpoint)) {
                if s == nca {
                    break;
                }
                path.push(s);
            }
        }
        if &nca == src || &nca == dst {
            path.push(nca.clone());
        }
        Some(path)
    }

    /// Price a set of transfers that are in flight **together**.
    ///
    /// The reason this exists rather than a loop over `derived_transfer_cost`: contention is a
    /// property of a *schedule*, not of a transfer. `derived_transfer_cost(src, dst, bytes)` has
    /// nowhere to put "and seven other transfers are crossing the same link", so no amount of
    /// improving it could ever express sharing. The set has to be the input.
    ///
    /// What this does **not** do is model contention. `FlowCost::cost` is still the isolated
    /// roofline even for a flow that `FlowCost::sharing` reports as contended, because the model
    /// has no sharing law and M3 (vx-review#17) has not been run. Inventing one before it is
    /// measured is precisely what the freeze exists to prevent. The value here is that the gap is
    /// now *visible and typed* -- a caller can see "this flow shares HBM with three others and was
    /// priced as though it were alone" -- instead of being invisible in a signature that could not
    /// have said otherwise.
    pub fn derived_transfer_costs(&self, flows: &[Flow]) -> Vec<FlowCost> {
        let routes: Vec<Option<Vec<MemorySpace>>> = flows
            .iter()
            .map(|f| self.route_spaces(&f.src, &f.dst))
            .collect();

        // How many flows traverse each space. Counted over the whole set first, because a flow's
        // own contention depends on every other flow and not on the ones before it.
        let mut users: HashMap<&MemorySpace, usize> = HashMap::new();
        for route in routes.iter().flatten() {
            for s in route {
                *users.entry(s).or_insert(0) += 1;
            }
        }

        flows
            .iter()
            .zip(routes.iter())
            .map(|(f, route)| {
                let route = route.clone().unwrap_or_default();
                let sharing = route
                    .iter()
                    .filter_map(|s| {
                        let n = *users.get(s).unwrap_or(&0);
                        (n > 1).then(|| Sharing {
                            space: s.clone(),
                            flows: n,
                        })
                    })
                    .collect();
                FlowCost {
                    cost: self.derived_transfer_cost(&f.src, &f.dst, f.bytes),
                    route,
                    sharing,
                }
            })
            .collect()
    }

    /// The bandwidth-derived cost of moving `bytes` from `src` to `dst` along the containment
    /// tree: the sum, over each space on the path, of `ceil(bytes / bandwidth)`. This is the
    /// paper's roofline (`T = bytes / (B/cyc)`).
    ///
    /// A shared ancestor that is neither endpoint is excluded — it is a reservoir the transfer
    /// passes through, not a rate it is charged. An ancestor that *is* an endpoint is included:
    /// on a containment hop like `SMEM within L2` the parent is where the data physically comes
    /// from, and its delivery rate is on the critical path. Charging only the child is what scored
    /// `L2->SMEM` at -91.3% against an H100, and `HBM->L2` at -85.1% (vx-review#15).
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
        let path = self.route_spaces(src, dst)?;
        // Collect the terms first, because whether the path can be summed natively depends on all
        // of them: a path is kept in cycles only if *every* space on it is cycle-denominated.
        #[allow(clippy::type_complexity)]
        let mut terms: Vec<(crate::syntax::Bandwidth, Option<u64>, Option<u64>)> =
            Vec::with_capacity(path.len());
        for s in &path {
            let d = self.descriptor(s)?;
            let bw = d.bandwidth?;
            if bw.bytes == 0 {
                return None;
            }
            terms.push((bw, d.clock_hz, d.replicas));
        }
        if terms.is_empty() {
            return None;
        }

        // Put every term in the same DENOMINATION before summing any of them.
        //
        // A `scope: sm` space quotes the rate of one instance; a device-scoped space quotes an
        // aggregate over all of them. Adding those directly is a unit error that looks like
        // arithmetic: on an H100 L2's declared 12 TB/s is 6060 B/cyc device-wide against SMEM's 128
        // per SM, so the L2 term contributed nothing and `L2->SMEM` sat at -87% even after the
        // containment fix. Dividing the aggregate by `replicas` puts both in per-instance terms,
        // which is the right denomination here because a transfer into an sm-scoped space is
        // performed by one SM.
        //
        // Only applied when a replicated space is actually on the path; a purely device-scoped
        // walk keeps its aggregate figures untouched.
        let replicas = terms
            .iter()
            .filter_map(|(_, _, r)| *r)
            .max()
            .filter(|r| *r > 1);
        let terms: Vec<(Bandwidth, Option<u64>)> = terms
            .into_iter()
            .map(|(mut bw, clock, own)| {
                if let Some(r) = replicas {
                    // A space that declares its own count is already per-instance.
                    if own.is_none() {
                        bw.bytes = (bw.bytes / r).max(1);
                    }
                }
                (bw, clock)
            })
            .collect();

        // How the legs combine, from the DESTINATION's declaration (vx-review#26).
        //
        // The destination decides because it is the destination's fill mechanism that determines
        // whether the walk is one hardware transaction or a chain of instructions. `crossing:
        // streamed` means a hardware engine fills it without staging through registers, so nothing
        // is written and read back and the narrowest leg alone sets the rate; `sequenced` means a
        // load followed by a store, which really do happen one after the other.
        //
        // Defaults to `sequenced`, which is what the algebra has always done -- so a machine file
        // that says nothing gets exactly its previous cost and the frozen cells do not move.
        let streamed =
            self.descriptor(dst).map(|d| d.crossing) == Some(crate::syntax::Crossing::Streamed);
        let combine = |acc: u64, term: u64| -> u64 {
            if streamed {
                acc.max(term)
            } else {
                acc.saturating_add(term)
            }
        };

        let all_cycles = terms.iter().all(|(bw, _)| bw.per == RatePer::Cycle);
        if all_cycles {
            // Stay in cycles. A cycle count is clock-invariant, so converting it to wall time here
            // would throw away the one property that survives an unpinned clock.
            let mut total: u64 = 0;
            for (bw, _) in &terms {
                total = combine(total, hop_cost(bytes, *bw)?);
            }
            return Some(DerivedCost {
                value: total,
                per: RatePer::Cycle,
            });
        }

        // Otherwise the path mixes `B/s` and `B/cyc` and everything is converted to picoseconds.
        // A cycle-denominated space must declare the clock those cycles are counted in; without it
        // the conversion would require inventing a frequency, which is exactly the silent
        // translation PREDICTIONS.md decision 5 forbids. Refusing is the honest answer, and the
        // fleet files declare `clock:` precisely so this path stays derivable.
        let mut total_ps: u128 = 0;
        for (bw, clock_hz) in &terms {
            let ps = match bw.per {
                RatePer::Second => hop_cost(bytes, *bw)? as u128,
                RatePer::Cycle => {
                    let hz = (*clock_hz)? as u128;
                    if hz == 0 {
                        return None;
                    }
                    let cycles = bytes.div_ceil(bw.bytes) as u128;
                    cycles.checked_mul(PICOS_PER_SEC as u128)?.div_ceil(hz)
                }
            };
            total_ps = if streamed {
                total_ps.max(ps)
            } else {
                total_ps.checked_add(ps)?
            };
        }
        Some(DerivedCost {
            value: u64::try_from(total_ps).ok()?,
            per: RatePer::Second,
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
            clock_hz: None,
            replicas: None,
            managed: Management::default(),
            granule: None,
            scope: None,
            overcommit: false,
            crossing: crate::syntax::Crossing::default(),
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

    fn flow(src: &str, dst: &str, bytes: u64) -> Flow {
        Flow {
            src: space(src),
            dst: space(dst),
            bytes,
        }
    }

    /// A three-level hierarchy: SMEM within L2 within HBM, all with bandwidths.
    fn three_level() -> Vec<MemoryDecl> {
        vec![
            mem_bw("HBM", None, 256, RatePer::Cycle),
            mem_bw("L2", Some("HBM"), 512, RatePer::Cycle),
            mem_bw("SMEM", Some("L2"), 128, RatePer::Cycle),
        ]
    }

    #[test]
    fn one_flow_costs_what_it_always_did() {
        // The set-taking entry point must not change any answer while it carries no contention
        // law. A single flow is the old question asked the new way.
        let decls = three_level();
        let h = MemoryHierarchy::build(&decls);
        let out = h.derived_transfer_costs(&[flow("L2", "SMEM", 16384)]);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].cost,
            h.derived_transfer_cost(&space("L2"), &space("SMEM"), 16384)
        );
        assert!(!out[0].is_contended(), "a lone flow shares nothing");
    }

    #[test]
    fn concurrent_flows_report_the_spaces_they_share() {
        // Two flows out of the same L2 into SMEM. They collide on both spaces, and the count
        // includes the flow itself, so each reports 2 users.
        let decls = three_level();
        let h = MemoryHierarchy::build(&decls);
        let out = h.derived_transfer_costs(&[flow("L2", "SMEM", 16384), flow("L2", "SMEM", 16384)]);
        for fc in &out {
            assert!(fc.is_contended());
            assert_eq!(fc.worst_sharing().map(|s| s.flows), Some(2));
            let mut spaces: Vec<String> = fc
                .sharing
                .iter()
                .map(|s| format!("{:?}", s.space))
                .collect();
            spaces.sort();
            assert_eq!(spaces.len(), 2, "both L2 and SMEM are shared");
        }
    }

    #[test]
    fn disjoint_flows_are_not_contended() {
        // HBM->L2 and L2->SMEM both touch L2, so they DO contend; HBM->L2 against a flow that
        // never leaves the SMEM subtree would not. Uses the asymmetry to check the counter is
        // per-space rather than per-set.
        let decls = three_level();
        let h = MemoryHierarchy::build(&decls);
        let out = h.derived_transfer_costs(&[flow("HBM", "L2", 4096), flow("L2", "SMEM", 4096)]);
        assert!(out[0].is_contended());
        assert!(out[1].is_contended());
        // L2 is on both routes; HBM is only on the first, SMEM only on the second.
        assert_eq!(out[0].sharing.len(), 1);
        assert_eq!(out[1].sharing.len(), 1);
        assert_eq!(out[0].sharing[0].space, space("L2"));
        assert_eq!(out[1].sharing[0].space, space("L2"));
    }

    #[test]
    fn contended_flows_are_still_priced_as_exclusive() {
        // The deliberate gap, pinned so it cannot be closed by accident. Until M3 measures a
        // sharing law (vx-review#17), a contended flow costs exactly what a lone one costs -- and
        // the type says so out loud rather than the signature hiding the question.
        let decls = three_level();
        let h = MemoryHierarchy::build(&decls);
        let alone = h.derived_transfer_costs(&[flow("L2", "SMEM", 16384)]);
        let crowded = h.derived_transfer_costs(&[
            flow("L2", "SMEM", 16384),
            flow("L2", "SMEM", 16384),
            flow("L2", "SMEM", 16384),
            flow("L2", "SMEM", 16384),
        ]);
        assert_eq!(crowded[0].worst_sharing().map(|s| s.flows), Some(4));
        assert_eq!(
            crowded[0].cost, alone[0].cost,
            "no contention term exists yet; four-way sharing must not silently invent one"
        );
    }

    #[test]
    fn an_underivable_flow_still_reports_its_neighbours() {
        // `src == dst` has no route and no cost. It must not poison the set: the other flow's
        // sharing is computed over the routes that do exist.
        let decls = three_level();
        let h = MemoryHierarchy::build(&decls);
        let out = h.derived_transfer_costs(&[flow("L2", "L2", 4096), flow("L2", "SMEM", 4096)]);
        assert_eq!(out[0].cost, None);
        assert!(out[0].route.is_empty());
        assert!(
            !out[1].is_contended(),
            "a routeless flow contends with nothing"
        );
        assert!(out[1].cost.is_some());
    }

    /// `three_level()` with `crossing: streamed` declared on SMEM.
    fn three_level_streamed() -> Vec<MemoryDecl> {
        let mut v = three_level();
        v[2].crossing = crate::syntax::Crossing::Streamed;
        v
    }

    #[test]
    fn sequenced_is_the_default_and_sums() {
        // The default must be the pre-existing behaviour, or introducing `crossing:` would move
        // every frozen cell at once instead of letting a machine file opt in.
        let decls = three_level();
        assert_eq!(decls[2].crossing, crate::syntax::Crossing::Sequenced);
        let h = MemoryHierarchy::build(&decls);
        // L2 (512 B/cyc) -> SMEM (128 B/cyc), 16384 B: 32 + 128 = 160 cycles.
        assert_eq!(
            h.derived_transfer_cost(&space("L2"), &space("SMEM"), 16384),
            Some(DerivedCost {
                value: 160,
                per: RatePer::Cycle
            })
        );
    }

    #[test]
    fn streamed_takes_the_slowest_leg_alone() {
        // Nothing stages, so the narrowest leg sets the rate: max(32, 128) = 128, not 160.
        let decls = three_level_streamed();
        let h = MemoryHierarchy::build(&decls);
        assert_eq!(
            h.derived_transfer_cost(&space("L2"), &space("SMEM"), 16384),
            Some(DerivedCost {
                value: 128,
                per: RatePer::Cycle
            })
        );
    }

    #[test]
    fn crossing_is_read_from_the_destination_not_the_source() {
        // It is the DESTINATION's fill mechanism that decides whether the walk is one hardware
        // transaction or a chain of instructions, so a streamed SMEM must not make a walk that
        // merely *starts* at SMEM behave differently.
        let decls = three_level_streamed();
        let h = MemoryHierarchy::build(&decls);
        // SMEM -> L2: destination L2 is sequenced, so this still sums.
        assert_eq!(
            h.derived_transfer_cost(&space("SMEM"), &space("L2"), 16384),
            Some(DerivedCost {
                value: 160,
                per: RatePer::Cycle
            })
        );
    }

    #[test]
    fn streamed_applies_across_a_multi_hop_walk() {
        // HBM (256) -> SMEM (128) through L2 (512): terms 64, 32, 128.
        // sequenced sums to 224; streamed takes 128.
        let seq = three_level();
        let str_ = three_level_streamed();
        assert_eq!(
            MemoryHierarchy::build(&seq).derived_transfer_cost(
                &space("HBM"),
                &space("SMEM"),
                16384
            ),
            Some(DerivedCost {
                value: 224,
                per: RatePer::Cycle
            })
        );
        assert_eq!(
            MemoryHierarchy::build(&str_).derived_transfer_cost(
                &space("HBM"),
                &space("SMEM"),
                16384
            ),
            Some(DerivedCost {
                value: 128,
                per: RatePer::Cycle
            })
        );
    }

    #[test]
    fn containment_hop_charges_both_endpoints() {
        // Leaf (128 B/cyc) within Root (256 B/cyc). Root -> Leaf is a containment hop: the data
        // has to come OUT of Root as well as INTO Leaf, so both terms count.
        // 16384/256 + 16384/128 = 64 + 128 = 192 cycles.
        //
        // This previously charged the child alone (128 cycles), which is the defect that scored
        // -91.3% against an H100 on the L2->SMEM seam (vx-review#15): the source's delivery rate
        // was priced at zero.
        let decls = vec![
            mem_bw("Root", None, 256, RatePer::Cycle),
            mem_bw("Leaf", Some("Root"), 128, RatePer::Cycle),
        ];
        let h = MemoryHierarchy::build(&decls);
        assert_eq!(
            h.derived_transfer_cost(&space("Root"), &space("Leaf"), 16384),
            Some(DerivedCost {
                value: 192,
                per: RatePer::Cycle
            })
        );
        // Symmetric: spilling out of Leaf costs the same two terms.
        assert_eq!(
            h.derived_transfer_cost(&space("Leaf"), &space("Root"), 16384),
            Some(DerivedCost {
                value: 192,
                per: RatePer::Cycle
            })
        );
    }

    #[test]
    fn containment_hop_needs_the_parents_bandwidth_too() {
        // A parent with no declared bandwidth makes the hop NOT derivable, rather than silently
        // pricing it at the child's rate alone. Under-pricing by omission is what produced the
        // -91.3% seam; refusing to answer is the honest failure.
        let decls = vec![
            mem("Root", None, None),
            mem_bw("Leaf", Some("Root"), 128, RatePer::Cycle),
        ];
        let h = MemoryHierarchy::build(&decls);
        assert_eq!(
            h.derived_transfer_cost(&space("Root"), &space("Leaf"), 16384),
            None
        );
    }

    #[test]
    fn containment_hop_matches_the_h100_measurement() {
        // The measured H100 case, in per-SM B/cyc (vx-review#15, measurements/EDGES.md):
        // L2 delivers 23 B/cyc to one SM and SMEM absorbs 77, so a 16 KiB tile costs
        // ceil(16384/23) + ceil(16384/77) = 713 + 213 = 926 cycles -> 17.7 B/cyc effective.
        // Hardware measured 16.6 B/cyc on that seam. Pricing SMEM alone would have said
        // 16384/128 = 128 cycles, i.e. 128 B/cyc -- off by 7.7x.
        let decls = vec![
            mem_bw("L2", None, 23, RatePer::Cycle),
            mem_bw("SMEM", Some("L2"), 77, RatePer::Cycle),
        ];
        let h = MemoryHierarchy::build(&decls);
        let cost = h
            .derived_transfer_cost(&space("L2"), &space("SMEM"), 16384)
            .expect("derivable");
        assert_eq!(cost.per, RatePer::Cycle);
        assert_eq!(cost.value, 926);
        let effective = 16384.0 / cost.value as f64;
        assert!(
            (16.0..=19.0).contains(&effective),
            "effective {effective} B/cyc should bracket the measured 16.6"
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
