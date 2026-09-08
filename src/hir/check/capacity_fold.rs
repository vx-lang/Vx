//===- check/capacity_fold.rs - Vx Compiler ---------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The cross-call half of the capacity check. Each function's check exports a
// summary -- its own per-space peak, and what stays resident across each call it
// makes. This pass folds those summaries over the call graph:
//
//   total(f, s) = max( self_peak(f, s),
//                      max over calls c in f:  live(f, c, s) + total(callee(c), s) )
//
// Sequential calls compose by max (a returned callee's tiles are gone before the
// next call's arrive); holding a tile across a call composes by +. A `spawn`
// region is inlined into its parent block by the lowering, so it is scope
// structure inside one summary rather than a graph edge; if an asynchronous
// spawn ever lands, it becomes a second edge kind whose operator is + across
// siblings.
//
// The fold reads only summaries, so the per-function phase stays embarrassingly
// parallel and this pass is O(functions + calls) integer arithmetic. It only ever adds refusals: nothing the per-function check
// admitted is re-admitted here.
//
//===----------------------------------------------------------------------===//

use std::collections::{BTreeMap, HashMap};

/// One resolved call a function makes, as its capacity summary records it.
#[derive(Debug, Clone)]
pub struct CallEdge {
    pub callee: String,
    /// Bytes still resident in each space when control transfers, granule-rounded.
    pub live: BTreeMap<String, u64>,
    pub span: crate::syntax::Span,
}

/// A function's capacity footprint: its own per-space peak and the calls it
/// makes with what stays live across each. The unit the cross-call fold reads.
#[derive(Debug, Clone)]
pub struct FnCapacitySummary {
    pub name: String,
    /// Peak resident bytes per space over this function's own placements.
    pub self_peak: BTreeMap<String, u64>,
    pub calls: Vec<CallEdge>,
}

/// A space the fold budgets against: its declared capacity and whether it is
/// `overcommit` (which downgrades the refusal to a warning, as for E6010).
struct SpaceBudget {
    cap: u64,
    overcommit: bool,
}

pub fn fold_cross_call_capacity(
    summaries: &[FnCapacitySummary],
    env: &crate::hir::env::GlobalAstEnv,
    errors: &mut crate::diagnostic::DiagnosticsVec,
) {
    // Spaces with a declared capacity. Omission means unconstrained, and an
    // unconstrained space needs no interprocedural analysis at all.
    let mut budgets: BTreeMap<String, SpaceBudget> = BTreeMap::new();
    for decl in env.memories.values() {
        if let Some(crate::syntax::ByteSize(cap)) = decl.capacity {
            let name = crate::syntax::MemorySpace::from_name(decl.name.as_ref()).name();
            budgets.insert(
                name,
                SpaceBudget {
                    cap,
                    overcommit: decl.overcommit,
                },
            );
        }
    }
    fold_with_budgets(summaries, &budgets, errors);
}

fn fold_with_budgets(
    summaries: &[FnCapacitySummary],
    budgets: &BTreeMap<String, SpaceBudget>,
    errors: &mut crate::diagnostic::DiagnosticsVec,
) {
    if budgets.is_empty() {
        return;
    }

    // First summary wins: the same monomorphized instance can be summarized by
    // several workers, and every copy is derived from the same body.
    let mut nodes: Vec<&FnCapacitySummary> = Vec::new();
    let mut index: HashMap<&str, usize> = HashMap::new();
    for s in summaries {
        if !index.contains_key(s.name.as_str()) {
            index.insert(s.name.as_str(), nodes.len());
            nodes.push(s);
        }
    }

    let sccs = tarjan_sccs(&nodes, &index);

    // Per node and space: the folded total, the call edge it peaks through (for
    // the path in the diagnostic), and whether a refusal was already emitted at
    // or below it (so one overflow reports once, at the smallest function that
    // exhibits it, rather than once more per caller).
    let mut total: Vec<BTreeMap<String, u64>> = vec![BTreeMap::new(); nodes.len()];
    let mut peak_edge: Vec<BTreeMap<String, usize>> = vec![BTreeMap::new(); nodes.len()];
    let mut reported: Vec<BTreeMap<String, bool>> = vec![BTreeMap::new(); nodes.len()];

    // Tarjan emits SCCs in reverse topological order: every callee's component
    // before its callers'. Fold in that order, so `total` of a callee is final
    // by the time a caller reads it.
    for scc in &sccs {
        let cyclic = scc.len() > 1
            || nodes[scc[0]]
                .calls
                .iter()
                .any(|c| index.get(c.callee.as_str()) == Some(&scc[0]));
        if cyclic {
            fold_cycle(
                scc,
                &nodes,
                &index,
                budgets,
                &mut total,
                &mut peak_edge,
                &mut reported,
                errors,
            );
            continue;
        }
        let n = scc[0];
        for (space, budget) in budgets {
            let space = space.as_str();
            let mut best = nodes[n].self_peak.get(space).copied().unwrap_or(0);
            let mut best_edge = None;
            for (ei, call) in nodes[n].calls.iter().enumerate() {
                let Some(&callee) = index.get(call.callee.as_str()) else {
                    // No summary: an extern or registry-imported callee. Its body
                    // was checked (or will be) in its own compilation; from here
                    // it contributes no known placements.
                    continue;
                };
                let held = call.live.get(space).copied().unwrap_or(0);
                let reach = held + total[callee].get(space).copied().unwrap_or(0);
                if reach > best {
                    best = reach;
                    best_edge = Some(ei);
                }
            }
            total[n].insert(space.to_string(), best);
            if let Some(ei) = best_edge {
                peak_edge[n].insert(space.to_string(), ei);
            }
            report_overflow(
                n,
                space,
                best,
                budget,
                &nodes,
                &index,
                &peak_edge,
                &mut reported,
                errors,
            );
        }
    }
}

/// Refuse a call path whose folded peak exceeds a space's capacity, unless the
/// overflow was already reported at this node or somewhere along the path.
#[allow(clippy::too_many_arguments)]
fn report_overflow(
    n: usize,
    space: &str,
    peak: u64,
    budget: &SpaceBudget,
    nodes: &[&FnCapacitySummary],
    index: &HashMap<&str, usize>,
    peak_edge: &[BTreeMap<String, usize>],
    reported: &mut [BTreeMap<String, bool>],
    errors: &mut crate::diagnostic::DiagnosticsVec,
) {
    if peak <= budget.cap {
        return;
    }
    // The per-function checks own the intra-function overflow: if this
    // function's own peak is already over, E6009/E6010 said so.
    if nodes[n].self_peak.get(space).copied().unwrap_or(0) > budget.cap {
        reported[n].insert(space.to_string(), true);
        return;
    }
    // Walk the peak path; a flagged node below means this is the same overflow
    // seen from one caller further up.
    let mut path: Vec<usize> = vec![n];
    let mut hops: Vec<(usize, u64)> = Vec::new(); // (node, bytes held across its call)
    let mut cur = n;
    while let Some(&ei) = peak_edge[cur].get(space) {
        let call = &nodes[cur].calls[ei];
        let Some(&next) = index.get(call.callee.as_str()) else {
            break;
        };
        if reported[next].get(space).copied().unwrap_or(false) {
            reported[n].insert(space.to_string(), true);
            return;
        }
        hops.push((cur, call.live.get(space).copied().unwrap_or(0)));
        path.push(next);
        cur = next;
    }
    reported[n].insert(space.to_string(), true);

    let names: Vec<&str> = path.iter().map(|&i| nodes[i].name.as_str()).collect();
    let terminal = *path.last().unwrap();
    let mut held: Vec<String> = hops
        .iter()
        .filter(|(_, bytes)| *bytes > 0)
        .map(|(i, bytes)| format!("'{}' holds {} bytes across its call", nodes[*i].name, bytes))
        .collect();
    held.push(format!(
        "'{}' itself peaks at {} bytes",
        nodes[terminal].name,
        nodes[terminal].self_peak.get(space).copied().unwrap_or(0)
    ));
    let msg = format!(
        "the working set along call path '{}' in memory space '{}' peaks at {} bytes, over its {} \
         byte capacity: {}",
        names.join(" -> "),
        space,
        peak,
        budget.cap,
        held.join(", ")
    );
    let span = peak_edge[n]
        .get(space)
        .map(|&ei| crate::diagnostic::SourceSpan::from_ast_span(&nodes[n].calls[ei].span));
    let facts = crate::diagnostic::DiagnosticFacts::Capacity {
        space: space.to_string(),
        required_bytes: peak,
        available_bytes: budget.cap,
        tiles: None,
    };
    if budget.overcommit {
        errors
            .warn(
                crate::diagnostic::DiagnosticCode::W1028,
                format!("{msg}; allowed because '{space}' is `overcommit`"),
                span,
            )
            .facts = Some(facts);
    } else {
        errors
            .error_with_code(crate::diagnostic::DiagnosticCode::E6027, msg, span)
            .facts = Some(facts);
    }
}

/// A cycle in the call graph. Depth is unknown at compile time, so any residency
/// per activation makes the true peak unbounded: refuse (or warn, for an
/// `overcommit` space) naming the cycle. A cycle with no residency in any
/// budgeted space folds to its outgoing calls alone -- plain recursion that never
/// touches a bounded space needs no annotation.
#[allow(clippy::too_many_arguments)]
fn fold_cycle(
    scc: &[usize],
    nodes: &[&FnCapacitySummary],
    index: &HashMap<&str, usize>,
    budgets: &BTreeMap<String, SpaceBudget>,
    total: &mut [BTreeMap<String, u64>],
    peak_edge: &mut [BTreeMap<String, usize>],
    reported: &mut [BTreeMap<String, bool>],
    errors: &mut crate::diagnostic::DiagnosticsVec,
) {
    let in_scc = |name: &str| index.get(name).is_some_and(|i| scc.contains(i));
    for (space, budget) in budgets {
        let space = space.as_str();
        let per_activation: u64 = scc
            .iter()
            .map(|&i| nodes[i].self_peak.get(space).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        // Conservative stand-in so callers above still fold: one activation plus
        // whatever the cycle reaches outside itself (those callees were folded
        // before it, in reverse topological order). The refusal below is what
        // carries the unboundedness.
        for &i in scc {
            let mut best = nodes[i].self_peak.get(space).copied().unwrap_or(0);
            for (ei, call) in nodes[i].calls.iter().enumerate() {
                if in_scc(&call.callee) {
                    continue;
                }
                let Some(&callee) = index.get(call.callee.as_str()) else {
                    continue;
                };
                let held = call.live.get(space).copied().unwrap_or(0);
                let reach = held + total[callee].get(space).copied().unwrap_or(0);
                if reach > best {
                    best = reach;
                    peak_edge[i].insert(space.to_string(), ei);
                }
            }
            total[i].insert(space.to_string(), best);
        }
        if per_activation == 0 {
            continue;
        }
        for &i in scc {
            reported[i].insert(space.to_string(), true);
        }
        let mut names: Vec<&str> = scc.iter().map(|&i| nodes[i].name.as_str()).collect();
        names.sort_unstable();
        let cycle = if names.len() == 1 {
            format!("'{}' calls itself", names[0])
        } else {
            format!("the cycle '{}' recurses", names.join(" -> "))
        };
        let msg = format!(
            "{} and places up to {} bytes in memory space '{}' per activation; the recursion \
             depth is not known at compile time, so its working set is unbounded",
            cycle, per_activation, space
        );
        let facts = crate::diagnostic::DiagnosticFacts::Capacity {
            space: space.to_string(),
            required_bytes: per_activation,
            available_bytes: budget.cap,
            tiles: None,
        };
        if budget.overcommit {
            errors
                .warn(
                    crate::diagnostic::DiagnosticCode::W1028,
                    format!("{msg}; allowed because '{space}' is `overcommit`"),
                    None,
                )
                .facts = Some(facts);
        } else {
            errors
                .error_with_code(crate::diagnostic::DiagnosticCode::E6028, msg, None)
                .facts = Some(facts);
        }
    }
}

/// Tarjan's strongly-connected components, iteratively (call graphs recurse; the
/// compiler should not). Emits components in reverse topological order.
fn tarjan_sccs(nodes: &[&FnCapacitySummary], index: &HashMap<&str, usize>) -> Vec<Vec<usize>> {
    let n = nodes.len();
    let succ: Vec<Vec<usize>> = nodes
        .iter()
        .map(|s| {
            s.calls
                .iter()
                .filter_map(|c| index.get(c.callee.as_str()).copied())
                .collect()
        })
        .collect();

    let mut ids = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut next_id = 0usize;
    let mut sccs: Vec<Vec<usize>> = Vec::new();

    for root in 0..n {
        if ids[root] != usize::MAX {
            continue;
        }
        // (node, next successor position) — an explicit DFS frame.
        let mut work: Vec<(usize, usize)> = vec![(root, 0)];
        while let Some(&mut (v, ref mut si)) = work.last_mut() {
            if *si == 0 {
                ids[v] = next_id;
                low[v] = next_id;
                next_id += 1;
                stack.push(v);
                on_stack[v] = true;
            }
            if *si < succ[v].len() {
                let w = succ[v][*si];
                *si += 1;
                if ids[w] == usize::MAX {
                    work.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(ids[w]);
                }
            } else {
                work.pop();
                if let Some(&(parent, _)) = work.last() {
                    low[parent] = low[parent].min(low[v]);
                }
                if low[v] == ids[v] {
                    let mut comp = Vec::new();
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w] = false;
                        comp.push(w);
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(comp);
                }
            }
        }
    }
    sccs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(
        name: &str,
        peaks: &[(&str, u64)],
        calls: &[(&str, &[(&str, u64)])],
    ) -> FnCapacitySummary {
        FnCapacitySummary {
            name: name.to_string(),
            self_peak: peaks.iter().map(|(s, b)| (s.to_string(), *b)).collect(),
            calls: calls
                .iter()
                .map(|(callee, live)| CallEdge {
                    callee: callee.to_string(),
                    live: live.iter().map(|(s, b)| (s.to_string(), *b)).collect(),
                    span: crate::syntax::Span::default(),
                })
                .collect(),
        }
    }

    fn budgets(caps: &[(&str, u64, bool)]) -> BTreeMap<String, SpaceBudget> {
        caps.iter()
            .map(|(s, cap, oc)| {
                (
                    s.to_string(),
                    SpaceBudget {
                        cap: *cap,
                        overcommit: *oc,
                    },
                )
            })
            .collect()
    }

    fn run(summaries: &[FnCapacitySummary], caps: &[(&str, u64, bool)]) -> Vec<String> {
        let mut errors = crate::diagnostic::DiagnosticsVec::new();
        fold_with_budgets(summaries, &budgets(caps), &mut errors);
        errors.iter().map(|d| format!("{}", d)).collect()
    }

    /// Sequential calls compose by max: two callees of 3 each under a cap of 4
    /// admit, because the first returns before the second runs.
    #[test]
    fn sequential_calls_take_the_max() {
        let s = [
            summary("main", &[], &[("f", &[]), ("g", &[])]),
            summary("f", &[("W", 3)], &[]),
            summary("g", &[("W", 3)], &[]),
        ];
        assert!(run(&s, &[("W", 4, false)]).is_empty());
    }

    /// Holding a tile across a call composes by +: the caller's 3 live across
    /// its call joins the callee's 3, over a cap of 4.
    #[test]
    fn holding_across_a_call_sums() {
        let s = [
            summary("main", &[], &[("f", &[])]),
            summary("f", &[("W", 3)], &[("g", &[("W", 3)])]),
            summary("g", &[("W", 3)], &[]),
        ];
        let diags = run(&s, &[("W", 4, false)]);
        assert_eq!(
            diags.len(),
            1,
            "one refusal rather than one per caller: {diags:?}"
        );
        assert!(
            diags[0].contains("E6027") && diags[0].contains("'f -> g'"),
            "{diags:?}"
        );
    }

    /// The path in the refusal walks every hop that contributes, from the
    /// smallest function that overflows.
    #[test]
    fn the_path_names_every_contributing_hop() {
        let s = [
            summary("main", &[("W", 2)], &[("f", &[("W", 2)])]),
            summary("f", &[("W", 2)], &[("g", &[("W", 2)])]),
            summary("g", &[("W", 1)], &[]),
        ];
        let diags = run(&s, &[("W", 4, false)]);
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert!(diags[0].contains("'main -> f -> g'"), "{diags:?}");
        assert!(diags[0].contains("peaks at 5"), "{diags:?}");
    }

    /// A function whose own peak already overflowed is E6010's report, and the
    /// fold stays silent for it and for every caller that reaches it.
    #[test]
    fn a_local_overflow_is_not_reported_again() {
        let s = [
            summary("main", &[], &[("f", &[])]),
            summary("f", &[("W", 9)], &[]),
        ];
        assert!(run(&s, &[("W", 4, false)]).is_empty());
    }

    /// An `overcommit` space downgrades the cross-call refusal to a warning,
    /// exactly as it does E6010.
    #[test]
    fn overcommit_downgrades_to_a_warning() {
        let s = [
            summary("f", &[("W", 3)], &[("g", &[("W", 3)])]),
            summary("g", &[("W", 3)], &[]),
        ];
        let diags = run(&s, &[("W", 4, true)]);
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert!(
            diags[0].contains("W1028") && diags[0].contains("overcommit"),
            "{diags:?}"
        );
    }

    /// Residency in a cycle is unbounded and refused; the same cycle with no
    /// placements in any budgeted space folds through silently.
    #[test]
    fn a_cycle_refuses_exactly_when_it_places() {
        let placing = [summary("deep", &[("W", 3)], &[("deep", &[])])];
        let diags = run(&placing, &[("W", 4, false)]);
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert!(diags[0].contains("E6028"), "{diags:?}");

        let clean = [
            summary("fact", &[], &[("fact", &[])]),
            summary("main", &[("W", 3)], &[("fact", &[])]),
        ];
        assert!(run(&clean, &[("W", 4, false)]).is_empty());
    }

    /// A residency-free cycle still conducts: what it reaches outside itself
    /// joins what its own callers hold.
    #[test]
    fn a_clean_cycle_still_conducts_downstream_residency() {
        let s = [
            summary("main", &[("W", 2)], &[("spin", &[("W", 2)])]),
            summary("spin", &[], &[("spin", &[]), ("g", &[])]),
            summary("g", &[("W", 3)], &[]),
        ];
        let diags = run(&s, &[("W", 4, false)]);
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert!(
            diags[0].contains("E6027") && diags[0].contains("peaks at 5"),
            "{diags:?}"
        );
    }

    /// Mutual recursion is one component whether or not any single function
    /// calls itself.
    #[test]
    fn mutual_recursion_is_one_cycle() {
        let s = [
            summary("a", &[("W", 1)], &[("b", &[])]),
            summary("b", &[], &[("a", &[])]),
        ];
        let diags = run(&s, &[("W", 4, false)]);
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert!(
            diags[0].contains("E6028") && diags[0].contains("a -> b"),
            "{diags:?}"
        );
    }

    /// A callee with no summary (an extern) contributes nothing rather than
    /// poisoning the fold.
    #[test]
    fn an_unknown_callee_contributes_nothing() {
        let s = [summary(
            "f",
            &[("W", 3)],
            &[("mystery_extern", &[("W", 3)])],
        )];
        assert!(run(&s, &[("W", 4, false)]).is_empty());
    }

    /// Spaces without a declared budget are not folded at all.
    #[test]
    fn unbudgeted_spaces_are_skipped() {
        let s = [
            summary("f", &[("DRAM", 100)], &[("g", &[("DRAM", 100)])]),
            summary("g", &[("DRAM", 100)], &[]),
        ];
        assert!(run(&s, &[("W", 4, false)]).is_empty());
    }
}
