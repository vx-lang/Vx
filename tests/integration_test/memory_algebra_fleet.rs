//===- memory_algebra_fleet.rs - Vx Compiler -------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// S1 (vx-review#9): staging-synthesis correctness, over the WHOLE fleet rather than hand-picked
// pairs.
//
// The calibration study's predictions are per-hop costs along a synthesized route. If synthesis is
// wrong anywhere -- a route that is not the cheapest, a hop that was never declared, a pair
// reported reachable that is not -- then every prediction downstream of it is wrong too, and the
// measurement campaign would be calibrating against a bug. These are cheap properties to state and
// they hold for every `(src, dst)` pair of every machine file, so they are checked that way.
//
// Each property is checked against an *independent* oracle where one is available: shortest paths
// are re-derived by Bellman-Ford in this file rather than by asking the same Dijkstra the compiler
// uses, because a test that calls the implementation to compute its own expectation agrees with
// the implementation by construction and can only catch panics.
//
//===----------------------------------------------------------------------===//

use std::collections::{HashMap, HashSet};
use vxc::arch::TransferCostGraph;
use vxc::hir::memory::MemoryHierarchy;
use vxc::syntax::MemorySpace;

/// Every machine file the fleet ships. `admit.vx` is deliberately absent: it is the admission
/// *program*, not a SKU model, and has no declarations to check.
const FLEET: &[&str] = &[
    "fleet/a100-40.vx",
    "fleet/a100-80.vx",
    "fleet/b200.vx",
    "fleet/h100-sxm.vx",
    "fleet/h200.vx",
    "fleet/m4-uma.vx",
    "fleet/mi300x.vx",
    "fleet/node-8gpu.vx",
];

struct Machine {
    module: vxc::syntax::VxModule,
}

fn load(path: &str) -> Machine {
    let src = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: cannot read machine file: {e}"));
    let module = vxc::parse_module(&src).unwrap_or_else(|e| panic!("{path}: parse failed: {e}"));
    Machine { module }
}

impl Machine {
    fn graph(&self) -> TransferCostGraph {
        let mut g = TransferCostGraph::default();
        g.seed_from_topologies(&self.module.topologies);
        g
    }

    fn hierarchy(&self) -> MemoryHierarchy<'_> {
        MemoryHierarchy::build(self.module.memories.iter())
    }

    /// Every space in the *graph the router actually searches* -- this file's declarations plus
    /// the built-in topology's, which contribute both edges and spaces of their own.
    ///
    /// Taking only the file's spaces was this test's own first bug: the oracle then searched a
    /// subgraph and reported `CPU_DRAM -> NIC_RAM` as costing 201 when the router had legitimately
    /// found 55 through `NPU_HBM`, a built-in space the file never names. The router was right.
    fn spaces(&self, graph: &TransferCostGraph) -> Vec<MemorySpace> {
        let mut seen = graph.nodes();
        for m in &self.module.memories {
            let s = MemorySpace::from_name(m.name.as_ref());
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
        seen
    }
}

/// Shortest path cost from `src` to every space, by Bellman-Ford over the declared edges.
///
/// The independent oracle. Bellman-Ford rather than Dijkstra on purpose: a different algorithm
/// with a different failure mode, so agreeing with it is evidence rather than tautology.
fn bellman_ford(
    graph: &TransferCostGraph,
    spaces: &[MemorySpace],
    src: &MemorySpace,
) -> HashMap<MemorySpace, u64> {
    let edges: Vec<(MemorySpace, MemorySpace, u64)> = spaces
        .iter()
        .flat_map(|u| {
            spaces.iter().filter_map(move |v| {
                graph
                    .direct_edge_weight(u, v)
                    .map(|w| (u.clone(), v.clone(), w as u64))
            })
        })
        .collect();

    let mut dist: HashMap<MemorySpace, u64> = HashMap::new();
    dist.insert(src.clone(), 0);
    // |V|-1 relaxation rounds. All weights are non-negative, so there are no negative cycles to
    // detect and the fixpoint is reached by then.
    for _ in 0..spaces.len().saturating_sub(1).max(1) {
        let mut changed = false;
        for (u, v, w) in &edges {
            if let Some(&du) = dist.get(u) {
                let cand = du + w;
                if dist.get(v).is_none_or(|&dv| cand < dv) {
                    dist.insert(v.clone(), cand);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    dist
}

/// **P1: a route exists exactly when the pair is reachable, and P2: it is cost-minimal.**
///
/// Both directions matter. A route where none should exist would let the compiler stage data over
/// a link the hardware does not have; a missing route where one exists rejects a legal program.
/// Minimality is what makes a *predicted* cost meaningful at all -- the model's claim is that it
/// picks the cheapest declared route, and an off-by-one-hop route is a silently wrong prediction
/// rather than a crash.
#[test]
fn fleet_routes_exist_iff_reachable_and_are_cost_minimal() {
    for path in FLEET {
        let m = load(path);
        let graph = m.graph();
        let spaces = m.spaces(&graph);
        assert!(
            spaces.len() >= 3,
            "{path}: only {} spaces found -- the sweep below would be vacuous",
            spaces.len()
        );

        for src in &spaces {
            let oracle = bellman_ford(&graph, &spaces, src);
            for dst in &spaces {
                let got = graph.transfer_path(src, dst);
                let want = oracle.get(dst).copied();

                match (&got, want) {
                    (Some((cost, route)), Some(best)) => {
                        assert_eq!(
                            *cost as u64, best,
                            "{path}: {:?} -> {:?}: router returned cost {cost}, but the cheapest \
                             declared route costs {best}",
                            src, dst
                        );
                        // The returned path must actually be the thing that costs that much.
                        assert_eq!(
                            route.first(),
                            Some(src),
                            "{path}: route does not start at src"
                        );
                        assert_eq!(route.last(), Some(dst), "{path}: route does not end at dst");
                        let walked: u64 = route
                            .windows(2)
                            .map(|w| {
                                graph.direct_edge_weight(&w[0], &w[1]).unwrap_or_else(|| {
                                    panic!(
                                        "{path}: route {:?} -> {:?} contains hop {:?} -> {:?}, \
                                         which is not a declared edge",
                                        src, dst, w[0], w[1]
                                    )
                                }) as u64
                            })
                            .sum();
                        assert_eq!(
                            walked, best,
                            "{path}: {:?} -> {:?}: the returned route walks to {walked} but was \
                             reported as {best}",
                            src, dst
                        );
                    }
                    (None, Some(best)) if src != dst => panic!(
                        "{path}: {:?} -> {:?} is reachable at cost {best} but the router found no \
                         route",
                        src, dst
                    ),
                    (Some(_), None) => panic!(
                        "{path}: {:?} -> {:?} has no declared route but the router synthesized one",
                        src, dst
                    ),
                    _ => {}
                }

                // The all-pairs matrix and the path finder must agree about existence, or a
                // reachability question gets a different answer depending on which is asked.
                if src != dst {
                    assert_eq!(
                        graph.can_transfer(src, dst),
                        got.is_some(),
                        "{path}: can_transfer and transfer_path disagree for {:?} -> {:?}",
                        src,
                        dst
                    );
                }
            }
        }
    }
}

/// **P3: every hop of every synthesized route is a declared edge.**
///
/// Checked inside P1/P2 above for the routes the router returns; this states it as its own
/// property over the reachable closure so a failure names the invariant rather than a cost
/// mismatch. A synthesized hop that nobody declared is the model inventing hardware.
#[test]
fn fleet_route_hops_are_all_declared_edges() {
    for path in FLEET {
        let m = load(path);
        let graph = m.graph();
        let spaces = m.spaces(&graph);
        let mut checked = 0usize;
        for src in &spaces {
            for dst in &spaces {
                let Some((_, route)) = graph.transfer_path(src, dst) else {
                    continue;
                };
                for hop in route.windows(2) {
                    assert!(
                        graph.has_direct_edge(&hop[0], &hop[1]),
                        "{path}: synthesized route {:?} -> {:?} uses undeclared hop {:?} -> {:?}",
                        src,
                        dst,
                        hop[0],
                        hop[1]
                    );
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 0,
            "{path}: no multi-space routes were examined -- the property held vacuously"
        );
    }
}

/// **P4: the NCA rule for sibling moves.**
///
/// Two spaces that nest under a common ancestor are moved by going up to their nearest common
/// ancestor and back down; the ancestor itself is the shared reservoir and contributes no
/// bandwidth term. The cost must therefore be the sum over the two legs *excluding* the NCA, and
/// it must be symmetric -- moving a tile SMEM->TMEM cannot cost more than TMEM->SMEM when the
/// model has no notion of direction.
/// The fleet's own hierarchies are linear chains (`HBM <- L2 <- SMEM`), so **no fleet file
/// contains a true sibling pair** and sweeping them alone would pass this property vacuously --
/// which is how the first version of this test "passed". A synthetic machine with two spaces under
/// one parent is therefore checked alongside, so the rule is exercised by something.
///
/// Kept as a machine file's text rather than a hand-built hierarchy so it goes through the same
/// parser the fleet does.
const SIBLING_MACHINE: &str = "
Memory GPU_HBM { capacity: 40 GiB, bandwidth: 3 TB/s }
Memory SMEM { within: Memory::GPU_HBM, capacity: 228 KiB, bandwidth: 128 B/cyc, scope: sm }
Memory TMEM { within: Memory::GPU_HBM, capacity: 256 KiB, bandwidth: 256 B/cyc, scope: sm }
Topology Dev {
  memory: Memory::GPU_HBM,
  visible: [Memory::GPU_HBM, Memory::SMEM, Memory::TMEM],
  transfer Memory::GPU_HBM -> Memory::SMEM,
  transfer Memory::GPU_HBM -> Memory::TMEM
}
fn main() -> i32 { return 0; }
";

#[test]
fn fleet_sibling_moves_meet_at_the_nearest_common_ancestor() {
    const BYTES: u64 = 16 * 1024;
    let mut sibling_pairs = 0usize;

    let mut cases: Vec<(&str, vxc::syntax::VxModule)> = vec![(
        "<siblings>",
        vxc::parse_module(SIBLING_MACHINE).expect("synthetic sibling machine parses"),
    )];
    cases.extend(FLEET.iter().map(|p| (*p, load(p).module)));

    for (path, module) in &cases {
        let h = MemoryHierarchy::build(module.memories.iter());
        let spaces: Vec<MemorySpace> = module
            .memories
            .iter()
            .map(|d| MemorySpace::from_name(d.name.as_ref()))
            .collect();

        for a in &spaces {
            for b in &spaces {
                if a == b {
                    continue;
                }
                let Some(nca) = h.nearest_common_ancestor(a, b) else {
                    continue;
                };
                // Genuine siblings only: neither is the other's ancestor, so the NCA is a third
                // space and the move really does go up and back down.
                if nca == *a || nca == *b {
                    continue;
                }
                sibling_pairs += 1;

                let Some(cost) = h.derived_transfer_cost(a, b, BYTES) else {
                    continue;
                };
                // Legs, computed here from the declarations rather than by re-calling the function
                // under test.
                let leg = |from: &MemorySpace| -> Option<u64> {
                    let mut total = 0u64;
                    let mut cur = from.clone();
                    loop {
                        if cur == nca {
                            return Some(total);
                        }
                        let bw = h.descriptor(&cur)?.bandwidth?;
                        total += vxc::hir::memory::hop_cost(BYTES, bw)?;
                        cur = h.parent(&cur)?;
                    }
                };
                if let (Some(x), Some(y)) = (leg(a), leg(b)) {
                    assert_eq!(
                        cost.value,
                        x + y,
                        "{path}: {:?} -> {:?} via NCA {:?}: cost {} is not the sum of its legs \
                         ({x} + {y}) -- the NCA must contribute no bandwidth term",
                        a,
                        b,
                        nca,
                        cost.value
                    );
                }

                // Symmetry: the model has no directional term, so asserting it here pins that no
                // future asymmetry creeps in unannounced.
                let back = h.derived_transfer_cost(b, a, BYTES);
                assert_eq!(
                    Some(cost),
                    back,
                    "{path}: {:?} -> {:?} and its reverse disagree",
                    a,
                    b
                );
            }
        }
    }

    assert!(
        sibling_pairs > 0,
        "no sibling pairs found anywhere in the fleet -- the NCA rule was never exercised"
    );
}

/// Every fleet file must be *usable*: parse, declare a topology, and leave no space stranded.
///
/// A machine file that declares a memory nothing can reach is a modelling error that no program
/// would surface until it tried to place a tile there and got "no hardware path exists".
#[test]
fn fleet_files_declare_reachable_hierarchies() {
    for path in FLEET {
        let m = load(path);
        assert!(
            !m.module.topologies.is_empty(),
            "{path}: declares no topology, so nothing can be staged into it"
        );
        assert!(
            !m.module.memories.is_empty(),
            "{path}: declares no memory spaces"
        );

        let graph = m.graph();
        let h = m.hierarchy();
        // Every declared space must be reachable from the host, directly or by climbing to an
        // ancestor that is -- which is exactly how the pipeline resolves a sub-space placement.
        for d in &m.module.memories {
            let space = MemorySpace::from_name(d.name.as_ref());
            let reachable = std::iter::once(space.clone())
                .chain(h.ancestors(&space))
                .any(|s| {
                    graph.can_transfer(&MemorySpace::CPUDRAM, &s) || s == MemorySpace::CPUDRAM
                });
            assert!(
                reachable,
                "{path}: memory space {:?} is unreachable from the host, so no program can place \
                 anything in it",
                space
            );
        }

        // Names are the fleet's shared vocabulary (fleet/README.md): one program text is admitted
        // against every SKU by swapping the flag, which only works if the spaces are named alike.
        let names: HashSet<String> = m
            .module
            .memories
            .iter()
            .map(|d| d.name.as_ref().to_string())
            .collect();
        for required in ["HBM", "L2", "SMEM"] {
            assert!(
                names.contains(required),
                "{path}: missing shared-vocabulary space '{required}' (declares {names:?})"
            );
        }
    }
}
