//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Whether the topologies a program declares, and the transfer lowerings written against
// them, hold together: identity, coherence obligations, and structurally malformed lowerings.
//
//===----------------------------------------------------------------------===//

use super::super::*;

/// How much standing the topology a transfer lowering names has in this compilation. The three
/// cases need different diagnostics, because only one of them is fixable by editing an edge
/// list: a built-in topology has no edge list to edit (`builtin_descriptors` gives every
/// built-in `transfers: Vec::new()`, because their edges live in the cost graph instead).
#[derive(Clone, Copy, PartialEq, Eq)]
enum TopologyStanding {
    /// Declared in source by a `Topology <Name> { ... }` block, so it has its own edge list.
    Declared,
    /// A built-in topology. It exists, but declares no edges of its own.
    BuiltIn,
    /// This compilation has never heard of it.
    Unknown,
}

impl TypeChecker<'_> {
    /// Check the coherence obligations of every user-defined topology registered for this
    /// compilation, emitting diagnostics for incoherent declarations.
    ///
    /// Graph-decidable obligations (`default_space` visible, memory reachable from the host)
    /// come from `arch::descriptor_coherence`. The consistency obligation is discharged
    /// through the seam engine: a declared `relaxed` edge is modeled as a relaxed transfer of
    /// a published payload and handed to `seam::check_seam_buffers`; a `Reject` (the buffer
    /// can be read stale) means the edge does not preserve visibility.
    pub fn check_topology_coherence(&mut self, declared: &[crate::arch::TopologyDecl]) {
        // Identity checks first (E6016): a topology whose declaration cannot take effect fails
        // here, before any coherence rule reads the declaration as though it were in force.
        //
        // Shadowing: the parser resolves built-in names unconditionally (`Topology::GPU` is the
        // built-in GPU whatever the program declares), so a declaration under such a name
        // registers as Custom("GPU") and is unreachable from every use site. Found by review on
        // Vx#352: `Topology GPU { arch: applegpu }` still produced an NVPTX image via the band,
        // the machine file's word silently discarded.
        const BUILTIN_TOPOLOGY_NAMES: &[&str] = &[
            "CPU",
            "Current",
            "NPU",
            "AccCore",
            "AMX",
            "ANE",
            "GPU",
            "CpuAvx512",
            "CPU_AVX512",
            "CpuNeon",
            "CPU_Neon",
        ];
        for decl in declared {
            if BUILTIN_TOPOLOGY_NAMES.contains(&decl.name.as_ref()) {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E6016,
                    format!(
                        "topology '{}' shadows the built-in topology of the same name; every \
                         `Topology::{}` resolves to the built-in, so this declaration (including \
                         its `arch:`) would be silently ignored -- rename it",
                        decl.name, decl.name
                    ),
                    None,
                );
            }
        }
        // Dispatch-id collision: the id is a hash of the name, so two names can share one id.
        // Everything keyed by the id -- spawn dispatch,
        // seam identity, and now the declared-arch table -- then depends on hash-iteration order
        // for which declaration wins; measured as the same program getting a device image on some
        // runs and not others. Refused, because a coin-flip identity is not an identity.
        {
            let mut by_id: std::collections::HashMap<i32, &crate::symbol::Symbol> =
                std::collections::HashMap::new();
            let mut sorted: Vec<&crate::arch::TopologyDecl> = declared.iter().collect();
            sorted.sort_by(|a, b| a.name.as_ref().cmp(b.name.as_ref()));
            for decl in sorted {
                let id = crate::arch::topology_dispatch_id(&crate::syntax::Topology::Custom(
                    decl.name.clone(),
                ));
                if let Some(prev) = by_id.insert(id, &decl.name) {
                    if prev != &decl.name {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E6016,
                            format!(
                                "topologies '{}' and '{}' collide on dispatch id {} (custom ids \
                                 are derived from the name by hashing); which declaration is in \
                                 force would be hash order -- rename one",
                                prev, decl.name, id
                            ),
                            None,
                        );
                    }
                }
            }
        }
        for decl in declared {
            let name = &decl.name;
            for issue in
                crate::arch::descriptor_coherence(&decl.descriptor, self.transfer_cost_graph)
            {
                match issue {
                    crate::arch::CoherenceIssue::DefaultNotVisible => {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E6005,
                            format!(
                                "topology '{name}' is incoherent: it cannot see its own default \
                                 memory space (not in its visibility set)"
                            ),
                            None,
                        );
                    }
                    crate::arch::CoherenceIssue::MemoryUnreachableFromHost => {
                        self.errors.warn(
                            crate::diagnostic::DiagnosticCode::W1026,
                            format!(
                                "topology '{name}': its memory is unreachable from the host; \
                                 declare a `transfer` edge so data can reach it"
                            ),
                            None,
                        );
                    }
                }
            }

            // S2 (vx-review#10): exactly one cost source per edge. An edge that declares a cost
            // while its endpoints already supply a derivable one has two, and the compiler used
            // both -- routing by the declared number and reporting the derived one. Rejected at
            // declaration time rather than resolved by a precedence rule, because a precedence
            // rule silently discards a figure someone wrote down on purpose.
            //
            // Probed with one byte: derivability is a property of the containment tree and the
            // presence of `bandwidth:`, not of the transfer size.
            {
                let hierarchy =
                    crate::hir::memory::MemoryHierarchy::build(self.env.memories.values().copied());
                for edge in &decl.descriptor.transfers {
                    if edge.cost == crate::arch::EdgeCost::Derived {
                        continue;
                    }
                    if hierarchy
                        .derived_transfer_cost(&edge.from, &edge.to, 1)
                        .is_some()
                    {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E6013,
                            format!(
                                "topology '{name}': transfer Memory::{} -> Memory::{} declares a \
                                 cost, but one is already derivable from the `bandwidth:` of the \
                                 spaces it connects -- an edge gets exactly one cost source. Drop \
                                 the `: …` to use the bandwidth-derived cost, or remove the \
                                 `bandwidth:` from a space if the declared figure is the one you \
                                 mean.",
                                edge.from.name(),
                                edge.to.name()
                            ),
                            None,
                        );
                    }
                }
            }

            // Consistency obligation, discharged via the seam engine: a relaxed edge that
            // carries a payload does not preserve the buffer's visibility.
            for edge in &decl.descriptor.transfers {
                if edge.sync {
                    continue;
                }
                use crate::hir::seam::{AbsState, Cell, Transfer, Verdict};
                let reached = AbsState {
                    cells: vec![("payload".into(), Cell::constant(1))],
                };
                let transfer = Transfer::Relaxed {
                    published: vec!["payload".into()],
                };
                if self.seam.solver.is_none() {
                    self.seam.solver = Some(crate::hir::seam::Solver::new());
                }
                let solver = self.seam.solver.as_mut().unwrap();
                // `if let Ok(Reject)` used to be the whole of this: an Err -- including "no
                // solver available" -- fell through the pattern and the edge was silently
                // treated as if it had been proved coherent (Vx#374). An obligation that could
                // not be discharged is reported as such.
                match solver.check_seam_buffers(&reached, &transfer, &["payload".to_string()]) {
                    Ok(Verdict::Reject { .. }) => {
                        self.errors.warn(
                            crate::diagnostic::DiagnosticCode::W1027,
                            format!(
                                "topology '{name}': declared relaxed transfer {} -> {} does \
                                 not preserve visibility; a consumer may read stale data",
                                edge.from.name(),
                                edge.to.name()
                            ),
                            None,
                        );
                    }
                    Ok(Verdict::Accept) => {}
                    Err(e) => {
                        // An obligation that could not be discharged fails the build, unless the
                        // user has explicitly accepted unverified compilation -- in which case it
                        // is still said out loud, every time. The two must not look alike.
                        // Spaces by name: `{:?}` renders a declared one as
                        // `Custom("SMEM")`, which names the compiler's representation
                        // rather than anything the program wrote.
                        let msg = format!(
                            "topology '{name}': the visibility of relaxed transfer {} -> {} \
                             was NOT verified: {e}",
                            edge.from.name(),
                            edge.to.name()
                        );
                        if crate::hir::solver::unverified_allowed() {
                            self.errors
                                .warn(crate::diagnostic::DiagnosticCode::W1031, msg, None);
                        } else {
                            self.errors.error_with_code(
                                crate::diagnostic::DiagnosticCode::E6024,
                                msg,
                                None,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Structural checks on transfer lowerings
    /// (`impl Transfer<Memory::A, Memory::B> for Topology::X { ... }`), E6015. Ways a lowering
    /// can be malformed before its body is even looked at:
    ///
    ///   * the same edge implemented twice FOR THE SAME TOPOLOGY (which one is in force would be
    ///     module load order -- the same ambiguity E6012 exists to refuse). Two topologies
    ///     implementing the same edge is now the normal case, not an error;
    ///   * an empty lowering (nothing to emit, so the `impl` would be an inert annotation);
    ///   * a `for Topology::X` naming a topology this compilation has never heard of;
    ///   * a topology that exists but does not declare the edge being implemented.
    ///
    /// The last two are new with the topology key. Before it, a lowering named no machine, so
    /// there was nothing to check it against -- `TransferImplDecl` carried a comment saying the
    /// edge should be matched against a topology's declared edges "(not yet wired)", and it
    /// could not be wired, because the declaration did not say which topology to look at.
    pub fn check_transfer_impls(&mut self) {
        // Read the topology facts first. Emitting a diagnostic needs `&mut self`, and the
        // descriptor lookup borrows `self`, so the two cannot be interleaved.
        let topo_facts: Vec<(TopologyStanding, bool)> = self
            .env
            .transfer_impls
            .iter()
            .map(|t| {
                // Source-declared topologies only. A BUILT-IN topology carries
                // `transfers: Vec::new()` -- its edges live in `TransferCostGraph::default()`
                // rather than on its descriptor -- so there is nothing on it to implement, and
                // no way for a program to add one. Consulting the cost graph instead is not the
                // fix: the graph is flat and merged across every topology, so any machine could
                // then claim any edge another machine declared, which is exactly the conflation
                // the topology key removes.
                let declared = self
                    .env
                    .topologies
                    .values()
                    .find(|d| d.name.as_ref() == t.topology.display_name());
                match declared {
                    Some(d) => (
                        TopologyStanding::Declared,
                        d.descriptor
                            .transfers
                            .iter()
                            .any(|e| e.from == t.from && e.to == t.to),
                    ),
                    None if self
                        .transfer_cost_graph
                        .descriptor(&t.topology.kind())
                        .is_some() =>
                    {
                        (TopologyStanding::BuiltIn, false)
                    }
                    None => (TopologyStanding::Unknown, false),
                }
            })
            .collect();
        for (i, t) in self.env.transfer_impls.clone().iter().enumerate() {
            let (standing, edge_declared) = topo_facts[i];
            let header = format!(
                "`impl Transfer<Memory::{}, Memory::{}> for Topology::{}`",
                t.from.name(),
                t.to.name(),
                t.topology.display_name()
            );
            let message = match standing {
                TopologyStanding::Unknown => Some(format!(
                    "{header}: no topology named '{}' is declared in this compilation, so this \
                     lowering is code for a machine that does not exist here",
                    t.topology.display_name()
                )),
                TopologyStanding::BuiltIn => Some(format!(
                    "{header}: '{}' is a built-in topology, and built-in topologies declare no \
                     edges of their own -- there is nothing here for a lowering to implement. \
                     Write the machine as a `Topology {} {{ ... transfer {} -> {} }}` \
                     declaration and implement that",
                    t.topology.display_name(),
                    t.topology.display_name(),
                    t.from.name(),
                    t.to.name()
                )),
                TopologyStanding::Declared if !edge_declared => Some(format!(
                    "{header}: topology '{}' does not declare the edge {} -> {}, so there is no \
                     movement for this lowering to implement -- declare the edge in the \
                     `Topology` block or implement an edge it has",
                    t.topology.display_name(),
                    t.from.name(),
                    t.to.name()
                )),
                TopologyStanding::Declared => None,
            };
            if let Some(message) = message {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E6015,
                    message,
                    None,
                );
            }
        }
        let mut seen: std::collections::HashMap<
            (
                crate::syntax::MemorySpace,
                crate::syntax::MemorySpace,
                String,
            ),
            usize,
        > = std::collections::HashMap::new();
        for t in &self.env.transfer_impls {
            // A generic fn inside a lowering can never be checked: generic bodies are checked at
            // instantiation, and a lowering fn is not callable, so it is never instantiated -- its
            // body would escape the checker forever. That silently reopens, for generics only, the
            // exact parses-clean-while-broken gap #353 A1 closes, so it is refused outright: a
            // lowering is instantiated per EDGE, not per type, and a type parameter has no meaning
            // there.
            for f in &t.methods {
                if !f.generics.is_empty() {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E6015,
                        format!(
                            "`impl Transfer<Memory::{}, Memory::{}> for Topology::{}`: fn '{}' is \
                             generic; a lowering is instantiated per edge, not per type, so this \
                             body could never be instantiated -- and an uninstantiated body is \
                             never type-checked",
                            t.from.name(),
                            t.to.name(),
                            t.topology.display_name(),
                            f.name
                        ),
                        None,
                    );
                }
            }
            if t.methods.is_empty() {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E6015,
                    format!(
                        "`impl Transfer<Memory::{}, Memory::{}> for Topology::{}` declares no \
                         functions; an empty lowering cannot move anything -- give it a body or \
                         remove it",
                        t.from.name(),
                        t.to.name(),
                        t.topology.display_name()
                    ),
                    None,
                );
            }
            *seen
                .entry((
                    t.from.clone(),
                    t.to.clone(),
                    t.topology.display_name().to_string(),
                ))
                .or_insert(0) += 1;
        }
        for ((from, to, topo), n) in seen {
            if n > 1 {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E6015,
                    format!(
                        "Topology::{} has {} lowerings for the edge {} -> {}; which one is in \
                         force would be load order, so exactly one per machine is allowed. Two \
                         DIFFERENT topologies implementing this edge is fine -- that is what the \
                         `for` clause is for",
                        topo,
                        n,
                        from.name(),
                        to.name()
                    ),
                    None,
                );
            }
        }
    }
}
