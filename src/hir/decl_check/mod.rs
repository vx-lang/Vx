//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The checks that are about the whole program rather than about one function. They ran
// inside `check/transfer.rs`, which is where the *expression* checks live; these never were.
//
//===----------------------------------------------------------------------===//

use super::*;

mod conflicts;
mod memory;
mod topology;

impl TypeChecker<'_> {
    /// Every check that is about the program as a whole rather than about one function: names
    /// declared twice, transfer lowerings, topology coherence, memory coherence.
    ///
    /// One method because there are two frontends. The driver ran these four and the parallel
    /// pipeline ran none, so a machine model the driver refused compiled without a diagnostic on
    /// the other path -- a 250-space corpus whose spaces collided onto shared dispatch ids went
    /// through clean. Both callers now run the same list in the same order, and adding a fifth
    /// check reaches both.
    ///
    /// Runs before any body is checked: these read the collapsed declaration tables, and a body
    /// checked against an ambiguous table has been checked against a coin flip.
    pub fn check_whole_program_declarations(&mut self) {
        // A name declared by two inputs (e.g. a `--machine` file and the program) is ambiguous.
        self.check_declaration_conflicts();
        // Structural validity of transfer lowerings: duplicate edge, empty body.
        self.check_transfer_impls();
        // Declared topologies, read from the env rather than one program: a topology arriving via
        // `--machine` is the primary case and is not in any single module's list.
        //
        // Sorted, because `env.topologies` is a HashMap: the coherence loop reports per
        // declaration, so an unsorted walk would emit the same diagnostics in a different order
        // on every run.
        let mut declared: Vec<crate::arch::TopologyDecl> =
            self.env.topologies.values().map(|t| (*t).clone()).collect();
        declared.sort_by(|a, b| a.name.as_ref().cmp(b.name.as_ref()));
        self.check_topology_coherence(&declared);
        // Declared memory spaces: `within:` cycles, oversized sub-spaces, colliding ids.
        self.check_memory_coherence();
    }
}
