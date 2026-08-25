//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Whether the memory spaces a program declares hold together: `within:` cycles, sub-space
// capacities, and colliding dispatch ids.
//
//===----------------------------------------------------------------------===//

use super::super::*;

impl TypeChecker<'_> {
    /// Check coherence of the program's declared memory spaces (`Memory <Name> { ... }`):
    /// `within:` is acyclic, a sub-space's capacity does not exceed its parent's, and declared
    /// properties are positive. Reads descriptors from the per-compilation env (`self.env`),
    /// not a process-global registry, so it needs no scoping list (unlike topologies).
    pub fn check_memory_coherence(&mut self) {
        use crate::hir::memory::{MemoryCoherenceIssue, MemoryHierarchy};
        // Dispatch-id collision, on the same terms as the topology check above (E6016). Two
        // declared spaces whose names hash to one id are one space to everything downstream: the
        // codegen keeps its descriptors in a map keyed by the id, so a `transfer` into the second
        // space is emitted with the FIRST space's name, capacity, granule and scope, and which one
        // wins is hash-iteration order. Measured on `L2_GPU5`/`GPU6_VMEM` (both 1277): the same
        // source file emitted `space = "L2_GPU5"` on 8 of 20 runs and `space = "GPU6_VMEM"` on the
        // other 12, with a 64 KiB tile placed in a 60 MiB L2 instead of 24 GiB of VRAM and no
        // diagnostic either way.
        {
            let mut by_id: std::collections::HashMap<i32, crate::symbol::Symbol> =
                std::collections::HashMap::new();
            let mut sorted: Vec<&crate::symbol::Symbol> = self.env.memories.keys().collect();
            sorted.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
            for name in sorted {
                let space = crate::syntax::MemorySpace::from_name(name.as_ref());
                let id = crate::arch::memory_space_dispatch_id(&space);
                if let Some(prev) = by_id.insert(id, name.clone()) {
                    if &prev != name {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E6016,
                            format!(
                                "memory spaces '{}' and '{}' collide on dispatch id {} (custom \
                                 ids are derived from the name by hashing); which descriptor a \
                                 `transfer` into either one carries would be hash order -- \
                                 rename one",
                                prev, name, id
                            ),
                            None,
                        );
                    }
                }
            }
        }
        let issues = MemoryHierarchy::build(self.env.memories.values().copied()).coherence_issues();
        for issue in issues {
            match issue {
                MemoryCoherenceIssue::Cycle { space } => {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E6006,
                        format!(
                            "memory space '{}' is in a `within:` cycle (a space cannot contain \
                             itself)",
                            space.name()
                        ),
                        None,
                    );
                }
                MemoryCoherenceIssue::CapacityExceedsParent {
                    child,
                    parent,
                    child_bytes,
                    parent_bytes,
                } => {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E6007,
                        format!(
                            "memory space '{}' ({} bytes) is larger than its parent '{}' ({} \
                             bytes); a sub-space cannot exceed what contains it",
                            child.name(),
                            child_bytes,
                            parent.name(),
                            parent_bytes
                        ),
                        None,
                    );
                }
                MemoryCoherenceIssue::NonPositiveProperty { space, property } => {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E6008,
                        format!(
                            "memory space '{}' has a non-positive `{}`",
                            space.name(),
                            property
                        ),
                        None,
                    );
                }
                MemoryCoherenceIssue::ScopeWidensInChild {
                    child,
                    parent,
                    child_scope,
                    parent_scope,
                } => {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E6011,
                        format!(
                            "memory space '{}' has scope {:?}, broader than its parent '{}' \
                             ({:?}); locality must narrow down the hierarchy, not widen",
                            child.name(),
                            child_scope,
                            parent.name(),
                            parent_scope
                        ),
                        None,
                    );
                }
            }
        }
    }
}
