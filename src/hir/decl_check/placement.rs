//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Whether the placements a program writes name places the machine has.
//
//===----------------------------------------------------------------------===//

use super::super::*;

impl TypeChecker<'_> {
    /// Every placement written in a type names a place the machine has.
    ///
    /// A placement carries a device and a space together and derives whichever the source did not
    /// write. Both derivations fall back to the like-named other half when the declarations do not
    /// answer, so an undeclared name resolves rather than failing: `Memory::Nowhere` becomes
    /// `Topology::Nowhere` and the reverse. The pair is then internally consistent and describes a
    /// machine that does not exist.
    ///
    /// Each spelling is checked on the half the source wrote: a device names a place when the
    /// compilation declares it, a space when some declared topology holds it. The two are refused
    /// alike, so which spelling a program used cannot decide whether it compiles.
    ///
    /// Name resolution is where the derivation happens and would be the earliest place to catch
    /// this, but it has no diagnostic channel -- an unresolved struct name is carried forward
    /// unreported there too, for the checker to find.
    ///
    /// Only a type that mentions a tensor can state a location, so that is asked first, before
    /// a clone, a string or a sort. This runs on the serial spine once per compile, and on a
    /// 16,000-function program with no placements at all it was 90 ms of it: building and
    /// sorting 27,000 sites, then walking 90,000 types and finding nothing in any of them.
    pub fn check_placements_name_a_place(&mut self) {
        // Sorted, because both tables are HashMaps and this reports per declaration: an unsorted
        // walk would emit the same diagnostics in a different order on every run.
        let mut sites: Vec<(String, Vec<crate::syntax::Type>)> = Vec::new();
        for (name, f) in &self.env.syntax_functions {
            let tys: Vec<crate::syntax::Type> = f
                .params
                .iter()
                .map(|(_, t)| t)
                .chain(std::iter::once(&f.return_type))
                .filter(|t| t.mentions_tensor())
                .cloned()
                .collect();
            if !tys.is_empty() {
                sites.push((format!("function '{name}'"), tys));
            }
        }
        for (name, s) in &self.env.structs {
            for (field, ty) in &s.fields {
                if ty.mentions_tensor() {
                    sites.push((
                        format!("field '{field}' of struct '{name}'"),
                        vec![ty.clone()],
                    ));
                }
            }
        }
        sites.sort_by(|a, b| a.0.cmp(&b.0));

        for (context, tys) in sites {
            for ty in &tys {
                // A declared position states where a tensor lives without allocating one here,
                // so it takes the per-tile verdict and stays out of any function's working set.
                self.check_declared_type_placement(ty, &context, &crate::syntax::Span::default());
            }
        }
    }

    /// Report every placement inside `ty` that names no place.
    ///
    /// Reads only the half the source wrote and derives the other from the cost graph's
    /// descriptors. A placement's derived half is filled in by name resolution, which a program
    /// that fails the type checker never reaches, so by the time this runs the derived half is
    /// still whatever the parser guessed -- for `Topology::Dev` that is the like-named
    /// `Memory::Dev`, not the `memory:` the declaration gives.
    pub(crate) fn report_unheld_placements(&mut self, ty: &crate::syntax::Type, context: &str) {
        use crate::syntax::Written;
        let mut bad = Vec::new();
        ty.for_each_placement(&mut |p| match p.written() {
            // A device names a place when the compilation declares it. Its space follows from the
            // declaration, so there is nothing further to check.
            Written::Device => {
                if self
                    .transfer_cost_graph
                    .descriptor(&p.topology.kind())
                    .is_none()
                {
                    bad.push((
                        p.as_written(),
                        "no topology of that name is declared".to_string(),
                    ));
                }
            }
            Written::Space => {
                if !self.transfer_cost_graph.is_space_held(&p.space) {
                    bad.push((
                        p.as_written(),
                        format!(
                            "no declared topology holds `Memory::{}`, as its own memory or in \
                             its `visible:` list",
                            p.space.name()
                        ),
                    ));
                }
            }
        });
        for (written, why) in bad {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E6025,
                format!("`{written}` in {context} names no location: {why}"),
                None,
            );
        }
    }
}
