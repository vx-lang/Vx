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
    /// Only a type that mentions a tensor or a placed pointer can state a location, so that is
    /// asked first, before a clone, a string or a sort. This runs on the serial spine once per
    /// compile, and on a 16,000-function program with no placements at all it was 90 ms of it:
    /// building and sorting 27,000 sites, then walking 90,000 types and finding nothing in any
    /// of them.
    pub fn check_placements_name_a_place(&mut self) {
        // Sorted, because both tables are HashMaps and this reports per declaration: an unsorted
        // walk would emit the same diagnostics in a different order on every run. Each site
        // keeps the file and the place that its diagnostics name.
        struct Site {
            order: (String, usize),
            context: String,
            ty: crate::syntax::Type,
            place: crate::syntax::DeclPlace,
        }
        let mut sites: Vec<Site> = Vec::new();
        for (name, f) in &self.env.syntax_functions {
            // One site for each parameter and one for the return type, so that each bad
            // placement has a message of its own.
            let params = f.params.iter().map(|(p, t)| (Some(p), t));
            let ret = std::iter::once((None, &f.return_type));
            for (i, (param, ty)) in params.chain(ret).enumerate() {
                if ty.states_a_location() {
                    let context = match param {
                        Some(p) => format!("parameter '{p}' of function '{name}'"),
                        None => format!("the return type of function '{name}'"),
                    };
                    sites.push(Site {
                        order: (format!("function '{name}'"), i),
                        context,
                        ty: ty.clone(),
                        place: crate::syntax::DeclPlace {
                            file: f.file.clone(),
                            span: f.span,
                        },
                    });
                }
            }
        }
        for (name, s) in &self.env.structs {
            for (field, ty) in &s.fields {
                if ty.states_a_location() {
                    let context = format!("field '{field}' of struct '{name}'");
                    sites.push(Site {
                        order: (context.clone(), 0),
                        context,
                        ty: ty.clone(),
                        place: s.place.clone(),
                    });
                }
            }
        }
        sites.sort_by(|a, b| a.order.cmp(&b.order));

        let outside = self.errors.here();
        for site in sites {
            self.errors.point_at(&site.place, &outside);
            // A declared position states where a tensor lives without allocating one here,
            // so it takes the per-tile verdict and stays out of any function's working set.
            self.check_declared_type_placement(&site.ty, &site.context, &site.place.span);
        }
        // The device in `on Topology::X[i]` names one device of a declared topology too.
        let mut on_clauses: Vec<_> = self
            .env
            .syntax_functions
            .iter()
            .filter_map(|(name, f)| {
                let why = self.declared_device_problem(&f.topology)?;
                let place = crate::syntax::DeclPlace {
                    file: f.file.clone(),
                    span: f.span,
                };
                Some((name.to_string(), f.topology.to_string(), why, place))
            })
            .collect();
        on_clauses.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, written, why, place) in on_clauses {
            self.errors.point_at(&place, &outside);
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E6025,
                format!("`on {written}` of function '{name}' names no device: {why}"),
                None,
            );
        }
        self.errors.point_back(outside);
    }

    /// Why `top` names no device of a declared topology, or `None` when it names one. A topology
    /// with `count: N` has the devices 0 to N - 1, and a use must pick one when N is more than 1.
    /// The built-in topologies have their own index rules, so this says nothing about them.
    pub(crate) fn declared_device_problem(&self, top: &crate::syntax::Topology) -> Option<String> {
        use crate::syntax::Topology;
        let (name, first, end) = match top {
            Topology::Custom(name, index) => (name, index.as_deref(), None),
            Topology::Slice(base, start, end) => match &**base {
                Topology::Custom(name, _) => (name, Some(&**start), Some(&**end)),
                _ => return None,
            },
            _ => return None,
        };
        let Some(decl) = self.env.topologies.get(name) else {
            return first.map(|_| {
                format!("no topology named '{name}' is declared, so it has no devices to index")
            });
        };
        let count = decl.descriptor.count.unwrap_or(1);
        let devices = if count == 1 {
            "1 device".to_string()
        } else {
            format!("{count} devices")
        };
        let constant = |e: &crate::syntax::Expr| crate::arch::const_topology_index(e);
        let Some(first) = first else {
            return (count > 1).then(|| {
                format!(
                    "'{name}' has {devices}, so name one of them: `Topology::{name}[0]` to \
                     `Topology::{name}[{}]`",
                    count - 1
                )
            });
        };
        if let Some(i) = constant(first) {
            if i < 0 || i as u64 >= count {
                return Some(format!(
                    "'{name}' has {devices}, so device {i} does not exist"
                ));
            }
        }
        let end = end.and_then(constant)?;
        if end < 0 || end as u64 > count {
            return Some(format!(
                "'{name}' has {devices}, so a slice of it ends at {count} or before, not at {end}"
            ));
        }
        match constant(first) {
            Some(start) if start >= end => Some(format!("the slice {start}..{end} has no devices")),
            _ => None,
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
                } else if let Some(why) = self.declared_device_problem(&p.topology) {
                    bad.push((p.as_written(), why));
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
        let named_no_place = !bad.is_empty();
        for (written, why) in bad {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E6025,
                format!("`{written}` in {context} names no location: {why}"),
                None,
            );
        }
        if !named_no_place {
            self.report_placements_with_no_address_space(ty, context);
        }
    }

    /// Report every placement inside `ty` whose memory has no `scope:` and is not built in.
    /// Such a memory has no address space on the target, and a default could put the data
    /// somewhere the program did not ask for.
    fn report_placements_with_no_address_space(&mut self, ty: &crate::syntax::Type, context: &str) {
        use crate::syntax::Written;
        let mut bad = Vec::new();
        ty.for_each_placement(&mut |p| {
            let space = match p.written() {
                Written::Device if matches!(p.topology, crate::syntax::Topology::Current) => return,
                Written::Device => match self.transfer_cost_graph.descriptor(&p.topology.kind()) {
                    Some(d) => d.default_space.clone(),
                    None => return,
                },
                Written::Space => p.space.clone(),
            };
            let decl = self
                .env
                .memories
                .values()
                .find(|d| crate::syntax::MemorySpace::from_name(d.name.as_ref()) == space)
                .copied();
            if crate::arch::declared_address_space(&space, decl).is_none() {
                bad.push((p.as_written(), space.name()));
            }
        });
        for (written, space) in bad {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E6037,
                format!(
                    "`{written}` in {context} has no address space on the target: \
                     `Memory::{space}` has no `scope:`. Declare its memory with a `scope:` \
                     (device, sm, cta or thread)"
                ),
                None,
            );
        }
    }
}
