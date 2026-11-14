//===- check submodule - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A channel end or a lock stays on the host (M6 in
// docs/discussions/waiting_tasks_channels_and_task_errors.md). A GPU does not promise that a
// block of threads makes progress, so a kernel that waits for the host while the host waits for
// the kernel never ends. A `spawn on` body for a device that names one gets E6038.
//
//===----------------------------------------------------------------------===//

use super::super::*;

impl TypeChecker<'_> {
    /// Whether code on `top` runs on a device, not on the host. A GPU that shares the memory of
    /// the host still runs on the device.
    pub(crate) fn runs_on_a_device(&self, top: &Topology) -> bool {
        match top {
            Topology::CPU
            | Topology::CpuAvx512
            | Topology::CpuNeon
            | Topology::AMX
            | Topology::Current => false,
            Topology::GPU(_) | Topology::NPU(_) | Topology::AccCore(_) | Topology::ANE => true,
            Topology::Slice(inner, ..) => self.runs_on_a_device(inner),
            Topology::Custom(..) => {
                self.transfer_cost_graph.default_memory_for(top)
                    != self.transfer_cost_graph.default_memory_for(&Topology::CPU)
            }
        }
    }

    /// E6038 when the device body being checked names `name`, of type `ty`, and `ty` holds a
    /// channel end or a lock. True when it reported the error.
    pub(crate) fn report_host_only_on_a_device(
        &mut self,
        name: &str,
        ty: &Type,
        span: &Span,
    ) -> bool {
        let Some(spawn) = self.device_spawn else {
            return false;
        };
        // A generic function that the body calls is checked with the topology of the host.
        if self.speculating || !self.runs_on_a_device(&self.active_topology) {
            return false;
        }
        let Some(why) = self.host_only_part(ty, &mut Vec::new()) else {
            return false;
        };
        self.unreadable_reported.insert(*span);
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E6038,
            format!(
                "the body of the `spawn on` at line {} runs on {}, and it uses '{name}': {why}. A \
                 channel end and a lock stay on the host, because a wait on the device for the \
                 host can wait forever. Use '{name}' before the `spawn on`, or after it",
                spawn.line,
                self.active_topology.display_name()
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
        true
    }

    /// The chain of fields from `ty` to a channel end or a lock in it, or `None` when it holds
    /// none.
    fn host_only_part(&mut self, ty: &Type, visiting: &mut Vec<Type>) -> Option<String> {
        match ty {
            Type::Borrow { inner, .. }
            | Type::Ref(inner, _)
            | Type::Verified(inner)
            | Type::Pinned(inner, _) => self.host_only_part(inner, visiting),
            Type::Struct(..) | Type::Enum(..) | Type::GenericInstance(..) => {
                if self.implements_host_only(ty) {
                    return Some(format!("'{ty}' can wait for the host"));
                }
                let (name, args) = match ty {
                    Type::GenericInstance(base, args) => match base.as_ref() {
                        Type::Struct(n, _) | Type::Enum(n, _) => (n.clone(), args.clone()),
                        _ => return None,
                    },
                    Type::Struct(n, _) | Type::Enum(n, _) => (n.clone(), Vec::new()),
                    _ => return None,
                };
                if visiting.contains(ty) {
                    return None;
                }
                visiting.push(ty.clone());
                let why = self
                    .held_types(&name, &args)
                    .into_iter()
                    .find_map(|(field, held)| {
                        self.host_only_part(&held, visiting).map(|inner| {
                            format!("{field} of '{ty}' has type '{held}', and {inner}")
                        })
                    });
                visiting.pop();
                why
            }
            _ => None,
        }
    }

    /// Whether the standard library marks `ty` with `impl HostOnly`, as it does for a channel end
    /// and a lock.
    fn implements_host_only(&mut self, ty: &Type) -> bool {
        let blocks = self.env.impls.get("HostOnly").cloned().unwrap_or_default();
        blocks
            .iter()
            .any(|ib| self.unify_types(&ib.target_type, ty, &mut std::collections::HashMap::new()))
    }
}
