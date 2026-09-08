//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
mod placement;
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
        // Placements written in signatures and struct fields name a space some topology holds.
        // After the two coherence checks, so a program whose declarations are wrong is told that
        // first rather than told its placements are unheld as a consequence.
        self.check_placements_name_a_place();
        // An `extern` signature has to be spellable in C, and a tensor is not.
        self.check_extern_signatures();
    }

    /// Refuse a tensor anywhere in an `extern` signature.
    ///
    /// Lowering expands a memref into its seven descriptor scalars, so such a declaration names a
    /// C symbol with a signature no C source could have written, and the mismatch shows up as
    /// corrupt arguments at run time rather than as a link error.
    fn check_extern_signatures(&mut self) {
        for ext in &self.env.externs {
            let span = crate::diagnostic::SourceSpan::from_ast_span(&ext.span);
            for (name, ty) in &ext.params {
                if let Some(found) = tensor_within(ty) {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3022,
                        format!(
                            "parameter `{}` of extern fn `{}` is {}; an extern signature cannot \
                             mention a tensor -- take a raw pointer and build the tensor in Vx",
                            name,
                            ext.name,
                            Self::short_type_name(found)
                        ),
                        Some(span.clone()),
                    );
                }
            }
            if let Some(found) = tensor_within(&ext.return_type) {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3022,
                    format!(
                        "extern fn `{}` returns {}; an extern signature cannot mention a tensor \
                         -- return a raw pointer and build the tensor in Vx",
                        ext.name,
                        Self::short_type_name(found)
                    ),
                    Some(span),
                );
            }
        }
    }
}

/// The tensor inside a type, through the wrappers that restate where a value lives, and through
/// one level of pointer or borrow -- `*mut Tensor<f32, [?, ?]>` is a descriptor pointer, which is no
/// more spellable in C than the descriptor itself.
fn tensor_within(ty: &crate::syntax::Type) -> Option<&crate::syntax::Type> {
    use crate::syntax::Type;
    match ty {
        Type::Tensor(..) => Some(ty),
        Type::Verified(inner) | Type::Pinned(inner, _) | Type::Ref(inner, _) => {
            tensor_within(inner)
        }
        Type::Borrow { inner, .. } => tensor_within(inner),
        Type::Pointer(inner, _, _) => tensor_within(inner),
        _ => None,
    }
}
