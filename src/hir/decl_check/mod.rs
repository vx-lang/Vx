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

mod assoc_types;
mod bounds;
mod conflicts;
mod copy_impls;
mod memory;
mod placement;
mod thread_safety_impls;
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
        // `impl Copy for X` where one of X's fields moves rather than copies.
        self.check_copy_impls();
        // `impl Send for X` with no `unsafe`, and `unsafe impl` of another trait.
        self.check_thread_safety_impls();
        // An impl binds every associated type its trait declares. Nothing downstream can catch
        // this: the binding is substituted into the signatures before name resolution runs.
        self.check_associated_type_bindings();
        // An impl writes every method its trait gives no default body.
        self.check_impls_write_required_methods();
        self.check_no_projections_in_fields();
        // `T : NoSuchTrait` on any declaration, called or not.
        self.check_bounds_name_a_trait();
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
        // A `const` whose value does not fit its type, or has no value while compiling, is
        // reported once here. Uses only substitute the value.
        self.check_const_declarations();
    }

    /// Report E3044 once per scalar `const` that has no usable value, at the declaration.
    ///
    /// The check used to run when a use asked for the value, so a `const` used twice got two
    /// errors and an unused overflowing `const` got none. Tables stay checked at the use that
    /// names them, where the diagnostic says how to read them.
    fn check_const_declarations(&mut self) {
        let mut consts: Vec<&crate::syntax::ConstDecl> =
            self.env.consts.values().copied().collect();
        // Source order, so FileCheck on multi-const fixtures stays stable.
        consts.sort_by(|a, b| {
            (a.span.line, a.span.column, a.name.as_ref()).cmp(&(
                b.span.line,
                b.span.column,
                b.name.as_ref(),
            ))
        });
        for decl in consts {
            match crate::registry::ConstTable::from_decl(decl) {
                Ok(None) => {
                    let _ = self.const_literal(decl, &decl.span, true);
                }
                Ok(Some(_)) => {}
                // Bad tables are reported when named; an unused bad table is rare and the use
                // site message says how to read a table.
                Err(_) => {}
            }
        }
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
                    Some(span.clone()),
                );
            }
            if let Some(c_type) = c_allocator_mismatch(ext) {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3059,
                    format!(
                        "extern fn `{}` does not have the type of C's `{}`. Declare it as \
                         `{}`. A size can also be `i64` or `u64`, and a pointer can be any raw \
                         pointer to host memory.",
                        ext.name, ext.name, c_type
                    ),
                    Some(span),
                );
            }
        }
    }
}

/// For a declaration of a C allocation function whose type is not C's, the declaration to write.
/// The compiler calls these functions itself, with C's type, and LLVM has one declaration of each.
fn c_allocator_mismatch(ext: &crate::syntax::ExternDecl) -> Option<&'static str> {
    use crate::syntax::{ElementType, Type};
    // Each parameter is a size (`true`) or a pointer (`false`); the bool after is whether the
    // function returns a pointer.
    let (params, returns_pointer, c_type): (&[bool], bool, &'static str) = match ext.name.as_ref() {
        "malloc" => (&[true], true, "fn malloc(size : usize) -> *mut i8"),
        "calloc" => (
            &[true, true],
            true,
            "fn calloc(count : usize, size : usize) -> *mut i8",
        ),
        "realloc" => (
            &[false, true],
            true,
            "fn realloc(ptr : *mut i8, size : usize) -> *mut i8",
        ),
        "aligned_alloc" => (
            &[true, true],
            true,
            "fn aligned_alloc(alignment : usize, size : usize) -> *mut i8",
        ),
        "free" => (&[false], false, "fn free(ptr : *mut i8) -> void"),
        _ => return None,
    };
    let is_size = |ty: &Type| {
        matches!(
            ty,
            Type::Scalar(ElementType::I64 | ElementType::U64 | ElementType::USize)
        )
    };
    let is_host_pointer = |ty: &Type| matches!(ty, Type::Pointer(_, None, _));
    let params_match = ext.params.len() == params.len()
        && ext.params.iter().zip(params).all(|((_, ty), size)| {
            if *size {
                is_size(ty)
            } else {
                is_host_pointer(ty)
            }
        });
    let result_matches = if returns_pointer {
        is_host_pointer(&ext.return_type)
    } else {
        crate::syntax::types::is_void_ty(&ext.return_type)
    };
    (!(params_match && result_matches)).then_some(c_type)
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
