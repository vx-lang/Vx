//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An impl of `Send` or `Sync` is `unsafe`, and only those two traits take `unsafe impl`.
//
//===----------------------------------------------------------------------===//

use super::super::*;

impl TypeChecker<'_> {
    /// Report `impl Send for X` with no `unsafe` (E3053), and `unsafe impl` of any other trait
    /// (E3054).
    ///
    /// The compiler derives `Send` and `Sync` from the fields. An impl of one adds a claim that
    /// the compiler cannot check, for a type that holds a raw pointer, so it must say `unsafe`.
    pub fn check_thread_safety_impls(&mut self) {
        let mut found: Vec<(crate::symbol::Symbol, String, bool)> = Vec::new();
        for (trait_name, blocks) in &self.env.impls {
            for ib in blocks {
                found.push((trait_name.clone(), ib.target_type.to_string(), ib.is_unsafe));
            }
        }
        // The impl table is a hash map. Sorted, the errors come in the same order each time.
        found.sort();
        for (trait_name, target, is_unsafe) in found {
            let is_thread_trait =
                crate::hir::check::thread_safety::ThreadTrait::from_name(&trait_name).is_some();
            if is_thread_trait && !is_unsafe {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3053,
                    format!(
                        "`impl {trait_name} for {target}` must be `unsafe impl`: the compiler \
                         derives `{trait_name}` from the fields, and cannot check this claim"
                    ),
                    None,
                );
            } else if is_unsafe && trait_name.as_ref() == "_inherent" {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3054,
                    format!(
                        "`unsafe impl {target}`: an impl with no trait cannot be `unsafe`. Remove \
                         `unsafe`, and mark each method that needs it `unsafe fn`"
                    ),
                    None,
                );
            } else if !is_thread_trait && is_unsafe {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3054,
                    format!(
                        "`unsafe impl {trait_name} for {target}`: only `Send`, `Sync` and `Owned` take \
                         `unsafe impl`. Remove `unsafe`"
                    ),
                    None,
                );
            }
        }
    }
}
