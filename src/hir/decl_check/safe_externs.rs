//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A `safe` extern function cannot return or change a value of a type with an invariant, because
// C code does not keep the invariant.
//
//===----------------------------------------------------------------------===//

use super::super::*;

impl TypeChecker<'_> {
    /// E8015 for a `safe` extern function that returns a value that holds a value of a type with
    /// an invariant, or takes a `&mut` or a `*mut` to such a value.
    pub fn check_safe_externs_keep_invariants(&mut self) {
        let outside = self.errors.here();
        for ext in self.env.externs.clone() {
            if !ext.is_safe {
                continue;
            }
            let returned = self.invariant_types_held_by(&ext.return_type);
            let written = ext.params.iter().find_map(|(name, ty)| {
                let inner = match ty {
                    Type::Borrow {
                        inner,
                        is_mut: true,
                        ..
                    }
                    | Type::Pointer(inner, _, true) => inner,
                    _ => return None,
                };
                let decl = self.invariant_types_held_by(inner).into_iter().next()?;
                Some((name, ty, decl))
            });
            let what = if let Some(decl) = returned.first() {
                format!(
                    "returns a value of `{}`, and `{}` has an invariant",
                    ext.return_type, decl.name
                )
            } else if let Some((name, ty, decl)) = written {
                format!(
                    "takes `{name} : {ty}`, so C code can change a value of `{0}`, and `{0}` has \
                     an invariant",
                    decl.name
                )
            } else {
                continue;
            };
            let place = crate::syntax::DeclPlace {
                file: ext.file.clone(),
                span: ext.span,
            };
            self.errors.point_at(&place, &outside);
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E8015,
                format!(
                    "the `safe` extern function `{}` {what}. C code does not keep the invariant. \
                     Remove `safe`, so that each call is in an `unsafe` block",
                    ext.name
                ),
                None,
            );
        }
        self.errors.point_back(outside);
    }
}
