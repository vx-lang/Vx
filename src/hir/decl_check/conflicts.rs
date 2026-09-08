//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One name declared by two inputs of the same compilation -- a `--machine` file and the
// program, most often.
//
//===----------------------------------------------------------------------===//

use super::super::*;

impl TypeChecker<'_> {
    /// Report a `Memory`/`Topology` name declared by two modules of this compilation unit with
    /// different declarations (E6012). The declaration tables are name-keyed, so one would
    /// otherwise silently shadow the other -- and since module order is a `HashMap` iteration,
    /// *which* one survived could vary between runs. The motivating case is a `--machine` file
    /// whose SKU declares a space the program also declares inline: the whole point of the flag is
    /// that the machine model is swappable, which requires knowing it is the one in force (#281).
    pub fn check_declaration_conflicts(&mut self) {
        for dup in &self.env.duplicate_decls {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E6012,
                format!(
                    "{} '{}' is declared in more than one input (also in '{}'); remove one or \
                     rename it -- a duplicate declaration would silently shadow the other",
                    dup.kind, dup.name, dup.module
                ),
                None,
            );
        }
    }
}
