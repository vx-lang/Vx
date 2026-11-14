//===- check submodule - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A closure copies a variable that it does not move. A write to that variable before the last
// use of the closure is a write that the closure does not see. K4 in
// docs/discussions/waiting_tasks_channels_and_task_errors.md makes it an error. This release
// warns (W1034).
//
//===----------------------------------------------------------------------===//

use super::super::*;

/// A variable that the closure in a `let` copied.
pub(crate) struct ClosureCopy {
    /// The variable that holds the closure.
    closure: String,
    name: String,
    /// The scope that declares the copied variable.
    depth: usize,
    line: usize,
}

impl TypeChecker<'_> {
    /// The closure in `let closure = |..| ..` at `at` copied `copied`, the captures that it does
    /// not move.
    pub(crate) fn note_closure_copies(
        &mut self,
        closure: String,
        copied: &[String],
        at: &crate::syntax::Span,
    ) {
        for name in copied {
            if let Some((_, _, depth)) = self.lookup_with_depth(name) {
                self.closure_copies.push(ClosureCopy {
                    closure: closure.clone(),
                    name: name.clone(),
                    depth,
                    line: at.line,
                });
            }
        }
    }

    /// W1034: the program writes `name`, and a closure that copied `name` is used later.
    pub(crate) fn warn_write_to_closure_copy(&mut self, name: &str, at: &crate::syntax::Span) {
        if self.speculating || self.closure_copies.is_empty() {
            return;
        }
        let Some(depth) = self.lookup_with_depth(name).map(|(_, _, d)| d) else {
            return;
        };
        let Some(c) = self.closure_copies.iter().find(|c| {
            c.name == name && c.depth == depth && self.borrow.is_variable_used_after(&c.closure)
        }) else {
            return;
        };
        let message = format!(
            "the closure in '{}' copied '{name}' on line {}, so the closure does not see this \
             write. A later release makes this an error. Write '{name}' before the closure, or \
             copy the value into a new variable for the closure",
            c.closure, c.line
        );
        self.errors.warn(
            crate::diagnostic::DiagnosticCode::W1034,
            message,
            Some(crate::diagnostic::SourceSpan::from_ast_span(at)),
        );
    }
}
