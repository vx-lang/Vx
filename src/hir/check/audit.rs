//===- audit.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `vxc --audit`: the facts that the compiler trusts without a proof. Each `assume`, each pointer
// clause that a call in `unsafe` code trusts, each `ensures` of an `extern` function, and each
// pointer `requires` of a `pub fn`, which C code can call with no proof. A1 in
// docs/discussions/contracts_for_extern_and_unsafe_code.md has the design.
//
//===----------------------------------------------------------------------===//

use crate::hir::env::TypeChecker;
use crate::syntax::{Expr, Program, Span, UnaryOp};

/// A fact that the compiler trusts without a proof, and where the program states it. The order
/// is the order of the lines of the audit.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Trusted {
    pub file: String,
    pub line: usize,
    pub column: usize,
    pub what: String,
}

impl std::fmt::Display for Trusted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}: {}",
            self.file, self.line, self.column, self.what
        )
    }
}

/// The clause `e` as the program wrote it, such as `valid(p, n)` or `!frees(s)`.
pub(crate) fn clause_text(e: &Expr) -> String {
    let (not, call) = match e {
        Expr::UnaryOp(u) if u.op == UnaryOp::Not => ("!", &*u.expr),
        e => ("", e),
    };
    if let Expr::FunctionCall(fc) = call {
        let args: Option<Vec<String>> = fc.args.iter().map(crate::hir::prover::to_source).collect();
        if let Some(args) = args {
            return format!("{not}{}({})", fc.name, args.join(", "));
        }
    }
    crate::hir::prover::to_source(e).unwrap_or_else(|| "..".to_string())
}

/// The facts that the declarations of `program` state and the compiler trusts: each `ensures`
/// of an `extern` function, whose body the compiler cannot read, and each pointer `requires` of
/// a `pub fn`, because a call from C proves nothing.
pub fn trusted_declarations(program: &Program) -> Vec<Trusted> {
    let file = program
        .file
        .as_ref()
        .map_or_else(|| "?".to_string(), |f| f.to_string());
    let at = |span: Span, what: String| Trusted {
        file: file.clone(),
        line: span.line,
        column: span.column,
        what,
    };
    let mut found = Vec::new();
    for ext in &program.externs {
        for clause in &ext.ensures {
            let what = format!(
                "the `extern` function `{}` ensures `{}`, and the compiler trusts it",
                ext.name,
                clause_text(clause)
            );
            found.push(at(clause.span(), what));
        }
    }
    let pointer_clause = |e: &&Expr| crate::hir::check::pointer_facts::read_clause(e).is_some();
    for f in program.functions.iter().filter(|f| f.is_pub) {
        for clause in f.requires.iter().filter(pointer_clause) {
            let what = format!(
                "`pub fn {}` requires `{}`, and a call from C proves nothing",
                f.name,
                clause_text(clause)
            );
            found.push(at(clause.span(), what));
        }
    }
    found
}

impl TypeChecker<'_> {
    /// For `vxc --audit`: the code at `at` trusts a fact, as `what` says.
    pub(crate) fn note_trusted(&mut self, at: Span, what: String) {
        if self.speculating {
            return;
        }
        let file = self
            .errors
            .file
            .as_ref()
            .map_or_else(|| "?".to_string(), |f| f.to_string());
        if let Some(trusted) = &mut self.trusted {
            trusted.insert(Trusted {
                file,
                line: at.line,
                column: at.column,
                what,
            });
        }
    }
}
