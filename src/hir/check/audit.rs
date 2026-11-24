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
    for f in program.functions.iter().filter(|f| f.exported) {
        for clause in f.requires.iter().filter(pointer_clause) {
            let what = format!(
                "the exported function `{}` requires `{}`, and a call from C proves nothing",
                f.name,
                clause_text(clause)
            );
            found.push(at(clause.span(), what));
        }
    }
    found
}

/// A kind of trusted code: code that the compiler does not check, and that a review reads. Code
/// with none of these cannot forge a capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustedCode {
    UnsafeBlock,
    UnsafeFn,
    MlirBlock,
    ExternFn,
    ExternImpl,
    UnsafeImpl,
}

impl TrustedCode {
    /// The kind as the count line of the audit names `n` of it.
    fn counted(self, n: usize) -> String {
        let (one, many) = match self {
            TrustedCode::UnsafeBlock => ("`unsafe` block", "`unsafe` blocks"),
            TrustedCode::UnsafeFn => ("`unsafe fn`", "`unsafe fn`"),
            TrustedCode::MlirBlock => ("`mlir!` block", "`mlir!` blocks"),
            TrustedCode::ExternFn => ("`extern` function", "`extern` functions"),
            TrustedCode::ExternImpl => ("`extern impl`", "`extern impl`"),
            TrustedCode::UnsafeImpl => ("`unsafe impl`", "`unsafe impl`"),
        };
        format!("{n} {}", if n == 1 { one } else { many })
    }
}

/// Each use of trusted code in `program`: an `unsafe` block or an `unsafe fn` that the program
/// wrote, an `mlir!` block, an `extern` function, an `extern impl` and an `unsafe impl`.
pub fn trusted_code(program: &Program) -> Vec<(TrustedCode, Trusted)> {
    let file = program
        .file
        .as_ref()
        .map_or_else(|| "?".to_string(), |f| f.to_string());
    let mut found = Vec::new();
    let mut add = |kind: TrustedCode, span: Span, what: String| {
        let trusted = Trusted {
            file: file.clone(),
            line: span.line,
            column: span.column,
            what: format!("trusted code: {what}"),
        };
        found.push((kind, trusted));
    };
    for ext in &program.externs {
        let what = format!("the `extern` function `{}`, which runs C code", ext.name);
        add(TrustedCode::ExternFn, ext.span, what);
    }
    for ib in &program.impls {
        let header = match &ib.trait_name {
            Some(t) if t.as_ref() != "_inherent" => format!("impl {t} for {}", ib.target_type),
            _ => format!("impl {}", ib.target_type),
        };
        let span = ib.place.span;
        match (ib.is_unsafe, ib.is_extern) {
            (true, true) => add(
                TrustedCode::UnsafeImpl,
                span,
                format!("`unsafe extern {header}`, a claim that the compiler does not check"),
            ),
            (true, false) => add(
                TrustedCode::UnsafeImpl,
                span,
                format!("`unsafe {header}`, a claim that the compiler does not check"),
            ),
            (false, true) => add(
                TrustedCode::ExternImpl,
                span,
                format!("`extern {header}`, an impl outside the file of its type"),
            ),
            (false, false) => {}
        }
    }
    let methods = program.impls.iter().flat_map(|ib| ib.methods.iter());
    for f in program.functions.iter().chain(methods) {
        if f.is_unsafe {
            add(
                TrustedCode::UnsafeFn,
                f.span,
                format!("`unsafe fn {}`", f.name),
            );
        }
        crate::syntax::visit_exprs(&f.body, &mut |e| match e {
            Expr::UnsafeBlock(b) if !b.generated && b.task.is_none() => add(
                TrustedCode::UnsafeBlock,
                b.span,
                "an `unsafe` block".to_string(),
            ),
            // The compiler verified a block outside `unsafe`, so the audit leaves it out.
            Expr::InlineMlir(m) if crate::hir::check::safe_mlir::unverified(m).is_some() => {
                add(TrustedCode::MlirBlock, m.span, "an `mlir!` block".into())
            }
            _ => {}
        });
    }
    found
}

/// The line of the audit for the trusted code of `core` and `std`, which it counts and does not
/// list. `None` when there is none.
pub fn library_count_line(
    counts: &std::collections::BTreeMap<TrustedCode, usize>,
) -> Option<String> {
    let parts: Vec<String> = counts.iter().map(|(k, n)| k.counted(*n)).collect();
    let (last, rest) = parts.split_last()?;
    let list = match rest.is_empty() {
        true => last.clone(),
        false => format!("{} and {last}", rest.join(", ")),
    };
    Some(format!(
        "core and std: {list} in the modules that this program uses, not listed. \
         `--audit-library` lists them"
    ))
}

/// E5003 for each use of trusted code in `program`, when `vxc --no-trusted-code` names its
/// package. Runs after macro expansion, so the code that a macro writes counts too.
pub fn refused_trusted_code(program: &Program) -> Vec<crate::diagnostic::Diagnostic> {
    let Some(package) = &program.no_trusted_code else {
        return Vec::new();
    };
    let mut found = trusted_code(program);
    found.sort_by(|a, b| a.1.cmp(&b.1));
    found
        .into_iter()
        .map(|(_, t)| {
            let what = t.what.strip_prefix("trusted code: ").unwrap_or(&t.what);
            let mut d = crate::diagnostic::Diagnostic::error(format!(
                "`--no-trusted-code {package}` refuses trusted code in the package `{package}`, \
                 and this is {what}"
            ));
            d.code = Some(crate::diagnostic::DiagnosticCode::E5003);
            d.file = program.file.clone();
            d.source_span = Some(crate::diagnostic::SourceSpan::new(t.line, t.column, 1));
            d
        })
        .collect()
}

/// Mark each module of a package that `vxc --no-trusted-code` names. A name that is not a
/// package of the compile is an error.
pub fn mark_refused_packages(
    programs: &mut [Program],
    refused: &[String],
    packages: &[(String, std::path::PathBuf)],
) -> Result<(), String> {
    for name in refused {
        if !packages.iter().any(|(p, _)| p == name) {
            return Err(format!(
                "`--no-trusted-code {name}` names no package. Give the package with \
                 `--package {name}=DIR`"
            ));
        }
    }
    for p in programs {
        let path = p.module_path.to_string();
        let package = refused
            .iter()
            .find(|name| path == **name || path.starts_with(&format!("{name}::")));
        p.no_trusted_code = package.map(|n| n.as_str().into());
    }
    Ok(())
}

/// Whether `program` is a module of `core` or `std`.
pub fn is_library(program: &Program) -> bool {
    let path = program.module_path.as_ref();
    ["core", "std"]
        .iter()
        .any(|l| path == *l || path.starts_with(&format!("{l}::")))
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
