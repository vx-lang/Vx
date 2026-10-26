//===- unused_functions.rs - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The warning for a function in the program's own file that nothing calls (W1002).
//
// A function is used when `main` or a `pub fn` names it, or a used function does: in a call, as a
// value, as a `grad`/`vjp`/`jvp` target or as a symbol in an `mlir!` block. Methods are called
// through types and traits, which this does not follow, so every method body, trait default,
// transfer lowering and `const` initializer in the file counts as used too.
//
// It reads the syntax after macro expansion, so a call a macro wrote counts, and before type
// checking, which renames calls to generic instances.
//
//===----------------------------------------------------------------------===//
use crate::diagnostic::{DiagnosticCode, DiagnosticsVec};
use crate::syntax::{visit_expr, visit_exprs, Expr, Function, Program, Statement, Topology};
use std::collections::{HashMap, HashSet};

/// Warn about each free function in `program` that `main` does not reach. A file without `main`
/// is a library, whose functions are there for its importers, so it gets no warning.
pub fn warn_functions_nothing_calls(program: &Program, errors: &mut DiagnosticsVec) {
    let functions: HashMap<&str, &Function> = program
        .functions
        .iter()
        .map(|f| (f.name.as_ref(), f))
        .collect();
    if !functions.contains_key("main") {
        return;
    }

    // A `pub fn` is exported, so code outside the program can call it, like `main`.
    let mut to_visit: Vec<String> = vec!["main".to_string()];
    to_visit.extend(
        program
            .functions
            .iter()
            .filter(|f| f.is_pub)
            .map(|f| f.name.to_string()),
    );
    for i in &program.impls {
        i.methods
            .iter()
            .for_each(|m| names_in_function(m, &mut to_visit));
    }
    for t in &program.traits {
        for m in &t.methods {
            if let Some(body) = &m.default_body {
                names_in_stmts(body, &mut to_visit);
            }
        }
    }
    for t in &program.transfer_impls {
        t.methods
            .iter()
            .for_each(|m| names_in_function(m, &mut to_visit));
    }
    for c in &program.consts {
        visit_expr(&c.expr, &mut |e| names_in_expr(e, &mut to_visit));
    }

    let mut used: HashSet<&str> = HashSet::new();
    while let Some(name) = to_visit.pop() {
        if let Some((&name, f)) = functions.get_key_value(name.as_str()) {
            if used.insert(name) {
                names_in_function(f, &mut to_visit);
            }
        }
    }

    for f in &program.functions {
        let name: &str = f.name.as_ref();
        if !used.contains(name) && !name.starts_with('_') {
            // The warning names the function's file and points at its name.
            errors.file = f.file.clone();
            let at =
                (f.span.line > 0).then(|| crate::diagnostic::SourceSpan::from_ast_span(&f.span));
            errors.warn(
                DiagnosticCode::W1002,
                format!(
                    "function `{name}` is never called, directly or through other functions, \
                     from `main`; remove it, or start its name with `_` to keep it"
                ),
                at,
            );
        }
    }
}

/// Every name `f` mentions that could be a function: in its body, its contracts and its topology.
fn names_in_function(f: &Function, out: &mut Vec<String>) {
    names_in_stmts(&f.body, out);
    for e in f.requires.iter().chain(&f.ensures) {
        visit_expr(e, &mut |e| names_in_expr(e, out));
    }
    names_in_topology(&f.topology, out);
}

fn names_in_stmts(stmts: &[Statement], out: &mut Vec<String>) {
    visit_exprs(stmts, &mut |e| names_in_expr(e, out));
}

/// The names `e` itself mentions; `visit_expr` reaches the expressions inside it.
fn names_in_expr(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::FunctionCall(c) => push_path(&c.name, out),
        Expr::Identifier(id) => push_path(&id.name, out),
        Expr::Grad(g) => push_path(&g.target_fn, out),
        Expr::Vjp(v) => push_path(&v.target_fn, out),
        Expr::Jvp(j) => push_path(&j.target_fn, out),
        // `a::f` without a call is read as an enum variant until the checker knows better.
        Expr::EnumVariant(v) => out.push(v.variant_name.to_string()),
        Expr::Topology(t) => names_in_topology(&t.top, out),
        Expr::SpawnOn(s) => names_in_topology(&s.top, out),
        Expr::TransferPredicate(p) => {
            names_in_topology(&p.from, out);
            names_in_topology(&p.to, out);
        }
        // `func.call @helper(...)` inside the block's MLIR text.
        Expr::InlineMlir(m) => {
            for piece in m.block_str.split('@').skip(1) {
                let end = piece
                    .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .unwrap_or(piece.len());
                out.push(piece[..end].to_string());
            }
        }
        _ => {}
    }
}

/// A device index such as `GPU[pick()]` is an expression too.
fn names_in_topology(top: &Topology, out: &mut Vec<String>) {
    match top {
        Topology::NPU(e) | Topology::AccCore(e) | Topology::GPU(e) => {
            visit_expr(e, &mut |e| names_in_expr(e, out))
        }
        Topology::Slice(base, start, end) => {
            names_in_topology(base, out);
            visit_expr(start, &mut |e| names_in_expr(e, out));
            visit_expr(end, &mut |e| names_in_expr(e, out));
        }
        _ => {}
    }
}

/// `helper`, `helper<i32>` and `path::helper` all may name the function `helper`.
fn push_path(name: &str, out: &mut Vec<String>) {
    let base = name.split('<').next().unwrap_or(name);
    if let Some((_, last)) = base.rsplit_once("::") {
        out.push(last.to_string());
    }
    out.push(base.to_string());
}
