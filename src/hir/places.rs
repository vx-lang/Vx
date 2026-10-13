//===- places.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Shared "place" helpers — one derivation of *which memory an lvalue names* and *whether two paths
//! overlap*, used by both the borrow checker (conflict detection, `expr.rs`) and the flat lowerer
//! (alias-scope disjointness, `flatten.rs`).
//!
//! These facts used to be computed twice -- the checker's `extract_base_and_path` and its three
//! inline overlap loops, and the lowerer's `place_base_path` / `paths_may_alias`. Both were sound
//! but duplicated; this module is the single source both now call. The borrow checker keeps its
//! `Vec<String>` representation (its `BorrowRecord` stores string paths) by adapting
//! `base_and_path` at the boundary; the overlap predicate is generic over the path element, so
//! `Vec<String>` and `Vec<Symbol>` share it without conversion.
use crate::symbol::Symbol;
use crate::syntax::Expr;

/// The path element for an index: `[3]` for an integer literal, and [`ANY_INDEX`] for anything
/// else, which may be any element. So `q[0]` and `q[1]` are different places, and `q[i]` may be
/// either of them.
pub fn index_element(index: &Expr) -> String {
    match index {
        Expr::Number(n) => match n.value.parse::<u64>() {
            Ok(k) => format!("[{k}]"),
            Err(_) => ANY_INDEX.to_string(),
        },
        _ => ANY_INDEX.to_string(),
    }
}

/// The path element of an index whose value is not known: it may name any element.
pub const ANY_INDEX: &str = "[?]";

/// The root local and projection path of an lvalue: `o.x.y` -> `(o, [x, y])`, `q[0].x` ->
/// `(q, [[0], x])`, a bare local -> `(o, [])`. An index is a path element, written by
/// [`index_element`]. `None` for a base with no nameable root (`&5`, a call result).
pub fn base_and_path(e: &Expr) -> Option<(Symbol, Vec<Symbol>)> {
    match e {
        Expr::Identifier(id) => Some((id.name.clone(), Vec::new())),
        Expr::MemberAccess(m) => {
            let (root, mut path) = base_and_path(&m.base)?;
            path.push(m.member.clone());
            Some((root, path))
        }
        Expr::IndexAccess(ix) => {
            let (root, mut path) = base_and_path(&ix.base)?;
            path.push(Symbol::from(index_element(&ix.index)));
            Some((root, path))
        }
        _ => None,
    }
}

/// Whether two projection paths under the **same** root may name overlapping memory: true when
/// one is a prefix of the other (`[inner]` vs `[inner, v]`) or they match at every shared level.
/// Distinct fields (`[x]` vs `[y]`) and distinct constant indices (`[[0]]` vs `[[1]]`) are
/// disjoint; [`ANY_INDEX`] matches any element. Generic over the path element so the checker's
/// `[String]` and the lowerer's [`Symbol`] paths share it. Returning `true` for paths that are in
/// fact disjoint is always the safe direction: it withholds a `noalias` or reports a conflict.
pub fn paths_may_alias<T: AsRef<str>>(a: &[T], b: &[T]) -> bool {
    a.iter().zip(b).all(|(x, y)| {
        let (x, y) = (x.as_ref(), y.as_ref());
        x == y || x == ANY_INDEX || y == ANY_INDEX
    })
}

/// How a place reads in a message: `q[0].x`, with a dot before each field and none before an
/// index. A tuple's fields, which the parser names `_0`, `_1`, .., read as the program writes
/// them: `t.0`.
pub fn display_place<T: AsRef<str>>(root: &str, path: &[T]) -> String {
    let mut out = root.to_string();
    for p in path {
        let p = p.as_ref();
        if !p.starts_with('[') {
            out.push('.');
        }
        match p.strip_prefix('_') {
            Some(n) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => out.push_str(n),
            _ => out.push_str(p),
        }
    }
    out
}

/// A moved place as the checker records it, `p.a` or `t._0`, as it reads in a message.
pub fn display_dotted(place: &str) -> String {
    let mut parts = place.split('.');
    let root = parts.next().unwrap_or_default();
    let path: Vec<&str> = parts.collect();
    display_place(root, &path)
}
