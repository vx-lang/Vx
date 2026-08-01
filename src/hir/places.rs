//===- places.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//! Shared "place" helpers — one derivation of *which memory an lvalue names* and *whether two paths
//! overlap*, used by both the borrow checker (conflict detection, `expr.rs`) and the flat lowerer
//! (§5.4 alias-scope disjointness, `flatten.rs`).
//!
//! `scalar_references_flat.md` §16.2 recorded that these facts were being computed twice — the checker's
//! `extract_base_and_path` / its three inline overlap loops, and the lowerer's `place_base_path` /
//! `paths_may_alias`. Both derivations were individually sound but duplicated; this module is the single
//! source both now call. The borrow checker keeps its `Vec<String>` representation (its `BorrowRecord`
//! stores string paths) by adapting `base_and_path` at the boundary; the overlap predicate is generic
//! over the path element, so `Vec<String>` and `Vec<Symbol>` share it without conversion.
use crate::symbol::Symbol;
use crate::syntax::Expr;

/// The root local and projection path of an lvalue: `o.x.y` -> `(o, [x, y])`, a bare local -> `(o, [])`.
/// An `IndexAccess` is *seen through* (`a[i].x` -> `(a, [x])`) — an index does not name a distinct field,
/// so two accesses that differ only in index share a path and are treated as *possibly* aliasing (the
/// conservative direction). `None` for a base with no nameable root (`&5`, a call result).
pub fn base_and_path(e: &Expr) -> Option<(Symbol, Vec<Symbol>)> {
    match e {
        Expr::Identifier(id) => Some((id.name.clone(), Vec::new())),
        Expr::MemberAccess(m) => {
            let (root, mut path) = base_and_path(&m.base)?;
            path.push(m.member.clone());
            Some((root, path))
        }
        Expr::IndexAccess(ix) => base_and_path(&ix.base),
        _ => None,
    }
}

/// Whether two projection paths under the **same** root may name overlapping memory: true when one is a
/// prefix of the other (`[inner]` vs `[inner, v]`) or they are equal; distinct fields at any shared level
/// (`[x]` vs `[y]`) are disjoint. Generic over the path element so the checker's `[String]` and the
/// lowerer's [`Symbol`] paths share it. Missing a real disjointness (returning `true` when the paths are
/// actually disjoint) is always the safe direction — it withholds a `noalias` / reports a conflict.
pub fn paths_may_alias<T: PartialEq>(a: &[T], b: &[T]) -> bool {
    let n = a.len().min(b.len());
    a[..n] == b[..n]
}
