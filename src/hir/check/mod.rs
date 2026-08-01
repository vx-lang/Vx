//===- check/mod.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Expression type-checking, split out of the former 5000-line `hir/expr.rs` along the
// `check_expr_type_flag` dispatch seam (frontend_refactoring_borrow_checker.md R2, #279). Each submodule
// holds the checks for one family of expressions as additional `impl TypeChecker` blocks; the dispatch and
// the shared type helpers stay in `hir/expr.rs`. Moves only — zero logic change.
//
//===----------------------------------------------------------------------===//

pub mod autodiff;
pub mod control;
pub mod literals;
pub mod operators;
