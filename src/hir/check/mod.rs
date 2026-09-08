//===- check/mod.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Expression type-checking, split out of the former 5000-line `hir/expr.rs` along the
// `check_expr_type_flag` dispatch seam (frontend_refactoring_borrow_checker.md R2, #279). Each submodule
// holds the checks for one family of expressions as additional `impl TypeChecker` blocks; the dispatch and
// the shared type helpers (`is_assignable`, `lower_to_type_id`) stay in `hir/expr.rs`. Moves only — zero
// logic change.
//
//===----------------------------------------------------------------------===//

pub mod access;
pub mod autodiff;
pub mod calls;
pub mod capacity_fold;
pub mod control;
pub mod literals;
pub mod operators;
pub mod raw;
pub mod region_traffic;
pub mod transfer;
