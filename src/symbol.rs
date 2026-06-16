//===- symbol.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This module provides the `Symbol` type, an interned/shared string type used
// across the AST to avoid heavy string allocations and deep cloning during compiler passes.
//
//===----------------------------------------------------------------------===//

pub type Symbol = std::sync::Arc<str>;
