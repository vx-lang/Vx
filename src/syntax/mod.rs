//===- mod.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Abstract Syntax Tree module for the Vx compiler. Defines the core representation of the parsed program.
//
//===----------------------------------------------------------------------===//

pub mod decl;
pub mod expr;
pub mod macro_rules;
pub mod resolve;
pub mod stmt;
pub mod types;

pub use decl::*;
pub use expr::*;
pub use macro_rules::*;
pub use stmt::*;
pub use types::*;
