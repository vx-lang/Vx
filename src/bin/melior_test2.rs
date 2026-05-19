//===- melior_test2.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file is a standalone executable used for testing the Melior MLIR bindings.
// It serves as an experimental sandbox for verifying the behavior of specific MLIR
// dialects (like arith and func) before they are formally integrated into the main
// Vx compiler lowering passes.
//
//===----------------------------------------------------------------------===//
fn main() {
    let context = melior::Context::new();
    let _ = melior::ir::attribute::DenseI64ArrayAttribute::new(&context, &[1, 2, 3]);
}
