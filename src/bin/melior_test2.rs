//===- melior_test2.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
fn main() {
    let context = melior::Context::new();
    let _ = melior::ir::attribute::DenseI64ArrayAttribute::new(&context, &[1, 2, 3]);
}
