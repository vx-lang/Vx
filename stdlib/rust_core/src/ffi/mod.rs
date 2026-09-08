//===- mod.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file aggregates the FFI (Foreign Function Interface) modules.
// It serves as the gateway between the dynamically compiled Vx MLIR code and the
// statically compiled Rust standard library, ensuring safe data boundary crossings.
//
//===----------------------------------------------------------------------===//
pub mod llama;
pub mod macros;
pub mod rt;
