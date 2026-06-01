#![allow(clippy::not_unsafe_ptr_arg_deref)]
//===- lib.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file is the root of the statically linked Vx standard library core.
// It aggregates all the low-level FFI implementations (math, networking, I/O,
// collections) that are natively compiled and linked against lowered Vx MLIR code.
//
//===----------------------------------------------------------------------===//
#![allow(clippy::box_default)]

pub mod collections;
pub mod env;
pub mod ffi;
pub mod googletest;
pub mod io;
pub mod math;
pub mod net;
pub mod simd;
