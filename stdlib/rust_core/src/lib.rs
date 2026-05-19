#![allow(clippy::not_unsafe_ptr_arg_deref)]
//===- lib.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
#![allow(clippy::box_default)]

pub mod collections;
pub mod ffi;
pub mod io;
pub mod math;
pub mod net;
pub mod simd;
