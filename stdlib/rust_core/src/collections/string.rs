//===- string.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the FFI bindings for the String type.
// It manages UTF-8 text allocations, providing safe interfaces for Vx programs
// to concatenate, slice, and manipulate heap-allocated text data.
//
//===----------------------------------------------------------------------===//
use crate::instantiate_string_ffi;

instantiate_string_ffi!();
