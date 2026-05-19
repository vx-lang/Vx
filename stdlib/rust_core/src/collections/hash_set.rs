//===- hash_set.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
use crate::instantiate_hash_set_ffi;

// Common hash set specializations
instantiate_hash_set_ffi!(i32, i32);
