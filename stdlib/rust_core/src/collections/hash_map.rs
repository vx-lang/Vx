//===- hash_map.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
use crate::instantiate_hash_map_ffi;

// Common hash map specializations
instantiate_hash_map_ffi!(i32_i32, i32, i32);
instantiate_hash_map_ffi!(i32_f32, i32, f32);
