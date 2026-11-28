//===- hash_map.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FFI bindings for the HashMap type.
// It wraps the standard Rust HashMap implementation, exposing a C-ABI compatible
// interface for Vx programs to create and manipulate associative arrays.
//
//===----------------------------------------------------------------------===//
use crate::instantiate_hash_map_ffi;

// Common hash map specializations
instantiate_hash_map_ffi!(i32_i32, i32, i32);
instantiate_hash_map_ffi!(i32_f32, i32, f32);

/// A random number for the seed of one `HashMap` written in Vx. Rust's `RandomState` reads the
/// operating system's randomness once for each thread, and then gives each call new keys. So the
/// seeds of two maps differ, and a program cannot know the seed of a map in advance.
#[no_mangle]
pub extern "C" fn vx_hash_map_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(0x9e37_79b9_7f4a_7c15);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_seeds_differ() {
        let seeds: std::collections::HashSet<u64> = (0..64).map(|_| vx_hash_map_seed()).collect();
        assert_eq!(seeds.len(), 64);
    }
}
