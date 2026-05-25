//===- borrow.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
// For a comprehensive overview of how this module interacts with the Lexical Borrow Checker
// in sema.rs, please read: `docs/discussions/borrow_checker_architecture.md`.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Borrow Checker for the Vx compiler.
// It statically verifies memory safety by enforcing borrowing rules, ensuring that
// aliasing and mutability invariants are respected, and guaranteeing that pinned
// memory (like NPU HBM) does not leak outside its required scope.
//
//===----------------------------------------------------------------------===//
//
// # The 256-Bit FastPath Borrow Checker Algorithm
//
// Traditional borrow checkers rely on complex Graph or AST traversal, which
// causes massive compiler slowdowns (e.g., Rust's NLL). The Vx compiler
// solves this by encoding lifetimes and subtyping variance directly into
// the 256-bit global TypeId structures and executing hardware-level
// bitwise register math to prove safety instantly!
//
// ## TypeId Bitpacking Structure
// A `TypeId` consists of 4 x 64-bit words (256 bits total).
// `Word 2` is exclusively reserved for the "FastPath Lifetime Hash".
// It can store up to 4 generic parameters/lifetimes simultaneously, packed into
// 16-bit slots (4 * 16 = 64 bits):
//
//     Word 2:  [ Param 3 (16) | Param 2 (16) | Param 1 (16) | Param 0 (16) ]
//
// Each 16-bit parameter payload is split as follows:
//     [ Variance Flags (4 bits) | Region ID (12 bits) ]
//
// ### Region IDs (Lexical Scope Depths)
// The Region ID represents the lifetime of the borrow. A Region ID is simply
// the depth of the lexical block it was instantiated in.
// - `Region 0`: The `'static` lifetime (Global scope, lives forever).
// - `Region 1`: Top-level function block.
// - `Region 2..N`: Nested inner blocks.
// **Math Rule**: A smaller Region ID outlives a larger Region ID.
//
// ### Variance Flags
// The Variance Flags determine how lifetimes coerce during assignment:
// - `0x0`: Invariant (e.g., `&mut T`). Exact match required!
// - `0x1`: Covariant (e.g., `&T`). A longer lifetime can be coerced to a shorter one.
// - `0x2`: Contravariant (e.g., function arguments).
//
// ## Arithmetic Examples
//
// 1. Covariant Subtyping (Immutable Borrows):
// Let `type_a` = `&'long T` (Region 1, Covariant) -> Param 0 = `0x1001`
// Let `type_b` = `&'short T` (Region 5, Covariant) -> Param 0 = `0x1005`
// The compiler checks:
//     `variance_a == variance_b` -> `0x1 == 0x1` (True)
//     `region_a <= region_b` -> `1 <= 5` (True! 'long can be assigned to 'short).
//
// 2. Invariant Subtyping (Mutable Borrows):
// Let `type_a` = `&'long mut T` (Region 1, Invariant) -> Param 0 = `0x0001`
// Let `type_b` = `&'short mut T` (Region 5, Invariant) -> Param 0 = `0x0005`
// The compiler checks:
//     `variance_a == variance_b` -> `0x0 == 0x0` (True)
//     BUT for Invariance, Vx enforces an exact match (if we strictly follow Invariance).
//     Currently, the fast-path checks `region_a <= region_b`. To make mutable borrows
//     strictly invariant, the bitwise check would demand `region_a == region_b`.
//     (Note: Vx allows re-borrowing, so covariant-like lifetime shrinking is often allowed
//     even for mutable references under certain strict models, but the flags enable distinction!)
//
//===----------------------------------------------------------------------===//
use crate::gid::{LifetimeSignature, TypeId, UnboundedFunctionMetadata};
use crate::session::LocalWorkerState;

/// High-performance verification check for variance and lifetime compatibility.
/// Encodes borrow checker math directly into the 256-bit registers.
///
/// **Region ID Mathematical Convention:**
/// The Region ID represents the lifetime scope depth.
/// - `Region 0`: The `'static` lifetime (lives forever).
/// - `Region N`: An inner block at depth N.
///   Therefore, a **smaller Region ID lives longer** than a larger Region ID.
///
/// **Variance Math:**
/// - `Invariant (0x0)`: Requires strict equality (`region_a == region_b`).
/// - `Covariant (0x1)`: Source must outlive target (`region_a <= region_b`).
/// - `Contravariant (0x2)`: Target must outlive source (`region_a >= region_b`).
pub fn verify_subtyping_bounds(
    type_a: &TypeId,
    type_b: &TypeId,
    worker: &LocalWorkerState,
) -> bool {
    match (
        worker.resolve_lifetime(type_a),
        worker.resolve_lifetime(type_b),
    ) {
        (LifetimeSignature::FastPath(bits_a), LifetimeSignature::FastPath(bits_b)) => {
            // FAST PATH: Check lifetime compatibility using register operations
            if bits_a == bits_b {
                return true; // Exact structural match, exit instantly
            }

            // Evaluate individual variance rules for Parameter 0
            // Note: For a detailed explanation of the math rules, see:
            // `docs/discussions/borrow_checker_fastpath_integration.md`

            // Mask out the lowest 16 bits to extract the full payload for Parameter 0
            // The 16 bits are structured as: [ 4 bits: Variance | 12 bits: Region ID ]
            let param_a = bits_a & 0xFFFF;
            let param_b = bits_b & 0xFFFF;

            // Shift right by 12 bits to isolate the top 4 bits representing the Variance flag
            let variance_a = param_a >> 12;
            let variance_b = param_b >> 12;

            if variance_a == variance_b {
                // Mask out the bottom 12 bits to isolate the Region ID
                // (e.g., 0x0FFF ensures we drop the 4-bit variance flag)
                let region_a = param_a & 0x0FFF;
                let region_b = param_b & 0x0FFF;

                // 0x0 represents Invariance (typically used for the inner type of &mut T)
                if variance_a == 0x0 {
                    // Invariant: Lifetimes must match EXACTLY.
                    return region_a == region_b;
                }
                // 0x1 represents Covariance (typically used for the outer lifetime of references: &'a)
                else if variance_a == 0x1 {
                    // Covariant: Source lifetime must outlive or equal target lifetime.
                    // Because Region 0 is 'static, a smaller Region ID actually lives longer.
                    // Therefore, region_a (source) <= region_b (target).
                    return region_a <= region_b;
                }
                // 0x2 represents Contravariance (typically used for function pointer arguments)
                else if variance_a == 0x2 {
                    // Contravariant: Target must outlive source.
                    return region_a >= region_b;
                }
            }
            false
        }
        (LifetimeSignature::SlowPath(meta_a), LifetimeSignature::SlowPath(meta_b)) => {
            // SLOW PATH: Iterate through deep vector elements sequentially
            evaluate_slow_path_variance(meta_a, meta_b)
        }
        _ => false, // Incompatible layout paths
    }
}

fn evaluate_slow_path_variance(
    a: &UnboundedFunctionMetadata,
    b: &UnboundedFunctionMetadata,
) -> bool {
    // Unbounded processing logic for complex signatures
    // For now, require exact structural match
    a.lifetime_regions == b.lifetime_regions && a.trait_vtables == b.trait_vtables
}
