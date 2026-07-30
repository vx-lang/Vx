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

const VARIANCE_MASK: u64 = 0xF000;
const REGION_MASK: u64 = 0x0FFF;
const PARAM_MASK: u64 = 0xFFFF;

/// The reserved "region not yet assigned" sentinel: the maximum value the 12-bit region field can
/// hold. A parsed reference type carries it until a scope depth is assigned (see
/// `src/parser/types.rs`), and it surfaces in generic-deduction diagnostics as `region_id: 4095`.
///
/// It is **not** a scope depth — `verify_subtyping_bounds` treats it as a wildcard, never comparing
/// it numerically, so an unset region neither satisfies nor fails subtyping by accident (#267). It
/// is reserved: real depths are clamped to [`REGION_MAX`] so none ever equals the sentinel. Anyone
/// narrowing this field (e.g. #265 shrinking slot 0 to 9 bits) must keep a reserved sentinel at the
/// new field's maximum and clamp real depths below it — the numeric value must never be trusted as a
/// region. See `docs/discussions/borrow_checker_architecture.md` §2.
pub const REGION_UNSET: u64 = REGION_MASK;

/// The largest assignable real region (scope depth): one below the [`REGION_UNSET`] sentinel, so a
/// genuine depth can never be mistaken for "unset". Deeper nesting is clamped to this (shortest-lived
/// valid region) rather than overflowing into the sentinel.
pub const REGION_MAX: u64 = REGION_MASK - 1;

/// Map a lexical scope depth to the region value stored in a `TypeId` (#267). The [`REGION_UNSET`]
/// sentinel is preserved as-is (a parsed reference carries it until a depth is bound); any *real*
/// depth is clamped to [`REGION_MAX`] so it can never equal the sentinel — nesting deeper than
/// `REGION_MAX` degrades to the shortest-lived valid region rather than overflowing into "unset".
pub fn region_for_depth(depth: u64) -> u64 {
    if depth == REGION_UNSET {
        REGION_UNSET
    } else {
        depth.min(REGION_MAX)
    }
}

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

            // Iterate through all 4 active slots in Word 2
            for i in 0..4 {
                let shift = i * 16;
                let slot_a = (bits_a >> shift) & PARAM_MASK;
                let slot_b = (bits_b >> shift) & PARAM_MASK;

                let variance_a = (slot_a & VARIANCE_MASK) >> 12;
                let variance_b = (slot_b & VARIANCE_MASK) >> 12;

                if variance_a != variance_b {
                    return false;
                }

                let region_a = slot_a & REGION_MASK;
                let region_b = slot_b & REGION_MASK;

                // An unset region (a parse-time placeholder not yet bound to a scope depth) does not
                // constrain subtyping: treat it as a wildcard rather than a concrete very-short-lived
                // region. Recognising it explicitly — instead of relying on its numeric position —
                // is what keeps the check correct if the region field is ever narrowed (#267/#265).
                if region_a == REGION_UNSET || region_b == REGION_UNSET {
                    continue;
                }

                let valid = match variance_a {
                    // 0x0 represents Invariance (typically used for the inner type of &mut T)
                    // Invariant: Lifetimes must match EXACTLY.
                    0x0 => region_a == region_b,
                    // 0x1 represents Covariance (typically used for the outer lifetime of references: &'a)
                    // Covariant: Source lifetime must outlive or equal target lifetime.
                    // Because Region 0 is 'static, a smaller Region ID actually lives longer.
                    // Therefore, region_a (source) <= region_b (target).
                    0x1 => region_a <= region_b,
                    // 0x2 represents Contravariance (typically used for function pointer arguments)
                    // Contravariant: Target must outlive source.
                    0x2 => region_a >= region_b,
                    _ => false,
                };

                if !valid {
                    return false;
                }
            }
            true
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gid::TypeId;
    use crate::session::{GlobalSession, LocalWorkerState};
    use std::sync::Arc;

    fn worker() -> LocalWorkerState {
        LocalWorkerState::new(Arc::new(GlobalSession::new(1)))
    }

    /// A fast-path lifetime GID: param 0 packs `(region, variance)` into word 2 -- exactly what the
    /// type checker's `lower_to_type_id` produces for a `Borrow`/`Pointer` in `is_assignable`.
    fn lifetime_gid(region: u16, variance: u8) -> TypeId {
        let mut id = TypeId::new(0, 0, 0, 0);
        id.try_set_fast_param(0, region, variance).unwrap();
        id
    }

    /// Covariance (`&'a T`): the source lifetime must outlive-or-equal the target. Region 0 is
    /// `'static`, so a *smaller* region id lives longer -> source region <= target region.
    #[test]
    fn covariant_borrow_source_outlives_target_is_assignable() {
        let w = worker();
        assert!(verify_subtyping_bounds(
            &lifetime_gid(1, 0x1),
            &lifetime_gid(3, 0x1),
            &w
        ));
    }

    #[test]
    fn covariant_borrow_source_shorter_than_target_is_rejected() {
        let w = worker();
        assert!(!verify_subtyping_bounds(
            &lifetime_gid(3, 0x1),
            &lifetime_gid(1, 0x1),
            &w
        ));
    }

    /// Invariance (the inner type of `&mut T`): lifetimes must match exactly.
    #[test]
    fn invariant_requires_exact_region() {
        let w = worker();
        assert!(verify_subtyping_bounds(
            &lifetime_gid(2, 0x0),
            &lifetime_gid(2, 0x0),
            &w
        ));
        assert!(!verify_subtyping_bounds(
            &lifetime_gid(2, 0x0),
            &lifetime_gid(3, 0x0),
            &w
        ));
    }

    /// A variance mismatch between the two sides is never compatible.
    #[test]
    fn variance_mismatch_is_rejected() {
        let w = worker();
        assert!(!verify_subtyping_bounds(
            &lifetime_gid(1, 0x1),
            &lifetime_gid(1, 0x2),
            &w
        ));
    }

    /// The reserved `REGION_UNSET` sentinel is a wildcard: it does not constrain the region
    /// dimension, so it neither satisfies nor fails subtyping by its numeric value (#267).
    #[test]
    fn unset_region_is_a_wildcard() {
        let w = worker();
        let unset = REGION_UNSET as u16;
        // Covariant: unset on either side passes regardless of the other operand's depth. (A
        // numeric read would fail `4095 <= 5`.)
        assert!(verify_subtyping_bounds(
            &lifetime_gid(unset, 0x1),
            &lifetime_gid(5, 0x1),
            &w
        ));
        assert!(verify_subtyping_bounds(
            &lifetime_gid(5, 0x1),
            &lifetime_gid(unset, 0x1),
            &w
        ));
        // Invariant: a real region would require exact equality, but the wildcard passes.
        assert!(verify_subtyping_bounds(
            &lifetime_gid(unset, 0x0),
            &lifetime_gid(5, 0x0),
            &w
        ));
        // Variance mismatch still fails even with unset regions.
        assert!(!verify_subtyping_bounds(
            &lifetime_gid(unset, 0x1),
            &lifetime_gid(unset, 0x2),
            &w
        ));
    }

    /// A just-below-sentinel region (`REGION_MAX` = 4094) and the issue's 510 are *ordinary* regions
    /// compared by depth, distinct from the sentinel — the property #265's narrowed field must keep.
    #[test]
    fn max_real_region_is_distinct_from_the_sentinel() {
        let w = worker();
        let max_real = REGION_MAX as u16; // 4094
        assert_ne!(REGION_MAX, REGION_UNSET);
        // Two real regions compare by depth: invariant requires exact equality.
        assert!(verify_subtyping_bounds(
            &lifetime_gid(max_real, 0x0),
            &lifetime_gid(max_real, 0x0),
            &w
        ));
        assert!(!verify_subtyping_bounds(
            &lifetime_gid(510, 0x0),
            &lifetime_gid(max_real, 0x0),
            &w
        ));
        // Covariant: a shorter-lived (larger) real region does not outlive a longer-lived one.
        assert!(!verify_subtyping_bounds(
            &lifetime_gid(max_real, 0x1),
            &lifetime_gid(510, 0x1),
            &w
        ));
        assert!(verify_subtyping_bounds(
            &lifetime_gid(510, 0x1),
            &lifetime_gid(max_real, 0x1),
            &w
        ));
        // 510 vs the sentinel passes (wildcard) — distinct from 510 vs a real 4094 (fails invariant).
        assert!(verify_subtyping_bounds(
            &lifetime_gid(510, 0x0),
            &lifetime_gid(REGION_UNSET as u16, 0x0),
            &w
        ));
    }

    /// The lowering rule (`region_for_depth`, used by `lower_to_type_id`) preserves the sentinel and
    /// keeps every real depth strictly below it, so no genuine region can ever be read back as
    /// "unset" — the invariant `verify_subtyping_bounds`'s wildcard relies on (#267).
    #[test]
    fn region_for_depth_preserves_sentinel_and_clamps_real_depths() {
        assert_eq!(region_for_depth(REGION_UNSET), REGION_UNSET); // sentinel preserved
        assert_eq!(region_for_depth(0), 0); // 'static unchanged
        assert_eq!(region_for_depth(5), 5); // ordinary depth unchanged
        assert_eq!(region_for_depth(REGION_MAX), REGION_MAX); // max real unchanged
                                                              // A depth at or beyond the sentinel is clamped *below* it — never becomes the sentinel.
        assert_eq!(region_for_depth(REGION_UNSET + 1), REGION_MAX);
        assert_eq!(region_for_depth(999_999), REGION_MAX);
        assert_ne!(region_for_depth(999_999), REGION_UNSET);
    }
}
