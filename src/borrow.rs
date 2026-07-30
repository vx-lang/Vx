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

/// Slot 0 (the return slot) reserves its top 3 region bits for the return-provenance code (#265), so
/// its region field is only **9 bits** — slots 1-3 keep the full 12. Any rule that reads slot 0's
/// region must mask with this, not [`REGION_MASK`]: reading the wide field would fold the provenance
/// bits into the lifetime and corrupt the comparison (exactly the truncation #267 warned of).
pub const REGION_MASK_0: u64 = 0x01FF;

/// The slot-0 counterpart of [`REGION_UNSET`]: the maximum of the narrowed 9-bit return-slot region
/// is its reserved "unset" sentinel (#265/#267). Recognised as a wildcard exactly like [`REGION_UNSET`].
pub const REGION_UNSET_0: u64 = REGION_MASK_0;

/// The largest assignable real region in slot 0 — one below [`REGION_UNSET_0`]. A return lifetime is
/// by construction a parameter's or `'static`, so it never needs deep nesting; anything deeper clamps
/// here rather than colliding with the sentinel or the provenance bits.
pub const REGION_MAX_0: u64 = REGION_MASK_0 - 1;

/// The slot-0 analogue of [`region_for_depth`] (#265): map a lexical scope depth to the region stored
/// in the **return slot**, whose field is 9 bits. The [`REGION_UNSET`] sentinel maps to the slot-0
/// sentinel [`REGION_UNSET_0`]; any real depth is clamped to [`REGION_MAX_0`] so it can neither reach
/// the sentinel nor spill into the provenance bits above the region.
pub fn region_for_depth_slot0(depth: u64) -> u64 {
    if depth == REGION_UNSET {
        REGION_UNSET_0
    } else {
        depth.min(REGION_MAX_0)
    }
}

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

                // Slot 0 is the return slot: its top 3 region bits carry the provenance code (#265),
                // so its region is 9 bits with its own unset sentinel; slots 1-3 keep the 12-bit
                // field. The mask must be slot-dependent — reading slot 0 with the wide `REGION_MASK`
                // would fold the provenance bits into the lifetime and corrupt the comparison.
                let (region_mask, region_unset) = if i == 0 {
                    (REGION_MASK_0, REGION_UNSET_0)
                } else {
                    (REGION_MASK, REGION_UNSET)
                };
                let region_a = slot_a & region_mask;
                let region_b = slot_b & region_mask;

                // An unset region (a parse-time placeholder not yet bound to a scope depth) does not
                // constrain subtyping: treat it as a wildcard rather than a concrete very-short-lived
                // region. Recognising it explicitly — instead of relying on its numeric position —
                // is what keeps the check correct if the region field is ever narrowed (#267/#265).
                if region_a == region_unset || region_b == region_unset {
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

    /// A fast-path lifetime GID with `(region, variance)` packed into an arbitrary slot — used to
    /// exercise the 12-bit parameter slots (1-3), which keep the full region width that slot 0 (the
    /// narrowed 9-bit return slot, #265) does not.
    fn lifetime_gid_at(slot: usize, region: u16, variance: u8) -> TypeId {
        let mut id = TypeId::new(0, 0, 0, 0);
        id.try_set_fast_param(slot, region, variance).unwrap();
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

    /// Real regions just below each slot's sentinel are *ordinary* depths compared numerically, never
    /// confused with the sentinel — the property #265's narrowed field must keep. Slot 0 (the return
    /// slot) has a 9-bit region (max real 510, sentinel 511); the parameter slots keep 12 bits (max
    /// real 4094, sentinel 4095).
    #[test]
    fn max_real_region_is_distinct_from_the_sentinel() {
        let w = worker();
        assert_ne!(REGION_MAX, REGION_UNSET);
        assert_ne!(REGION_MAX_0, REGION_UNSET_0);

        // --- Slot 0: 9-bit region, max real = 510. ---
        let max0 = REGION_MAX_0 as u16; // 510
                                        // Two real regions compare by depth: invariant requires exact equality.
        assert!(verify_subtyping_bounds(
            &lifetime_gid(max0, 0x0),
            &lifetime_gid(max0, 0x0),
            &w
        ));
        assert!(!verify_subtyping_bounds(
            &lifetime_gid(509, 0x0),
            &lifetime_gid(max0, 0x0),
            &w
        ));
        // Covariant: a shorter-lived (larger) real region does not outlive a longer-lived one.
        assert!(!verify_subtyping_bounds(
            &lifetime_gid(max0, 0x1),
            &lifetime_gid(509, 0x1),
            &w
        ));
        assert!(verify_subtyping_bounds(
            &lifetime_gid(509, 0x1),
            &lifetime_gid(max0, 0x1),
            &w
        ));
        // Max real vs the slot-0 sentinel passes (wildcard) — distinct from vs a real 510 (invariant).
        assert!(verify_subtyping_bounds(
            &lifetime_gid(max0, 0x0),
            &lifetime_gid(REGION_UNSET_0 as u16, 0x0),
            &w
        ));

        // --- Parameter slot 1: 12-bit region, max real = 4094, unchanged by the narrowing. ---
        let max_real = REGION_MAX as u16; // 4094
        assert!(verify_subtyping_bounds(
            &lifetime_gid_at(1, max_real, 0x0),
            &lifetime_gid_at(1, max_real, 0x0),
            &w
        ));
        assert!(!verify_subtyping_bounds(
            &lifetime_gid_at(1, 510, 0x0),
            &lifetime_gid_at(1, max_real, 0x0),
            &w
        ));
        // A real 4094 in a param slot is a wildcard only against the 4095 sentinel.
        assert!(verify_subtyping_bounds(
            &lifetime_gid_at(1, max_real, 0x0),
            &lifetime_gid_at(1, REGION_UNSET as u16, 0x0),
            &w
        ));
    }

    /// The provenance bits packed into slot 0 must not leak into the lifetime comparison (#265): two
    /// return types with the same region but different provenance codes still satisfy subtyping, and
    /// the region still governs once codes are set. This is the invariant the slot-dependent mask in
    /// `verify_subtyping_bounds` exists to guarantee — the corruption #267 warned a narrowed field
    /// would cause if read with the wide mask.
    #[test]
    fn slot0_provenance_bits_do_not_corrupt_region_compare() {
        let w = worker();
        // Same region (covariant, equal) but different provenance codes -> still compatible.
        let mut a = lifetime_gid(4, 0x1);
        a.set_return_provenance(1); // derives from param 0
        let mut b = lifetime_gid(4, 0x1);
        b.set_return_provenance(7); // conservative top
        assert_ne!(a, b, "the codes make the raw bits differ");
        assert!(verify_subtyping_bounds(&a, &b, &w));
        assert!(verify_subtyping_bounds(&b, &a, &w));

        // The region still decides once codes are present: a longer-lived source (smaller region) is
        // covariantly assignable; a shorter-lived one is not.
        let mut src_ok = lifetime_gid(2, 0x1);
        src_ok.set_return_provenance(3);
        let mut tgt = lifetime_gid(6, 0x1);
        tgt.set_return_provenance(3);
        assert!(verify_subtyping_bounds(&src_ok, &tgt, &w));
        let mut src_bad = lifetime_gid(9, 0x1);
        src_bad.set_return_provenance(3);
        assert!(!verify_subtyping_bounds(&src_bad, &tgt, &w));
    }

    /// The slot-0 lowering rule maps the unset sentinel to the slot-0 sentinel and clamps real depths
    /// below it, so a return region can never be read back as "unset" nor spill into the provenance
    /// bits above the 9-bit field (#265).
    #[test]
    fn region_for_depth_slot0_maps_sentinel_and_clamps() {
        assert_eq!(region_for_depth_slot0(REGION_UNSET), REGION_UNSET_0); // sentinel -> slot-0 sentinel
        assert_eq!(region_for_depth_slot0(0), 0); // 'static unchanged
        assert_eq!(region_for_depth_slot0(5), 5); // ordinary depth unchanged
        assert_eq!(region_for_depth_slot0(REGION_MAX_0), REGION_MAX_0); // slot-0 max real unchanged
                                                                        // Deeper than the 9-bit field clamps to max real — never to, or past, the sentinel.
        assert_eq!(region_for_depth_slot0(512), REGION_MAX_0);
        assert_eq!(region_for_depth_slot0(999_999), REGION_MAX_0);
        assert_ne!(region_for_depth_slot0(999_999), REGION_UNSET_0);
        // The clamped value fits the 9-bit field, so it cannot touch the provenance bits.
        assert!(region_for_depth_slot0(999_999) <= REGION_MASK_0);
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
