//===- borrow_test.rs - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file contains integration tests for the Borrow Checker.
// It ensures that the compiler correctly catches mutability violations, lifetime
// escapes, and invalid aliases, confirming the static memory safety guarantees
// of the Vx language.
//
//===----------------------------------------------------------------------===//
use vxc::borrow::verify_subtyping_bounds;
use vxc::gid::TypeId;

#[test]
fn test_fast_path_variance_checks() -> Result<(), String> {
    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let worker = vxc::session::LocalWorkerState::new(global_session.clone());

    // Type A: Variance = 1 (Covariant), Region = 0 ('static)
    let mut type_a = TypeId::new(0, 0, 0, 0);
    type_a.try_set_fast_param(0, 0, 1).unwrap();

    // Type B: Variance = 1 (Covariant), Region = 1 ('a)
    let mut type_b = TypeId::new(0, 0, 0, 0);
    type_b.try_set_fast_param(0, 1, 1).unwrap();

    // 'static (0) can be coerced to 'a (1) for Covariant types
    if !verify_subtyping_bounds(&type_a, &type_b, &worker) {
        return Err("Subtyping bound failed".into());
    }

    // 'a (1) cannot be coerced to 'static (0)
    if verify_subtyping_bounds(&type_b, &type_a, &worker) {
        return Err("Subtyping bound failed".into());
    }

    // Type C: Variance = 0 (Invariant), Region = 0
    let mut type_c = TypeId::new(0, 0, 0, 0);
    type_c.try_set_fast_param(0, 0, 0).unwrap();
    let mut type_d = TypeId::new(0, 0, 0, 0);
    type_d.try_set_fast_param(0, 1, 0).unwrap();

    // Invariant requires EXACT match
    if !verify_subtyping_bounds(&type_c, &type_c, &worker) {
        return Err("Subtyping bound failed".into());
    }
    if verify_subtyping_bounds(&type_c, &type_d, &worker) {
        return Err("Subtyping bound failed".into());
    }

    Ok(())
}

#[test]
fn test_fast_path_contravariant() -> Result<(), String> {
    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let worker = vxc::session::LocalWorkerState::new(global_session.clone());

    // Contravariant (0x2): Target must outlive source (region_a >= region_b)
    // Type A: Region 3, Contravariant
    let mut type_a = TypeId::new(0, 0, 0, 0);
    type_a.try_set_fast_param(0, 3, 2).unwrap();

    // Type B: Region 1, Contravariant
    let mut type_b = TypeId::new(0, 0, 0, 0);
    type_b.try_set_fast_param(0, 1, 2).unwrap();

    // region_a(3) >= region_b(1) → valid for Contravariant
    if !verify_subtyping_bounds(&type_a, &type_b, &worker) {
        return Err("Contravariant: source(3) >= target(1) should pass".into());
    }

    // region_a(1) >= region_b(3) → invalid
    if verify_subtyping_bounds(&type_b, &type_a, &worker) {
        return Err("Contravariant: source(1) >= target(3) should fail".into());
    }

    Ok(())
}

#[test]
fn test_fast_path_mismatched_variance() -> Result<(), String> {
    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let worker = vxc::session::LocalWorkerState::new(global_session.clone());

    // Type A: Covariant (0x1), Region 0
    let mut type_a = TypeId::new(0, 0, 0, 0);
    type_a.try_set_fast_param(0, 0, 1).unwrap();

    // Type B: Invariant (0x0), Region 0
    let mut type_b = TypeId::new(0, 0, 0, 0);
    type_b.try_set_fast_param(0, 0, 0).unwrap();

    // Variance flags differ → must reject
    if verify_subtyping_bounds(&type_a, &type_b, &worker) {
        return Err("Mismatched variance flags should be rejected".into());
    }

    Ok(())
}

#[test]
fn test_fast_path_multi_slot() -> Result<(), String> {
    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let worker = vxc::session::LocalWorkerState::new(global_session.clone());

    // Type A: Slot 0 = Covariant/Region 0, Slot 1 = Covariant/Region 1
    let mut type_a = TypeId::new(0, 0, 0, 0);
    type_a.try_set_fast_param(0, 0, 1).unwrap();
    type_a.try_set_fast_param(1, 1, 1).unwrap();

    // Type B: Slot 0 = Covariant/Region 2, Slot 1 = Covariant/Region 3
    let mut type_b = TypeId::new(0, 0, 0, 0);
    type_b.try_set_fast_param(0, 2, 1).unwrap();
    type_b.try_set_fast_param(1, 3, 1).unwrap();

    // Both slots: source region <= target region → valid
    if !verify_subtyping_bounds(&type_a, &type_b, &worker) {
        return Err("Multi-slot covariant with smaller source regions should pass".into());
    }

    // Reverse: slot 0 region(2) > target(0) → invalid
    if verify_subtyping_bounds(&type_b, &type_a, &worker) {
        return Err("Multi-slot covariant with larger source regions should fail".into());
    }

    // Mixed: one slot passes, one fails
    let mut type_c = TypeId::new(0, 0, 0, 0);
    type_c.try_set_fast_param(0, 0, 1).unwrap(); // slot 0: region 0 (outlives)
    type_c.try_set_fast_param(1, 5, 1).unwrap(); // slot 1: region 5 (shorter)

    // type_c -> type_a: slot 0 OK (0<=0), slot 1 FAIL (5 > 1)
    if verify_subtyping_bounds(&type_c, &type_a, &worker) {
        return Err("Multi-slot with one failing slot should reject".into());
    }

    Ok(())
}
