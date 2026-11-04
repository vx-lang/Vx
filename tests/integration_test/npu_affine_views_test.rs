//===- npu_affine_views_test.rs - Vx Compiler --------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds `tests/runtime/npu_affine_views_test.cpp` with the Mac dispatcher and
//! runs it. It checks that the CoreML model `affine_4` gets only the views it
//! can read, and that every view gives the right elements.

use std::path::PathBuf;
use std::process::Command;

use crate::integration_test::runtime_test_build;

#[test]
fn an_affine_kernel_on_a_view_gives_the_right_elements() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let models = root.join("ane-primitives");
    // Without the model, every case runs on the CPU, so the test cannot check
    // which views the model gets.
    let model = models.join("affine_4.mlmodelc");
    if !model.is_dir() {
        eprintln!("skipping: no {} was built", model.display());
        return;
    }

    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("npu_affine_views_test");
    let cxx = runtime_test_build::cxx();
    let build = runtime_test_build::command(&cxx, "c++17")
        .current_dir(&root)
        // build.rs allows the deprecated CBLAS calls in npu_dispatch.mm too.
        .args(["-Wno-deprecated-declarations", "-fobjc-arc"])
        .arg("-DVX_ENABLE_CPU_FALLBACK")
        .args([
            "runtime/npu_dispatch.mm",
            "tests/runtime/npu_affine_views_test.cpp",
        ])
        .args(["-framework", "Accelerate", "-framework", "Foundation"])
        .args([
            "-framework",
            "Metal",
            "-framework",
            "MetalPerformanceShaders",
        ])
        .args(["-framework", "CoreML", "-lffi", "-o"])
        .arg(&binary)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {cxx}: {e}"));
    assert!(
        build.status.success(),
        "compiling the affine view test failed:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );

    let run = Command::new(&binary)
        .env("VX_ANE_MODEL_DIR", &models)
        .output()
        .expect("failed to run the compiled affine view test");
    assert!(
        run.status.success(),
        "an affine kernel on a view gave the wrong elements:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
