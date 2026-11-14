//===- collective_test.rs - Vx Compiler -------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/collective_test.cpp`, which checks the
//! all-reduce and all-gather of a job of several GPUs against a fake NCCL.

use std::path::PathBuf;
use std::process::Command;

use crate::integration_test::runtime_test_build;

fn compile(cxx: &str, args: &[&str]) {
    let build = runtime_test_build::command(cxx, "c++17")
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {cxx}: {e}"));
    assert!(
        build.status.success(),
        "{cxx} {args:?} failed:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
}

#[test]
fn collectives_pass_their_arguments_to_nccl() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("collective_test");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(out.join("ids")).unwrap();
    let cxx = runtime_test_build::cxx();
    let fake = out.join("libfake_nccl.so");
    let binary = out.join("collective_test");
    let path = |p: &PathBuf| p.to_str().unwrap().to_string();

    compile(
        &cxx,
        &[
            "-shared",
            "-fPIC",
            &path(&root.join("tests/runtime/fake_nccl.cpp")),
            "-o",
            &path(&fake),
        ],
    );
    compile(
        &cxx,
        &[
            "-pthread",
            &path(&root.join("tests/runtime/collective_test.cpp")),
            "-ldl",
            "-o",
            &path(&binary),
        ],
    );

    let run = Command::new(&binary)
        .arg(&fake)
        .arg(out.join("ids"))
        .output()
        .expect("failed to run the compiled collective test");
    assert!(
        run.status.success(),
        "collective checks failed:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
