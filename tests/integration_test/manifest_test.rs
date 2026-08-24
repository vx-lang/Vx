//===- manifest_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/manifest_test.cpp`, which turns the
//! `toponame=DecodeWorker` a dispatch carries into a machine (#348).
//!
//! The rule under test is what the location-transparency claim rests on: a name
//! that is absent means *local*. A program with no manifest runs entirely on
//! this machine, and adding an entry is what distributes it, which is why
//! `llama2.vx` is the same text either way.
//!
//! That rule is also why a malformed line is refused rather than skipped.
//! Skipping one leaves a worker silently local -- the program runs, the numbers
//! are right, and the distribution it was meant to demonstrate did not happen,
//! with no error anywhere to notice afterwards.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn a_worker_name_resolves_to_a_machine() {
    let root = repo_root();
    let source = root.join("tests/runtime/manifest_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("manifest_test");
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "clang++".to_string());

    let build = Command::new(&cxx)
        .args(["-std=c++17", "-O1", "-Wall", "-Werror"])
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {cxx}: {e}"));

    assert!(
        build.status.success(),
        "compiling {} failed:\n{}",
        source.display(),
        String::from_utf8_lossy(&build.stderr)
    );

    let run = Command::new(&binary)
        .output()
        .expect("failed to run the compiled region test");

    assert!(
        run.status.success(),
        "remote region checks failed:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

/// The C mirror of `fnv_dispatch_id` must agree with the compiler for *any* name, not only the
/// handful `manifest_test.cpp` pins as constants.
///
/// Those constants are hand-maintained, which is the problem: widening the id range meant
/// recomputing six of them by hand, and a wrong recomputation would have looked exactly like a
/// passing test. Here the two implementations are run against the same names and compared, so a
/// divergence fails whatever the names are.
#[test]
fn the_c_mirror_of_the_dispatch_hash_agrees_with_the_compiler() {
    let root = repo_root();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let source = dir.join("dispatch_id_mirror.c");
    let binary = dir.join("dispatch_id_mirror");

    // Prints the mirror's answer for each name given on the command line.
    std::fs::write(
        &source,
        format!(
            r#"#include "{}"
#include <stdio.h>
int main(int argc, char **argv) {{
  for (int i = 1; i < argc; ++i)
    printf("%d\n", vx_manifest_dispatch_id(argv[i]));
  return 0;
}}
"#,
            root.join("runtime/vx_manifest.h").display()
        ),
    )
    .expect("write the mirror probe");

    // Spread over lengths, cases and separators, plus the empty name and the spellings a real
    // machine file uses. None of these are built-in topology names, so every one takes the hash.
    let mut names: Vec<String> = vec![
        String::new(),
        "A".into(),
        "DecodeWorker".into(),
        "PrefillWorker".into(),
        "AcmeCore".into(),
        "MyTPU".into(),
        "L2_GPU5".into(),
        "GPU6_VMEM".into(),
        "DO2TX".into(),
        "DSC0A".into(),
        "a_very_long_declared_topology_name_that_keeps_going".into(),
    ];
    for i in 0..64 {
        names.push(format!("H{i}"));
        names.push(format!("SMEM_dev{i}"));
    }

    let cxx = std::env::var("CC").unwrap_or_else(|_| "clang".to_string());
    let build = Command::new(&cxx)
        .args(["-std=c11", "-O1", "-Wall", "-Werror"])
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {cxx}: {e}"));
    assert!(
        build.status.success(),
        "compiling the mirror probe failed:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );

    let run = Command::new(&binary)
        .args(&names)
        .output()
        .expect("failed to run the mirror probe");
    assert!(run.status.success(), "the mirror probe did not run");

    let from_c: Vec<i32> = String::from_utf8_lossy(&run.stdout)
        .lines()
        .map(|l| l.trim().parse().expect("an integer per name"))
        .collect();
    assert_eq!(from_c.len(), names.len(), "one id per name");

    for (name, c_id) in names.iter().zip(&from_c) {
        let rust_id = vxc::arch::topology_dispatch_id(&vxc::syntax::Topology::Custom(
            vxc::symbol::Symbol::from(name.as_str()),
        ));
        assert_eq!(
            rust_id, *c_id,
            "src/arch.rs and runtime/vx_manifest.h disagree on '{name}'"
        );
        assert!(
            rust_id >= vxc::arch::CUSTOM_DISPATCH_ID_BASE,
            "'{name}' landed below the declared range at {rust_id}"
        );
    }
}
