//===- compile_time_of_string_literals.rs - Vx Compiler ---------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A function with many string literals generates code in seconds. The code generator used to
// search the whole text of the function once for each string, so the time grew with the
// square of the number of strings.
//===----------------------------------------------------------------------===//

use std::process::Command;
use std::time::{Duration, Instant};

/// A `main` that prints `lines` different string literals.
fn program(lines: usize) -> String {
    let mut src = String::from("fn main() -> i32 {\n");
    for k in 0..lines {
        src.push_str(&format!("  println!(\"line {k}\");\n"));
    }
    src.push_str("  return 0;\n}\n");
    src
}

const LINES: usize = 40_000;

#[test]
fn forty_thousand_string_literals_generate_code_in_seconds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many_string_literals.vx");
    std::fs::write(&path, program(LINES)).unwrap();

    let start = Instant::now();
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg(&path)
        .args(["--action", "emit-mlir"])
        .output()
        .expect("failed to run vxc");
    let took = start.elapsed();

    // A string with no global fails the MLIR verifier, so this also checks that each string has
    // its global.
    assert!(
        out.status.success(),
        "vxc failed on the program:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // About 4 s in a test build. The old search took 78 s. The limit leaves room for a slow,
    // busy machine.
    assert!(
        took < Duration::from_secs(30),
        "{LINES} string literals took {took:?} to compile"
    );
}
