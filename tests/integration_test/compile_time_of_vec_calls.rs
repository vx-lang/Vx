//===- compile_time_of_vec_calls.rs - Vx Compiler ---------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A program with many calls of a `Vec` method type-checks in seconds. Each call used to
// copy impl blocks, with their method bodies, so 3000 calls took 55 s in a debug build.
//===----------------------------------------------------------------------===//

use std::process::Command;
use std::time::{Duration, Instant};

/// A `main` that calls `Vec::get` `calls` times.
fn program(calls: usize) -> String {
    let mut src = String::from(
        "import std::vec;\n\
         fn main() -> i32 {\n  \
           let mut v = Vec<i32>::with_capacity(8);\n  \
           for i in 0..8 { v.push(i); }\n  \
           let mut t = 0;\n",
    );
    for k in 0..calls {
        src.push_str(&format!("  t += v.get({});\n", k % 8));
    }
    src.push_str("  return t;\n}\n");
    src
}

#[test]
fn three_thousand_vec_get_calls_type_check_in_seconds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many_vec_get_calls.vx");
    std::fs::write(&path, program(3000)).unwrap();

    let start = Instant::now();
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg(&path)
        .args(["--action", "print-ast"])
        .output()
        .expect("failed to run vxc");
    let took = start.elapsed();

    assert!(
        out.status.success(),
        "vxc failed on the program:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // About 0.3 s in a debug build now. The limit leaves room for a slow machine.
    assert!(
        took < Duration::from_secs(20),
        "3000 calls of Vec::get took {took:?} to type-check"
    );
}
