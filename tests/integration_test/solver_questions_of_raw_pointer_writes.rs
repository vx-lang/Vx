//===- solver_questions_of_raw_pointer_writes.rs - Vx Compiler --*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Each write through a raw pointer gives the compiler a fact, and the compiler asks the solver
// whether the fact joins an earlier one. The number of questions must grow linearly with the
// number of writes. It grew with the square: 64 writes asked 16,128 questions and took minutes.
// These tests run vxc with a `z3` first on PATH that counts its runs.
//===----------------------------------------------------------------------===//

#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::process::Command;

const WRITES: usize = 64;

/// Compiles `src` and returns how many times vxc started the solver.
fn solver_runs(src: &str) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let count = dir.path().join("count");
    let z3 = String::from_utf8(
        Command::new("sh")
            .args(["-c", "command -v z3"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();
    let z3 = z3.trim();
    assert!(!z3.is_empty(), "these tests need z3 on PATH");
    let wrapper = dir.path().join("z3");
    std::fs::write(
        &wrapper,
        format!(
            "#!/bin/sh\necho x >> '{}'\nexec '{z3}' \"$@\"\n",
            count.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
    let program = dir.path().join("writes.vx");
    std::fs::write(&program, src).unwrap();

    let path = format!(
        "{}:{}",
        dir.path().display(),
        std::env::var("PATH").unwrap()
    );
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("PATH", path)
        .arg(&program)
        .args(["--action", "emit-mlir"])
        .output()
        .expect("failed to run vxc");
    assert!(
        out.status.success(),
        "vxc failed on the program:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read_to_string(&count).map_or(0, |s| s.lines().count())
}

/// The writes `out[(token + i) * rows + r]`. The compiler cannot tell whether two of them touch,
/// because `rows` is not known, so only the solver can answer.
#[test]
fn writes_at_unknown_distances_ask_few_questions() {
    let mut src = String::from(
        "pub fn stores(out : *mut f32, rows : i32, token : i32) -> i32 {\n  unsafe {\n    for r in 0..rows {\n",
    );
    for i in 0..WRITES {
        src.push_str(&format!("      out[(token + {i}) * rows + r] = 0.0;\n"));
    }
    src.push_str("    }\n  }\n  return 0;\n}\nfn main() -> i32 {\n  return 0;\n}\n");
    let runs = solver_runs(&src);
    assert!(
        runs <= 10 * WRITES,
        "{WRITES} writes started the solver {runs} times"
    );
}

/// The writes `p[0]` to `p[63]`, then a call that needs all 64 elements. The compiler sees from
/// the indexes alone that each write touches the one before, so it needs no solver.
#[test]
fn writes_next_to_each_other_need_no_solver() {
    let mut src = String::from(
        "import core::ptr;\n\
         fn first(p : *const i64, n : i64) -> i64\n\
         requires valid(p, n)\n\
         requires initialized(p, n)\n\
         requires 1i64 <= n\n\
         {\n  return unsafe { p[0] };\n}\n\
         fn write_all(p : *mut i64) -> i64\n",
    );
    src.push_str(&format!(
        "requires valid(p, {WRITES}i64)\n{{\n  unsafe {{\n"
    ));
    for i in 0..WRITES {
        src.push_str(&format!("    p[{i}] = 1i64;\n"));
    }
    src.push_str(&format!(
        "  }}\n  return first(p as *const i64, {WRITES}i64);\n}}\nfn main() -> i32 {{\n  return 0;\n}}\n"
    ));
    let runs = solver_runs(&src);
    assert!(
        runs <= 4,
        "{WRITES} writes next to each other started the solver {runs} times"
    );
}
