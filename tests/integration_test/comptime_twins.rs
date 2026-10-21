//===- comptime_twins.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A `comptime` block must fold to the value that the same code computes when the program runs.
// Each program in `tests/comptime_twins/` runs twice: as written, with its blocks folded while
// compiling, and with each `comptime {` changed to `unsafe {`, so that the same code runs at
// run time. Both runs must print the same lines, give the same errors, and end the same way.
//
//===----------------------------------------------------------------------===//

use std::path::{Path, PathBuf};
use std::process::Command;

/// What a run printed, without the JIT's progress lines, its errors, and how vxc ended.
fn run(root: &Path, program: &Path) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .current_dir(root)
        .arg(program)
        .output()
        .expect("failed to run vxc");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let printed: Vec<&str> = stdout.lines().filter(|l| !l.starts_with("[JIT]")).collect();
    let errors: Vec<&str> = stdout
        .lines()
        .chain(stderr.lines())
        .filter(|l| l.starts_with("Error") || l.contains("exited with non-zero code"))
        .collect();
    format!(
        "printed:\n{}\nerrors: {:?}\nvxc exit: {:?}",
        printed.join("\n"),
        errors,
        out.status.code()
    )
}

#[test]
fn comptime_blocks_print_what_their_run_time_twins_print() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let twins = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("comptime_twins");
    std::fs::create_dir_all(&twins).expect("cannot make the directory for the twins");
    let mut programs: Vec<PathBuf> = std::fs::read_dir(root.join("tests/comptime_twins"))
        .expect("tests/comptime_twins is missing")
        .map(|entry| entry.expect("cannot read tests/comptime_twins").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "vx"))
        .collect();
    programs.sort();
    assert!(
        programs.len() >= 6,
        "expected the six programs, found {}",
        programs.len()
    );

    let mut differ = Vec::new();
    for program in &programs {
        let source = std::fs::read_to_string(program).expect("cannot read a program");
        assert!(
            source.contains("comptime {"),
            "{} has no `comptime` block to compare",
            program.display()
        );
        let twin = twins.join(program.file_name().expect("a program has a file name"));
        std::fs::write(&twin, source.replace("comptime {", "unsafe {"))
            .expect("cannot write the twin");
        let folded = run(&root, program);
        let at_run_time = run(&root, &twin);
        if folded != at_run_time {
            differ.push(format!(
                "{}\n--- with comptime blocks folded:\n{}\n--- run at run time:\n{}",
                program.display(),
                folded,
                at_run_time
            ));
        }
    }
    assert!(
        differ.is_empty(),
        "a comptime block folded to a different result than the program computes:\n\n{}",
        differ.join("\n\n")
    );
}
