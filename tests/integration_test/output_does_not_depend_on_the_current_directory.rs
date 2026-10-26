//===- output_does_not_depend_on_the_current_directory.rs -------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `vxc` reads only the files its command line names, so one command gives the same output
// wherever it runs. Before, `vxc` read the standard library and imports from the current
// directory, and failed outside the repository root.
//
//===----------------------------------------------------------------------===//

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn empty_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vx_cwd_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Compile `program` to MLIR with `vxc` running in `cwd`. Returns whether it succeeded, and
/// everything it printed.
fn emit_mlir(cwd: &Path, program: &Path, flags: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .current_dir(cwd)
        .arg(program)
        .args(flags)
        .args(["--action", "emit-mlir"])
        .output()
        .expect("failed to run vxc");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), log)
}

/// A program that imports `std::vec` compiles to the same MLIR from an empty directory.
#[test]
fn a_program_that_imports_the_standard_library() {
    let program = repo_root().join("tests/backend/pass/bare_return.vx");
    let from_the_root = emit_mlir(&repo_root(), &program, &[]);
    assert!(
        from_the_root.0,
        "the program should compile: {}",
        from_the_root.1
    );
    let from_elsewhere = emit_mlir(&empty_dir("std"), &program, &[]);
    assert_eq!(
        from_the_root, from_elsewhere,
        "the output depends on the current directory"
    );
}

/// A program imports a file beside it while `vxc` runs in another directory.
#[test]
fn a_program_that_imports_a_file_beside_it() {
    let dir = empty_dir("beside");
    std::fs::write(dir.join("helper.vx"), "fn seven() -> i32 { return 7; }\n").unwrap();
    let main = dir.join("main.vx");
    std::fs::write(
        &main,
        "import helper;\nfn main() -> i32 { return seven(); }\n",
    )
    .unwrap();
    let from_its_dir = emit_mlir(&dir, &main, &[]);
    assert!(
        from_its_dir.0,
        "the program should compile: {}",
        from_its_dir.1
    );
    let from_elsewhere = emit_mlir(&empty_dir("beside_cwd"), &main, &[]);
    assert_eq!(
        from_its_dir, from_elsewhere,
        "the output depends on the current directory"
    );
}

/// `--std-path` replaces the standard library, so with an empty directory `std::vec` is missing.
#[test]
fn std_path_replaces_the_standard_library() {
    let program = repo_root().join("tests/backend/pass/bare_return.vx");
    let empty = empty_dir("std_path");
    let (ok, log) = emit_mlir(
        &repo_root(),
        &program,
        &["--std-path", empty.to_str().unwrap()],
    );
    assert!(!ok, "the program compiled without a standard library");
    assert!(
        log.contains("Could not resolve import 'std::vec'"),
        "unexpected output: {log}"
    );
}

/// `vxc` no longer reads `VX_STD_PATH`. Pointed at an empty directory, it changes nothing.
#[test]
fn the_vx_std_path_variable_is_not_read() {
    let program = repo_root().join("tests/backend/pass/bare_return.vx");
    let empty = empty_dir("vx_std_path");
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&program)
        .args(["--action", "emit-mlir"])
        .env("VX_STD_PATH", &empty)
        .output()
        .expect("failed to run vxc");
    assert!(
        out.status.success(),
        "VX_STD_PATH still changes where the standard library is: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
