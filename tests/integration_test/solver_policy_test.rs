//===- solver_policy_test.rs - Vx Compiler ---------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A missing SMT solver must never look like a discharged obligation (Vx#374).
//
// This is a regression test for a silence, which is the hard kind to keep. The compiler used to
// answer "z3 could not be started" with the verdict that means *proved*, so a machine without the
// solver compiled programs whose seam contracts were never checked and said nothing about it. Unit
// tests cannot catch that -- the whole failure is that nothing is emitted -- so the check has to
// run the real binary with the solver genuinely unreachable.
//
//===----------------------------------------------------------------------===//

use std::path::{Path, PathBuf};
use std::process::Command;

/// A PATH containing the tools the compiler needs to link, and deliberately not `z3`.
///
/// Built by symlinking the toolchain into a scratch directory rather than by emptying PATH: the
/// point is to isolate the *solver*, and a compiler that failed because it could not find `clang`
/// would pass this test for the wrong reason.
fn path_without_z3(dir: &Path) -> String {
    std::fs::create_dir_all(dir).expect("scratch dir");
    for tool in [
        "clang",
        "clang++",
        "cc",
        "ld",
        "mlir-translate",
        "llvm-config",
    ] {
        if let Ok(p) = which(tool) {
            let link = dir.join(tool);
            let _ = std::fs::remove_file(&link);
            let _ = std::os::unix::fs::symlink(&p, &link);
        }
    }
    assert!(
        which_in(dir, "z3").is_none(),
        "the stripped PATH must not contain z3, or this test proves nothing"
    );
    dir.display().to_string()
}

fn which(tool: &str) -> Result<PathBuf, ()> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {tool}"))
        .output()
        .map_err(|_| ())?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        Err(())
    } else {
        Ok(PathBuf::from(s))
    }
}

fn which_in(dir: &Path, tool: &str) -> Option<PathBuf> {
    let p = dir.join(tool);
    p.exists().then_some(p)
}

/// A topology whose `relaxed` edge has a visibility obligation. Whether it is coherent is exactly
/// what the solver decides, so with no solver the answer is unknown -- not "fine".
const RELAXED: &str = r#"Topology RelaxedTPU {
  memory: Memory::Local_SRAM
  transfer Memory::CPU_DRAM -> Memory::Local_SRAM : 5 relaxed
}

fn main() -> i32 {
  spawn on(Topology::RelaxedTPU) {
    let x = 1;
  }
  return 0;
}
"#;

fn write_program(tmp: &Path) -> PathBuf {
    std::fs::create_dir_all(tmp).expect("tmp");
    let prog = tmp.join("relaxed.vx");
    std::fs::write(&prog, RELAXED).expect("write program");
    prog
}

fn run(prog: &Path, path: &str, allow_unverified: bool) -> (bool, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vxc"));
    cmd.arg(prog)
        .args(["--action", "emit-mlir"])
        .env("PATH", path);
    if allow_unverified {
        cmd.env("VX_ALLOW_UNVERIFIED", "1");
    } else {
        cmd.env_remove("VX_ALLOW_UNVERIFIED");
    }
    let out = cmd.output().expect("run vxc");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

/// With the solver present the obligation is actually decided, and this edge is incoherent, so
/// W1027 fires. Establishes that the fixture exercises the solver at all -- without this, the
/// two cases below could both pass on a program that never had an obligation.
#[test]
fn with_a_solver_the_obligation_is_decided() {
    if which("z3").is_err() {
        eprintln!("skipping: no z3 on this machine");
        return;
    }
    let tmp = std::env::temp_dir().join("vx-solver-policy-present");
    let prog = write_program(&tmp);
    let path = std::env::var("PATH").unwrap_or_default();
    let (ok, text) = run(&prog, &path, false);
    assert!(ok, "compilation should succeed with a solver:\n{text}");
    assert!(
        text.contains("W1027"),
        "the solver should have found this edge incoherent:\n{text}"
    );
}

/// The regression this file exists for. No solver, no opt-out: the build must FAIL, and say why.
/// Before Vx#374 this compiled cleanly and emitted nothing at all.
#[test]
fn without_a_solver_the_build_fails_rather_than_claiming_success() {
    let tmp = std::env::temp_dir().join("vx-solver-policy-missing");
    let path = path_without_z3(&tmp.join("bin"));
    let prog = write_program(&tmp);
    let (ok, text) = run(&prog, &path, false);

    assert!(
        !ok,
        "an undischarged obligation must not compile successfully:\n{text}"
    );
    assert!(
        text.contains("E6024"),
        "the failure must be the undischarged-obligation error:\n{text}"
    );
    assert!(
        text.contains("z3"),
        "the diagnostic must name the missing tool:\n{text}"
    );
    assert!(
        text.contains("VX_ALLOW_UNVERIFIED"),
        "the diagnostic must name the escape hatch:\n{text}"
    );
    assert!(
        !text.contains("W1027"),
        "W1027 means the solver ran and found a violation; it cannot fire with no solver:\n{text}"
    );
}

/// Opting out compiles, but never silently: the obligation is reported as unverified every time.
/// The distinction from the case above is the whole point -- "I accepted this" must not look like
/// "this was proved".
#[test]
fn opting_out_compiles_but_says_so_every_time() {
    let tmp = std::env::temp_dir().join("vx-solver-policy-optout");
    let path = path_without_z3(&tmp.join("bin"));
    let prog = write_program(&tmp);
    let (ok, text) = run(&prog, &path, true);

    assert!(ok, "the opt-out should permit compilation:\n{text}");
    assert!(
        text.contains("W1031"),
        "the opt-out must still report the obligation as unverified:\n{text}"
    );
    assert!(
        text.contains("NOT verified"),
        "the wording must not be mistakable for a proof:\n{text}"
    );
    assert!(
        !text.contains("E6024"),
        "the opt-out downgrades the error, it does not keep it:\n{text}"
    );
}
