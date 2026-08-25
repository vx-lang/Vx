//===- codegen_determinism.rs - Vx Compiler -------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
// Compiling one program twice must give byte-identical MLIR. Both backends used to emit
// functions in whatever order a HashMap happened to iterate -- the flat path through the
// driver's lowering worklist, the AST path through `generate_module` -- and a HashMap's order
// depends on a per-process random seed. Nothing failed, because either order is valid MLIR,
// so the property is checked here directly (Vx#389).
//===----------------------------------------------------------------------===//
use std::path::{Path, PathBuf};
use std::process::Command;

fn corpus_programs(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    collect(dir, &mut found);
    found.sort();
    found
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read corpus directory {}: {e}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "vx") {
            out.push(path);
        }
    }
}

/// Compile to MLIR and return it. `--action emit-mlir` stops before the JIT, so this needs no
/// accelerator. `None` for a program that does not reach MLIR at all -- that is the corpus
/// sweep's business, not this test's.
fn emit_mlir(program: &Path) -> Option<String> {
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(program)
        .args(["--action", "emit-mlir"])
        .output()
        .unwrap_or_else(|e| panic!("could not run vxc: {e}"));
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Each run is a fresh process, so each gets its own hash seed -- which is exactly what used to
/// reorder the output. Two runs caught every case; a third is cheap insurance against a seed
/// that happens to agree with the first.
const RUNS: usize = 3;

#[test]
fn the_same_program_always_compiles_to_the_same_mlir() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/backend/pass");
    let mut unstable = Vec::new();
    let mut checked = 0usize;

    for program in corpus_programs(&root) {
        let name = program
            .strip_prefix(&root)
            .unwrap_or(&program)
            .to_string_lossy()
            .to_string();
        let Some(first) = emit_mlir(&program) else {
            continue;
        };
        checked += 1;
        for run in 2..=RUNS {
            let Some(again) = emit_mlir(&program) else {
                continue;
            };
            if again != first {
                let (a, b) = first_difference(&first, &again);
                unstable.push(format!(
                    "{name}: run {run} differs\n    run 1: {a}\n    run {run}: {b}"
                ));
                break;
            }
        }
    }

    assert!(
        checked > 0,
        "no corpus program reached MLIR -- this test checked nothing"
    );
    assert!(
        unstable.is_empty(),
        "these programs compile to different MLIR on different runs, so the compiler's \
         output depends on something other than its input:\n  {}",
        unstable.join("\n  ")
    );
}

/// The first line that differs, so a failure names what moved instead of dumping two modules.
fn first_difference(a: &str, b: &str) -> (String, String) {
    for (la, lb) in a.lines().zip(b.lines()) {
        if la != lb {
            return (la.trim().to_string(), lb.trim().to_string());
        }
    }
    (
        format!("{} lines", a.lines().count()),
        format!("{} lines", b.lines().count()),
    )
}
