//===- shipped_programs_compile.rs - Vx Compiler ----------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Every `.vx` the repository ships outside its test tiers still compiles.
//
// `examples/` and `benchmarks/` were reachable from no test and no CI job. Nothing
// enumerated them, so they rotted quietly against a language that moved: the retired
// `Tensor_f32(n, m)` and `.with_memory(..)` spellings, a `Memory::Host_DRAM` that no
// longer exists, and two implicit int-to-float conversions from before the numeric
// model required a cast. Six of thirteen files did not compile, including
// `examples/llama.vx` -- the only file in `examples/`, and the one the README's repo
// tour points a new reader at.
//
// The rot is also not new: `benchmarks/llama2_100.vx` was repaired once before, in
// August, and had broken again by the time this test was written. A directory nothing
// walks is a directory that breaks.
//
//===----------------------------------------------------------------------===//

use std::path::{Path, PathBuf};
use std::process::Command;

/// Files that do not compile yet, each with the reason.
///
/// This list is checked in BOTH directions: a file on it that starts compiling fails the
/// test, so the list cannot quietly become a permanent exemption. Removing an entry is
/// part of fixing the thing it names.
const KNOWN_BROKEN: &[(&str, &str)] = &[];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.vx` under `dir`, recursively, sorted so failures report in a stable order.
fn vx_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            vx_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "vx") {
            out.push(path);
        }
    }
}

/// Compile one file as far as MLIR. Returns the combined output when it failed.
fn compile_failure(root: &Path, file: &Path) -> Option<String> {
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .current_dir(root)
        .args([
            file.to_str().unwrap(),
            "--action",
            "emit-mlir",
            "-o",
            "/dev/null",
        ])
        .output()
        .expect("failed to run vxc");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // An internal error exits non-zero; a semantic error is reported on stdout and, on some
    // actions, still exits 0 -- so the verdict reads the diagnostics, not only the status.
    let failed = !out.status.success()
        || log
            .lines()
            .any(|l| l.starts_with("Error") || l.contains("Internal Error"));
    failed.then_some(log)
}

#[test]
fn every_shipped_example_and_benchmark_compiles() {
    let root = repo_root();
    let mut files = Vec::new();
    vx_files(&root.join("examples"), &mut files);
    vx_files(&root.join("benchmarks"), &mut files);
    assert!(
        files.len() >= 10,
        "expected to find the shipped .vx corpus, found {} files -- has the walk broken?",
        files.len()
    );

    let mut unexpected_failures = Vec::new();
    let mut unexpected_successes = Vec::new();

    for file in &files {
        let rel = file
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let known = KNOWN_BROKEN.iter().find(|(p, _)| *p == rel);
        match (compile_failure(&root, file), known) {
            (Some(log), None) => {
                let first = log
                    .lines()
                    .find(|l| l.starts_with("Error") || l.contains("Internal Error"))
                    .unwrap_or("(no diagnostic; non-zero exit)");
                unexpected_failures.push(format!("  {rel}\n      {first}"));
            }
            (None, Some((_, why))) => {
                unexpected_successes.push(format!("  {rel}\n      was listed as broken: {why}"))
            }
            _ => {}
        }
    }

    assert!(
        unexpected_failures.is_empty(),
        "shipped programs that no longer compile:\n{}",
        unexpected_failures.join("\n")
    );
    assert!(
        unexpected_successes.is_empty(),
        "these compile now -- remove them from KNOWN_BROKEN so the list keeps meaning \
         something:\n{}",
        unexpected_successes.join("\n")
    );
}
