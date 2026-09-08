//===- cargo-vx-bench.rs - Vx Compiler -------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs the benchmarks named in `benchmarks/manifest.txt` and records what they
// report.
//
// It drives `vxc` rather than compiling anything itself. It used to build its own
// reduced pipeline -- Lexer, parser, GlobalAstEnv, TypeChecker, MeliorGenerator --
// which never loaded stdlib modules, so `import std::time` did not resolve and
// every benchmark failed on an undefined `now`. It then spliced in a harness
// calling `vx_print_float`, a symbol that does not exist, so the two files that
// got past the checker failed to link. All ten failed, the summary printed no
// rows, and the process exited 0.
//
// A benchmark reports its own measurements now (`bench_report` in `std::time`),
// so there is nothing to splice: this runs the program and reads the lines.
//
//===----------------------------------------------------------------------===//

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// One measurement, as a benchmark reported it plus the context it cannot know.
struct Record {
    benchmark: String,
    name: String,
    unit: String,
    value: f64,
}

/// The `vxc` built alongside this binary.
fn vxc_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("vxc")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("vxc"))
}

fn capture(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// What a number has to carry to be comparable with another one: which commit
/// produced it and which machine ran it. No existing schema recorded either, so a
/// figure could not be told from a figure taken on different silicon.
fn provenance() -> (String, String) {
    let commit = capture("git", &["rev-parse", "--short", "HEAD"]);
    let dirty = !capture("git", &["status", "--porcelain"]).is_empty();
    let commit = if dirty {
        format!("{commit}-dirty")
    } else {
        commit
    };
    let machine = format!(
        "{} {}",
        capture("uname", &["-s"]),
        capture("uname", &["-m"])
    );
    (commit, machine)
}

/// Parse `vx-bench <name> <unit> <value>` out of a run's stdout, ignoring the JIT
/// chatter it is mixed into.
fn records_from(benchmark: &str, stdout: &str) -> Vec<Record> {
    stdout
        .lines()
        .filter_map(|l| l.strip_prefix("vx-bench "))
        .filter_map(|rest| {
            let mut f = rest.split_whitespace();
            let (name, unit, value) = (f.next()?, f.next()?, f.next()?);
            Some(Record {
                benchmark: benchmark.to_string(),
                name: name.to_string(),
                unit: unit.to_string(),
                value: value.parse().ok()?,
            })
        })
        .collect()
}

fn manifest_entries(root: &Path) -> Result<Vec<PathBuf>, String> {
    let path = root.join("benchmarks/manifest.txt");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| root.join(l))
        .collect())
}

fn main() -> ExitCode {
    let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let entries = match manifest_entries(&root) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    if entries.is_empty() {
        eprintln!("error: the manifest names no benchmarks");
        return ExitCode::FAILURE;
    }

    let vxc = vxc_path();
    let (commit, machine) = provenance();
    println!("commit {commit}   machine {machine}");
    println!();

    let mut records = Vec::new();
    let mut failed = Vec::new();

    for path in &entries {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .display()
            .to_string();
        if !path.exists() {
            failed.push(format!("{rel}: listed in the manifest but does not exist"));
            continue;
        }
        let out = match Command::new(&vxc).arg(path).arg("--run").output() {
            Ok(o) => o,
            Err(e) => {
                failed.push(format!("{rel}: could not run {}: {e}", vxc.display()));
                continue;
            }
        };
        let stdout = String::from_utf8_lossy(&out.stdout);
        let mine = records_from(&rel, &stdout);
        if !out.status.success() {
            failed.push(format!("{rel}: exited {}", out.status));
        } else if mine.is_empty() {
            // The case the old runner could not see: it ran, it succeeded, and it
            // measured nothing.
            failed.push(format!("{rel}: ran but reported no measurement"));
        }
        records.extend(mine);
    }

    if !records.is_empty() {
        let w = records
            .iter()
            .map(|r| r.name.len())
            .max()
            .unwrap_or(4)
            .max(4);
        println!("{:<w$}  {:>14}  unit", "name", "value", w = w);
        for r in &records {
            println!(
                "{:<w$}  {:>14.9}  {}   ({})",
                r.name,
                r.value,
                r.unit,
                r.benchmark,
                w = w
            );
        }
        println!();
    }

    if failed.is_empty() {
        println!(
            "{} measurements from {} benchmarks",
            records.len(),
            entries.len()
        );
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "{} of {} benchmarks did not report:",
            failed.len(),
            entries.len()
        );
        for f in &failed {
            eprintln!("  {f}");
        }
        ExitCode::FAILURE
    }
}
