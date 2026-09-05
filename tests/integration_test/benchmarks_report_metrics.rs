//===- benchmarks_report_metrics.rs - Vx Compiler ---------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Every benchmark the manifest names still runs and still reports a measurement.
//
// What is gated is that a number arrives, never what the number is: timings on a
// shared runner are noise, and a gate that fails on noise teaches people to ignore
// it. Compiling is gated separately, by shipped_programs_compile.rs -- a benchmark
// can compile perfectly and measure nothing, which is the state the whole directory
// was in. `benchmarks/run_benchmarks.sh` carried the comment "Run and time the
// execution" above code that did no timing and piped every run to /dev/null, and
// `cargo vx-bench` reported all ten as FAILED, printed no rows, and exited 0.
//
//===----------------------------------------------------------------------===//

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The benchmarks named in `benchmarks/manifest.txt`.
fn manifest() -> Vec<String> {
    let path = repo_root().join("benchmarks/manifest.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

#[test]
fn every_manifest_benchmark_reports_a_measurement() {
    let root = repo_root();
    let entries = manifest();
    assert!(
        entries.len() >= 4,
        "the manifest names {} benchmarks; has it been emptied?",
        entries.len()
    );

    let mut silent = Vec::new();
    for rel in &entries {
        let path = root.join(rel);
        if !path.exists() {
            silent.push(format!("  {rel}: named in the manifest but does not exist"));
            continue;
        }
        let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
            .current_dir(&root)
            .arg(&path)
            .arg("--run")
            .output()
            .expect("failed to run vxc");
        let stdout = String::from_utf8_lossy(&out.stdout);

        // `vx-bench <name> <unit> <value>`: three fields, and the value must parse
        // as a number. A line that merely starts with the prefix is not a
        // measurement -- `println!` emits its format string literally (Vx#450), so
        // a report written that way would produce a prefix with `{}` after it.
        let measurements: Vec<&str> = stdout
            .lines()
            .filter_map(|l| l.strip_prefix("vx-bench "))
            .filter(|rest| {
                let f: Vec<&str> = rest.split_whitespace().collect();
                f.len() == 3 && f[2].parse::<f64>().is_ok()
            })
            .collect();

        if !out.status.success() {
            silent.push(format!("  {rel}: exited {}", out.status));
        } else if measurements.is_empty() {
            silent.push(format!(
                "  {rel}: ran and reported nothing parseable{}",
                match stdout.lines().find(|l| l.starts_with("vx-bench ")) {
                    Some(l) => format!(" (closest line: {l:?})"),
                    None => String::new(),
                }
            ));
        }
    }

    assert!(
        silent.is_empty(),
        "benchmarks that no longer report a measurement:\n{}",
        silent.join("\n")
    );
}
