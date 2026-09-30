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
// `cargo vx-bench compare` runs the suite in `benchmarks/stdlib` instead, where each
// benchmark is written in Vx and in C++; see `benchmarks/stdlib/README.md`.
//
//===----------------------------------------------------------------------===//

use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

#[derive(Parser)]
#[command(name = "cargo-vx-bench", about = "Runs the Vx benchmarks")]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Runs benchmarks written in Vx and in C++, and compares them.
    Compare(CompareArgs),
}

#[derive(Args)]
struct CompareArgs {
    /// Benchmarks to run, by directory name. All of them when none are named.
    names: Vec<String>,
    /// The directory holding one directory per benchmark.
    #[arg(long, default_value = "benchmarks/stdlib")]
    suite: PathBuf,
    /// How many times each program runs. Runs of different programs alternate.
    #[arg(long, default_value_t = 10)]
    rounds: usize,
    /// Run every program on this CPU only (Linux, with `taskset`).
    #[arg(long)]
    cpu: Option<usize>,
    /// The C++ compilers to build with.
    #[arg(long, value_delimiter = ',', default_value = "clang++,g++")]
    cxx: Vec<String>,
    /// Build the C++ only without `-ffast-math`, not also with it.
    #[arg(long)]
    no_fast_math: bool,
    /// The largest relative difference between two answers to one quantity that is not
    /// reported.
    #[arg(long, default_value_t = 1e-6)]
    tolerance: f64,
    /// Also write the results to this file, as JSON.
    #[arg(long)]
    json: Option<PathBuf>,
    /// The `vxc` to compile with. The one beside this binary by default.
    #[arg(long)]
    vxc: Option<PathBuf>,
}

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
    let cli = Cli::parse();
    match cli.command {
        Some(Cmd::Compare(args)) => compare::run(&args),
        None => run_manifest(),
    }
}

fn run_manifest() -> ExitCode {
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

/// `cargo vx-bench compare`: build each benchmark in Vx and in C++, run the programs in
/// alternating rounds, and compare what they report.
mod compare {
    use super::{capture, provenance, records_from, vxc_path, CompareArgs, Record};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::process::{Command, ExitCode};

    /// One program to run: a benchmark built one way.
    struct Build {
        benchmark: String,
        /// `vx`, or the C++ compiler and its extra flag.
        label: String,
        exe: PathBuf,
    }

    /// Everything one build reported for one `quantity/implementation`, over all rounds.
    #[derive(Default)]
    struct Samples {
        seconds: Vec<f64>,
        results: Vec<f64>,
    }

    fn median(v: &[f64]) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        let n = s.len();
        if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        }
    }

    fn run_to_completion(mut cmd: Command, what: &str) -> Result<String, String> {
        let out = cmd
            .output()
            .map_err(|e| format!("could not start {what}: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "{what} failed ({}):\n{}{}",
                out.status,
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// Compiles `src` with `vxc --action emit-obj -O3` and links it the way `--run` does.
    ///
    /// Not with `--run`: that path compiles for a generic CPU, which made the same program
    /// about a quarter slower than this one.
    fn build_vx(vxc: &Path, src: &Path, out_dir: &Path, stem: &str) -> Result<PathBuf, String> {
        let obj = out_dir.join(format!("{stem}.o"));
        let exe = out_dir.join(format!("{stem}-vx"));
        let mut cmd = Command::new(vxc);
        cmd.arg(src)
            .args(["--action", "emit-obj", "-O3", "-o"])
            .arg(&obj);
        run_to_completion(cmd, &format!("vxc on {}", src.display()))?;
        if !obj.exists() {
            return Err(format!("vxc wrote no object for {}", src.display()));
        }

        let libs = vxc::jit::shared_library_paths()?;
        let dispatch = std::env::var("VX_DISPATCH_LIB")
            .unwrap_or_else(|_| std::env!("NPU_SHARED_LIB_PATH").to_string());
        let clang = std::env::var("CLANG_PATH").unwrap_or_else(|_| "clang".to_string());
        let mut cmd = Command::new(&clang);
        cmd.arg(&obj).arg("-o").arg(&exe);
        for lib in &libs {
            if let Some(dir) = Path::new(lib).parent() {
                cmd.arg(format!("-Wl,-rpath,{}", dir.display()));
            }
            cmd.arg(lib);
        }
        // The dispatch runtime needs libffi, and finds outlined kernels in the executable's
        // own symbol table, as in the `--run` link.
        if !dispatch.is_empty() && libs.contains(&dispatch) {
            cmd.arg("-lffi");
            if cfg!(target_os = "linux") {
                cmd.arg("-rdynamic");
            }
        }
        cmd.arg("-lm");
        run_to_completion(cmd, &format!("{clang} linking {}", obj.display()))?;
        Ok(exe)
    }

    fn build_cxx(
        compiler: &str,
        fast_math: bool,
        src: &Path,
        out_dir: &Path,
        stem: &str,
    ) -> Result<PathBuf, String> {
        let tag = compiler.replace(['+', '/'], "p") + if fast_math { "-fast" } else { "" };
        let exe = out_dir.join(format!("{stem}-{tag}"));
        let mut cmd = Command::new(compiler);
        cmd.args(["-O3", "-march=native", "-std=c++20"]);
        if fast_math {
            cmd.arg("-ffast-math");
        }
        cmd.arg(src).arg("-o").arg(&exe);
        run_to_completion(cmd, &format!("{compiler} on {}", src.display()))?;
        Ok(exe)
    }

    /// The benchmarks in the suite, as (name, directory), sorted by name.
    fn benchmarks(suite: &Path, wanted: &[String]) -> Result<Vec<(String, PathBuf)>, String> {
        let mut found = Vec::new();
        let entries = std::fs::read_dir(suite)
            .map_err(|e| format!("cannot read {}: {e}", suite.display()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if path.is_dir() && (wanted.is_empty() || wanted.contains(&name)) {
                found.push((name, path));
            }
        }
        found.sort();
        for w in wanted {
            if !found.iter().any(|(n, _)| n == w) {
                return Err(format!("no benchmark named {w} in {}", suite.display()));
            }
        }
        Ok(found)
    }

    fn json_str(s: &str) -> String {
        let mut out = String::from("\"");
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    pub fn run(args: &CompareArgs) -> ExitCode {
        let root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let suite = root.join(&args.suite);
        let list = match benchmarks(&suite, &args.names) {
            Ok(l) if !l.is_empty() => l,
            Ok(_) => {
                eprintln!("error: {} holds no benchmarks", suite.display());
                return ExitCode::FAILURE;
            }
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        let vxc = args.vxc.clone().unwrap_or_else(vxc_path);
        let (commit, machine) = provenance();
        let cpu_model = capture(
            "sh",
            &["-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
        );
        println!("commit {commit}   machine {machine}   cpu {cpu_model}");

        // Build everything first, so a round measures only running.
        let mut failures: Vec<String> = Vec::new();
        let mut builds: Vec<Build> = Vec::new();
        for (name, dir) in &list {
            let out_dir = root.join("target/vx-bench").join(name);
            if let Err(e) = std::fs::create_dir_all(&out_dir) {
                failures.push(format!("{name}: cannot create {}: {e}", out_dir.display()));
                continue;
            }
            let vx_src = dir.join(format!("{name}.vx"));
            if vx_src.exists() {
                match build_vx(&vxc, &vx_src, &out_dir, name) {
                    Ok(exe) => builds.push(Build {
                        benchmark: name.clone(),
                        label: "vx".into(),
                        exe,
                    }),
                    Err(e) => failures.push(format!("{name} (vx): {e}")),
                }
            }
            let cxx_src = dir.join(format!("{name}.cpp"));
            if cxx_src.exists() {
                for compiler in &args.cxx {
                    for fast in [false, true] {
                        if fast && args.no_fast_math {
                            continue;
                        }
                        let label = if fast {
                            format!("{compiler} -ffast-math")
                        } else {
                            compiler.clone()
                        };
                        match build_cxx(compiler, fast, &cxx_src, &out_dir, name) {
                            Ok(exe) => builds.push(Build {
                                benchmark: name.clone(),
                                label,
                                exe,
                            }),
                            Err(e) => failures.push(format!("{name} ({label}): {e}")),
                        }
                    }
                }
            }
        }
        println!(
            "{} programs, {} rounds{}",
            builds.len(),
            args.rounds,
            args.cpu
                .map(|c| format!(", on cpu {c}"))
                .unwrap_or_default()
        );
        println!();

        // (benchmark, label, record name) -> samples. Rounds alternate between programs, so
        // a slow stretch of the machine is spread over all of them rather than one.
        let mut samples: BTreeMap<(String, String, String), Samples> = BTreeMap::new();
        let mut broken = vec![false; builds.len()];
        for _ in 0..args.rounds {
            for (i, b) in builds.iter().enumerate() {
                if broken[i] {
                    continue;
                }
                let mut cmd = match args.cpu {
                    Some(c) => {
                        let mut t = Command::new("taskset");
                        t.arg("-c").arg(c.to_string()).arg(&b.exe);
                        t
                    }
                    None => Command::new(&b.exe),
                };
                cmd.current_dir(&root);
                let what = format!("{} ({})", b.benchmark, b.label);
                let stdout = match run_to_completion(cmd, &what) {
                    Ok(s) => s,
                    Err(e) => {
                        failures.push(e);
                        broken[i] = true;
                        continue;
                    }
                };
                let records: Vec<Record> = records_from(&b.benchmark, &stdout);
                if records.is_empty() {
                    failures.push(format!("{what}: ran but reported no measurement"));
                    broken[i] = true;
                    continue;
                }
                for r in records {
                    let s = samples
                        .entry((b.benchmark.clone(), b.label.clone(), r.name.clone()))
                        .or_default();
                    match r.unit.as_str() {
                        "seconds" => s.seconds.push(r.value),
                        "result" => s.results.push(r.value),
                        other => failures.push(format!(
                            "{what}: {} reported unit {other}; only seconds and result are read",
                            r.name
                        )),
                    }
                }
            }
        }

        // One table per benchmark and quantity; the quantity is the record name up to `/`.
        struct Row {
            benchmark: String,
            quantity: String,
            label: String,
            implementation: String,
            median: f64,
            min: f64,
            max: f64,
            runs: usize,
            result: Option<f64>,
            result_varies: bool,
        }
        let mut rows: Vec<Row> = Vec::new();
        for ((benchmark, label, name), s) in &samples {
            if s.seconds.is_empty() {
                continue;
            }
            let (quantity, implementation) = name.split_once('/').unwrap_or((name, ""));
            let result = s.results.first().copied();
            rows.push(Row {
                benchmark: benchmark.clone(),
                quantity: quantity.to_string(),
                label: label.clone(),
                implementation: implementation.to_string(),
                median: median(&s.seconds),
                min: s.seconds.iter().copied().fold(f64::INFINITY, f64::min),
                max: s.seconds.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                runs: s.seconds.len(),
                result,
                result_varies: s.results.iter().any(|r| Some(*r) != result),
            });
        }
        rows.sort_by(|a, b| {
            (&a.benchmark, &a.quantity)
                .cmp(&(&b.benchmark, &b.quantity))
                .then(a.median.total_cmp(&b.median))
        });

        let mut differ = false;
        let mut i = 0;
        while i < rows.len() {
            let j = rows[i..]
                .iter()
                .position(|r| {
                    (&r.benchmark, &r.quantity) != (&rows[i].benchmark, &rows[i].quantity)
                })
                .map_or(rows.len(), |k| i + k);
            let group = &rows[i..j];
            // The fastest Vx implementation is the yardstick: `vs vx` above 1 is slower.
            let vx = group
                .iter()
                .filter(|r| r.label == "vx")
                .min_by(|a, b| a.median.total_cmp(&b.median));
            let reference = vx.and_then(|r| r.result).or_else(|| group[0].result);
            println!("{} / {}", group[0].benchmark, group[0].quantity);
            println!(
                "  {:<22} {:<30} {:>11} {:>11} {:>11} {:>7}  result",
                "build", "implementation", "median ms", "min ms", "max ms", "vs vx"
            );
            for r in group {
                let ratio = vx.map_or("-".to_string(), |v| format!("{:.2}x", r.median / v.median));
                let mark = match (r.result, reference) {
                    (Some(x), Some(y))
                        if (x - y).abs() > args.tolerance * y.abs().max(f64::MIN_POSITIVE) =>
                    {
                        differ = true;
                        " *"
                    }
                    _ => "",
                };
                let varies = if r.result_varies {
                    " (changed between rounds)"
                } else {
                    ""
                };
                println!(
                    "  {:<22} {:<30} {:>11.3} {:>11.3} {:>11.3} {:>7}  {}{mark}{varies}",
                    r.label,
                    r.implementation,
                    r.median * 1e3,
                    r.min * 1e3,
                    r.max * 1e3,
                    ratio,
                    r.result.map_or("-".to_string(), |x| format!("{x}")),
                );
            }
            println!();
            i = j;
        }
        if differ {
            println!(
                "* differs from the fastest Vx answer by more than {} (relative). Not a failure: \
                 a sum taken in another order is a different number.",
                args.tolerance
            );
            println!();
        }

        if let Some(path) = &args.json {
            let mut out = format!(
                "{{\n  \"commit\": {},\n  \"machine\": {},\n  \"cpu_model\": {},\n  \"rounds\": {},\n  \"cpu\": {},\n  \"rows\": [\n",
                json_str(&commit),
                json_str(&machine),
                json_str(&cpu_model),
                args.rounds,
                args.cpu.map_or("null".to_string(), |c| c.to_string()),
            );
            let body: Vec<String> = rows
                .iter()
                .map(|r| {
                    format!(
                        "    {{\"benchmark\": {}, \"quantity\": {}, \"build\": {}, \"implementation\": {}, \"median_s\": {}, \"min_s\": {}, \"max_s\": {}, \"runs\": {}, \"result\": {}}}",
                        json_str(&r.benchmark),
                        json_str(&r.quantity),
                        json_str(&r.label),
                        json_str(&r.implementation),
                        r.median,
                        r.min,
                        r.max,
                        r.runs,
                        r.result.map_or("null".to_string(), |x| format!("{x}")),
                    )
                })
                .collect();
            out.push_str(&body.join(",\n"));
            out.push_str("\n  ],\n  \"failures\": [");
            out.push_str(
                &failures
                    .iter()
                    .map(|f| json_str(f))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            out.push_str("]\n}\n");
            if let Err(e) = std::fs::write(path, out) {
                failures.push(format!("cannot write {}: {e}", path.display()));
            } else {
                println!("wrote {}", path.display());
            }
        }

        if failures.is_empty() {
            ExitCode::SUCCESS
        } else {
            eprintln!("{} problems:", failures.len());
            for f in &failures {
                eprintln!("  {f}");
            }
            ExitCode::FAILURE
        }
    }
}
