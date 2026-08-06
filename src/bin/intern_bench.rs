//===- intern_bench.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
//              EVAL-ONLY. `parallel-frontend-eval` BRANCH. NEVER MERGED.
//
// E1 + E2 of the CGO 2027 R2 measurement plan (#295, #296): time the frontend under both interning
// strategies across a corpus sweep and a thread ladder, so the deferred design's speedup has a
// baseline to attribute against and a shape to attribute it to.
//
//   cargo run --release --bin intern_bench -- \
//       --modules 8,64,512 --fns 16,128 --density 0,1 --threads 1,2,4,8 --reps 10
//
// Every swept dimension appears in the CSV, so a raw log is self-describing.
//
//===----------------------------------------------------------------------===//
#[path = "corpus/mod.rs"]
mod corpus;

use std::time::{Duration, Instant};
use vxc::intern_mode::{self, InternMode};

fn median_iqr(mut xs: Vec<f64>) -> (f64, f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| -> f64 {
        if xs.is_empty() {
            return 0.0;
        }
        let i = (p * (xs.len() - 1) as f64).round() as usize;
        xs[i.min(xs.len() - 1)]
    };
    (q(0.5), q(0.25), q(0.75))
}

/// Phase medians for one cell, in the order the pipeline runs them.
fn phase_report(samples: Vec<Vec<(&'static str, Duration)>>) -> Vec<(&'static str, f64)> {
    let mut names: Vec<&'static str> = Vec::new();
    for s in &samples {
        for (n, _) in s {
            if !names.contains(n) {
                names.push(n);
            }
        }
    }
    names
        .into_iter()
        .map(|n| {
            let xs: Vec<f64> = samples
                .iter()
                .map(|s| {
                    s.iter()
                        .filter(|(m, _)| *m == n)
                        .map(|(_, d)| d.as_secs_f64() * 1e3)
                        .sum::<f64>()
                })
                .collect();
            (n, median_iqr(xs).0)
        })
        .collect()
}

/// What one rep compiled: its wall clock, and how many bytes of MLIR came out.
///
/// `Some(0)` means codegen ran and the flat emitter *declined* — the frontend still did all its
/// work, but nothing was generated, so the rep timed an incomplete compile and must not be quoted
/// as one. `None` means codegen was not asked for (`--emit=none`).
struct Rep {
    wall: Duration,
    mlir_bytes: Option<usize>,
}

fn run_once(paths: &[String], threads: usize, emit_mlir: bool) -> Rep {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    let _ = intern_mode::take_phases(); // drop any timings from a prior rep
    let t = Instant::now();
    let mlir_bytes = pool.install(|| {
        if emit_mlir {
            Some(
                vxc::pipeline::compile_pipeline_mlir(paths)
                    .expect("pipeline")
                    .map(|t| t.len())
                    .unwrap_or(0),
            )
        } else {
            let _ = vxc::pipeline::compile_pipeline_type_stream(paths).expect("pipeline");
            None
        }
    });
    Rep {
        wall: t.elapsed(),
        mlir_bytes,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let module_counts: Vec<usize> = corpus::arg_list(&args, "--modules", vec![64]);
    let fn_counts: Vec<usize> = corpus::arg_list(&args, "--fns", vec![16]);
    // The generics-density sweep is the comparison E2 exists for: the same function count with and
    // without instantiations in the signatures.
    let densities: Vec<f64> = corpus::arg_list(&args, "--density", vec![1.0]);
    let reps: usize = corpus::arg(&args, "--reps", 10);
    let threads: Vec<usize> = corpus::arg_list(&args, "--threads", vec![1, 2, 4, 8]);
    // Codegen is on by default (#311). A sweep that stops at the SIMD patch times a frontend, not a
    // compile, and the number it produces cannot be quoted as a compile-time speedup no matter how
    // carefully the rest of the harness is built. `--emit=none` reproduces the old frontend-only
    // measurement when the frontend is deliberately what is under study.
    let emit_mlir = corpus::arg::<String>(&args, "--emit", "mlir".to_string()) != "none";

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let over = threads.iter().filter(|&&t| t > cores).count();
    if over > 0 {
        eprintln!(
            "WARNING: {over} of {} thread counts exceed this machine's {cores} cores. Past the \
             core count you are measuring oversubscription, not scaling — those rows are not \
             publishable data. Run the full ladder on a machine with at least as many cores as \
             the largest thread count.",
            threads.len()
        );
    }

    println!(
        "mode,modules,fns,density,generic_slots,distinct_keys,threads,median_ms,q1_ms,q3_ms,reps"
    );

    for &n in &module_counts {
        for &m in &fn_counts {
            for &d in &densities {
                let params = corpus::params_from_args(&args, n, m);
                let params = corpus::CorpusParams {
                    density: d,
                    ..params
                };
                let c = corpus::generate(&params, None);
                eprintln!("\n{}", c.log_line());

                let mut medians: std::collections::HashMap<(&str, usize), f64> =
                    std::collections::HashMap::new();

                for (label, mode) in [
                    ("deferred", InternMode::Deferred),
                    ("content", InternMode::Content),
                ] {
                    intern_mode::set_mode(mode);
                    for &t in &threads {
                        // One warm-up rep, discarded: the first run pays page faults and
                        // filesystem-cache misses that have nothing to do with interning.
                        let warm = run_once(&c.paths, t, emit_mlir);
                        if warm.mlir_bytes == Some(0) {
                            eprintln!(
                                "  NOTE: the flat emitter declined this corpus, so every rep below \
                                 times a frontend plus a codegen attempt that produced nothing. Not \
                                 a compile-time measurement. Run with VX_FLAT_DBG=1 to see which \
                                 construct declined, or --emit=none to measure the frontend on \
                                 purpose."
                            );
                        }
                        let mut samples = Vec::with_capacity(reps);
                        let mut phase_samples = Vec::with_capacity(reps);
                        let mut mlir_bytes = 0usize;
                        for _ in 0..reps {
                            let rep = run_once(&c.paths, t, emit_mlir);
                            mlir_bytes = rep.mlir_bytes.unwrap_or(0);
                            samples.push(rep.wall.as_secs_f64() * 1e3);
                            phase_samples.push(intern_mode::take_phases());
                        }
                        let (med, q1, q3) = median_iqr(samples);
                        medians.insert((label, t), med);
                        println!(
                            "{label},{n},{m},{d},{},{},{t},{med:.2},{q1:.2},{q3:.2},{reps}",
                            c.generic_slots, c.distinct_keys
                        );
                        let ph = phase_report(phase_samples);
                        let total: f64 = ph.iter().map(|(_, v)| v).sum();
                        let share = |name: &str| -> f64 {
                            let v: f64 =
                                ph.iter().filter(|(n, _)| *n == name).map(|(_, v)| v).sum();
                            if total > 0.0 {
                                v / total * 100.0
                            } else {
                                0.0
                            }
                        };
                        eprintln!(
                            "  [{label} t={t}] {} | measured-phase total {:.1} ms, type_check \
                             {:.0}%, codegen {:.0}% (the rest is serial or barrier work){}",
                            ph.iter()
                                .map(|(n, v)| format!("{n} {v:.1}"))
                                .collect::<Vec<_>>()
                                .join("  "),
                            total,
                            share("type_check"),
                            share("codegen"),
                            if emit_mlir {
                                format!(", {mlir_bytes} bytes of MLIR")
                            } else {
                                String::new()
                            }
                        );
                    }
                }
                intern_mode::set_mode(InternMode::Deferred);

                // The attribution number. Speedup is measured against each mode's *own*
                // single-thread time, so the ratio isolates how well each design parallelises
                // rather than conflating that with a constant-factor difference in serial work.
                eprintln!("  threads  sp(deferred)  sp(content)   ms(deferred/content)");
                let base = |mm: &str| medians.get(&(mm, threads[0])).copied().unwrap_or(f64::NAN);
                let at = |mm: &str, t: usize| medians.get(&(mm, t)).copied().unwrap_or(f64::NAN);
                for &t in &threads {
                    let flag = if t > cores { "  <- oversubscribed" } else { "" };
                    eprintln!(
                        "  {t:>7}  {:>12.2}  {:>11.2}   {:>8.1}/{:>8.1}{flag}",
                        base("deferred") / at("deferred", t),
                        base("content") / at("content", t),
                        at("deferred", t),
                        at("content", t),
                    );
                }
                eprintln!(
                    "  single-thread: deferred {:.1} ms, content {:.1} ms. Absolute times matter \
                     as much as the speedup ratios here: `content` removes work rather than \
                     parallelising it, so a win shows up first in the single-thread column, not in \
                     the scaling column.",
                    base("deferred"),
                    base("content"),
                );
            }
        }
    }
}
