//===- intern_bench.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
use vxc::config::Schedule;
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

/// One compile, timed, inside a caller-supplied pool.
///
/// The pool is the caller's because thread *placement* is the dominant nuisance variable on a
/// heterogeneous machine, and it is decided when the pool is built. On an Apple M4 (4 performance +
/// 6 efficiency cores) a two-thread pool lands either on two P-cores or not, and the difference is a
/// factor of 1.7 on this corpus -- measured: the same binary, corpus and mode gives a median of
/// 30 ms in one process and 50 ms in the next, with under 2% spread *within* each. A per-rep pool
/// makes that draw a hidden per-rep variable; a per-cell pool makes it a constant that every mode in
/// the cell shares, so a mode comparison is paired against it rather than confounded by it.
///
/// This is not hypothetical caution. Running each mode's whole ladder in sequence produced a table
/// showing deferred gaining nothing from a second core while content gained 1.6x, reproducibly
/// across reps and across three runs. Both modes show the same bimodality when measured alone: the
/// effect was the placement draw, not the design.
fn run_in(
    pool: Option<&rayon::ThreadPool>,
    paths: &[String],
    emit_mlir: bool,
    sched: Schedule,
    mode: InternMode,
) -> Rep {
    let _ = intern_mode::take_phases(); // drop any timings from a prior rep
    let t = Instant::now();
    let compile = || {
        if emit_mlir {
            Some(
                vxc::pipeline::compile_pipeline_mlir_in(paths, sched, mode)
                    .expect("pipeline")
                    .map(|t| t.len())
                    .unwrap_or(0),
            )
        } else {
            let _ = vxc::pipeline::compile_pipeline_type_stream_in(paths, sched, mode)
                .expect("pipeline");
            None
        }
    };
    // The sequential arm runs *outside* any pool. Installing into a rayon pool that is then never
    // asked to do parallel work would still be a rayon run, and the point of the arm is to have
    // rayon nowhere on the path -- otherwise its cost is charged to both sides of the comparison and
    // cancels out of every ratio.
    let mlir_bytes = match pool {
        Some(p) if sched != Schedule::Sequential => p.install(compile),
        _ => compile(),
    };
    Rep {
        wall: t.elapsed(),
        mlir_bytes,
    }
}

/// The machine's own scaling ceiling: a parallel-for that allocates nothing, touches no shared
/// state, and fits its whole working set in registers.
///
/// Whatever this reaches at N threads is the most *any* phase can reach, so a phase's scaling should
/// be read as a fraction of it rather than as a fraction of N. Two effects are charged here where
/// they are visible instead of silently inflating the compiler's apparent serial fraction: clock
/// scaling (one active core boosts higher than four) and shared cache/bandwidth. Without this
/// control, "4 threads gave 2.6x" reads as 35% of the compiler failing to parallelise, when much of
/// it may be a ceiling no program on this machine can beat.
///
/// The work is a splitmix64 chain, which is a dependent sequence of integer ops -- no memory
/// traffic, no branch prediction to win, and impossible to vectorise away. `black_box` on the result
/// keeps it from being optimised out entirely.
fn calibrate(pool: Option<&rayon::ThreadPool>, items: usize, iters: u64) -> Duration {
    use rayon::prelude::*;
    let burn = |seed: u64| -> u64 {
        let mut x = seed;
        for _ in 0..iters {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x = x.wrapping_add(z ^ (z >> 31));
        }
        x
    };
    let t = Instant::now();
    let total: u64 = match pool {
        Some(p) => p.install(|| (0..items).into_par_iter().map(|i| burn(i as u64)).sum()),
        None => (0..items).map(|i| burn(i as u64)).sum(),
    };
    std::hint::black_box(total);
    t.elapsed()
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
    // `both` (default) runs a rayon-free arm before the thread ladder, so the ladder's baseline is
    // "the same compiler, not parallelised" rather than "the same compiler, parallelised, on one
    // thread". Without it the parallel machinery's own cost sits on both sides of every ratio and
    // cancels, and the sweep can only answer whether more threads help -- never whether any of this
    // beats not doing it. `par` / `seq` run one arm alone.
    let schedule_arg = corpus::arg::<String>(&args, "--schedule", "both".to_string());
    let run_seq = schedule_arg != "par";
    let run_par = schedule_arg != "seq";
    // Which interning modes to run, and in which order. Both in one process is convenient and
    // produces the comparison table, but it is also a confound: the second mode's cells run on a
    // heap the first mode fragmented and a machine the first mode warmed. `--modes content` /
    // `--modes deferred` puts each in its own process, and `--modes content,deferred` reverses the
    // order -- between them, whether an effect follows the *mode* or its *position in the run* is
    // decidable rather than assumed.
    let modes: Vec<(&str, InternMode)> =
        corpus::arg::<String>(&args, "--modes", "deferred,content".to_string())
            .split(',')
            .filter_map(|m| match m.trim() {
                "deferred" => Some(("deferred", InternMode::Deferred)),
                "content" => Some(("content", InternMode::Content)),
                other => {
                    eprintln!("unknown --modes entry '{other}', expected deferred or content");
                    None
                }
            })
            .collect();
    assert!(!modes.is_empty(), "--modes selected nothing to run");

    // The ladder, with `0` standing for the sequential arm: one key space for the medians map, and
    // `0 threads` reads as "no thread pool", which is what it is.
    let mut ladder: Vec<usize> = Vec::new();
    if run_seq {
        ladder.push(0);
    }
    if run_par {
        ladder.extend(threads.iter().copied());
    }

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

    // Calibrate first, on the same ladder, so every scaling number below can be read against what
    // this machine can actually deliver rather than against N.
    {
        let items = 1600; // one work item per function in the default corpus
        let mut cal: Vec<(usize, f64)> = Vec::new();
        for &t in &ladder {
            let pool = (t > 0).then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .expect("thread pool")
            });
            let _ = calibrate(pool.as_ref(), items, 4_000); // warm up
            let mut xs: Vec<f64> = (0..5)
                .map(|_| calibrate(pool.as_ref(), items, 4_000).as_secs_f64() * 1e3)
                .collect();
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            cal.push((t, xs[xs.len() / 2]));
        }
        let base = cal[0].1;
        eprint!("machine ceiling (pure compute, no allocation):");
        for (t, ms) in &cal {
            eprint!(
                "  {}={:.2}x",
                if *t == 0 {
                    "seq".to_string()
                } else {
                    format!("t{t}")
                },
                base / ms
            );
        }
        eprintln!(
            "\n  Read every phase speedup below as a fraction of this, not of the thread \
                   count: clock scaling and shared cache are charged here, where they belong."
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
                // Per-phase medians, so each phase gets its own scaling curve. The total's speedup
                // says only *that* something is not parallelising; this says *which*, and a phase
                // whose own curve is flat is a different problem from one that scales but is small.
                let mut phase_medians: std::collections::HashMap<
                    (&str, usize),
                    Vec<(&'static str, f64)>,
                > = std::collections::HashMap::new();

                // Thread count outermost, modes paired inside it. One pool per cell, and every mode
                // in the cell runs in that pool, alternating rep by rep -- so whatever placement the
                // pool drew, both modes drew it, and the mode comparison is paired rather than
                // confounded by a nuisance variable worth 1.7x on this machine. Running each mode's
                // ladder end to end instead once produced a clean, reproducible, and entirely false
                // finding; see `run_in`.
                for &t in &ladder {
                    let sched = if t == 0 {
                        Schedule::Sequential
                    } else {
                        Schedule::Parallel
                    };
                    let tag = if t == 0 {
                        "seq".to_string()
                    } else {
                        t.to_string()
                    };
                    let pool = (t > 0).then(|| {
                        rayon::ThreadPoolBuilder::new()
                            .num_threads(t)
                            .build()
                            .expect("thread pool")
                    });

                    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); modes.len()];
                    let mut phase_samples: Vec<Vec<_>> =
                        vec![Vec::with_capacity(reps); modes.len()];
                    let mut mlir_bytes = vec![0usize; modes.len()];

                    // One warm-up rep per mode, discarded: the first run pays page faults and
                    // filesystem-cache misses that have nothing to do with interning.
                    for &(_, mode) in &modes {
                        let warm = run_in(pool.as_ref(), &c.paths, emit_mlir, sched, mode);
                        if warm.mlir_bytes == Some(0) {
                            eprintln!(
                                "  NOTE: the flat emitter declined this corpus, so every rep below \
                                 times a frontend plus a codegen attempt that produced nothing. Not \
                                 a compile-time measurement. Run with VX_FLAT_DBG=1 to see which \
                                 construct declined, or --emit=none to measure the frontend on \
                                 purpose."
                            );
                        }
                    }
                    for _ in 0..reps {
                        for (i, &(_, mode)) in modes.iter().enumerate() {
                            let rep = run_in(pool.as_ref(), &c.paths, emit_mlir, sched, mode);
                            mlir_bytes[i] = rep.mlir_bytes.unwrap_or(0);
                            samples[i].push(rep.wall.as_secs_f64() * 1e3);
                            phase_samples[i].push(intern_mode::take_phases());
                        }
                    }

                    for (i, &(label, _)) in modes.iter().enumerate() {
                        let mlir_bytes = mlir_bytes[i];
                        let (med, q1, q3) = median_iqr(samples[i].clone());
                        medians.insert((label, t), med);
                        println!(
                            "{label},{n},{m},{d},{},{},{tag},{med:.2},{q1:.2},{q3:.2},{reps}",
                            c.generic_slots, c.distinct_keys
                        );
                        let ph = phase_report(std::mem::take(&mut phase_samples[i]));
                        phase_medians.insert((label, t), ph.clone());
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
                            "  [{label} t={tag}] {} | measured-phase total {:.1} ms, type_check \
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

                // The attribution number. Speedup is measured against each mode's own *sequential*
                // run when there is one -- the same compiler with rayon off the path -- so the ratio
                // answers "is parallelising this worth it", not merely "does adding threads help a
                // design that is already paying for parallelism". Falling back to the first thread
                // count under `--schedule par` keeps that arm readable, but its ratios are the
                // weaker claim and the header says which is in force.
                let at = |mm: &str, t: usize| medians.get(&(mm, t)).copied().unwrap_or(f64::NAN);
                let base_key = ladder[0];
                let base_tag = if base_key == 0 {
                    "sequential, no rayon"
                } else {
                    "1 thread (rayon)"
                };
                eprintln!(
                    "  threads  sp(deferred)  sp(content)   ms(deferred/content)   [baseline: {base_tag}]"
                );
                let base = |mm: &str| at(mm, base_key);
                let mut suspect = Vec::new();
                for &t in &ladder {
                    // A row slower than a row with *fewer* threads is not a result. More workers
                    // cannot make the same work take longer, so the row is measuring something other
                    // than the compiler -- on a heterogeneous machine, almost always which cores the
                    // pool drew. Flagged rather than silently tabulated, because such a row is
                    // internally consistent (tight IQR, reproducible across reps) and reads exactly
                    // like a finding.
                    let regressed = modes.iter().any(|&(mm, _)| {
                        ladder
                            .iter()
                            .take_while(|&&u| u < t)
                            .any(|&u| at(mm, t) > at(mm, u) * 1.02)
                    });
                    if regressed {
                        suspect.push(t);
                    }
                    let flag = if t > cores {
                        "  <- oversubscribed"
                    } else if regressed {
                        "  <- SUSPECT: slower than a lower thread count"
                    } else {
                        ""
                    };
                    let tag = if t == 0 { "seq".into() } else { t.to_string() };
                    eprintln!(
                        "  {tag:>7}  {:>12.2}  {:>11.2}   {:>8.1}/{:>8.1}{flag}",
                        base("deferred") / at("deferred", t),
                        base("content") / at("content", t),
                        at("deferred", t),
                        at("content", t),
                    );
                }
                if run_seq && run_par {
                    // What one rayon thread costs over no rayon at all. If it is near 1.00 the two
                    // baselines are interchangeable and the scaling column can be read as speedup;
                    // if it is not, every ratio taken against the 1-thread column was quietly
                    // discounting the parallel machinery's own overhead.
                    let overhead = |mm: &str| at(mm, threads[0]) / at(mm, 0);
                    eprintln!(
                        "  parallel-machinery overhead at {} thread(s) vs no rayon: deferred {:.3}x, \
                         content {:.3}x (1.00 = the parallel structure is free when not used)",
                        threads[0],
                        overhead("deferred"),
                        overhead("content"),
                    );
                }
                eprintln!(
                    "  baseline: deferred {:.1} ms, content {:.1} ms. Absolute times matter as much \
                     as the speedup ratios here: `content` removes work rather than parallelising \
                     it, so a win shows up first in the baseline column, not in the scaling column.",
                    base("deferred"),
                    base("content"),
                );
                if !suspect.is_empty() {
                    eprintln!(
                        "  WARNING: thread count(s) {suspect:?} came out slower than a lower one. \
                         Adding workers cannot lengthen the same work, so such a row is measuring \
                         something other than the compiler. Two causes seen so far: (a) a \
                         heterogeneous host, where a small pool that draws efficiency cores runs \
                         ~1.7x slower for the whole life of the pool with a tight per-rep spread \
                         that reads exactly like a result; (b) too little work per thread, where the \
                         corpus is small enough that pool dispatch dominates -- check the corpus \
                         line above for functions-per-thread before blaming the machine. Re-run the \
                         cell. Do not quote it."
                    );
                }

                // Per-phase scaling. The total's speedup says only *that* something is not
                // parallelising; this says which. A phase that scales but is tiny and a phase that
                // is large and flat both show up as "the total fell short", and they are entirely
                // different problems -- the first is nothing to do, the second is the whole job.
                //
                // `unaccounted` is the wall clock the phase timers do not name. It is not a rounding
                // artifact to be ignored: work outside every `timed()` region is invisible to the
                // attribution, so a large unaccounted share means the phase table is answering a
                // narrower question than it appears to.
                for &(label, _) in &modes {
                    let Some(base_ph) = phase_medians.get(&(label, base_key)) else {
                        continue;
                    };
                    eprintln!("\n  per-phase scaling [{label}], baseline = {}:", base_tag);
                    eprint!("  {:<16}", "phase");
                    for &t in &ladder {
                        eprint!(
                            "{:>9}",
                            if t == 0 {
                                "seq".to_string()
                            } else {
                                format!("t{t}")
                            }
                        );
                    }
                    eprintln!("{:>9}{:>10}", "sp(max)", "share");
                    // Sub-phases are indented and nested *inside* their parent, so they must not
                    // join the total or the shares exceed 100% and `unaccounted` goes negative.
                    let top = |p: &Vec<(&'static str, f64)>| -> f64 {
                        p.iter()
                            .filter(|(n, _)| !n.starts_with(' '))
                            .map(|(_, v)| v)
                            .sum()
                    };
                    let base_total: f64 = top(base_ph);
                    let max_t = *ladder.last().unwrap();
                    for (name, base_ms) in base_ph {
                        eprint!("  {name:<16}");
                        for &t in &ladder {
                            let v = phase_medians
                                .get(&(label, t))
                                .and_then(|p| p.iter().find(|(n, _)| n == name))
                                .map(|(_, v)| *v)
                                .unwrap_or(f64::NAN);
                            eprint!("{v:>9.1}");
                        }
                        let at_max = phase_medians
                            .get(&(label, max_t))
                            .and_then(|p| p.iter().find(|(n, _)| n == name))
                            .map(|(_, v)| *v)
                            .unwrap_or(f64::NAN);
                        eprintln!(
                            "{:>9.2}{:>9.1}%",
                            base_ms / at_max,
                            if base_total > 0.0 {
                                base_ms / base_total * 100.0
                            } else {
                                0.0
                            }
                        );
                    }
                    eprint!("  {:<16}", "unaccounted");
                    for &t in &ladder {
                        let named = phase_medians.get(&(label, t)).map(top).unwrap_or(f64::NAN);
                        eprint!("{:>9.1}", at(label, t) - named);
                    }
                    let named_base: f64 = base_total;
                    let named_max: f64 = phase_medians
                        .get(&(label, max_t))
                        .map(top)
                        .unwrap_or(f64::NAN);
                    eprintln!(
                        "{:>9.2}{:>9.1}%",
                        (at(label, base_key) - named_base) / (at(label, max_t) - named_max),
                        (at(label, base_key) - named_base) / at(label, base_key) * 100.0
                    );
                }
            }
        }
    }
}
