//===- parallel_demo.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A live demonstration of Vx's data-oriented, zero-lock *parallel* compiler
// front-end. Generates a corpus of independent modules, then runs the whole
// parallel pipeline (parse -> name-resolution -> registry freeze -> parallel
// type-check + flat-HIR lowering -> dedup -> SIMD patch) under 1, 2, 4, and 8
// `rayon` threads, and shows:
//
//   * scaling      — wall time drops as threads are added (parse/check/lower run
//                    per-module in parallel with no shared mutable state);
//   * determinism  — the flat 256-bit GID stream is *byte-identical, in order*
//                    across every thread count (identity is a content hash, not a
//                    scheduling-dependent counter);
//   * zero-lock    — no lock primitives anywhere on the path (CI enforces it), so
//                    the speedup comes from genuine isolation, not lock contention.
//
// The corpus comes from the shared generator in `src/bin/corpus` (#296), which the
// `intern_bench` measurements also use — a demo and a benchmark that disagree about
// the workload produce two numbers nobody can compare. Its knobs are documented in
// `corpus/mod.rs`; the ones that matter here:
//
//   --modules N --fns M          corpus size
//   --density 0.0 .. 1.0         fraction of parameter slots holding a generic
//                                instantiation (0 = plain, 1 = instantiation-dense)
//   --arity A --shared-frac S    key-space size, and how often two threads mint the
//                                same interner key
//   --seed S --out DIR           reproducibility
//
// The pipeline's own phase logging goes to stdout; this demo's report goes to
// stderr, so `cargo run --bin parallel_demo >/dev/null` shows just the report.
//
//===----------------------------------------------------------------------===//
#[path = "corpus/mod.rs"]
mod corpus;

use std::io::Write;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Positional N is kept working: `parallel_demo 400` predates the flags and still means
    // "400 modules".
    let positional = args
        .get(1)
        .filter(|a| !a.starts_with("--"))
        .and_then(|a| a.parse().ok());
    let modules = positional.unwrap_or_else(|| corpus::arg(&args, "--modules", 400));
    let fns = corpus::arg(&args, "--fns", 8);
    let params = corpus::params_from_args(&args, modules, fns);
    let out = args
        .windows(2)
        .find(|w| w[0] == "--out")
        .map(|w| std::path::PathBuf::from(&w[1]));
    let c = corpus::generate(&params, out.as_deref());

    let report = std::io::stderr();
    let mut w = report.lock();
    let hw = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    writeln!(
        w,
        "\n╔══════════════════════════════════════════════════════════════════╗"
    )
    .unwrap();
    writeln!(
        w,
        "║   Vx parallel compiler — live demonstration                        ║"
    )
    .unwrap();
    writeln!(
        w,
        "╚══════════════════════════════════════════════════════════════════╝"
    )
    .unwrap();
    // The corpus line carries every parameter plus the two counts that decide how much interning
    // there is to do, so this report is self-describing when it is pasted into an issue.
    writeln!(w, "{}", c.log_line()).unwrap();
    writeln!(w, "corpus dir: {}   (machine: {hw} cores)", c.dir.display()).unwrap();
    writeln!(w, "pipeline: parse → resolve → freeze registry → parallel type-check + flat-HIR → dedup → SIMD-patch\n").unwrap();
    writeln!(
        w,
        "  threads      wall-time    speedup    GID stream    order vs 1-thread"
    )
    .unwrap();
    writeln!(
        w,
        "  ───────────────────────────────────────────────────────────────────"
    )
    .unwrap();

    let mut baseline: Option<Vec<[u64; 4]>> = None;
    let mut single_time = 0.0f64;

    for &threads in &[1usize, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let start = Instant::now();
        let stream: Vec<[u64; 4]> = pool.install(|| {
            vxc::pipeline::compile_pipeline_type_stream(&c.paths)
                .expect("pipeline")
                .into_iter()
                .map(|id| id.words)
                .collect()
        });
        let elapsed = start.elapsed().as_secs_f64();
        if threads == 1 {
            single_time = elapsed;
        }

        let order = match &baseline {
            None => {
                baseline = Some(stream.clone());
                "baseline".to_string()
            }
            Some(b) if *b == stream => "✓ identical".to_string(),
            Some(_) => "✗ DIVERGED".to_string(),
        };

        writeln!(
            w,
            "  {threads:>2} thread{}   {elapsed:>8.3} s   {:>6.2}×    {:>7} GIDs    {order}",
            if threads == 1 { " " } else { "s" },
            single_time / elapsed,
            stream.len(),
        )
        .unwrap();
    }

    writeln!(w, "\n  What this shows").unwrap();
    writeln!(
        w,
        "  • Identity is a 256-bit content hash (module⊕symbol), not a counter — so the flat"
    )
    .unwrap();
    writeln!(
        w,
        "    GID stream is byte-identical, *in order*, no matter how the work is scheduled."
    )
    .unwrap();
    writeln!(
        w,
        "  • Every phase is a rayon parallel-for over modules/functions with a *frozen*,"
    )
    .unwrap();
    writeln!(
        w,
        "    read-only session — zero lock primitives on the path (CI lint enforces it)."
    )
    .unwrap();
    writeln!(
        w,
        "  • Determinism is verified in the test suite across thread counts"
    )
    .unwrap();
    writeln!(
        w,
        "    (pipeline.rs: flat_type_stream_order_is_deterministic_across_thread_counts).\n"
    )
    .unwrap();

    // The corpus is left on disk. It is named by a digest of its parameters, so it is reused by the
    // next run with the same flags rather than regenerated — and it can be inspected after the fact,
    // which a PID-keyed directory deleted on exit could not be.
}
