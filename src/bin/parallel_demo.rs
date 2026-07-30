//===- parallel_demo.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
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
// The pipeline's own phase logging goes to stdout; this demo's report goes to
// stderr, so `cargo run --bin parallel_demo >/dev/null` shows just the report.
//
//===----------------------------------------------------------------------===//
use std::io::Write;
use std::time::Instant;

/// Write `n_modules` independent `.vx` modules, each with `fns_per` non-trivial scalar functions
/// (params, a loop, an if/else, a mutable local — real work for the parallel type-checker and the
/// flat-HIR lowering). Returns the file paths.
fn generate_corpus(dir: &std::path::Path, n_modules: usize, fns_per: usize) -> Vec<String> {
    std::fs::create_dir_all(dir).unwrap();
    let mut paths = Vec::with_capacity(n_modules);
    for m in 0..n_modules {
        let mut src = String::new();
        for f in 0..fns_per {
            src.push_str(&format!(
                "fn m{m}_f{f}(a: i32, b: i32) -> i32 {{\n\
                 \x20 let mut s: i32 = a;\n\
                 \x20 for k in 0..b {{\n\
                 \x20   if a < k {{ s = s + a * k; }} else {{ s = s - k; }}\n\
                 \x20 }}\n\
                 \x20 return s + a * {f};\n\
                 }}\n"
            ));
        }
        let path = dir.join(format!("m{m}.vx"));
        std::fs::write(&path, src).unwrap();
        paths.push(path.to_string_lossy().into_owned());
    }
    paths
}

fn main() {
    let n_modules: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(400);
    let fns_per = 8;

    let dir = std::env::temp_dir().join(format!("vx_parallel_demo_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let paths = generate_corpus(&dir, n_modules, fns_per);

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
    writeln!(
        w,
        "corpus: {n_modules} modules × {fns_per} functions = {} functions   (machine: {hw} cores)",
        n_modules * fns_per
    )
    .unwrap();
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
            vxc::pipeline::compile_pipeline_type_stream(&paths)
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

    let _ = std::fs::remove_dir_all(&dir);
}
