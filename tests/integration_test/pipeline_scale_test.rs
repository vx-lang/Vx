//===- pipeline_scale_test.rs - Vx Compiler --------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
// Runs the parallel pipeline at the scale its scaling result was measured at and
// checks the emitted MLIR is byte-identical however many threads produced it.
// The determinism tests in src/pipeline.rs use two modules and four functions;
// a 5.89x measured over 16,000 functions needs a check over 16,000 functions.
//===----------------------------------------------------------------------===//

// The generator the ladder and the demo already share, included the same way, so the
// workload CI checks is the workload the number was measured on.
#[path = "../../src/bin/corpus/mod.rs"]
mod corpus;

use vxc::config::Schedule;
use vxc::intern_mode::InternMode;
use vxc::pipeline::compile_pipeline_mlir_in;

fn from_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The full corpus in a release build (2.7s), a tenth of it in a debug build.
/// Debug is ~30x slower and this runs on every `cargo test`; CI gates the release
/// build, so the 16,000-function claim is still checked on every commit.
fn default_modules() -> usize {
    if cfg!(debug_assertions) {
        100
    } else {
        1000
    }
}

#[test]
fn pipeline_emits_identical_mlir_at_benchmark_scale() {
    let modules = from_env("VX_SCALE_MODULES", default_modules());
    let fns_per_module = from_env("VX_SCALE_FNS", 16);

    let mut params = corpus::params_from_args(&[], modules, fns_per_module);
    // A corpus with no machine declarations left a serial phase unmeasured for two
    // weeks once. Keep some here so the memory-algebra surface is covered as well.
    params.memalg_frac = 0.25;

    let generated = corpus::generate(&params, None);
    let paths = generated.paths.clone();
    assert_eq!(
        paths.len(),
        modules,
        "generator produced the requested corpus"
    );

    let compile = |threads: usize, sched: Schedule| -> String {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        pool.install(|| {
            compile_pipeline_mlir_in(&paths, sched, InternMode::Deferred)
                .expect("pipeline should compile the generated corpus")
                .expect("the flat emitter should cover the generated corpus")
        })
    };

    let one_thread = compile(1, Schedule::Parallel);
    let many_threads = compile(4, Schedule::Parallel);
    // Not "one rayon thread": the sequential schedule takes rayon off the path entirely,
    // which is the baseline the scaling number is quoted against.
    let no_rayon = compile(1, Schedule::Sequential);

    let describe = |label: &str, text: &String| {
        format!(
            "{label}: {} bytes, {} functions",
            text.len(),
            text.matches("func.func").count()
        )
    };

    // A determinism failure over 36 MB of MLIR is useless without the offending line,
    // so dump both and name the first place they diverge.
    let first_divergence = |a: &str, b: &str| -> String {
        let dir = std::env::temp_dir().join("vx_scale_determinism");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("a.mlir"), a);
        let _ = std::fs::write(dir.join("b.mlir"), b);
        for (n, (la, lb)) in a.lines().zip(b.lines()).enumerate() {
            if la != lb {
                return format!(
                    "first difference at line {}:\n  a: {}\n  b: {}\n  both written to {}",
                    n + 1,
                    la.trim(),
                    lb.trim(),
                    dir.display()
                );
            }
        }
        format!(
            "no differing line; lengths {} vs {} (one is a prefix of the other). both written to {}",
            a.len(),
            b.len(),
            dir.display()
        )
    };

    assert!(
        one_thread == many_threads,
        "the pipeline emitted different MLIR on 1 thread and 4 threads over {} functions.\n{}\n{}\n{}",
        modules * fns_per_module,
        describe("1 thread", &one_thread),
        describe("4 threads", &many_threads),
        first_divergence(&one_thread, &many_threads)
    );
    assert!(
        one_thread == no_rayon,
        "the pipeline emitted different MLIR with rayon on the path and with it off.\n{}\n{}\n{}",
        describe("parallel", &one_thread),
        describe("sequential", &no_rayon),
        first_divergence(&one_thread, &no_rayon)
    );

    println!(
        "identical MLIR at {} modules / {} functions / {} lines: {} bytes, {} func.func",
        modules,
        modules * fns_per_module,
        generated.lines,
        one_thread.len(),
        one_thread.matches("func.func").count()
    );
}
