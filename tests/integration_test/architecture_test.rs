//===- architecture_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file contains tests verifying compiler architecture constraints.
// It runs static analysis checks against the compiler source code itself to ensure
// that prohibited locking primitives are not accidentally introduced into the
// highly parallel, lock-free pipeline.
//
//===----------------------------------------------------------------------===//
use std::fs;

#[test]
fn test_pipeline_architecture_hooks() -> Result<(), String> {
    // A temp dir keyed by process id, not a fixed path inside the repo (#304). Two concurrent
    // `cargo test` runs -- an editor's and a terminal's, or a TSan build alongside a normal one --
    // otherwise share one directory: the second `remove_dir_all` deletes the first run's inputs
    // mid-compile, and the failure looks like a compiler bug rather than a harness bug.
    let dir = std::env::temp_dir().join(format!("vx_architecture_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test dir");

    // File 1: Generic functions and structs by value
    let file1_path = dir.join("module_a.vx");
    fs::write(
        &file1_path,
        r#"
        struct Config {
            id: i32,
            active: bool,
        }

        fn process_config<T>(val: T, cfg: &Config, count: u32, limit: u64, ratio: f32) -> &Config {
            return cfg;
        }

        fn run_module_a() -> Config {
            let c = Config { id: 1i32, active: true };
            let count = 100u32;
            let limit = 5000u64;
            let ratio = 1.5f32;
            let _ref = process_config(10i32, &c, count, limit, ratio);
            return c;
        }
        "#,
    )
    .unwrap();

    // File 2: Function with 10+ parameters to trigger Slow Path (UnboundedFunctionMetadata)
    // Also tests ownership references (&mut)
    let file2_path = dir.as_path().join("module_b.vx");
    fs::write(
        &file2_path,
        r#"
        fn compute_heavy(
            p1: &Tensor<f32, [?, ?]>, p2: &Tensor<f32, [?, ?]>, p3: &Tensor<f32, [?, ?]>, p4: &Tensor<f32, [?, ?]>,
            p5: &Tensor<f32, [?, ?]>, p6: &Tensor<f32, [?, ?]>, p7: &Tensor<f32, [?, ?]>, p8: &Tensor<f32, [?, ?]>,
            p9: &Tensor<f32, [?, ?]>, p10: &Tensor<f32, [?, ?]>, p11: &mut Tensor<f32, [?, ?]>
        ) -> f32 {
            return 1.0f32;
        }

        fn run_module_b(t: Tensor<f32, [?, ?]>) -> Tensor<f32, [?, ?]> {
            let mut result = t;
            let mut result2 = Tensor<f32>();
            compute_heavy(&result, &result, &result, &result, &result, &result, &result, &result, &result, &result, &mut result2);
            return result;
        }
        "#,
    )
    .unwrap();

    // Collect paths
    let paths = vec![
        file1_path.to_string_lossy().to_string(),
        file2_path.to_string_lossy().to_string(),
    ];

    // Execute pipeline. This will run through the Verification Engine hooks.
    // We expect it to succeed, which means all invariants (Phase 1-8) held true.
    let result = vxc::pipeline::compile_pipeline(&paths);
    if result.is_err() {
        return Err(format!("Pipeline failed: {:?}", result.err()));
    }

    let _ = fs::remove_dir_all(&dir);

    Ok(())
}

/// End-to-end determinism: parsing a fileset (in parallel) and minting its 256-bit GIDs twice
/// must yield an *identical* set. Because GIDs are content hashes (module + symbol), not
/// scheduling-dependent counters, `rayon`'s work-stealing cannot change the result -- this is the
/// reproducibility guarantee the whole GID scheme exists to provide. Drives the real file-based
/// pipeline entry (`compile_pipeline_symbol_gids`), complementing the in-process
/// `build_symbol_map` unit test in `resolver.rs`.
#[test]
fn compile_pipeline_gid_stream_is_deterministic() -> Result<(), String> {
    // Process-keyed for the same reason as above (#304).
    let dir = std::env::temp_dir().join(format!("vx_determinism_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create test dir");

    // A few modules with structs + generic functions, to mint a rich GID stream (fast path) plus
    // a slow-path (5+ param) signature.
    let modules = [
        (
            "alpha.vx",
            r#"
            struct Widget { id: i32, active: bool }
            fn identity<T>(v: T) -> T { return v; }
            fn run_alpha() -> Widget {
                let w = Widget { id: 1i32, active: true };
                let _r = identity(7i32);
                return w;
            }
            "#,
        ),
        (
            "beta.vx",
            r#"
            struct Widget { id: i32, active: bool }
            fn combine(a: i32, b: i32, c: i32, d: i32, e: i32, f: u64) -> i32 { return a; }
            fn run_beta() -> i32 { return combine(1i32, 2i32, 3i32, 4i32, 5i32, 6u64); }
            "#,
        ),
    ];

    let mut paths = Vec::new();
    for (name, src) in modules {
        let p = dir.join(name);
        fs::write(&p, src).unwrap();
        paths.push(p.to_string_lossy().to_string());
    }

    let run = || -> Result<Vec<[u64; 4]>, String> {
        let mut stream = vxc::pipeline::compile_pipeline_type_stream(&paths)
            .map_err(|e| format!("pipeline failed: {:?}", e))?
            .into_iter()
            .map(|id| id.words)
            .collect::<Vec<_>>();
        // Order-independent: the *set* of emitted GIDs is the determinism guarantee, robust to
        // parallel scheduling of the phases.
        stream.sort_unstable();
        Ok(stream)
    };

    let first = run()?;
    let second = run()?;

    assert!(!first.is_empty(), "expected a non-empty GID stream");
    assert_eq!(
        first, second,
        "compile_pipeline emitted a different GID stream on a second run (non-deterministic)"
    );

    let _ = fs::remove_dir_all(&dir);
    Ok(())
}

/// The core parallel-architecture guarantee: compilations are isolated because the compiler holds
/// no process-global *mutable* state (docs/parallel_compiler_architecture.md §2.7). Many threads
/// each compile a program that declares a topology with the *same name* `Dev` but a *different*
/// default memory. Each thread must see only its own declaration in its own per-compilation
/// `TransferCostGraph`. A global registry (the old `RwLock`, or the `thread_local` stopgap) would
/// let these declarations race/leak across threads and this assertion would fire.
#[test]
fn concurrent_compilations_have_isolated_topologies() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    const THREADS: usize = 64;
    let gate = Arc::new(Barrier::new(THREADS));

    let handles: Vec<_> = (0..THREADS)
        .map(|i| {
            let gate = gate.clone();
            thread::spawn(move || {
                // Same topology *name*, different default memory per thread.
                let mem = if i % 2 == 0 { "Local_SRAM" } else { "GPU_HBM" };
                let src = format!(
                    "Topology Dev {{ memory: Memory::{mem}, visible: [Memory::CPU_DRAM, Memory::{mem}] }}\n\
                     fn main() -> i32 {{ return 0; }}"
                );

                // Release all threads into the parse+check at once, for maximum contention on any
                // (accidental) shared state.
                gate.wait();

                let mut lexer = vxc::lexer::Lexer::new(&src);
                let tokens = lexer.tokenize();
                let mut parser = vxc::parser::Parser::new(&tokens, &src);
                let program = parser.parse().expect("parse failed");
                let programs = vec![program];

                let env = vxc::hir::GlobalAstEnv::build(&programs);
                let session = Arc::new(vxc::session::GlobalSession::new(1));
                let mut worker = vxc::session::LocalWorkerState::new(session);
                let checker = vxc::hir::TypeChecker::new(&env, &mut worker);

                // The topology descriptor this compilation sees for `Dev`.
                let desc = checker
                    .transfer_cost_graph
                    .descriptor(&vxc::syntax::TopologyKind::Custom("Dev".into()))
                    .cloned();
                (i, mem, desc)
            })
        })
        .collect();

    for h in handles {
        let (i, mem, desc) = h.join().expect("worker thread panicked");
        let desc = desc.unwrap_or_else(|| panic!("thread {i}: its own topology `Dev` is missing"));
        let expected = vxc::syntax::MemorySpace::from_name(mem);
        assert_eq!(
            desc.default_space, expected,
            "thread {i} declared Dev on {mem} but saw a leaked topology from another compilation"
        );
    }
}
