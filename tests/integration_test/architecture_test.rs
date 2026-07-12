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
use std::path::PathBuf;

#[test]
fn test_pipeline_architecture_hooks() -> Result<(), String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/modules/architecture_test");
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
            p1: &Tensor, p2: &Tensor, p3: &Tensor, p4: &Tensor,
            p5: &Tensor, p6: &Tensor, p7: &Tensor, p8: &Tensor,
            p9: &Tensor, p10: &Tensor, p11: &mut Tensor
        ) -> f32 {
            return 1.0f32;
        }

        fn run_module_b(t: Tensor) -> Tensor {
            let mut result = t;
            let mut result2: Tensor = Tensor<f32>();
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
