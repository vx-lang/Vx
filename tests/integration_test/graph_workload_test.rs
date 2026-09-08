//===- graph_workload_test.rs - Vx Compiler --------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Uses the `stdlib/graph` library (10 graph algorithms across 5 modules) as a
// real, rich workload for the parallel front-end. Runs the parallel resolution
// phases (`build_symbol_map` + `resolve_names`, both `par_iter`) over the graph
// modules and asserts the output is invariant to thread count and stable under
// contention. This is the intended ThreadSanitizer payload (#202): run these
// tests under `-Zsanitizer=thread` to dynamically prove race freedom on
// realistic multi-module input.
//
//===----------------------------------------------------------------------===//
use rayon::prelude::*;
use vxc::resolver::build_symbol_map;
use vxc::syntax::Program;

fn parse_graph_module(path: &str) -> Program {
    let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut lexer = vxc::lexer::Lexer::new(&src);
    let tokens = lexer.tokenize();
    let mut parser = vxc::parser::Parser::new(&tokens, &src);
    let mut prog = parser.parse().expect("graph module parses");
    prog.module_path = path.into();
    prog
}

fn graph_modules() -> Vec<Program> {
    ["core", "traversal", "shortest_path", "analysis", "tests"]
        .iter()
        .map(|f| parse_graph_module(&format!("stdlib/graph/{f}.vx")))
        .collect()
}

/// The parallel resolution phases run over the graph library; under a 1-thread and an 8-thread
/// rayon pool the minted GIDs must match exactly. Race-free parallelism => identical output
/// regardless of thread count.
#[test]
fn graph_library_parallel_resolution_is_deterministic() {
    let modules = graph_modules();

    let run = |threads: usize| -> Vec<[u64; 4]> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut mods = modules.clone();
            let symbol_map = build_symbol_map(&mods);
            mods.par_iter_mut()
                .for_each(|m| m.resolve_names(&symbol_map, &[]));
            let mut gids: Vec<[u64; 4]> = symbol_map
                .values()
                .flat_map(|t| t.values().map(|id| id.words))
                .collect();
            gids.sort_unstable();
            gids
        })
    };

    let single = run(1);
    let many = run(8);
    assert!(!single.is_empty(), "graph library minted no GIDs");
    assert_eq!(
        single, many,
        "graph library resolution differs across thread counts (data race?)"
    );
}

/// Stress: hammer the parallel resolution over the graph library from many threads at once (the
/// payload TSan instruments), asserting the minted symbol count never diverges under contention.
#[test]
fn graph_library_parallel_resolution_stress() {
    let modules = graph_modules();
    let baseline: usize = {
        let map = build_symbol_map(&modules);
        map.values().map(|t| t.len()).sum()
    };
    assert!(baseline > 0, "expected at least the Graph symbol");

    (0..64).into_par_iter().for_each(|_| {
        let mut mods = modules.clone();
        let map = build_symbol_map(&mods);
        mods.par_iter_mut().for_each(|m| m.resolve_names(&map, &[]));
        let n: usize = map.values().map(|t| t.len()).sum();
        assert_eq!(n, baseline, "symbol count diverged under contention");
    });
}
