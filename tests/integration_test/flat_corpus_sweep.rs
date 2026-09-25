//===- flat_corpus_sweep.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Sweeps every `pass` family through the shipping compiler and records which codegen path
// each program took (Vx#383). Declining to the AST path is correct; coverage moving without
// anyone noticing is not, so the declining set is compared exactly, in both directions.
// Programs with a `// REQUIRES:` line are skipped -- their path choice differs by host --
// as are the few that cannot compile as a bare `vxc file.vx` (each names why, in-line).
//===----------------------------------------------------------------------===//
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Programs the flat path declines today, relative to `tests/backend/pass/`.
/// A worklist, not an exemption list: shrinking it is Vx#383.
const KNOWN_DECLINES: &[&str] = &[
    "backend/pass/custom_topology_user_lowering.vx",
    "backend/pass/matmul_assign_alias.vx",
    // "A borrow of something that is not a tensor": the flat path borrows tensors only.
    "backend/pass/borrow_of_a_value.vx",
    // `Option::or` and its neighbours, which answer with an `Option<T>`. The flat path
    // declines them as "a non-scalar default return" -- the same shape as the file below,
    // and the AST path handles both. The module's other methods answer with a `T` or a
    // `bool` and compile through the flat path; adding these three is what moved the file.
    "backend/pass/core_option.vx",
    // The flat path declines `main` here as "a callee return type": the adaptors it
    // builds answer with a generic struct. Its answers come from the AST path.
    "backend/pass/core_iter.vx",
    // Same decline as `core_iter.vx`, one adaptor deeper: the answers come from the AST path.
    "backend/pass/iter_adaptors_chain.vx",
    // Same decline as `core_iter.vx`: `main` builds adaptors before consuming them.
    "backend/pass/core_iter_fold_sum_collect.vx",
    // Same decline again, for the same reason.
    "backend/pass/core_iter_consumers.vx",
    "backend/pass/core_iter_adapters.vx",
    "backend/pass/core_iter_stateful_adapters.vx",
    "backend/pass/core_iter_rev.vx",
    "backend/pass/core_iter_zip.vx",
    "backend/pass/core_iter_extend.vx",
    "backend/pass/core_iter_exact_size.vx",
    // A closure passed to a generic function: the flat emitter has no path for the call yet.
    "warnings/pass/w1001_a_called_local_is_used.vx",
    // "An enum with no modelled instance layout": `Option` of a tuple.
    "backend/pass/tuple_in_an_option.vx",
    // Same decline again: two chains whose `Map`s differ, for a name clash in the AST path.
    "backend/pass/generic_struct_instances_nested.vx",
    // A generic struct, `Cap<Count>`, which the flat path declines as "a struct with no GID".
    // The projections are resolved by the checker, before either code generator runs.
    "backend/pass/associated_type_projection.vx",
    // The flat path declines `main` as "a callee return type", since `wrap` and `some_pair`
    // answer with a generic struct and a generic enum. The answers come from the AST path.
    "backend/pass/nested_generic_names_in_generic_fns.vx",
    // `ok`, `err`, `map`, `map_err` and `and_then` all answer with an `Option` or a
    // `Result`, which the flat path declines as "a non-scalar default return" -- the same
    // shape as the file above. The AST path handles them, and that is where the answers
    // come from.
    "backend/pass/core_result.vx",
    // `partial_cmp` answers with an `Option<Ordering>`, and `then_with` and `max_by` take
    // closures that answer with an `Ordering`. The flat path declines the first as "a
    // callee return type" and the others as "an indirect callee returning a non-scalar".
    // These live apart from `core_cmp.vx` so that file keeps compiling through the flat
    // path; putting them together would have moved it here instead.
    "backend/pass/core_cmp_partial.vx",
    // `right_opt` and `left_opt` answer with an `Option<T>`, which the flat path declines
    // as "a non-scalar default return" -- the same shape as the file above. Those two
    // methods are the point of the file: a parameter only reaches the code that binds it
    // when it is handed to another generic type, so the AST path is where it is asked.
    "backend/pass/enum_binds_every_type_parameter.vx",
    // A generic enum returned from a match whose arms each return. The flat path declines it
    // as "a non-scalar default return" -- the same shape the AST path used to mis-lower, and
    // the reason that file exists. Its answers come from the AST path, and it states no
    // directive about emitted IR for that reason.
    "backend/pass/generic_enum_returned_from_match.vx",
    // A generic struct, which the flat path declines as "a struct with no GID". The file
    // exists to pin that `>>` still closes two generics now that it is also the right
    // shift, and that question is settled in the parser, so the decline costs it nothing.
    // It states no directive about emitted IR, precisely because it is on the AST path.
    // Indexing the result of a rank-1 elementwise operator. The rank-1 lowering
    // reads both operands as one `vector.load` and writes the result with one
    // arithmetic op, and records no memref type for the value it produced, so
    // `op_tensor_index` has nothing to load from and declines. The operator
    // itself lowers fine; only reading an element back does not. Rank 2 keeps
    // its buffer and is unaffected, which is why the rank-2 half of these
    // checks lives in `tensor_arithmetic_run.vx` and still runs here.
    "backend/pass/tensor_arithmetic_rank1_run.vx",
    "backend/pass/user_lowering_name_collisions.vx",
    "backend/pass/user_lowering_uncountable.vx",
    "backend/pass/user_lowering_waste.vx",
    "frontend/pass/closure_fat_ptr.vx",
    "frontend/pass/control_flow_rigorous.vx",
    // The `vxc -j` fallback fixture: a program the flat path declines, chosen so the parallel
    // frontend has something to hand back to the sequential driver. Same shape as
    // generic_enum_returned_from_match.vx, and it declines for the same reason.
    "frontend/pass/jobs_falls_back_outside_the_flat_subset.vx",
    // The same shape again, with a warning added: it states that a program checked by both
    // frontends has its warnings reported once, which needs a program that declines.
    "frontend/pass/jobs_warns_once_when_it_falls_back.vx",
    "frontend/pass/trait_topologies.vx",
    // A parameter with run-time extents (Vx#409). It used to compile through the flat path
    // while the dims-less spelling let it read as rank-0: `topology.vx` got a `memref<f32>`
    // signature where the AST oracle gives `memref<?x?xf32>`, two ABIs for one function, plus
    // a dropped vx.transfer. Declining is the honest answer until the flat lowerer carries
    // run-time extents.
    "warnings/pass/lowering_declined_for_dynamic_tile.vx",
];

/// Every `.vx` file under `dir`, recursively, sorted for a stable report.
fn corpus_programs(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    collect(dir, &mut found);
    found.sort();
    found
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read corpus directory {}: {e}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "vx") {
            out.push(path);
        }
    }
}

/// Programs where the flat path declines AND the AST fallback then fails to compile --
/// invalid MLIR or a panic on the default `--action emit-mlir` (Vx#398). Each is a real
/// compiler defect; the list exists so the set can only shrink, never silently grow.
const KNOWN_BROKEN: &[&str] = &[
    "frontend/pass/control_flow_rigorous.vx", // multi-payload variant binding (Vx#233)
];

/// Which codegen path the compiler took for one program, and -- when it fell back -- the reasons
/// the flat path gave.
#[derive(PartialEq)]
enum CodegenPath {
    Flat,
    Ast(Vec<String>),
    /// The flat path declined and the AST fallback then FAILED to compile (nonzero exit):
    /// the program does not build on the default path at all. Tracked in KNOWN_BROKEN (Vx#398)
    /// so the set can only shrink.
    AstBroken(Vec<String>),
}

/// The bracketed grouping key the driver prints after each decline: `... [unsupported-expr(Grad)]`.
fn decline_keys(log: &str) -> Vec<String> {
    log.lines()
        .filter(|l| {
            l.starts_with("[flat-codegen] declined")
                || l.starts_with("[flat-codegen] emit declined")
        })
        .filter_map(|l| {
            let start = l.rfind('[')?;
            Some(l[start + 1..].trim_end().trim_end_matches(']').to_string())
        })
        .collect()
}

/// Compile one program and report the path it took. `--action emit-mlir` stops at
/// MLIR, so no accelerator or JIT is needed and the answer is the same everywhere.
fn path_taken(program: &Path) -> Result<CodegenPath, String> {
    let output = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(program)
        .args(["--action", "emit-mlir"])
        .output()
        .map_err(|e| format!("could not run vxc: {e}"))?;

    let log = String::from_utf8_lossy(&output.stderr);
    if log.contains("emitted module via the flat path") {
        Ok(CodegenPath::Flat)
    } else if log.contains("program outside the flat subset") {
        // Falling back is not the same as the fallback working: the AST path can emit
        // invalid MLIR or panic after the decline, and an exit-blind classification counted
        // that as success for months (Vx#398).
        if output.status.success() {
            Ok(CodegenPath::Ast(decline_keys(&log)))
        } else {
            Ok(CodegenPath::AstBroken(decline_keys(&log)))
        }
    } else {
        // Neither marker: the program never reached codegen. Two ways that happens,
        // and both are worth failing on. Either the program stopped compiling, or a
        // debug assertion fired -- `driver.rs` deliberately panics in debug and test
        // builds when the flat emitter returns MLIR that will not parse, so this test
        // sees a panic where a release build would have fallen back and said nothing.
        let tail: Vec<&str> = log.lines().rev().take(12).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        Err(format!(
            "compiled through neither path (exit {:?}). In a debug build this is often \
             the flat emitter producing unparseable MLIR rather than a missing feature. \
             stderr tail:\n{}",
            output.status.code(),
            tail.join("\n")
        ))
    }
}

#[test]
fn flat_path_coverage_of_the_backend_corpus_holds() {
    // Every `pass` family, not just the backend's: the flat path compiles most of the tree
    // now, and gating only backend/pass is how two programs (llama2_math.vx, closures.vx)
    // panicked at HEAD with nothing noticing. `fail` directories stay out -- their programs
    // must not compile -- as do multi-file module roots.
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let families = [
        "backend/pass",
        "frontend/pass",
        "middle_end/pass",
        "optimizations/pass",
        "warnings/pass",
    ];
    let expected: BTreeSet<String> = KNOWN_DECLINES.iter().map(|s| s.to_string()).collect();

    let mut declined = BTreeSet::new();
    let mut fallback_broken = BTreeSet::new();
    let mut flat_count = 0usize;
    let mut broken = Vec::new();
    let mut by_reason: std::collections::BTreeMap<String, usize> = Default::default();
    let mut unexplained: Vec<String> = Vec::new();

    let mut corpus: Vec<(String, PathBuf)> = Vec::new();
    for family in families {
        for program in corpus_programs(&tests.join(family)) {
            let rel = program
                .strip_prefix(&tests)
                .unwrap_or(&program)
                .to_string_lossy()
                .to_string();
            corpus.push((rel, program));
        }
    }
    for (name, program) in corpus {
        // Programs that cannot compile as a bare `vxc file.vx` for reasons that are not the
        // flat path's business. Each names why; shrinking this list is separate work.
        const NOT_STANDALONE: &[&str] = &[
            "optimizations/pass/array_literal_nested.vx", // expects failure by design (RUN: not vxc)
            "optimizations/pass/codegen_error_diagnostics.vx", // expects failure by design (RUN: not vxc)
            "optimizations/pass/host_flag_scope.vx",           // needs --host
            "optimizations/pass/device_transfer_plugin.vx",    // needs --machine and a plugin
            "optimizations/pass/numa_peer_node_is_priced.vx",  // needs --host and --machine
            // needs --host and --machine
            "optimizations/pass/dtcm_tiles_in_sibling_blocks_fit_a_cortex_m7.vx",
        ];
        if NOT_STANDALONE.contains(&name.as_str()) {
            continue;
        }
        let source = std::fs::read_to_string(&program).unwrap_or_default();
        if source.contains("// REQUIRES:") {
            continue;
        }
        // `// XFAIL: *` marks a RUN line that cannot pass yet. Some of those programs still
        // compile (the RUN line fails on what the IR says), and those are swept like any other;
        // one written ahead of the compiler reaches neither path, and that is the state it
        // claims, so it is skipped until the marker comes off.
        let xfail = source.contains("// XFAIL: *");

        match path_taken(&program) {
            Ok(CodegenPath::Flat) => flat_count += 1,
            Ok(CodegenPath::Ast(reasons)) => {
                if reasons.is_empty() {
                    unexplained.push(name.clone());
                }
                // One program declines for one reason -- the first construct the flat path could
                // not carry. Counting them all would weight a program by how many later
                // constructs it happens to contain.
                if let Some(first) = reasons.first() {
                    *by_reason.entry(first.clone()).or_insert(0) += 1;
                }
                declined.insert(name);
            }
            Ok(CodegenPath::AstBroken(reasons)) => {
                if let Some(first) = reasons.first() {
                    *by_reason.entry(first.clone()).or_insert(0) += 1;
                }
                declined.insert(name.clone());
                fallback_broken.insert(name);
            }
            Err(_) if xfail => continue,
            Err(why) => broken.push(format!("{name}: {why}")),
        }
    }

    assert!(
        broken.is_empty(),
        "these corpus programs no longer compile at all:\n{}",
        broken.join("\n")
    );

    let newly_declining: Vec<_> = declined.difference(&expected).cloned().collect();
    let newly_covered: Vec<_> = expected.difference(&declined).cloned().collect();

    assert!(
        newly_declining.is_empty(),
        "flat-path coverage regressed. These programs used to compile through the \
         flat path and now fall back to the AST path:\n  {}\n\
         Run `VX_FLAT_DBG=1 cargo run --bin vxc -- tests/backend/pass/<program> \
         --action emit-mlir` to see which construct declined.",
        newly_declining.join("\n  ")
    );

    assert!(
        newly_covered.is_empty(),
        "flat-path coverage improved -- please record it. These programs now compile \
         through the flat path, so delete them from KNOWN_DECLINES in this file:\n  {}\n\
         (coverage is now {} of {} portable corpus programs)",
        newly_covered.join("\n  "),
        flat_count,
        flat_count + declined.len()
    );

    let broken_expected: BTreeSet<String> = KNOWN_BROKEN.iter().map(|s| s.to_string()).collect();
    let newly_broken: Vec<_> = fallback_broken
        .difference(&broken_expected)
        .cloned()
        .collect();
    let newly_fixed: Vec<_> = broken_expected
        .difference(&fallback_broken)
        .cloned()
        .collect();
    assert!(
        newly_broken.is_empty(),
        "these programs decline the flat path and the AST fallback then fails to compile: \
         the default path is broken for them (Vx#398 class):\n  {}",
        newly_broken.join(
            "
  "
        )
    );
    assert!(
        newly_fixed.is_empty(),
        "the AST fallback now compiles these -- please delete them from KNOWN_BROKEN:
  {}",
        newly_fixed.join(
            "
  "
        )
    );

    // A floor as well as an exact set: if the corpus itself shrinks, the exact-set
    // check above still passes while real coverage quietly drops.
    assert!(
        flat_count >= KNOWN_DECLINES.len(),
        "the corpus should exercise the flat path far more than it declines; \
         flat {flat_count}, declined {}",
        declined.len()
    );

    // Every decline says why. A silent one is a bail-out that was added without a reason, which is
    // the state this whole mechanism exists to prevent -- it would shrink the histogram below
    // without shrinking the gap.
    assert!(
        unexplained.is_empty(),
        "these programs fall back to the AST path without saying why:\n  {}",
        unexplained.join("\n  ")
    );

    println!(
        "flat path: {} of {} portable corpus programs ({} declined, tracked in Vx#383)",
        flat_count,
        flat_count + declined.len(),
        declined.len()
    );
    // What is missing, rather than which files are missing it. This is a report, not a gate: the
    // exact-set assertions above are what catch a regression, and duplicating them as expected
    // counts would be a second table to maintain that catches strictly less (a program starting to
    // decline and another stopping, for the same reason, leaves every count unchanged).
    let mut ranked: Vec<_> = by_reason.iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (reason, count) in ranked {
        println!("flat decline: {count:3} {reason}");
    }
}

/// Run the backend corpus through the FLAT path and check the answers (Vx#566).
///
/// `flat_codegen_differential` already executes the flat path against the AST oracle, but
/// reaches only 5 corpus programs by name. `test_backend` checks all 84 `// EXPECT:` lines
/// and builds them with `MeliorGenerator`, the legacy path. The sweep above compiles the
/// corpus on the flat path but records only which path each program took. So most of the
/// corpus had its answers checked on one code generator and its path choice on the other.
///
/// This runs the same expectations on the flat path -- 68 programs, no new assertions.
#[test]
fn flat_path_answers_match_the_backend_expectations() {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let declines: BTreeSet<&str> = KNOWN_DECLINES.iter().copied().collect();

    // The freshly built vxc leads; the rest of PATH carries the MLIR tools the JIT shells
    // out to (mlir-translate, llc, the linker).
    let bin_dir = Path::new(env!("CARGO_BIN_EXE_vxc")).parent().unwrap();
    let path_var = format!(
        "{}:{}",
        bin_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );

    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();

    // `read_dir` rather than the recursive `corpus_programs` above, because this must be
    // the SAME corpus `run_backend_test` runs, and that walks the directory without
    // descending. Subdirectories like backend/pass/autodiff/ are reached by their own RUN
    // lines instead, and their `// EXPECT:` lines assert on emitted IR rather than on what
    // the program printed -- recursing would quietly test a different thing against
    // assertions that were never about program output.
    let backend_dir = tests.join("backend/pass");
    let mut programs: Vec<PathBuf> = std::fs::read_dir(&backend_dir)
        .expect("tests/backend/pass is missing")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("vx"))
        .collect();
    programs.sort();

    for program in programs {
        let rel = program
            .strip_prefix(&tests)
            .unwrap_or(&program)
            .to_string_lossy()
            .to_string();
        if declines.contains(rel.as_str()) {
            continue;
        }
        let source = std::fs::read_to_string(&program).unwrap_or_default();

        // The same gates `run_backend_test` applies, so the two harnesses agree on what a
        // fixture has opted out of -- all but `// REQUIRES: flat-codegen`, which says the
        // legacy path gets the answers wrong and is therefore a reason to run a file here,
        // not to skip it. Every other `// REQUIRES:` is about which codegen path a
        // program takes, which is this file's other test, not a reason to skip running one.
        // `NO_EXEC` was the gate this loop first left out: a fixture that has never been run
        // can carry EXPECT lines nothing has ever compared, and `matmul_bf16.vx` did.
        if source.contains("// NO_EXEC") {
            continue;
        }
        if source.contains("// REQUIRES: macos") && !cfg!(target_os = "macos") {
            continue;
        }
        if source.contains("// REQUIRES: ane") {
            continue;
        }

        let expects: Vec<String> = source
            .lines()
            .filter(|l| l.trim().starts_with("// EXPECT:"))
            .map(|l| l.split_once("EXPECT:").unwrap().1.trim().to_string())
            .collect();
        if expects.is_empty() {
            continue;
        }

        let output = match Command::new(env!("CARGO_BIN_EXE_vxc"))
            .arg(&program)
            .env("PATH", &path_var)
            .output()
        {
            Ok(o) => o,
            Err(e) => {
                failures.push(format!("{rel}: could not run vxc: {e}"));
                continue;
            }
        };
        let log = String::from_utf8_lossy(&output.stderr);

        // Refuse to pass on a program that quietly took the AST path. Without this the test
        // decays into `test_backend` the moment the flat path declines something new -- which
        // is precisely the failure this test exists because of, so it is checked rather than
        // assumed.
        if !log.contains("emitted module via the flat path") {
            failures.push(format!(
                "{rel}: states EXPECT lines but did not compile through the flat path, so \
                 this test asserted nothing about it. Either the flat path started declining \
                 it -- add it to KNOWN_DECLINES with a reason -- or it failed to compile."
            ));
            continue;
        }

        // Both streams, because a fixture's expected output is not always on stdout --
        // ffi_stdio.vx writes deliberately to stderr, and an unwind message goes there too.
        // `run_backend_test` sees one combined string from the in-process JIT, so checking
        // only stdout here would fail programs the legacy path passes for no real reason.
        let out = format!("{}{}", String::from_utf8_lossy(&output.stdout), log);
        for expect in &expects {
            if !crate::integration_test::compile_test::expect_matches(&out, expect) {
                failures.push(format!(
                    "{rel}: flat path did not produce the expected answer.\n  \
                     expected to find: {expect}\n  actual stdout:\n{out}"
                ));
            }
        }
        checked += 1;
    }

    // A corpus that silently emptied would make every assertion above vacuous. The count is
    // a floor, not the exact number, so adding fixtures does not edit this line.
    assert!(
        checked >= 60,
        "only {checked} backend fixtures ran through the flat path; the corpus or the \
         skip rules above have changed enough that this test covers far less than it did"
    );
    assert!(
        failures.is_empty(),
        "{} of {} flat-path executions disagreed with the backend expectations:\n\n{}",
        failures.len(),
        checked,
        failures.join("\n\n")
    );
}
