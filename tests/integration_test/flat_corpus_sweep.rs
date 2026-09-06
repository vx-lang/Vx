//===- flat_corpus_sweep.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
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
    "backend/pass/llama2_v2.vx",
    "backend/pass/matmul_assign_alias.vx",
    "backend/pass/matmul_into_buffer.vx",
    "backend/pass/matvec_view_routing.vx",
    "backend/pass/option_unwrap.vx",
    "backend/pass/spliced_block_tails.vx",
    "backend/pass/tensor_view_2d.vx",
    "backend/pass/unwind.vx",
    "backend/pass/user_lowering_name_collisions.vx",
    "backend/pass/user_lowering_uncountable.vx",
    "backend/pass/user_lowering_waste.vx",
    "backend/pass/vec_option_elem.vx",
    "frontend/pass/borrow_closure_return_param.vx",
    "frontend/pass/closure_fat_ptr.vx",
    "frontend/pass/const_generics.vx",
    "frontend/pass/const_generics_multiple.vx",
    "frontend/pass/const_generics_nested.vx",
    "frontend/pass/control_flow.vx",
    "frontend/pass/control_flow_rigorous.vx",
    "frontend/pass/coverage_advanced_types_pass.vx",
    "frontend/pass/custom_matmul.vx",
    "frontend/pass/enum_match.vx",
    "frontend/pass/env_args.vx",
    "frontend/pass/gen_tensor_math_pass.vx",
    "frontend/pass/generics.vx",
    "frontend/pass/if_comptime_and_topology.vx",
    "frontend/pass/impl_most_specific_pattern_wins.vx",
    "frontend/pass/indirect_call.vx",
    "frontend/pass/inline_mlir_const_generics.vx",
    "frontend/pass/legal_acccore_transfer.vx",
    "frontend/pass/macro_vec_nested.vx",
    "frontend/pass/memory_algebra_implicit.vx",
    "frontend/pass/rubin_disaggregated.vx",
    "frontend/pass/spawn_result_located.vx",
    "frontend/pass/tensor_methods.vx",
    "frontend/pass/tensor_operations.vx",
    "frontend/pass/topology_spawn.vx",
    "frontend/pass/trait_topologies.vx",
    "frontend/pass/transfer_cost_advanced_dijkstra.vx",
    "frontend/pass/vector_algorithms.vx",
    "middle_end/pass/closure_return_ref.vx",
    "middle_end/pass/fnval_indirect_call.vx",
    "middle_end/pass/implicit_transfer.vx",
    "middle_end/pass/match_int_literal_arms.vx",
    "middle_end/pass/pinned_annotation_struct_field.vx",
    "middle_end/pass/reshape_pad.vx",
    // A parameter with run-time extents (Vx#409). It used to compile through the flat path
    // while the dims-less spelling let it read as rank-0: `topology.vx` got a `memref<f32>`
    // signature where the AST oracle gives `memref<?x?xf32>`, two ABIs for one function, plus
    // a dropped vx.transfer. Declining is the honest answer until the flat lowerer carries
    // run-time extents.
    "middle_end/pass/reshape_transpose.vx",
    "middle_end/pass/topology_polymorphism.vx",
    "optimizations/pass/cpu_lowering.vx",
    "optimizations/pass/kernel_kind_matmul.vx",
    "optimizations/pass/matmul_into_roles.vx",
    "optimizations/pass/matvec_roles.vx",
    "optimizations/pass/npu_lowering.vx",
    "optimizations/pass/topology_name_in_payload.vx",
    "optimizations/pass/vectorize.vx",
    "warnings/pass/lowering_declined_for_dynamic_tile.vx",
    "warnings/pass/w1024_implicit_transfer.vx",
    "warnings/pass/w1029_dynamic_shape_unverified.vx",
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
    "frontend/pass/control_flow.vx", // extractvalue on i32 (enum payload, Vx#233)
    "frontend/pass/control_flow_rigorous.vx", // multi-payload variant binding (Vx#233)
    "frontend/pass/enum_match.vx",   // extractvalue on i32 (enum payload, Vx#233)
    "frontend/pass/generics.vx",     // extractvalue on i32 (enum payload, Vx#233)
    "frontend/pass/memory_algebra_implicit.vx", // insertvalue of memref (Vx#356)
    "frontend/pass/transfer_cost_advanced_dijkstra.vx", // `.topology()` has no lowering
    "middle_end/pass/implicit_transfer.vx", // insertvalue of memref (Vx#356)
    "middle_end/pass/pinned_annotation_struct_field.vx", // insertvalue of memref (Vx#356)
    "warnings/pass/w1024_implicit_transfer.vx", // insertvalue of memref (Vx#356)
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
            "frontend/pass/const_generics_methods.vx", // checker rejects standalone (E2001 on N)
            "optimizations/pass/array_literal_nested.vx", // expects failure by design (RUN: not vxc)
            "optimizations/pass/codegen_error_diagnostics.vx", // expects failure by design (RUN: not vxc)
            "optimizations/pass/host_flag_scope.vx",           // needs --host
            "optimizations/pass/device_transfer_plugin.vx",    // needs --machine and a plugin
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
