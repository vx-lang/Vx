//===- compile_test.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the file-driven compilation test harness.
// It walks the `tests/` directory, compiles `.vx` files, and verifies that the
// compiler appropriately emits expected error messages for negative tests or
// successfully generates MLIR for positive tests.
//
//===----------------------------------------------------------------------===//
use std::fs;
use std::path::Path;
use std::sync::{Mutex, Once};

static INIT_RAYON: Once = Once::new();
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn init_rayon() {
    INIT_RAYON.call_once(|| {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let threads = if cores > 2 { cores - 2 } else { 1 };
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
    });
}

use rayon::prelude::*;

use vxc::hir::TypeChecker;
use vxc::jit::execute_mlir;

// Whether the ANE/CoreML backend is actually available: the CoreML primitive models are
// built by build.rs (needs coremltools + `xcrun coremlc`) into the project root. When they
// are absent the dispatcher falls back to CPU, so tests asserting ANE execution are skipped.
fn ane_models_available() -> bool {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("matmul_4x4.mlmodelc")
        .exists()
}

// Minimal FileCheck-style match for an EXPECT line: a `{{...}}` hole matches any text, so
// the literal segments around the holes must appear in order in `out`. Without a hole this
// is a plain substring check. (Lets tests use `{{[0-9]+}}` for non-deterministic values like
// a JIT kernel counter without pulling in a regex engine.)
fn expect_matches(out: &str, expect: &str) -> bool {
    if !expect.contains("{{") {
        return out.contains(expect);
    }
    let mut segments: Vec<&str> = Vec::new();
    let mut rest = expect;
    while let Some(start) = rest.find("{{") {
        segments.push(&rest[..start]);
        match rest[start..].find("}}") {
            Some(end) => rest = &rest[start + end + 2..],
            None => {
                rest = "";
                break;
            }
        }
    }
    segments.push(rest);
    let mut idx = 0;
    for seg in segments {
        if seg.is_empty() {
            continue;
        }
        match out[idx..].find(seg) {
            Some(pos) => idx += pos + seg.len(),
            None => return false,
        }
    }
    true
}

// Frontend Runner
fn run_frontend_test(path: &Path, expect_pass: bool) -> Result<(), String> {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    // A `pass` fixture written ahead of the compiler -- `// XFAIL: *` -- may not parse or
    // check yet. That is the state it claims, so it is not a failure here; the RUN-line
    // runner below reports the day it starts passing, which is the other half of the claim.
    let xfail = source.contains("// XFAIL: *");

    let mut loader = vxc::module_loader::ModuleLoader::new();
    if let Err(e) = loader.load_main(path.to_str().unwrap()) {
        if !expect_pass || xfail {
            return Ok(());
        }
        return Err(format!("Parse failed on {:?}: {}", path, e));
    }
    let mut program_arr = loader.into_programs();

    let syntax_idx = program_arr
        .iter()
        .position(|p| p.module_path.as_ref() == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(syntax_idx);

    let mut global_macros = std::collections::HashMap::new();
    for p in &program_arr {
        for mac in &p.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let expander = vxc::parser::MacroExpander::new(&global_macros);
    for p in &mut program_arr {
        if let Err(e) = expander.expand_module(p) {
            if !expect_pass || xfail {
                return Ok(());
            }
            return Err(format!(
                "Macro expansion failed on {}: {}",
                p.module_path, e
            ));
        }
    }
    if let Err(e) = expander.expand_module(&mut program) {
        if !expect_pass || xfail {
            return Ok(());
        }
        return Err(format!("Macro expansion failed on {:?}: {}", path, e));
    }

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::hir::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    // Opt-in directive: discharge per-seam boundary obligations (as `--verify-seams` does).
    checker.seam.verify = source.contains("// VERIFY-SEAMS");
    checker.check_topology_coherence(&program.topologies);
    checker.check_memory_coherence();
    for f in &mut program.functions {
        checker.check_function(f);
    }
    let is_valid = !checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error);

    if expect_pass {
        if !is_valid {
            if xfail {
                return Ok(());
            }
            return Err(format!(
                "Semantic analysis failed on {:?}:\n{:#?}",
                path, checker.errors
            ));
        }
    } else {
        if is_valid {
            return Err(format!(
                "Expected semantic failure on {:?}, but it passed",
                path
            ));
        }
    }

    // A file that states CHECK lines gets them executed. This runner used to stop at the
    // semantic verdict, so 281 CHECK lines across this directory asserted nothing and a file
    // could pass with its claims about the emitted IR flatly false (Vx#407). Running them is
    // what makes the header mean what it says. XFAIL files come along because that marker is
    // itself an assertion -- that the RUN line still cannot pass.
    if source.lines().any(|l| l.trim().starts_with("// CHECK")) || source.contains("// XFAIL: *") {
        run_lit_test(path, false)?;
    }
    Ok(())
}

/// Match `input` against a file's CHECK directives using the real FileCheck.
///
/// The middle-end runner compiles in process -- it drives passes the CLI has no flag for --
/// so there is no RUN line to hand a shell. Piping what it produced into FileCheck gets the
/// same matching the lit tiers have: `{{regex}}` holes, CHECK-NEXT, CHECK-DAG, capture
/// variables. It replaces an ordered substring scan that understood only plain `CHECK:` and
/// had to refuse everything else to avoid passing vacuously (Vx#407).
fn filecheck(input: &str, match_file: &Path, prefix: Option<&str>) -> Result<(), String> {
    use std::io::Write;
    use std::process::Stdio;
    let mut cmd = std::process::Command::new("FileCheck");
    cmd.arg(match_file);
    if let Some(p) = prefix {
        cmd.arg(format!("--check-prefix={}", p));
    }
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run FileCheck (is it on PATH?): {}", e))?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .map_err(|e| format!("writing to FileCheck: {}", e))?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("waiting on FileCheck: {}", e))?;
    if out.status.success() {
        return Ok(());
    }
    Err(format!(
        "FileCheck failed on {:?}:\n{}{}",
        match_file,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    ))
}

// Middle-End Runner
fn run_middle_end_test(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    // Host-gated MLIR checks (e.g. a device plugin only registered on macOS) are skipped
    // where the precondition does not hold, mirroring the backend runner's lit-style gate.
    if source.contains("// REQUIRES: macos") && !cfg!(target_os = "macos") {
        return Ok(());
    }

    let mut loader = vxc::module_loader::ModuleLoader::new();
    loader
        .load_main(path.to_str().unwrap())
        .expect("Failed to parse");
    let mut program_arr = loader.into_programs();

    let syntax_idx = program_arr
        .iter()
        .position(|p| p.module_path.as_ref() == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(syntax_idx);

    let mut global_macros = std::collections::HashMap::new();
    for p in &program_arr {
        for mac in &p.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let expander = vxc::parser::MacroExpander::new(&global_macros);
    for p in &mut program_arr {
        if let Err(e) = expander.expand_module(p) {
            return Err(format!(
                "Macro expansion failed on {}: {}",
                p.module_path, e
            ));
        }
    }
    if let Err(e) = expander.expand_module(&mut program) {
        return Err(format!("Macro expansion failed on {:?}: {}", path, e));
    }

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::hir::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }
    // Transfer lowerings, exactly as the driver checks them (#353 A1/A2): bodies with the
    // edge set so `raw::` resolves, then the structural pass (E6015) and the whole-body
    // pass (E6019/E6021/E6022). Without this the harness never emits a lowering
    // diagnostic and every lowering fail test passes vacuously -- the A2 review proved
    // it by planting a fully legal program with a bogus CHECK line, and it passed.
    for t in &mut program.transfer_impls {
        checker.seam.lowering_edge = Some((
            t.from.clone(),
            t.to.clone(),
            t.topology.display_name().to_string(),
        ));
        for f in &mut t.methods {
            checker.check_function(f);
        }
        checker.seam.lowering_edge = None;
    }
    checker.check_transfer_impl_bodies(&program.transfer_impls);
    // The whole-program declaration checks, by calling the method the compiler calls rather than
    // listing them again here. This list was hand-rolled and had already drifted once: the E6016
    // tests passed vacuously until topology coherence was added back, and a later check went
    // missing the same way. Calling the method means it cannot drift a third time.
    checker.check_whole_program_declarations();
    if checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)
    {
        return Err(format!("Sema failed on {:?}: {:#?}", path, checker.errors));
    }

    // Structs generated during checking (e.g. a closure's `Closure_N` environment) must reach
    // codegen, exactly as the driver does (`ast.structs.extend(checker.mono.generated_structs)`).
    // Without this the harness cannot lower any closure, unlike the real compiler.
    let generated_structs = std::mem::take(&mut checker.mono.generated_structs);

    let mut monomorphized_program = program;
    let mut orig_functions = monomorphized_program.functions;
    orig_functions.retain(|f| f.generics.is_empty());

    let mut new_functions: Vec<_> = checker.mono.functions.into_iter().map(|(f, _)| f).collect();
    new_functions.extend(orig_functions);
    monomorphized_program.functions = new_functions;
    monomorphized_program.structs.extend(generated_structs);

    let context = melior::Context::new();
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    vxc::codegen::register_vx_dialect(&context);

    let module_syntaxes = std::collections::HashMap::new();

    let mut codegen = vxc::codegen::MeliorGenerator::new(&context, "test".to_string());
    // Opt-in directive: exercise seam-certificate emission (`--emit-seam-certs`).
    codegen.emit_seam_certs = source.contains("// EMIT-SEAM-CERTS");
    codegen
        .generate(&monomorphized_program, &module_syntaxes)
        .unwrap();
    let mut module = codegen.into_module();
    let mlir_str = module.as_operation().to_string();

    // Lower, as the compiler does. The harness used to stop at the MLIR text: a fixture could
    // emit something that reads correctly, satisfy its CHECK lines, and be rejected by the real
    // binary on the same input. Three fixtures were in exactly that state, each hiding a defect
    // the suite reported as passing.
    //
    // `// XFAIL-LOWER:` marks one whose MLIR is what the fixture means to pin but which the
    // lowering cannot yet accept. It is read in both directions, as the RUN-line XFAIL is: a
    // marked fixture that starts lowering fails too, so the marker cannot outlive its bug.
    let xfail_lower = source
        .lines()
        .find(|l| l.trim().starts_with("// XFAIL-LOWER:"))
        .map(|l| l.split_once("XFAIL-LOWER:").unwrap().1.trim().to_string());
    match (
        vxc::codegen::lower_to_llvm(&context, &mut module),
        &xfail_lower,
    ) {
        (Err(e), None) => {
            return Err(format!(
                "{:?} emits MLIR its own compiler cannot lower: {}",
                path, e
            ))
        }
        (Ok(_), Some(reason)) => {
            return Err(format!(
                "{:?} is marked `XFAIL-LOWER: {}`, but it lowers now. Remove the marker.",
                path, reason
            ))
        }
        _ => {}
    }

    filecheck(&mlir_str, path, None)
}

// Warning Runner: type-checks the file (which must succeed with no errors) and asserts
// that every `// WARN:` directive appears in the emitted warning diagnostics (matched as a
// substring against the rendered warning, so either the code `W1024` or its message works).
fn run_warning_test(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path).expect("Failed to read test file");
    let want: Vec<String> = source
        .lines()
        .filter(|l| l.trim().starts_with("// WARN:"))
        .map(|l| l.split_once("WARN:").unwrap().1.trim().to_string())
        .collect();
    if want.is_empty() {
        return Err(format!(
            "Warning test {:?} has no `// WARN:` directives",
            path
        ));
    }

    let mut loader = vxc::module_loader::ModuleLoader::new();
    loader
        .load_main(path.to_str().unwrap())
        .map_err(|e| format!("Parse failed on {:?}: {}", path, e))?;
    let mut program_arr = loader.into_programs();
    let syntax_idx = program_arr
        .iter()
        .position(|p| p.module_path.as_ref() == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(syntax_idx);

    let mut global_macros = std::collections::HashMap::new();
    for p in &program_arr {
        for mac in &p.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let expander = vxc::parser::MacroExpander::new(&global_macros);
    for p in &mut program_arr {
        expander
            .expand_module(p)
            .map_err(|e| format!("Macro expansion failed on {}: {}", p.module_path, e))?;
    }
    expander
        .expand_module(&mut program)
        .map_err(|e| format!("Macro expansion failed on {:?}: {}", path, e))?;

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::hir::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    // Opt-in directive: discharge per-seam boundary obligations (as `--verify-seams` does).
    checker.seam.verify = source.contains("// VERIFY-SEAMS");
    checker.check_topology_coherence(&program.topologies);
    checker.check_memory_coherence();
    for f in &mut program.functions {
        checker.check_function(f);
    }

    let errors: Vec<String> = checker
        .errors
        .iter()
        .filter(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)
        .map(|d| d.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(format!(
            "Warning test {:?} must type-check cleanly, but got errors:\n{}",
            path,
            errors.join("\n")
        ));
    }

    let warnings: Vec<String> = checker
        .errors
        .iter()
        .filter(|d| d.level == vxc::diagnostic::DiagnosticLevel::Warning)
        .map(|d| d.to_string())
        .collect();
    for w in &want {
        if !warnings.iter().any(|got| got.contains(w.as_str())) {
            return Err(format!(
                "WARN check failed on {:?}: expected a warning containing `{}`.\nGot warnings:\n{}",
                path,
                w,
                warnings.join("\n")
            ));
        }
    }
    Ok(())
}

// Backend Runner
fn run_backend_test(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    // Extract // EXPECT: lines (assuming just one for simplicity right now)
    let expect_lines: Vec<String> = source
        .lines()
        .filter(|line| line.trim().starts_with("// EXPECT:"))
        .map(|line| line.split_once("EXPECT:").unwrap().1.trim().to_string())
        .collect();

    let mut loader = vxc::module_loader::ModuleLoader::new();
    if let Err(e) = loader.load_main(path.to_str().unwrap()) {
        return Err(format!(
            "Frontend failed to parse '{}': {}",
            path.display(),
            e
        ));
    }
    let mut program_arr = loader.into_programs();

    let syntax_idx = program_arr
        .iter()
        .position(|p| p.module_path.as_ref() == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(syntax_idx);

    let mut global_macros = std::collections::HashMap::new();
    for p in &program_arr {
        for mac in &p.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let expander = vxc::parser::MacroExpander::new(&global_macros);
    for p in &mut program_arr {
        if let Err(e) = expander.expand_module(p) {
            return Err(format!(
                "Macro expansion failed on {}: {}",
                p.module_path, e
            ));
        }
    }
    if let Err(e) = expander.expand_module(&mut program) {
        return Err(format!("Macro expansion failed on {:?}: {}", path, e));
    }

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::hir::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }
    // #203: also check imported modules' non-generic bodies in place, so methods/generics they call
    // only transitively get monomorphized for codegen (the env borrows `all_programs` clones, so
    // `program_arr` is free to mutate). Mirrors `driver.rs::run_semantic_analysis`; library-internal
    // diagnostics are dropped (this pass collects instantiations, it does not re-validate deps).
    let errors_before_imports = checker.errors.len();
    for p in &mut program_arr {
        for f in &mut p.functions {
            if f.generics.is_empty() {
                checker.check_function(f);
            }
        }
    }
    checker.errors.inner.truncate(errors_before_imports);
    if checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)
    {
        return Err(format!(
            "Semantic check failed on '{}':\n{:#?}",
            path.display(),
            checker.errors
        ));
    }
    let mut monomorphized_program = program;
    let mut orig_functions = monomorphized_program.functions;
    orig_functions.retain(|f| f.generics.is_empty());

    let mut new_functions: Vec<_> = checker.mono.functions.into_iter().map(|(f, _)| f).collect();
    new_functions.extend(orig_functions);
    monomorphized_program.functions = new_functions;
    monomorphized_program
        .structs
        .extend(checker.mono.generated_structs);
    let mut module_syntaxes = std::collections::HashMap::new();
    for mut p in program_arr {
        let before = p.functions.len();
        p.functions.retain(|f| f.generics.is_empty());
        let after = p.functions.len();
        println!(
            "Module {}: retained {} out of {} functions",
            p.module_path, after, before
        );
        for f in &p.functions {
            if f.name.as_ref() == "expect_eq" {
                println!(
                    "WARNING: expect_eq was retained! Generics: {:?}",
                    f.generics
                );
            }
        }
        module_syntaxes.insert(p.module_path.clone(), p);
    }

    let context = melior::Context::new();
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    vxc::codegen::register_vx_dialect(&context);

    println!("DEBUG: MeliorGenerator::new");
    let mut codegen = vxc::codegen::MeliorGenerator::new(&context, "test".to_string());
    println!("DEBUG: codegen.generate");
    codegen
        .generate(&monomorphized_program, &module_syntaxes)
        .unwrap();
    println!("DEBUG: codegen.into_module");
    let mut module = codegen.into_module();
    println!("DEBUG: lower_to_llvm");
    if let Err(e) = vxc::codegen::lower_to_llvm(&context, &mut module) {
        println!("MLIR Before Lowering Error:\n{}", module.as_operation());
        return Err(format!(
            "Lowering to LLVM failed for {}: {:?}",
            path.display(),
            e
        ));
    }
    println!("DEBUG: lower_to_llvm ok");
    let mlir_str = module.as_operation().to_string();
    println!("DEBUG: module_to_string ok");

    if source.contains("// NO_EXEC") {
        return Ok(());
    }

    if source.contains("// REQUIRES: macos") && !cfg!(target_os = "macos") {
        return Ok(());
    }

    // Tests that assert real ANE execution need the CoreML models (built by build.rs).
    // Absent them the dispatcher runs on the CPU fallback, which won't print the ANE output.
    if source.contains("// REQUIRES: ane") && !ane_models_available() {
        return Ok(());
    }

    let out = execute_mlir(&mlir_str, vec![], 0, false).expect("JIT execution failed");

    for expect in expect_lines {
        if !expect_matches(&out, &expect) {
            return Err(format!(
                "Backend output mismatch on {:?}.\nExpected to find: `{}`\nActual Output:\n{}",
                path, expect, out
            ));
        }
    }
    Ok(())
}

#[test]
fn test_frontend_pass() -> Result<(), String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/frontend/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        let errors: Vec<String> = entries
            .into_par_iter()
            .filter_map(|entry| {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("vx") {
                    if let Err(e) = run_frontend_test(&path, true) {
                        return Some(e);
                    }
                }
                None
            })
            .collect();
        if !errors.is_empty() {
            return Err(format!(
                "The following tests failed:\n\n{}",
                errors.join("\n\n")
            ));
        }
    }
    Ok(())
}

#[test]
fn test_frontend_fail() -> Result<(), String> {
    let has_z3 = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .is_ok();
    let _ = has_z3;
    // `formal_verification` and `unimplemented_smt` have runners of their own, which gate on z3.
    // Everything else under here is walked, including the subdirectories that had no runner at
    // all: 39 fixtures that never executed, and so drifted.
    run_directory_tests_recursive(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/frontend/fail"),
        &["formal_verification", "unimplemented_smt"],
        run_shell_tests,
    )
}

#[test]
fn test_optimizations() -> Result<(), String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/optimizations/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        let errors: Vec<String> = entries
            .into_par_iter()
            .filter_map(|entry| {
                let path = entry.path();
                let ext = path.extension().and_then(|s| s.to_str());
                if path.is_file() && (ext == Some("mlr") || ext == Some("vx")) {
                    println!("Running test_optimizations on {:?}", path);
                    if let Err(e) = run_optimization_test(&path) {
                        return Some(e);
                    }
                }
                None
            })
            .collect();
        if !errors.is_empty() {
            return Err(format!(
                "The following tests failed:\n\n{}",
                errors.join("\n\n")
            ));
        }
    }
    Ok(())
}

/// Execute a fixture's `RUN:` lines the way lit would, in order.
///
/// The middle-end tiers generate and check MLIR in process, so the command a fixture documents
/// was never run: 18 of 111 fixtures carried a RUN line that could not pass, and nothing noticed.
/// This runs it in addition to the in-process checks rather than instead of them -- the two see
/// different things, and the point is that they agree.
///
/// `// XFAIL-RUN:` marks a fixture whose RUN line cannot pass yet. Read in both directions, as
/// `XFAIL-LOWER` is: a marked fixture whose RUN line starts passing fails too.
fn run_the_run_lines(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path).map_err(|e| format!("{:?}: {}", path, e))?;
    let runs: Vec<&str> = source
        .lines()
        .filter_map(|l| l.trim().strip_prefix("// RUN:"))
        .map(|c| c.trim())
        .collect();
    let xfail_run = source
        .lines()
        .find(|l| l.trim().starts_with("// XFAIL-RUN:"))
        .map(|l| l.split_once("XFAIL-RUN:").unwrap().1.trim().to_string());
    if runs.is_empty() {
        return Err(format!(
            "{:?} has no RUN line, so nothing states what the compiler should do with it",
            path
        ));
    }

    let vxc = env!("CARGO_BIN_EXE_vxc");
    let tmp_base = std::env::temp_dir().join(format!(
        "vx-runline-{}-{}",
        std::process::id(),
        path.file_stem().unwrap().to_string_lossy()
    ));
    let mut failure: Option<String> = None;
    for run in &runs {
        let cmd = run
            .replace("%s", &path.to_string_lossy())
            .replace("%t", &tmp_base.to_string_lossy());
        // `vxc` as a bare word is the binary under test, not whatever is on PATH.
        let cmd = cmd.replacen("vxc ", &format!("{} ", vxc), 1);
        let out = std::process::Command::new("bash")
            .arg("-c")
            .arg(&cmd)
            .output()
            .map_err(|e| format!("{:?}: could not run `{}`: {}", path, run, e))?;
        if !out.status.success() {
            failure = Some(format!(
                "{:?}: RUN line failed: {}\n{}",
                path,
                run,
                String::from_utf8_lossy(&out.stderr)
                    .lines()
                    .take(6)
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
            break;
        }
    }
    let _ = std::fs::remove_file(&tmp_base);
    match (failure, xfail_run) {
        (Some(why), None) => Err(why),
        (None, Some(reason)) => Err(format!(
            "{:?} is marked `XFAIL-RUN: {}`, but its RUN line passes now. Remove the marker.",
            path, reason
        )),
        _ => Ok(()),
    }
}

#[test]
fn test_middle_end() -> Result<(), String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/middle_end/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        let errors: Vec<String> = entries
            .into_par_iter()
            .filter_map(|entry| {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                    if let Err(e) = run_middle_end_test(&path) {
                        return Some(e);
                    }
                    if let Err(e) = run_the_run_lines(&path) {
                        return Some(e);
                    }
                }
                None
            })
            .collect();
        if !errors.is_empty() {
            return Err(format!(
                "The following tests failed:\n\n{}",
                errors.join("\n\n")
            ));
        }
    }
    Ok(())
}

#[test]
fn test_warnings() -> Result<(), String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/warnings/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        let errors: Vec<String> = entries
            .into_par_iter()
            .filter_map(|entry| {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                    if let Err(e) = run_warning_test(&path) {
                        return Some(e);
                    }
                }
                None
            })
            .collect();
        if !errors.is_empty() {
            return Err(format!(
                "The following warning tests failed:\n\n{}",
                errors.join("\n\n")
            ));
        }
    }
    Ok(())
}

// Optimization Test Runner
fn run_optimization_test(path: &Path) -> Result<(), String> {
    // These files pin the AST codegen's MLIR structure, so they force the legacy path.
    run_lit_test(path, true)
}

/// Execute a file's `// RUN:` lines through a shell, so `| FileCheck %s` runs the real
/// FileCheck and every directive it understands works: `{{regex}}` holes, `CHECK-NEXT`,
/// `CHECK-SAME`, `CHECK-DAG`, `CHECK-COUNT`, capture variables.
///
/// This used to reimplement FileCheck as an ordered substring scan over `CHECK:` and
/// `CHECK-NOT:`, which meant the pipe in the RUN line was decoration -- anything else a
/// file wrote had to be refused, or it would pass while asserting nothing (Vx#407).
///
/// `force_legacy` pins the AST codegen, which the optimizations and backend tests want
/// because they assert that path's exact IR. The frontend tests run the default path
/// instead: they assert what a user actually gets, and forcing legacy there would report
/// the AST path's own bugs as frontend failures (three such programs are recorded on
/// Vx#398).
fn run_lit_test(path: &Path, force_legacy: bool) -> Result<(), String> {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    // Same lit-style gate the middle-end and backend runners apply. A check against IR that
    // only a macOS-registered plugin can produce is not a failure elsewhere, it is a test
    // that does not apply.
    if source.contains("// REQUIRES: macos") && !cfg!(target_os = "macos") {
        return Ok(());
    }

    let run_lines: Vec<&str> = source
        .lines()
        .filter(|l| l.trim_start().starts_with("// RUN:"))
        .collect();
    if run_lines.is_empty() {
        return Err(format!("{:?} has no // RUN: line", path));
    }

    // `// XFAIL: *` marks a RUN line that cannot pass yet because the compiler is wrong,
    // not because the test is. It keeps the command running, so the day the bug is fixed
    // the file reports "expected to fail, but passed" and its assertions get restored --
    // which deleting the RUN line would not do.
    let xfail = source.contains("// XFAIL: *");

    // A file that states CHECK lines has to feed something into FileCheck, or the directives
    // assert nothing. FileCheck itself catches the other half of this -- a prefix named on
    // the command line with no directives behind it is an error, not an empty pass.
    let states_checks = source
        .lines()
        .any(|l| l.trim_start().starts_with("// CHECK"));
    if states_checks && !xfail && !run_lines.iter().any(|l| l.contains("FileCheck")) {
        return Err(format!(
            "{:?} states CHECK lines, but no RUN line pipes into FileCheck, so nothing \
             matches them.",
            path
        ));
    }

    // The freshly built vxc/vx-opt lead; the rest of PATH is what carries FileCheck.
    let bin_dir = Path::new(env!("CARGO_BIN_EXE_vxc")).parent().unwrap();
    let path_var = format!(
        "{}:{}",
        bin_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );

    // %t: a scratch path unique to this file, inside the repo's own target directory.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&path, &mut hasher);
    let hash = std::hash::Hasher::finish(&hasher);
    let tmp_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test_tmp");
    std::fs::create_dir_all(&tmp_dir).unwrap_or_default();
    let t_val = tmp_dir.join(format!(
        "vxc_test_{}_{:x}",
        path.file_stem().unwrap().to_string_lossy(),
        hash
    ));

    for run_line in run_lines {
        let mut cmd = run_line
            .split_once("RUN:")
            .unwrap()
            .1
            .trim()
            .replace("%s", path.to_str().unwrap())
            .replace("%t", t_val.to_str().unwrap());

        // `// REQUIRES: flat-codegen` opts a file out: some constructs exist ONLY on the flat
        // path (`flash_attention_into`, whose note-and-nest emission is what
        // kernel_kind_attention.vx pins), and forcing legacy there tests a lowering that
        // deliberately does not exist.
        // Skip a command that already names the flag: clap rejects it twice.
        if force_legacy
            && !cmd.contains("--legacy-codegen")
            && !source.contains("// REQUIRES: flat-codegen")
        {
            let split = cmd.find('|').unwrap_or(cmd.len());
            let (head, tail) = cmd.split_at(split);
            let mut words = head.split_whitespace();
            let first = words.next();
            if first == Some("vxc") || (first == Some("not") && words.next() == Some("vxc")) {
                cmd = format!("{} --legacy-codegen {}", head.trim_end(), tail);
            }
        }

        // pipefail, because a RUN line fails if any stage fails. A shell reports only the
        // last stage, so a compiler that crashed would still pass whenever its stderr
        // happened to satisfy the CHECK lines.
        let output = std::process::Command::new("bash")
            .arg("-c")
            .arg(format!("set -o pipefail; {}", cmd))
            .env("PATH", &path_var)
            .env("RUST_BACKTRACE", "1")
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("Failed to execute RUN line");

        if xfail {
            if !output.status.success() {
                return Ok(());
            }
            continue;
        }

        if !output.status.success() {
            return Err(format!(
                "RUN line failed for {:?}:\n  {}\nStdout:\n{}\nStderr:\n{}",
                path,
                cmd.trim(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }
    if xfail {
        return Err(format!(
            "{:?} is marked XFAIL, but every RUN line succeeded. The bug it waits on is \
             fixed: drop the XFAIL and restore the assertions.",
            path
        ));
    }
    Ok(())
}

#[test]
fn test_middle_end_fail() -> Result<(), String> {
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/middle_end/fail"),
        |path| {
            // A fail test must fail AND fail for the declared reason: the error text has
            // to satisfy the file's CHECK directives. "Any error passes" let a test keep
            // passing while the diagnostic it pinned was deleted -- the A2 review planted
            // a legal program with `// CHECK: E9999` and it passed.
            let source = fs::read_to_string(path).expect("Failed to read test file");
            // `// REQUIRES: z3`: the pinned diagnostic needs the prover, and the prover
            // fails OPEN without z3 (the program compiles). Skip rather than fail on a
            // machine without it, the same policy test_frontend_fail applies.
            if source.contains("// REQUIRES: z3") {
                let has_z3 = std::process::Command::new("z3")
                    .arg("--version")
                    .output()
                    .is_ok();
                if !has_z3 {
                    return Ok(());
                }
            }
            match run_middle_end_test(path) {
                Ok(()) => Err(format!(
                    "Expected {} to fail, but it succeeded!",
                    path.display()
                )),
                Err(e) => filecheck(&e, path, None).map_err(|why| {
                    format!(
                        "{} failed, but not for the declared reason.\n{}",
                        path.display(),
                        why
                    )
                }),
            }?;
            // And the command the fixture documents has to refuse it too, for the same reason.
            run_the_run_lines(path)
        },
    )
}

#[test]
fn test_crash() -> Result<(), String> {
    for i in 0..5 {
        println!("test_crash iteration {}", i);
        run_backend_test(Path::new("tests/backend/pass/plugin_npe.vx"))?;
        println!("test_crash iteration {} done", i);
    }
    Ok(())
}

#[test]
fn test_backend() -> Result<(), String> {
    std::env::set_var("RUST_BACKTRACE", "1");
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/backend/pass"),
        |path| {
            println!("Running test_backend on {:?}", path);
            run_backend_test(path)?;
            // Any file that states CHECK lines, or whose RUN line names FileCheck, gets
            // those RUN lines executed. Matching on one exact spelling of the RUN line
            // left most of this directory's CHECK lines inert, and they had drifted
            // (Vx#407). CHECK lines alone are enough: a file that states them and has no
            // RUN line should say so, not slip past.
            let source = std::fs::read_to_string(path).unwrap_or_default();
            let has_run = source
                .lines()
                .any(|l| l.trim_start().starts_with("// RUN:"));
            let states_checks = source
                .lines()
                .any(|l| l.trim_start().starts_with("// CHECK"));
            if states_checks || (has_run && source.contains("FileCheck")) {
                run_optimization_test(path)?;
            }
            Ok(())
        },
    )
}

// The multi-file module roots. `run_directory_tests` reads one level and takes only files, so
// a subdirectory of `frontend/pass` is walked by nothing unless it is named here -- which is
// how these three sat for a long time importing a `.ak` extension that no longer exists,
// against a `comptime { import(..) }` form the parser had stopped accepting (Vx#412).
#[test]
fn test_frontend_pass_modules() -> Result<(), String> {
    for dir in [
        "tests/frontend/pass/modules_basic",
        "tests/frontend/pass/modules_nested",
    ] {
        run_directory_tests(Path::new(env!("CARGO_MANIFEST_DIR")).join(dir), |path| {
            println!("Running test_frontend_pass_modules on {:?}", path);
            run_frontend_test(path, true)
        })?;
    }
    Ok(())
}

#[test]
fn test_frontend_pass_formal_verification() -> Result<(), String> {
    if std::process::Command::new("z3")
        .arg("--version")
        .output()
        .is_err()
    {
        return Err("z3 is not installed".to_string());
    }
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/frontend/pass/formal_verification"),
        |path| {
            println!(
                "Running test_frontend_pass_formal_verification on {:?}",
                path
            );
            run_frontend_test(path, true)
        },
    )
}

#[test]
fn test_frontend_fail_formal_verification() -> Result<(), String> {
    if std::process::Command::new("z3")
        .arg("--version")
        .output()
        .is_err()
    {
        return Err("z3 is not installed".to_string());
    }
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/frontend/fail/formal_verification"),
        |path| {
            println!(
                "Running test_frontend_fail_formal_verification on {:?}",
                path
            );
            run_shell_tests(path)
        },
    )
}

#[test]
fn test_frontend_fail_unimplemented_smt() -> Result<(), String> {
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/frontend/fail/unimplemented_smt"),
        |path| {
            println!("Running test_frontend_fail_unimplemented_smt on {:?}", path);
            run_shell_tests(path)
        },
    )
}

#[test]
fn test_backend_pass_autodiff() -> Result<(), String> {
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/backend/pass/autodiff"),
        |path| {
            println!("Running test_backend_pass_autodiff on {:?}", path);
            run_backend_autodiff_test(path)
        },
    )
}

// Backend Autodiff Runner
fn run_backend_autodiff_test(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    let expect_lines: Vec<String> = source
        .lines()
        .filter(|line| line.trim().starts_with("// EXPECT:"))
        .map(|line| line.split_once("EXPECT:").unwrap().1.trim().to_string())
        .collect();

    let mut loader = vxc::module_loader::ModuleLoader::new();
    if let Err(e) = loader.load_main(path.to_str().unwrap()) {
        return Err(format!(
            "Frontend failed to parse '{}': {}",
            path.display(),
            e
        ));
    }
    let mut program_arr = loader.into_programs();

    let syntax_idx = program_arr
        .iter()
        .position(|p| p.module_path.as_ref() == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(syntax_idx);

    let mut global_macros = std::collections::HashMap::new();
    for p in &program_arr {
        for mac in &p.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let expander = vxc::parser::MacroExpander::new(&global_macros);
    for p in &mut program_arr {
        if let Err(e) = expander.expand_module(p) {
            return Err(format!(
                "Macro expansion failed on {}: {}",
                p.module_path, e
            ));
        }
    }
    if let Err(e) = expander.expand_module(&mut program) {
        return Err(format!("Macro expansion failed on {:?}: {}", path, e));
    }

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::hir::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }
    // #203: also check imported modules' non-generic bodies in place, so methods/generics they call
    // only transitively get monomorphized for codegen (the env borrows `all_programs` clones, so
    // `program_arr` is free to mutate). Mirrors `driver.rs::run_semantic_analysis`; library-internal
    // diagnostics are dropped (this pass collects instantiations, it does not re-validate deps).
    let errors_before_imports = checker.errors.len();
    for p in &mut program_arr {
        for f in &mut p.functions {
            if f.generics.is_empty() {
                checker.check_function(f);
            }
        }
    }
    checker.errors.inner.truncate(errors_before_imports);
    if checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)
    {
        return Err(format!(
            "Semantic check failed on '{}':\n{:#?}",
            path.display(),
            checker.errors
        ));
    }
    let mut monomorphized_program = program;
    let mut orig_functions = monomorphized_program.functions;
    orig_functions.retain(|f| f.generics.is_empty());

    let mut new_functions: Vec<_> = checker.mono.functions.into_iter().map(|(f, _)| f).collect();
    new_functions.extend(orig_functions);
    monomorphized_program.functions = new_functions;

    let mut module_syntaxes = std::collections::HashMap::new();
    for mut p in program_arr {
        p.functions.retain(|f| f.generics.is_empty());
        module_syntaxes.insert(p.module_path.clone(), p);
    }

    let context = melior::Context::new();
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    vxc::codegen::register_vx_dialect(&context);

    let mut codegen = vxc::codegen::MeliorGenerator::new(&context, "test".to_string());
    codegen
        .generate(&monomorphized_program, &module_syntaxes)
        .unwrap();

    let mut module = codegen.into_module();
    if let Err(e) = vxc::codegen::lower_to_llvm(&context, &mut module) {
        println!("MLIR Before Lowering Error:\n{}", module.as_operation());
        return Err(format!(
            "Lowering to LLVM failed for {}: {:?}",
            path.display(),
            e
        ));
    }
    let mlir_str = module.as_operation().to_string();

    // This runner read `// EXPECT:` and nothing else, so every CHECK line in this directory
    // was inert -- including twelve in autodiff_basic.vx that had never once run (Vx#407 left
    // this directory out). A file that states them gets them matched, like every other tier.
    if source
        .lines()
        .any(|l| l.trim_start().starts_with("// CHECK"))
    {
        filecheck(&mlir_str, path, None)?;
    }

    if source.contains("// NO_EXEC") {
        for expect in expect_lines {
            if !mlir_str.contains(&expect) {
                return Err(format!(
                    "MLIR Output mismatch on {:?}.\nExpected to find: `{}`\nActual MLIR:\n{}",
                    path, expect, mlir_str
                ));
            }
        }
        return Ok(());
    }

    if std::env::var("ENZYME_LIB").is_err() {
        println!(
            "Skipping JIT execution for {} because ENZYME_LIB is not set.",
            path.display()
        );
        return Ok(());
    }

    let out = execute_mlir(&mlir_str, vec![], 0, false).expect("JIT execution failed");

    for expect in expect_lines {
        if !mlir_str.contains(&expect) && !out.contains(&expect) {
            return Err(format!(
                "Output mismatch on {:?}.\nExpected to find: `{}`\nActual Output:\n{}\nMLIR:\n{}",
                path, expect, out, mlir_str
            ));
        }
    }
    Ok(())
}

#[test]
fn test_backend_fail() -> Result<(), String> {
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/backend/fail"),
        run_shell_tests,
    )
}

// --- Test Runners ---
/// Every `.vx` under `dir`, recursively, except inside a directory named in `skip`.
///
/// `skip` exists because a subdirectory can have a runner of its own that does something this
/// one does not -- a z3 gate, or a different command entirely. Walking it here would run those
/// fixtures the wrong way, so they are left to their own test.
fn vx_files_under(dir: &Path, skip: &[&str], out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !skip.contains(&name) {
                vx_files_under(&path, skip, out);
            }
        } else if path.extension().and_then(|s| s.to_str()) == Some("vx") {
            out.push(path);
        }
    }
}

fn run_directory_tests<F>(dir: std::path::PathBuf, test_fn: F) -> Result<(), String>
where
    F: Fn(&std::path::Path) -> Result<(), String> + Sync + Send,
{
    run_directory_tests_inner(dir, &[], false, test_fn)
}

/// As [`run_directory_tests`], but walking subdirectories too, except those named in `skip`.
fn run_directory_tests_recursive<F>(
    dir: std::path::PathBuf,
    skip: &[&str],
    test_fn: F,
) -> Result<(), String>
where
    F: Fn(&std::path::Path) -> Result<(), String> + Sync + Send,
{
    run_directory_tests_inner(dir, skip, true, test_fn)
}

fn run_directory_tests_inner<F>(
    dir: std::path::PathBuf,
    skip: &[&str],
    recurse: bool,
    test_fn: F,
) -> Result<(), String>
where
    F: Fn(&std::path::Path) -> Result<(), String> + Sync + Send,
{
    init_rayon();
    let _guard = TEST_MUTEX.lock().unwrap();

    if dir.exists() {
        let entries: Vec<std::path::PathBuf> = if recurse {
            let mut v = Vec::new();
            vx_files_under(&dir, skip, &mut v);
            v
        } else {
            fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect()
        };
        let errors: Vec<String> = entries
            .into_par_iter()
            .filter_map(|path| {
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                    let result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| test_fn(&path)));
                    match result {
                        Ok(Err(msg)) => return Some(format!("Test {:?} failed: {}", path, msg)),
                        Err(e) => {
                            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                                s.to_string()
                            } else if let Some(s) = e.downcast_ref::<String>() {
                                s.to_string()
                            } else {
                                "Unknown panic".to_string()
                            };
                            return Some(format!("Test {:?} panicked: {}", path, msg));
                        }
                        Ok(Ok(())) => {}
                    }
                }
                None
            })
            .collect();
        if !errors.is_empty() {
            return Err(format!(
                "The following tests failed:\n\n{}",
                errors.join("\n\n")
            ));
        }
    }
    Ok(())
}

// The fail tiers name their own command, so nothing is pinned for them; everything else
// the lit runner does -- the real FileCheck, pipefail, %s/%t -- applies here too.
fn run_shell_tests(path: &Path) -> Result<(), String> {
    run_lit_test(path, false)
}

#[test]
fn test_melior_matmul() -> Result<(), String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/middle_end/pass/matmul.vx");
    let mut loader = vxc::module_loader::ModuleLoader::new();
    loader
        .load_main(path.to_str().unwrap())
        .expect("Failed to parse");
    let mut program_arr = loader.into_programs();

    let syntax_idx = program_arr
        .iter()
        .position(|p| p.module_path.as_ref() == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(syntax_idx);

    let mut global_macros = std::collections::HashMap::new();
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let expander = vxc::parser::MacroExpander::new(&global_macros);
    if let Err(e) = expander.expand_module(&mut program) {
        return Err(format!("Macro expansion failed on {:?}: {}", path, e));
    }

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::hir::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    let mut checked_program = program.clone();
    for f in &mut checked_program.functions {
        checker.check_function(f);
    }
    if checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)
    {
        return Err(format!("Sema failed on {:?}: {:#?}", path, checker.errors));
    }

    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);

    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    vxc::codegen::register_vx_dialect(&context);

    let mut gen = vxc::codegen::MeliorGenerator::new(&context, "test".to_string());
    let mlir_str = gen
        .generate(&checked_program, &std::collections::HashMap::new())
        .unwrap();

    filecheck(&mlir_str, &path, None)
}

extern "C" {
    fn registerVxDialect(ctx: mlir_sys::MlirContext);
}

#[test]
fn test_vx_dialect_registration() -> Result<(), String> {
    let registry = melior::dialect::DialectRegistry::new();
    let context = melior::Context::new();

    // Register our custom dialect using the FFI
    unsafe {
        registerVxDialect(context.to_raw());
    }

    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // We can't directly check the registered dialects easily in melior without parsing,
    // but we can parse a dummy module that requires the `vx` dialect.
    let mlir_source = r#"
        module {
            "vx.spawn"() <{topology = 100 : i32}> ({}) : () -> ()
        }
    "#;

    // If the dialect wasn't registered, parsing this would fail.
    let module = melior::ir::Module::parse(&context, mlir_source);
    if module.is_none() {
        return Err(
            "Failed to parse module containing vx.spawn. Dialect may not be registered!"
                .to_string(),
        );
    }
    Ok(())
}
