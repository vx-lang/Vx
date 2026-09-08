//===- flat_codegen_differential.rs - Vx Compiler --------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Differential testing of the *flat* codegen path against the AST path, the
// oracle (#200). For a function the flat HIR lowers today, the flat path
// (local_hir_stream -> `flat::emit_module_mlir`) must produce MLIR that, run
// through the *same* production lowering (`lower_to_llvm`) and JIT, yields the
// *same* result as the AST path. Parity here is what lets `vxc` eventually flip
// to the flat path. Coverage grows as the flat HIR + emitter widen (control
// flow, calls, then the non-scalar surface #199).
//===----------------------------------------------------------------------===//
use vxc::codegen::{lower_to_llvm, register_vx_dialect, MeliorGenerator};
use vxc::hir::flatten::lower_function_to_hir;
use vxc::hir::{GlobalAstEnv, TypeChecker};
use vxc::jit::execute_mlir;
use vxc::lexer::Lexer;
use vxc::parser::Parser;
use vxc::session::{GlobalSession, LocalWorkerState};
use vxc::syntax::Program;

fn make_context() -> melior::Context {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    register_vx_dialect(&context);
    context
}

fn parse(src: &str) -> Program {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(&tokens, src);
    let mut program = parser.parse().expect("parse failed");
    // Expand macros (`print!`/`println!` -> `Expr::Print`/`Expr::Println`, plus any user macros)
    // before name resolution, exactly as the real pipeline does — otherwise a `MacroCall` reaches
    // resolution and panics.
    let mut global_macros = std::collections::HashMap::new();
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let expander = vxc::parser::MacroExpander::new(&global_macros);
    expander
        .expand_module(&mut program)
        .expect("macro expansion failed");
    program
}

/// JIT the given LLVM-dialect MLIR and return the process exit code (`main`'s
/// return value): `execute_mlir` yields `Ok` for a zero exit and an `Err`
/// carrying the code otherwise.
fn exit_code(llvm_mlir: &str) -> i32 {
    exit_code_at_opt(llvm_mlir, 0)
}

/// The process exit code of the given LLVM-dialect MLIR JIT-compiled at a chosen optimization level.
/// `execute_mlir` runs `mlir-translate --mlir-to-llvmir` (which lowers MLIR `alias_scopes`/`noalias_scopes`
/// to LLVM IR `!alias.scope`/`!noalias`) then `opt -passes=default<O{opt}>` — so `opt > 0` is where an
/// optimizer actually *consumes* the alias metadata. Used by the §16.3 `-O0`-vs-`-O2` differential.
fn exit_code_at_opt(llvm_mlir: &str, opt: u8) -> i32 {
    match execute_mlir(llvm_mlir, vec![], opt, false) {
        Ok(_) => 0,
        Err(e) => e
            .rsplit(':')
            .next()
            .and_then(|s| s.trim().parse::<i32>().ok())
            .unwrap_or_else(|| panic!("unexpected execute_mlir error at -O{opt}: {e}")),
    }
}

/// Normalize JIT stdout for comparison: the `printMemref*` helper prints a
/// non-deterministic heap pointer (`base@ = 0x...`); replace every `0x<hex>` with
/// a placeholder so the shape/strides/data (which *are* deterministic) compare.
fn normalize(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(pos) = rest.find("0x") {
        out.push_str(&rest[..pos]);
        out.push_str("0x<ptr>");
        let after = &rest[pos + 2..];
        let hexlen = after.chars().take_while(|c| c.is_ascii_hexdigit()).count();
        rest = &after[hexlen..];
    }
    out.push_str(rest);
    out
}

/// The (normalized) stdout of running the given LLVM-dialect MLIR through the JIT.
/// Panics if the program exits non-zero (a printing program returns 0).
fn run_output(llvm_mlir: &str) -> String {
    match execute_mlir(llvm_mlir, vec![], 0, false) {
        Ok(out) => normalize(&out),
        Err(e) => panic!("printing program exited non-zero: {e}"),
    }
}

/// The lowered LLVM-dialect MLIR for `main` compiled through the AST codegen (the oracle).
fn ast_llvm(src: &str) -> String {
    let mut program = parse(src);
    let program_arr = [program.clone()];
    let env = GlobalAstEnv::build(&program_arr);
    let mut worker = LocalWorkerState::new(std::sync::Arc::new(GlobalSession::new(1)));
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }
    assert_eq!(
        checker.errors.error_count(),
        0,
        "AST type-checks:\n{}",
        checker
            .errors
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );
    // Append the monomorphs the checker collected (method-call rewrites like `x.sq()` -> `f32$sq`,
    // generic instances) so their bodies emit and the rewritten calls resolve — as the driver does.
    for (f, _) in std::mem::take(&mut checker.mono.functions) {
        program.functions.push(f);
    }
    // The structs the checker synthesized (a closure's `Closure_N` environment), as the driver
    // adds them: a named `!llvm.struct` with no body does not parse. A generic template stays
    // out, as in the driver: its instances are the monomorphs appended above.
    program
        .structs
        .extend(std::mem::take(&mut checker.mono.generated_structs));
    program.functions.retain(|f| f.generics.is_empty());

    let context = make_context();
    let mut codegen = MeliorGenerator::new(&context, "diff_ast".to_string());
    let module_syntaxes = std::collections::HashMap::new();
    codegen.generate(&program, &module_syntaxes).unwrap();
    let mut module = codegen.into_module();
    lower_to_llvm(&context, &mut module).expect("AST lower_to_llvm");
    module.as_operation().to_string()
}

/// Exit code of `main` compiled through the AST codegen (the oracle).
fn ast_exit_code(src: &str) -> i32 {
    exit_code(&ast_llvm(src))
}

/// The lowered LLVM-dialect MLIR for the whole program compiled through the *flat*
/// path, or `None` if the flat HIR / emitter declines any function (outside the
/// current subset). Lowers *all* functions (so calls resolve to their callee
/// `func.func`) through the frozen registry, then emits one module via
/// `flat::emit_module_mlir`.
fn flat_llvm(src: &str) -> Option<String> {
    let mut program = parse(src);
    program.module_path = "crate::diff".into();
    let mut mods = vec![program];
    let symbol_map = vxc::resolver::build_symbol_map(&mods);
    mods[0].resolve_names(&symbol_map, &[]);
    let registry = vxc::pipeline::build_frozen_registry(&mods).ok()?;
    let session = std::sync::Arc::new(GlobalSession::with_registry(1, registry));

    // Type-check so the checker annotates each `StructInit` with its struct GID and collects
    // monomorphizations (method-call rewrites like `x.sq()` -> `f32$sq`, generic instances). A scratch
    // worker; the annotation lands on the AST, which the per-function lowering below then reads.
    let env_mods = mods.clone();
    let env = GlobalAstEnv::build(&env_mods);
    let (monos, generated_structs) = {
        let mut scratch = LocalWorkerState::new(session.clone());
        let mut checker = TypeChecker::new(&env, &mut scratch);
        for f in &mut mods[0].functions {
            checker.check_function(f);
        }
        (checker.mono.functions, checker.mono.generated_structs)
    };
    if !monos.is_empty() || !generated_structs.is_empty() {
        // Append the monomorph bodies and the structs the checker synthesized (a closure's
        // `Closure_N` environment), then re-resolve + rebuild the registry so they land in
        // `fn_sigs` and `layouts` (a rewritten `f32$sq(x)` resolves its callee) -- mirroring the
        // driver's flat path.
        for (f, _) in monos {
            mods[0].functions.push(f);
        }
        mods[0].structs.extend(generated_structs);
        let symbol_map = vxc::resolver::build_symbol_map(&mods);
        mods[0].resolve_names(&symbol_map, &[]);
    }
    // A generic template is not lowered, as in the driver: its instances are the monomorphs.
    mods[0].functions.retain(|f| f.generics.is_empty());
    let registry = vxc::pipeline::build_frozen_registry(&mods).ok()?;
    let session = std::sync::Arc::new(GlobalSession::with_registry(1, registry));

    // Lower every function into its own worker; decline the whole program if any
    // function is outside the flat subset (module-level keep-green atomicity).
    let mut lowered: Vec<LocalWorkerState> = Vec::new();
    for f in &mods[0].functions {
        let mut worker = LocalWorkerState::new(session.clone());
        if let Err(e) = lower_function_to_hir(f, &mut worker) {
            eprintln!("flat path declined `{}`: {e:?}", f.name);
            return None;
        }
        lowered.push(worker);
    }
    let funcs: Vec<(&_, &[_], &[_])> = mods[0]
        .functions
        .iter()
        .zip(&lowered)
        .map(|(f, w)| {
            (
                f,
                w.local_hir_stream.as_slice(),
                w.local_type_stream.as_slice(),
            )
        })
        .collect();
    let tensor_types: Vec<_> = lowered
        .iter()
        .flat_map(|w| w.local_tensor_types.iter().cloned())
        .collect();
    let string_tables: Vec<&[String]> = lowered
        .iter()
        .map(|w| w.local_string_table.as_slice())
        .collect();
    let inline_tables: Vec<&[vxc::hir::flatten::InlineBlock]> = lowered
        .iter()
        .map(|w| w.local_inline_blocks.as_slice())
        .collect();
    let agg_layouts: Vec<_> = lowered
        .iter()
        .flat_map(|w| w.local_agg_layouts.iter().cloned())
        .collect();
    let alias_tables: Vec<&[(usize, usize, Vec<usize>)]> = lowered
        .iter()
        .map(|w| w.local_place_alias_stores.as_slice())
        .collect();
    let body = vxc::codegen::flat::emit_module_mlir(
        &funcs,
        &session.registry,
        &tensor_types,
        &string_tables,
        &inline_tables,
        &agg_layouts,
        &alias_tables,
        &[],
        &[],
        vxc::config::Schedule::Parallel,
    )
    .ok()?;

    let context = make_context();
    let mut module = melior::ir::Module::parse(&context, &body).expect("flat MLIR parses");
    lower_to_llvm(&context, &mut module).expect("flat lower_to_llvm");
    Some(module.as_operation().to_string())
}

/// Exit code of the program compiled through the *flat* path (`None` if declined).
fn flat_exit_code(src: &str) -> Option<i32> {
    Some(exit_code(&flat_llvm(src)?))
}

/// The raw flat-path MLIR (`vx.transfer` and friends, *before* the lowering to LLVM), with sub-space
/// scheduling metadata attached — the P0-1 companion to `flat_llvm`. Mirrors the driver's flat build:
/// builds the `SubspaceInfo` descriptors from the per-compilation env (the frozen registry has no
/// memory decls) so a `vx.transfer` carries `space`/`within`/`granule`/`capacity`/`scope` +
/// bump-allocated `offset`/`slots`, exactly as the AST path does. `None` if outside the flat subset.
/// (No monomorphization handling — the callers place plain tensor transfers, no generics/methods.)
fn flat_module_mlir(src: &str) -> Option<String> {
    let mut program = parse(src);
    program.module_path = "crate::diff".into();
    let mut mods = vec![program];
    let symbol_map = vxc::resolver::build_symbol_map(&mods);
    mods[0].resolve_names(&symbol_map, &[]);
    let registry = vxc::pipeline::build_frozen_registry(&mods).ok()?;
    let session = std::sync::Arc::new(GlobalSession::with_registry(1, registry));
    let env_mods = mods.clone();
    let env = GlobalAstEnv::build(&env_mods);
    {
        let mut scratch = LocalWorkerState::new(session.clone());
        let mut checker = TypeChecker::new(&env, &mut scratch);
        for f in &mut mods[0].functions {
            checker.check_function(f);
        }
    }
    // The same mapping the compiler uses, not a copy of it.
    //
    // This was a hand-rolled duplicate of `subspaces_from_env`, and a field
    // added to one is a field silently missing from the other -- which is the
    // exact failure that function's own doc comment warns about, since dropping
    // an attribute on the flat path is invisible until a program behaves
    // differently between the two codegens. A differential test least of all
    // should be reimplementing the thing it differentiates.
    let subspaces: Vec<vxc::codegen::flat::SubspaceInfo> =
        vxc::codegen::flat::subspaces_from_env(&env);
    let mut lowered: Vec<LocalWorkerState> = Vec::new();
    for f in &mods[0].functions {
        let mut worker = LocalWorkerState::new(session.clone());
        if let Err(e) = lower_function_to_hir(f, &mut worker) {
            eprintln!("flat path declined `{}`: {e:?}", f.name);
            return None;
        }
        lowered.push(worker);
    }
    let funcs: Vec<(&_, &[_], &[_])> = mods[0]
        .functions
        .iter()
        .zip(&lowered)
        .map(|(f, w)| {
            (
                f,
                w.local_hir_stream.as_slice(),
                w.local_type_stream.as_slice(),
            )
        })
        .collect();
    let tensor_types: Vec<_> = lowered
        .iter()
        .flat_map(|w| w.local_tensor_types.iter().cloned())
        .collect();
    let string_tables: Vec<&[String]> = lowered
        .iter()
        .map(|w| w.local_string_table.as_slice())
        .collect();
    let inline_tables: Vec<&[vxc::hir::flatten::InlineBlock]> = lowered
        .iter()
        .map(|w| w.local_inline_blocks.as_slice())
        .collect();
    let agg_layouts: Vec<_> = lowered
        .iter()
        .flat_map(|w| w.local_agg_layouts.iter().cloned())
        .collect();
    let alias_tables: Vec<&[(usize, usize, Vec<usize>)]> = lowered
        .iter()
        .map(|w| w.local_place_alias_stores.as_slice())
        .collect();
    vxc::codegen::flat::emit_module_mlir(
        &funcs,
        &session.registry,
        &tensor_types,
        &string_tables,
        &inline_tables,
        &agg_layouts,
        &alias_tables,
        &subspaces,
        &[],
        vxc::config::Schedule::Parallel,
    )
    .map_err(|e| eprintln!("flat path failed to emit: {e:?}"))
    .ok()
}

/// The parity assertion for a *printing* program: the flat path lowers it, and its
/// (normalized) JIT stdout equals the AST path's.
fn assert_output_parity(src: &str) {
    let flat = flat_llvm(src).expect("flat path lowers this printing program");
    let ast = ast_llvm(src);
    let flat_out = run_output(&flat);
    let ast_out = run_output(&ast);
    assert_eq!(
        flat_out, ast_out,
        "flat print output diverged from the AST path for `{src}`\nflat:\n{flat_out}\nast:\n{ast_out}"
    );
}

/// The core parity assertion: the flat path lowers `main`, and its JIT exit code
/// equals both the AST path's and the expected value.
fn assert_parity(src: &str, expected: i32) {
    let flat = flat_exit_code(src).expect("flat path lowers this main");
    let ast = ast_exit_code(src);
    assert_eq!(
        ast, expected,
        "AST oracle differs from expected for `{src}`"
    );
    assert_eq!(
        flat, ast,
        "flat codegen diverged from the AST path for `{src}` (flat={flat}, ast={ast})"
    );
}

#[test]
fn flat_matches_ast_integer_arithmetic() {
    assert_parity("fn main() -> i32 { return 3 + 4 * 5; }", 23);
}

#[test]
fn flat_matches_ast_subtraction_and_div() {
    assert_parity("fn main() -> i32 { return 100 - 84 / 2; }", 58);
}

#[test]
fn flat_matches_ast_with_a_scalar_param_chain() {
    // A helper-free chain so the whole `main` stays in the flat subset.
    assert_parity("fn main() -> i32 { return (2 + 3) * (10 - 3); }", 35);
}

/// #230 mutable slice: mutation through a `&mut` reference — the `increment` showcase. The AST path
/// cannot compile a borrowed mutable scalar local (an unresolved `memref -> !llvm.ptr` cast), so there
/// is no *direct* AST oracle for the reference form. Ground the result on the **value-semantics
/// equivalent** (the inlined mutation), which both paths compile — per `scalar_references_flat.md`
/// §6.2. `inc(&mut x)` twice on 41 must equal `x = x + 1` twice: 43.
#[test]
fn flat_runs_mutation_through_a_reference() {
    let ref_src = "fn inc(p : &mut i32) -> void { *p = *p + 1; }\n\
                   fn main() -> i32 { let mut x = 41; inc(&mut x); inc(&mut x); return x; }";
    let val_src = "fn main() -> i32 { let mut x = 41; x = x + 1; x = x + 1; return x; }";
    // The value-semantics version grounds the expected result on both paths.
    assert_eq!(ast_exit_code(val_src), 43, "value-semantics AST oracle");
    assert_eq!(
        flat_exit_code(val_src),
        Some(43),
        "value version lowers on flat"
    );
    // The reference version lowers on the flat path and matches that result.
    assert_eq!(
        flat_exit_code(ref_src),
        Some(43),
        "mutation through `&mut` on the flat path equals the value-semantics result",
    );
}

/// #230 mutable slice: two-`&mut`-parameter mutation — the `swap` showcase (a signature rustc needs
/// two lifetimes to express). `x * 2 + y` after swapping (10, 20) reads the swapped values (20, 10) ->
/// 50 — which distinguishes a full swap from a no-op (40) or a half swap (`*a = *b` only -> 60). Kept
/// under 256 so it survives the process exit-code truncation.
#[test]
fn flat_runs_swap_through_mutable_references() {
    let ref_src = "fn swap(a : &mut i32, b : &mut i32) -> void { let t = *a; *a = *b; *b = t; }\n\
         fn main() -> i32 { let mut x = 10; let mut y = 20; swap(&mut x, &mut y); return x * 2 + y; }";
    let val_src = "fn main() -> i32 { let x = 20; let y = 10; return x * 2 + y; }";
    assert_eq!(ast_exit_code(val_src), 50, "value-semantics AST oracle");
    assert_eq!(
        flat_exit_code(ref_src),
        Some(50),
        "swap through two `&mut` params reads the swapped values on the flat path",
    );
}

#[test]
fn flat_matches_ast_if_without_else() {
    // `if` with a fall-through merge: the branch runs, then both edges reconverge
    // and the function returns the mutated local.
    assert_parity(
        "fn main() -> i32 { let mut x = 5; if x < 10 { x = x + 100; } return x; }",
        105,
    );
}

#[test]
fn flat_matches_ast_if_else() {
    // A full if/else diamond; the taken branch decides the returned value.
    assert_parity(
        "fn main() -> i32 { let a = 3; let mut y = 0; if a < 0 { y = 1; } else { y = 2; } return y; }",
        2,
    );
}

/// #230 step 2: the per-local slot rule. A control-flow function no longer slots *every* local — a
/// non-mutated local stays a dominating SSA register. Correctness is the flat-vs-AST parity; the
/// memory-traffic win is asserted directly on the flat MLIR (no `memref.alloca` for the non-mutated
/// locals `k`/`c`, which the old function-global memory mode would have slotted).
#[test]
fn flat_registers_non_mutated_locals_under_control_flow() {
    let src =
        "fn main() -> i32 { let k = 40; let c = 1; if c > 0 { return k + 2; } return k + 1; }";
    assert_parity(src, 42);
    let mlir = flat_module_mlir(src).expect("lowers on the flat path");
    assert!(
        !mlir.contains("memref.alloca"),
        "non-mutated locals should be SSA registers, not memref slots:\n{mlir}"
    );
}

/// #230 step 2, the other half: a *mutated* local under control flow still gets its slot (its value
/// must cross the branch merge), so the optimization does not over-reach. Parity with the AST oracle.
#[test]
fn flat_still_slots_a_mutated_local_under_control_flow() {
    let src = "fn main() -> i32 { let mut x = 5; if x < 10 { x = x + 100; } return x; }";
    assert_parity(src, 105);
    let mlir = flat_module_mlir(src).expect("lowers on the flat path");
    assert!(
        mlir.contains("memref.alloca"),
        "the mutated `x` must live in a slot to cross the merge:\n{mlir}"
    );
}

/// #275 §5 Example A: a non-escaping `&x` binds a symbolic place, so `x` never materializes an
/// address — `let r = &x; return *r` compiles with NO `alloca` (the "alloca that should not exist"),
/// yet still matches the AST oracle. This is the precision win places buy over §9's demote-on-any-`&x`.
#[test]
fn flat_non_escaping_borrow_needs_no_alloca() {
    let src = "fn main() -> i32 { let x = 5; let r = &x; return *r; }";
    assert_parity(src, 5);
    let mlir = flat_module_mlir(src).expect("lowers on the flat path");
    assert!(
        !mlir.contains("alloca"),
        "a non-escaping borrow needs no materialized address:\n{mlir}"
    );
}

/// The escaping counterpart: passing the reference on (`id(r)`, where `r` is used as a value, not
/// `*r`) forces the base to materialize a real address. Parity with the AST oracle; the `alloca` is
/// expected here — the escape analysis correctly distinguishes it from Example A.
#[test]
fn flat_escaping_borrow_materializes_an_address() {
    let src = "fn id(p : &i32) -> i32 { return *p; }\n\
               fn main() -> i32 { let x = 5; let r = &x; return id(r); }";
    assert_parity(src, 5);
    let mlir = flat_module_mlir(src).expect("lowers on the flat path");
    assert!(
        mlir.contains("alloca"),
        "an escaping reference needs a real address:\n{mlir}"
    );
}

/// #275 §5 Example C: disjoint field borrows through a `&mut` parameter. `let bx = &mut p.x; let by =
/// &mut p.y; *bx = 1; *by = 2` binds `bx`/`by` as symbolic *field places*; each `*b = v` re-lowers as a
/// `FieldStore` through `p`. The borrow checker proves `x` and `y` disjoint; this brings the whole
/// program onto the flat path, matching the AST oracle (`x*10 + y == 12`).
#[test]
fn flat_runs_disjoint_field_borrows_through_places() {
    let src = "struct Point { x : i32, y : i32 }\n\
               fn update(p : &mut Point) -> void { let bx = &mut p.x; let by = &mut p.y; *bx = 1; *by = 2; }\n\
               fn main() -> i32 { let mut pt = Point { x : 0, y : 0 }; update(&mut pt); return pt.x * 10 + pt.y; }";
    assert_parity(src, 12);
}

/// #275 §5.4 (M2b-2): the two disjoint place-writes above carry alias-scope metadata — each `*b = v`
/// store belongs to its own `alias_scopes` scope and lists the other as a `noalias_scopes` sibling
/// (the borrow checker proved `p.x` and `p.y` disjoint). Verified structurally: an `-O0` differential
/// is blind to alias metadata (it only bites under optimization), so `assert_parity` proves the
/// attributes don't break translation while this asserts they are present + mutually non-aliasing.
#[test]
fn flat_tags_disjoint_field_stores_with_alias_scopes() {
    let src = "struct Point { x : i32, y : i32 }\n\
               fn update(p : &mut Point) -> void { let bx = &mut p.x; let by = &mut p.y; *bx = 1; *by = 2; }\n\
               fn main() -> i32 { let mut pt = Point { x : 0, y : 0 }; update(&mut pt); return pt.x * 10 + pt.y; }";
    let mlir = flat_module_mlir(src).expect("lowers on the flat path");
    // `{alias_scopes` is the own-scope attribute; matching the `{` avoids also counting the tail of
    // `noalias_scopes` (which ends in the same `alias_scopes` substring).
    assert_eq!(
        mlir.matches("{alias_scopes = [").count(),
        2,
        "each disjoint place-write store declares its own alias scope:\n{mlir}"
    );
    assert_eq!(
        mlir.matches("noalias_scopes = [").count(),
        2,
        "each store lists the disjoint sibling as noalias:\n{mlir}"
    );
    // The two stores' own scopes must be *distinct* ids (they name different fields), and each store's
    // noalias sibling must be the *other* store's own scope — mutual non-aliasing.
    assert!(
        mlir.contains("distinct[1]") && mlir.contains("distinct[2]"),
        "the two disjoint fields get distinct alias-scope ids:\n{mlir}"
    );
}

/// A *single* place-write field store has no disjoint sibling, so it declares its own `alias_scopes`
/// scope but carries **no** `noalias_scopes` (there is nothing proven disjoint from it) — the reduction
/// only emits a `noalias` relationship when the frontend actually proved one. Parity at 42.
#[test]
fn flat_tags_a_lone_field_store_without_noalias() {
    let src = "struct P { x : i32, y : i32 }\n\
               fn main() -> i32 { let mut p = P { x : 1, y : 2 }; let r = &mut p.x; *r = 42; return *r; }";
    assert_parity(src, 42);
    let mlir = flat_module_mlir(src).expect("lowers on the flat path");
    assert!(
        mlir.contains("{alias_scopes = ["),
        "the lone place-write store declares its alias scope:\n{mlir}"
    );
    assert!(
        !mlir.contains("noalias_scopes = ["),
        "a lone place-write has no proven-disjoint sibling, so no noalias scope:\n{mlir}"
    );
}

/// §16.3 — the `-O0`-vs-`-O2` differential, the check the M2b caveat lacked. The disjoint-field
/// `noalias` metadata (§15) changes no result at `-O0` (the pipeline ignores alias metadata), so the
/// `-O0` flat differential cannot tell a *sound* `noalias` from an unsound one. Re-run the same lowered
/// module at `-O2`, where `opt` actually consumes `!alias.scope`/`!noalias`: a wrong `noalias` would
/// license the optimizer to reorder or drop a store and diverge from the `-O0` ground truth. Example C
/// carries the metadata (two mutually-`noalias` field writes); both levels must return 12.
#[test]
fn flat_alias_metadata_is_sound_under_o2() {
    let src = "struct Point { x : i32, y : i32 }\n\
               fn update(p : &mut Point) -> void { let bx = &mut p.x; let by = &mut p.y; *bx = 1; *by = 2; }\n\
               fn main() -> i32 { let mut pt = Point { x : 0, y : 0 }; update(&mut pt); return pt.x * 10 + pt.y; }";
    let llvm = flat_llvm(src).expect("Example C lowers on the flat path");
    // The metadata must survive `lower_to_llvm` to reach the optimizer, else this differential is
    // vacuous. (`alias_scope` appears in the LLVM-dialect `llvm.store` attribute.)
    assert!(
        llvm.contains("alias_scope"),
        "alias-scope metadata must survive lower_to_llvm to reach -O2:\n{llvm}"
    );
    let o0 = exit_code_at_opt(&llvm, 0);
    let o2 = exit_code_at_opt(&llvm, 2);
    assert_eq!(o0, 12, "Example C ground truth at -O0");
    assert_eq!(
        o2, o0,
        "the disjoint-field noalias metadata must not change the result under -O2 (soundness)"
    );
}

/// #275 §5 field place, read + write through a single-level field reference. `&mut p.x; *r = 42;
/// return *r` resolves both the write and the read through `p.x` — no pointer materialized. Parity.
#[test]
fn flat_reads_and_writes_a_field_through_a_place() {
    let src = "struct P { x : i32, y : i32 }\n\
               fn main() -> i32 { let mut p = P { x : 1, y : 2 }; let r = &mut p.x; *r = 42; return *r; }";
    assert_parity(src, 42);
}

/// #275 M4 / #278: nested references (`&&T`). `let rr = &r` binds `rr : &&i32`; `**rr` derefs twice — the
/// inner load yields the `&i32` pointer, the outer loads its `i32`. The AST codegen used to miscompile
/// nested address-of and segfault (§18.2), so this once had to ground on the value-semantics equivalent.
/// #278 fixed `BorrowExpr::lower` to materialize a slot for `&r`, restoring the AST oracle, so this is now
/// a real flat-vs-AST differential (the `nested_reference.vx` backend fixture is the AST-path guard).
#[test]
fn flat_runs_a_nested_reference() {
    assert_parity(
        "fn main() -> i32 { let x = 5; let r = &x; let rr = &r; return **rr; }",
        5,
    );
}

/// #278 Defect 2: the *mutable* nested reference (`&mut &mut i32`). Unlike the immutable form (which the
/// flat path resolves as nested symbolic places), a mutable reference local can't be a place, so `rr` is a
/// **materialized** pointer-to-pointer: `r` gets a real `!llvm.ptr` slot (`bind_local` now slots an
/// address-taken pointer local) and `**rr` derefs through two `PtrIndex`es — the inner one loading a
/// *pointer* element (`pointer_elem_ty`/`PtrIndex` now accept a `Ptr` pointee). On the AST path `&mut r`
/// extracts the memref's aligned pointer so the same chain stays llvm-pointer-based. Read form: `**rr` = 5.
#[test]
fn flat_runs_a_nested_mutable_reference() {
    assert_parity(
        "fn main() -> i32 { let mut x = 5; let r = &mut x; let rr = &mut r; return **rr; }",
        5,
    );
}

/// #278 Defect 2: mutation *through* a mutable nested reference. `**rr = 10` writes 10 to `x` via the
/// pointer-to-pointer, and `return x` observes it — proving `rr` aliases `x`'s real storage (not a copy)
/// on both paths. The flat path stores through the inner `PtrIndex` element pointer; the AST path stores
/// through the extracted aligned pointer, which aliases `x`'s memref cell.
#[test]
fn flat_writes_through_a_nested_mutable_reference() {
    assert_parity(
        "fn main() -> i32 { let mut x = 5; let r = &mut x; let rr = &mut r; **rr = 10; return x; }",
        10,
    );
}

/// #275 M4: a reference-typed struct field (`struct Holder { r : &i32 }`). Constructing `Holder { r : &x }`
/// stores the address of `x` into the reference field, and `*h.r` loads the field pointer and derefs it.
/// Previously declined: the escape scan didn't descend into aggregate literals, so the `&x` never
/// materialized `x`. Now it does; the whole program lowers and matches the AST oracle (5).
#[test]
fn flat_runs_a_reference_typed_struct_field() {
    let src = "struct Holder { r : &i32 }\n\
               fn main() -> i32 { let x = 5; let h = Holder { r : &x }; return *h.r; }";
    assert_parity(src, 5);
}

/// #275 M3b part 2: a *reference-returning* function whose result is the address of a scalar field of a
/// reference parameter (`probe(m : &Map) -> &i32 { return &m.slot; }`). Previously declined — the flat
/// path only addressed nested-aggregate fields, so a scalar field's `&` fell through. Now it GEPs to the
/// element and returns the pointer; the borrow checker's return-provenance (#243) is the safety proof.
/// `probe(&mm)` returns `&mm.slot` (7); `*r` reads it. Parity with the AST oracle.
#[test]
fn flat_returns_a_reference_to_a_scalar_field() {
    let src = "struct Map { slot : i32, present : i32 }\n\
               fn probe(m : &Map) -> &i32 { return &m.slot; }\n\
               fn main() -> i32 { let mm = Map { slot : 7, present : 0 }; let r = probe(&mm); return *r; }";
    assert_parity(src, 7);
}

/// The reference-return address must GEP the *right* field: returning `&m.present` (the second field,
/// non-zero offset) reads back 9, not `slot`. Guards the `field_idx`/offset resolution the first-field
/// test above cannot distinguish from a hardcoded `[0, 0]`. (#275 M3b)
#[test]
fn flat_returns_a_reference_to_a_non_first_field() {
    let src = "struct Map { slot : i32, present : i32 }\n\
               fn probe(m : &Map) -> &i32 { return &m.present; }\n\
               fn main() -> i32 { let mm = Map { slot : 3, present : 9 }; let r = probe(&mm); return *r; }";
    assert_parity(src, 9);
}

/// #275 §5 Example B (#276 + #277): a two-level `&mut o.inner.v` place over a by-value nested
/// aggregate. The nested construction stores the inner struct *value* (#277 — not the inner slot's
/// address), the borrow check accepts the read after the loan is dead (#276), and the two-level
/// projection resolves the write and read through `o.inner.v`. Parity: writes 42 through `*r`, reads it
/// back. This is the case §12 originally named as M2's target.
#[test]
fn flat_runs_nested_field_place_example_b() {
    let src = "struct Inner { v : i32 }\n\
               struct Outer { inner : Inner }\n\
               fn main() -> i32 { let mut o = Outer { inner : Inner { v : 1 } }; let r = &mut o.inner.v; *r = 42; return *r; }";
    assert_parity(src, 42);
}

/// #277 in isolation: by-value nested-aggregate construction must store the inner struct *value*, so a
/// later read of the nested field yields it rather than a stored pointer reinterpreted as an `i32`. No
/// reference involved — this pins the constructor fix independently of the place machinery.
#[test]
fn flat_constructs_a_nested_aggregate_by_value() {
    let src = "struct Inner { v : i32 }\n\
               struct Outer { inner : Inner }\n\
               fn main() -> i32 { let mut o = Outer { inner : Inner { v : 7 } }; let r = &mut o.inner.v; return *r; }";
    assert_parity(src, 7);
}

/// An immutable single-level field reference read through a place. `&p.x; return *r` — the field is
/// `FieldLoad`ed at the deref. Parity with the AST oracle.
#[test]
fn flat_reads_an_immutable_field_through_a_place() {
    let src = "struct P { x : i32, y : i32 }\n\
               fn main() -> i32 { let p = P { x : 7, y : 2 }; let r = &p.x; return *r; }";
    assert_parity(src, 7);
}

/// #275 §5.4: carry the borrow checker's exclusivity into the signature. A `&mut T` parameter is an
/// exclusive borrow → `llvm.noalias` — the aliasing guarantee LLVM cannot re-derive (the callee can't
/// see the caller's borrows). This is what rustc emits; verified structurally (an `-O0` differential is
/// blind to alias attributes) plus a parity run that the attribute doesn't break translation.
#[test]
fn flat_marks_mut_ref_param_noalias() {
    let src = "struct P { x : i32, y : i32 }\n\
               fn set(p : &mut P) -> void { p.x = 9; }\n\
               fn main() -> i32 { let mut p = P { x : 0, y : 0 }; set(&mut p); return p.x; }";
    assert_parity(src, 9);
    let mlir = flat_module_mlir(src).expect("lowers on the flat path");
    assert!(
        mlir.contains("!llvm.ptr {llvm.noalias}"),
        "the `&mut P` param should carry llvm.noalias:\n{mlir}"
    );
}

/// A `&T` shared reference cannot be written *through* (type-guaranteed) → `llvm.readonly`; but two
/// `&T` may alias, so it is deliberately *not* `noalias`. Structural + parity.
#[test]
fn flat_marks_shared_ref_param_readonly() {
    let src = "fn rd(x : &i32) -> i32 { return *x; }\n\
               fn main() -> i32 { let a = 7; let r = &a; return rd(r); }";
    assert_parity(src, 7);
    let mlir = flat_module_mlir(src).expect("lowers on the flat path");
    assert!(
        mlir.contains("!llvm.ptr {llvm.readonly}"),
        "the `&i32` param should carry llvm.readonly:\n{mlir}"
    );
    assert!(
        !mlir.contains("noalias"),
        "a shared reference must not be marked noalias (two `&T` may alias):\n{mlir}"
    );
}

#[test]
fn flat_matches_ast_if_expression() {
    // A value-position `if` (`let m: i32 = if a > b { a } else { b }`, #201): lowered to blocks + a
    // result slot each branch stores into, then loaded. JIT-matches the AST oracle. This is the
    // `let m_new = if tm > m { tm } else { m }` shape from the attention corpus.
    assert_parity(
        "fn main() -> i32 { let a = 3; let b = 7; let m: i32 = if a > b { a } else { b }; return m; }",
        7,
    );
    // With side-effecting leading statements in a branch before the trailing value.
    assert_parity(
        "fn main() -> i32 { let x = 5; \
         let r: i32 = if x > 0 { let t = x * 2; t + 1 } else { 0 }; return r; }",
        11,
    );
}

#[test]
fn flat_matches_ast_assert_is_checked() {
    // `assert(cond, msg)` emits a real runtime check in BOTH paths (Vx#361): the flat path a
    // `cf.assert` from `Opcode::Assert`, the AST path the same op from `emit_runtime_assert`.
    // A satisfied assertion must therefore change nothing about the result, on either path.
    //
    // Named for what it now tests. Until Vx#361 this was `..._assert_is_a_noop` and asserted
    // the opposite -- correctly, at the time, when neither path emitted anything.
    assert_parity(
        "fn main() -> i32 { let x = 5; assert(x == 5, \"ok\"); assert(x > 0, \"pos\"); return x; }",
        5,
    );
}

#[test]
fn flat_matches_ast_for_range_accumulator() {
    // A `for` range loop accumulating 0+1+2+3+4 through slot-backed locals.
    assert_parity(
        "fn main() -> i32 { let mut s = 0; for i in 0..5 { s = s + i; } return s; }",
        10,
    );
}

#[test]
fn flat_matches_ast_loop_with_break() {
    // An infinite `loop` exited by `break` from a nested `if` — exercises the
    // loop back-edge, the break target, and a compare inside the body.
    assert_parity(
        "fn main() -> i32 { let mut i = 0; loop { if i >= 3 { break; } i = i + 1; } return i; }",
        3,
    );
}

#[test]
fn flat_matches_ast_scalar_helper_call() {
    // A `main` that calls a scalar helper — exercises the module-level emitter
    // (both `func.func`s in one module) + callee resolution via `fn_sigs`.
    assert_parity(
        "fn add(a: i32, b: i32) -> i32 { return a + b; }\n\
         fn main() -> i32 { return add(20, 22); }",
        42,
    );
}

#[test]
fn flat_matches_ast_call_inside_expression() {
    // The call is one operand of a larger arithmetic expression in `main`.
    assert_parity(
        "fn mul(a: i32, b: i32) -> i32 { return a * b; }\n\
         fn main() -> i32 { return mul(6, 7) - 2; }",
        40,
    );
}

#[test]
fn flat_matches_ast_extern_call() {
    // An `extern` (libm `sqrtf`, linked by the JIT via `-lm`) now resolves + emits through the flat
    // path: registered in `fn_sigs`, called, and declared `func.func private` from the call's own
    // signature. `sqrtf(16.0) = 4.0`; compare the printed output flat-vs-AST.
    assert_output_parity(
        "extern { safe fn sqrtf(x: f32) -> f32; }\n\
         fn main() -> i32 { print(sqrtf(16.0)); return 0; }",
    );
}

#[test]
fn flat_matches_ast_scalar_method_call() {
    // Method dispatch through the flat path (#217): the type checker rewrites `(3.0).sq()` ->
    // `f32$sq(3.0)` and monomorphizes the body; the harness appends + registers the monomorph, so the
    // flat path lowers the caller + the `f32$sq` body together. `3*3 = 9`; output matches the AST path.
    assert_output_parity(
        "trait Sq { fn sq(self: Self) -> f32; }\n\
         impl Sq for f32 { fn sq(self: f32) -> f32 { return self * self; } }\n\
         fn main() -> i32 { print((3.0f32).sq()); return 0; }",
    );
}

#[test]
fn flat_matches_ast_unsafe_extern_call() {
    // The stdlib math-wrapper shape: a (non-safe) extern called inside an `unsafe { .. }` value block
    // (`impl Math for f32 { fn sqrt(self) { return unsafe { sqrtf(self) }; } }`). `unsafe` is
    // transparent to lowering, so both lower through the flat path; stdout matches the AST oracle.
    assert_output_parity(
        "extern { fn sqrtf(x: f32) -> f32; }\n\
         fn main() -> i32 { print(unsafe { sqrtf(16.0) }); return 0; }",
    );
}

#[test]
fn flat_matches_ast_print_string_literal() {
    // A bare string `print!` (the `print!("Success: ")` shape from the ffi corpus, #225): the flat
    // path records the bytes in a string side table, emits an `llvm.mlir.global` for them, and calls
    // `@print_str`. Stdout must match the AST oracle, which does the same.
    assert_output_parity("fn main() -> i32 { print!(\"Stdio Success: \"); return 0; }");
}

#[test]
fn flat_matches_ast_print_string_and_scalar() {
    // The common `print!("label", value)` shape: a string arg emits `PrintStr`, the scalar arg emits
    // `Print`, in order and with no separators — byte-identical to the AST path.
    assert_output_parity("fn main() -> i32 { let v = 42; print!(\"x=\", v); return 0; }");
}

#[test]
fn flat_matches_ast_println_string() {
    // `println!` prints its args then a trailing newline. The flat path reuses `PrintStr` for the
    // newline (a `"\n"` string is byte-identical to the AST path's `println()` runtime call), so the
    // full line — label, value, newline — matches the oracle.
    assert_output_parity("fn main() -> i32 { let v = 7; println!(\"count: \", v); return 0; }");
}

/// The text a printing program actually writes, through both lowerings, asserted to be equal to
/// each other AND to what it should be.
///
/// Parity alone cannot see a formatting bug: the substitution is a desugaring both paths share, so
/// a broken one leaves them agreeing on the same wrong text. The expected string is what makes
/// these tests able to fail.
fn assert_prints(src: &str, expected: &str) {
    let flat = run_output(&flat_llvm(src).expect("flat path lowers this printing program"));
    let ast = run_output(&ast_llvm(src));
    assert_eq!(flat, ast, "flat and AST disagree for `{src}`");
    assert_eq!(flat.trim_end(), expected, "wrong text for `{src}`");
}

/// `println!` substitutes its `{}` placeholders.
///
/// The format string used to be printed verbatim with the arguments appended after it, so
/// `println!("value={}", x)` wrote `value={}7`.
#[test]
fn a_format_placeholder_is_substituted() {
    assert_prints(
        "fn main() -> i32 { let v = 7; println!(\"value={}\", v); return 0; }",
        "value=7",
    );
    assert_prints(
        "fn main() -> i32 { let a = 1; let b = 2; println!(\"a={} b={}\", a, b); return 0; }",
        "a=1 b=2",
    );
}

/// A brace escape reaches the output as one brace, and a string with no placeholders is a label
/// printed ahead of its arguments -- the spelling the corpus already uses, which the substitution
/// had to keep working.
#[test]
fn braces_escape_and_a_label_still_prints_its_arguments() {
    assert_prints(
        "fn main() -> i32 { println!(\"{{literal}}\"); return 0; }",
        "{literal}",
    );
    assert_prints(
        "fn main() -> i32 { let v = 7; println!(\"count: \", v); return 0; }",
        "count: 7",
    );
}

#[test]
fn flat_matches_ast_placed_tensor() {
    // A tensor whose type names a space runs the same on both paths. It used to be spelled
    // `Tensor<..>(..).with_memory(Memory::X)`, a method the type checker wrapped in a `Ref` and
    // both backends then ignored; the placement says the same thing in the type (Vx#429). The
    // device-placement transfer *methods* (`to_device`/`to_host`) are a different thing and are
    // rewritten to `Expr::Transfer` by the type checker.
    assert_output_parity(
        "fn main() -> i32 { let mut c = Tensor<f32, [2], Memory::NPU_HBM>::uninit(); \
         c[0] = 3.0; c[1] = 4.0; print(c[0]); return 0; }",
    );
}

#[test]
fn flat_matches_ast_two_tensors_differing_only_in_placement() {
    // The flat path's tensor identity hashes the element and the shape and not the
    // placement, which rides on the alloc instruction instead. So these two share a GID and
    // a memref type, and the program has to come out the same on both paths anyway: the
    // space is not a property the type carries into codegen, and this pins that it is not
    // one the type needs to.
    assert_output_parity(
        "fn main() -> i32 { let mut h = Tensor<f32, [2]>::uninit(); \
         let mut d = Tensor<f32, [2], Memory::NPU_HBM>::uninit(); \
         h[0] = 1.0; h[1] = 2.0; d[0] = 3.0; d[1] = 4.0; \
         print(h[1]); print(d[0]); return 0; }",
    );
}

#[test]
fn flat_matches_ast_run_time_allocation() {
    // `Tensor<f32>([n])` with `n` a parameter allocates a `memref<?xf32>` sized from the value
    // on the flat path; the AST path allocates fully dynamic and casts. Both fill and read it
    // back the same: 1 + 2 + 3.
    assert_parity(
        "fn build(n : i32) -> Tensor<f32, [?]> { let mut a = Tensor<f32>([n]);            a[0] = 1.0; a[1] = 2.0; a[2] = 3.0; return a; }\n\
         fn total(t : Tensor<f32, [?]>) -> f32 { let mut s = 0.0;            for i in 0..t.extent(0) { s = s + t[i]; } return s; }\n\
         fn main() -> i32 { let a = build(3); return total(a) as i32; }",
        6,
    );
    // The typed form: the `?` position takes its extent from the argument, the static one is
    // the type's, and `::new` zeroes the whole buffer before the one store.
    assert_parity(
        "fn build(n : i32) -> Tensor<f32, [?, 4]> {            let mut m = Tensor<f32, [?, 4]>::new([n, 4]); m[1][2] = 5.0; return m; }\n\
         fn main() -> i32 { let m = build(2); return (m[1][2] + m[0][0] + m[1][3]) as i32; }",
        5,
    );
}

#[test]
fn flat_lowers_rows_of_mixed_and_deeper_rank_dynamic_tensors() {
    // Flat only: the AST oracle reads a row's length out of the memref text and cannot take
    // `memref<?x4x..>`. A row of a `[?, ?]` tensor has a `?` extent, so its width is read off
    // the base and the row offset is the index times it. `m[0][3]` is read back too: an
    // offset that is off by the stride rather than the width lands `m[1][2]` on it.
    assert_flat_exit(
        "fn build(n : i32) -> Tensor<f32, [?, ?]> { \
           let mut m = Tensor<f32>([n, 4]); m[1][2] = 5.0; m[0][3] = 7.0; return m; }\n\
         fn main() -> i32 { let m = build(2); return (m[1][2] * 10.0 + m[0][3]) as i32; }",
        57,
    );
    // A row of a `[?, 4]` tensor has a static shape, so it takes the literal-stride path over a
    // dynamic base, and `dot` over two rows loads `vector<4xf32>`.
    assert_flat_exit(
        "fn build(n : i32) -> Tensor<f32, [?, 4]> { \
           let mut m = Tensor<f32, [?, 4]>::new([n, 4]); m[0][2] = 2.0; m[1][2] = 5.0; return m; }\n\
         fn main() -> i32 { let m = build(2); return dot(m[0], m[1]) as i32; }",
        10,
    );
    // A rank-3 dynamic tensor's row is a rank-2 view; both of its extents come from the base.
    assert_flat_exit(
        "fn build(n : i32) -> Tensor<f32, [?, ?, ?]> { let m = Tensor<f32>([n, 3, 4]); return m; }\n\
         fn main() -> i32 { let m = build(2); let r = m[1]; return r.extent(0) * 10 + r.extent(1); }",
        34,
    );
}

#[test]
fn flat_lowers_a_row_of_a_row() {
    // Three indices deep, static and dynamic: the second index takes a row of a strided row,
    // whose own offset comes back through `memref.extract_strided_metadata`. `t[0][2][3]` is
    // read back too: a lost base offset lands `t[1][2][3]` on it.
    assert_flat_exit(
        "fn main() -> i32 { let mut t = Tensor<f32, [2, 3, 4]>::new(); t[1][2][3] = 5.0; \
           t[0][2][3] = 7.0; return (t[1][2][3] * 10.0 + t[0][2][3]) as i32; }",
        57,
    );
    assert_flat_exit(
        "fn build(n : i32) -> Tensor<f32, [?, ?, ?]> { let m = Tensor<f32>([n, 3, 4]); return m; }\n\
         fn main() -> i32 { let mut m = build(2); m[1][2][3] = 5.0; m[0][2][3] = 7.0; \
           return (m[1][2][3] * 10.0 + m[0][2][3]) as i32; }",
        57,
    );
}

#[test]
fn flat_matches_ast_spawn_regions_with_a_value_a_tail_and_nesting() {
    // A spawn in expression position: the region's tail is its value, pinned to the
    // topology, which a comparison reads through.
    assert_parity(
        "fn main() -> i32 { let v = spawn on(Topology::CPU) { 40 + 2 }; \
           if v == 42 { return 42; } return 0; }",
        42,
    );
    // A trailing `if` with no semicolon is the region's tail as the parser reads it, and in
    // statement position it is the last thing the region does, not a value.
    assert_parity(
        "fn main() -> i32 { let mut a = Tensor<f32, [1, 1]>::uninit(); a[0][0] = 0.0; \
           let x : f32 = 2.0; \
           spawn on(Topology::CPU) { a[0][0] = 1.0; if x > 1.0 { a[0][0] = 3.0; } } \
           return a[0][0] as i32; }",
        3,
    );
    // A function whose value is a spawn's: `Pinned<i32, ..>` is the scalar it wraps on both
    // sides of the call.
    assert_parity(
        "fn get() -> Pinned<i32, Topology::CPU> { let v = spawn on(Topology::CPU) { 40 + 2 }; return v; }\n\
         fn main() -> i32 { let p = get(); if p == 42 { return 42; } return 0; }",
        42,
    );
    // A nested region: the inner one is the outer's tail, and each closes with its own
    // topology.
    assert_parity(
        "fn main() -> i32 { let mut a = Tensor<f32, [1, 1]>::uninit(); a[0][0] = 0.0; \
           spawn on(Topology::CPU) { let y = 1; spawn on(Topology::CPU) { a[0][0] = 5.0; } } \
           return a[0][0] as i32; }",
        5,
    );
}

#[test]
fn flat_matches_ast_elementwise_ops_beyond_rank_one() {
    // Rank 2, static: `a + b` and `c * e` as named linalg ops into fresh buffers, read back
    // by element. Tensors are linear, so each operand is used once: 1 + 2 = 3, 3 * 3 = 9,
    // and two elements of the product: 18.
    assert_parity(
        "fn main() -> i32 { let a = Tensor<f32, [4, 4]>::fill(1.0); \
           let b = Tensor<f32, [4, 4]>::fill(2.0); let c = a + b; \
           let e = Tensor<f32, [4, 4]>::fill(3.0); let d = c * e; \
           return (d[1][1] + d[0][3]) as i32; }",
        18,
    );
    // Rank 2 with run-time extents: the result buffer takes its sizes from the first operand.
    assert_parity(
        "fn build(n : i32, m : i32) -> Tensor<f32, [?, ?]> { \
           let mut a = Tensor<f32, [?, ?]>::new([n, m]); a[0][1] = 2.0; return a; }\n\
         fn main() -> i32 { let a = build(2, 3); let mut b = build(2, 3); b[0][1] = 5.0; \
           let c = b - a; return c[0][1] as i32; }",
        3,
    );
}

#[test]
fn flat_matches_ast_closure_capturing_nothing() {
    // A closure that captures nothing has an environment struct with no fields. Its slot is
    // still allocated -- as the empty `!llvm.struct<()>` -- and the call goes through the
    // adapter like any closure's. 41 + 1.
    assert_parity(
        "fn main() -> i32 { let f = |x : i32| x + 1; return f(41); }",
        42,
    );
    // The corpus shape: a closure over a borrowed struct returning a borrow of a field.
    assert_parity(
        "struct Map { slot : i32, present : i32 }\n\
         fn good(m : &Map) -> &i32 { let f = |q : &Map| &q.slot; return f(m); }\n\
         fn main() -> i32 { let x = Map { slot : 7, present : 1 }; return *good(&x); }",
        7,
    );
}

#[test]
fn flat_matches_ast_const_generic_struct_instance() {
    // A struct generic over a `const`, with a by-value generic field: its base layout is a stub,
    // so the instance `Pair<f32, 4>` gets a synthesized one -- as a return type, at the
    // construction, and at each field read. 2 * 10 + 1.
    assert_parity(
        "struct Pair<T, const N : i32> { a : T, b : T }\n\
         fn make<const N : i32>() -> Pair<f32, N> { let p = Pair<f32, N> { a : 1.0, b : 2.0 }; return p; }\n\
         fn main() -> i32 { let p = make<4>(); return (p.b * 10.0 + p.a) as i32; }",
        21,
    );
}

#[test]
fn flat_matches_ast_borrow_of_a_call_result() {
    // A closure returned by a closure is called directly: the returned environment is a
    // by-value aggregate with no slot, and the call borrows it, so it is given one.
    assert_parity(
        "fn main() -> i32 { let get = || || 42; let a = (get())(); return a; }",
        42,
    );
}

#[test]
fn flat_matches_ast_vec_of_options() {
    // `Vec<Option<i32>>`: the storage pointer's element is a synthesized enum instance, on the
    // write side (`push` takes the construction by value, loaded off its slot) and the read side
    // (`get` returns it whole, and a `match` reads the payload). The corpus programs
    // `vec_option_elem.vx` and `option_unwrap.vx` print the same on both paths through the CLI;
    // this is the shape the harness can compile without the stdlib. 7 + 30.
    assert_parity(
        &format!(
            "{VEC_MINI}\nenum Option<T> {{ Some(T), None }}\n\
             fn main() -> i32 {{ let mut v: Vec<Option<i32>> = Vec<Option<i32>>::new(); \
               v.push(Option<i32>::Some(7)); v.push(Option<i32>::None); v.push(Option<i32>::Some(30)); \
               let mut r = 0; let a = v.get(0); \
               match a {{ Option<i32>::Some(x) => {{ r = r + x; }} Option<i32>::None => {{ r = -100; }} }} \
               let b = v.get(1); \
               match b {{ Option<i32>::Some(x) => {{ r = r + x; }} Option<i32>::None => {{ r = r + 0; }} }} \
               let c = v.get(2); \
               match c {{ Option<i32>::Some(x) => {{ r = r + x; }} Option<i32>::None => {{ r = -100; }} }} \
               return r; }}"
        ),
        37,
    );
}

#[test]
fn flat_matches_ast_inline_mlir() {
    // Scalar inputs and a scalar result: the block becomes a private wrapper the function
    // calls. Subtraction, so the operand order is observable. 44 - 2.
    assert_parity(
        "fn sub(a : i32, b : i32) -> i32 { let r = mlir!(inputs : (%x = a : i32, %y = b : i32), \
           clobbers : [], returns : i32, dialects : [\"arith\"]) { \
           %s = arith.subi %x, %y : i32 \n func.return %s : i32 }; return r; }\n\
         fn main() -> i32 { return sub(44, 2); }",
        42,
    );
    // Tensor inputs: the corpus shape, a reduction over two rank-2 memrefs into a bool. The
    // tensors are linear and passed by value, so each call gets fresh ones.
    assert_parity(
        "fn same(a : Tensor<f32, [2, 2]>, b : Tensor<f32, [2, 2]>) -> i32 { \
           let r = mlir!(inputs : (%lhs = a : memref<2x2xf32>, %rhs = b : memref<2x2xf32>), \
           clobbers : [], returns : bool, dialects : [\"linalg\", \"arith\", \"memref\"]) { \
           %t = arith.constant 1 : i1 \n %acc = memref.alloca() : memref<i1> \n \
           memref.store %t, %acc[] : memref<i1> \n \
           linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, \
             affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>], \
             iterator_types = [\"reduction\", \"reduction\"]} \
           ins(%lhs, %rhs : memref<2x2xf32>, memref<2x2xf32>) outs(%acc : memref<i1>) { \
             ^bb0(%p : f32, %q : f32, %o : i1): \n %c = arith.cmpf oeq, %p, %q : f32 \n \
             %n = arith.andi %o, %c : i1 \n linalg.yield %n : i1 } \n \
           %v = memref.load %acc[] : memref<i1> \n func.return %v : i1 }; \
           if r { return 1; } return 0; }\n\
         fn main() -> i32 { let a = Tensor<f32, [2, 2]>::fill(1.0); \
           let b = Tensor<f32, [2, 2]>::fill(1.0); let s = same(a, b); \
           let c = Tensor<f32, [2, 2]>::fill(1.0); let mut d = Tensor<f32, [2, 2]>::fill(1.0); \
           d[1][1] = 2.0; let t = same(c, d); return s * 10 + t; }",
        10,
    );
    // A void block with a `clobbers` list: the stdlib's `fill` shape.
    assert_parity(
        "fn fill(t : &mut Tensor<f32, [?, ?]>, v : f32) -> void { \
           mlir!(inputs : (%m = t : memref<?x?xf32>, %x = v : f32), clobbers : [t], \
           dialects : [\"linalg\"]) { linalg.fill ins(%x : f32) outs(%m : memref<?x?xf32>) \n \
           macro.yield }; }\n\
         fn main() -> i32 { let mut t = Tensor<f32, [2, 3]>::fill(0.0); fill(&mut t, 7.0); \
           return (t[1][2] as i32) * 6; }",
        42,
    );
}

#[test]
fn flat_matches_ast_inline_mlir_as_an_if_branch() {
    // A value-position `if` whose branches are `mlir!` blocks, the stdlib's topology-dispatched
    // shape: the branch type is the block's declared result. |44 - 2| both ways. Flat-only: the
    // oracle fails verification on this program (a block-argument type mismatch at the merge).
    assert_flat_exit(
        "fn dist(a : i32, b : i32) -> i32 { let r = if a > b { \
           mlir!(inputs : (%x = a : i32, %y = b : i32), clobbers : [], returns : i32, \
           dialects : [\"arith\"]) { %s = arith.subi %x, %y : i32 \n func.return %s : i32 } \
         } else { \
           mlir!(inputs : (%x = a : i32, %y = b : i32), clobbers : [], returns : i32, \
           dialects : [\"arith\"]) { %s = arith.subi %y, %x : i32 \n func.return %s : i32 } \
         }; return r; }\n\
         fn main() -> i32 { return dist(44, 2) + dist(2, 44) - 42; }",
        42,
    );
    // The `comptime` form the stdlib writes: the checker folds the `if` to its surviving branch,
    // in either direction, and that branch alone is the value.
    assert_parity(
        "fn dist(a : i32, b : i32) -> i32 { let r = if comptime true { \
           mlir!(inputs : (%x = a : i32, %y = b : i32), clobbers : [], returns : i32, \
           dialects : [\"arith\"]) { %s = arith.subi %x, %y : i32 \n func.return %s : i32 } \
         } else { \
           mlir!(inputs : (%x = a : i32, %y = b : i32), clobbers : [], returns : i32, \
           dialects : [\"arith\"]) { %s = arith.subi %y, %x : i32 \n func.return %s : i32 } \
         }; return r; }\n\
         fn dist2(a : i32, b : i32) -> i32 { let r = if comptime false { \
           mlir!(inputs : (%x = a : i32, %y = b : i32), clobbers : [], returns : i32, \
           dialects : [\"arith\"]) { %s = arith.subi %x, %y : i32 \n func.return %s : i32 } \
         } else { \
           mlir!(inputs : (%x = a : i32, %y = b : i32), clobbers : [], returns : i32, \
           dialects : [\"arith\"]) { %s = arith.subi %y, %x : i32 \n func.return %s : i32 } \
         }; return r; }\n\
         fn main() -> i32 { return dist(44, 2) + dist2(2, 44) - 42; }",
        42,
    );
}

#[test]
fn flat_matches_ast_match_over_integer_literals() {
    // Literal arms over an `i32` subject, a wildcard default, and a fall-through arm body.
    // 100 + 111 + 200 + 200 - 569.
    assert_parity(
        "fn pick(n : i32) -> i32 { let mut r = 0; match n { 0 => { r = 100; } 1 => { r = 111; } \
           _ => { r = 200; } } return r; }\n\
         fn main() -> i32 { return pick(0) + pick(1) + pick(7) + pick(0 - 1) - 569; }",
        42,
    );
    // An identifier arm binds the subject as the default. Flat-only: the oracle has no
    // lowering for a bare identifier pattern. 1 + 40 + 1.
    assert_flat_exit(
        "fn f(n : i32) -> i32 { let mut r = 0; match n { 3 => { r = 1; } k => { r = k * 2; } } \
           return r; }\n\
         fn main() -> i32 { return f(3) + f(20) + 1; }",
        42,
    );
}

#[test]
fn flat_matches_ast_reshape_and_transpose() {
    // A reshape reads the same buffer under new extents (t[1][2] is r[5]); a transpose copies
    // the permuted view (u[1][0] is p[0][1]); a rank-3 transpose then reshape reads the copy
    // (w[1][0][2] is q[2][1][0], flat index 10). The tensors are parameters: the oracle panics
    // on a reshape of a mutable local. 5 + 40 + 100.
    assert_parity(
        "fn rs(t : Tensor<f32, [2, 3]>) -> f32 { let r = t.reshape([6]); return r[5]; }\n\
         fn tr(u : Tensor<f32, [2, 3]>) -> f32 { let p = u.transpose([1, 0]); return p[0][1]; }\n\
         fn tr3(w : Tensor<f32, [2, 2, 3]>) -> f32 { let q = w.transpose([2, 0, 1]); \
           let s = q.reshape([12]); return s[10]; }\n\
         fn main() -> i32 { let mut t = Tensor<f32, [2, 3]>::fill(0.0); t[1][2] = 5.0; \
           let mut u = Tensor<f32, [2, 3]>::fill(0.0); u[1][0] = 4.0; \
           let mut w = Tensor<f32, [2, 2, 3]>::fill(0.0); w[1][0][2] = 1.0; \
           return (rs(t) + tr(u) * 10.0 + tr3(w) * 100.0) as i32; }",
        145,
    );
}

#[test]
fn flat_runs_a_tensor_typed_struct_field() {
    // A tensor field is its memref descriptor by value: constructed, passed with the struct,
    // and read back as a memref. Flat-only: the AST path inserts the memref itself into the
    // struct and fails verification. 4 * 10 + 2.
    assert_flat_exit(
        "struct Holder { t : Tensor<f32, [2, 2]>, k : i32 }\n\
         fn get(h : Holder) -> f32 { return h.t[1][0]; }\n\
         fn main() -> i32 { let mut a = Tensor<f32, [2, 2]>::fill(0.0); a[1][0] = 4.0; \
           let h = Holder { t : a, k : 2 }; let k = h.k; return (get(h) * 10.0) as i32 + k; }",
        42,
    );
    // A field with run-time extents: the descriptor carries the sizes the type does not.
    assert_flat_exit(
        "struct Dyn { w : Tensor<f32, [?, ?]> }\n\
         fn build(n : i32, m : i32) -> Tensor<f32, [?, ?]> { \
           let mut t = Tensor<f32, [?, ?]>::new([n, m]); t[1][2] = 7.0; return t; }\n\
         fn main() -> i32 { let d = Dyn { w : build(2, 3) }; return (d.w[1][2] * 6.0) as i32; }",
        42,
    );
}

#[test]
fn flat_matches_ast_tensor_map() {
    // `map` with a closure capturing a local: a fresh tensor, each element the closure's
    // adapter applied to the source's. 12 * 3 + 3 * 2.
    assert_parity(
        "fn main() -> i32 { let mut x = Tensor<f32, [2, 3]>::fill(1.0); x[1][2] = 4.0; \
           let k = 3.0; let y = x.map(|v| v * k); \
           return (y[1][2] * 3.0 + y[0][0] * 2.0) as i32; }",
        42,
    );
}

#[test]
fn flat_runs_a_data_enum_without_generics() {
    // `Result { Ok(i32), Err(i32) }` is the enum instance with no arguments: constructed,
    // returned, passed, and matched with its payload bound. Flat-only: the AST path fails on a
    // payload extraction. 2 + 4 * 10.
    assert_flat_exit(
        "enum R { Ok(i32), Err(i32) }\n\
         fn c(x : i32) -> R { if x >= 0 { return R::Ok(x); } return R::Err(0 - x); }\n\
         fn u(r : R, d : i32) -> i32 { let mut o = d; \
           match r { R::Ok(v) => { o = v; } R::Err(e) => { o = e * 10; } } return o; }\n\
         fn main() -> i32 { return u(c(2), 0) + u(c(0 - 4), 0); }",
        42,
    );
}

#[test]
fn flat_matches_ast_generic_struct_receiver() {
    // A method on a generic struct instance: `self : &Pair<i32>` resolves to the synthesized
    // instance layout, so the field reads through it. 40 + 2.
    assert_parity(
        "struct Pair<T> { first : T, second : T }\n\
         impl<T> Pair<T> { fn sum(self : &Pair<T>) -> T { return self.first + self.second; } }\n\
         fn main() -> i32 { let p = Pair<i32> { first : 40, second : 2 }; return p.sum(); }",
        42,
    );
}

#[test]
fn flat_matches_ast_function_bound_to_a_let() {
    // `let f = probe;` gives `f` the function's type, so the indirect call knows its result.
    assert_parity(
        "struct Map { slot : i32, present : i32 }\n\
         fn probe(m : &Map) -> &i32 { return &m.slot; }\n\
         fn main() -> i32 { let x = Map { slot : 42, present : 1 }; let f = probe; return *f(&x); }",
        42,
    );
}

#[test]
fn flat_matches_ast_spawn_with_no_value() {
    // A spawn region with no value as the trailing expression of a block: lowered as the effect
    // it is, and the block's value is the placeholder a void call gets.
    assert_parity(
        "fn main() -> i32 { unsafe { let k = 1; spawn on(Topology::CPU) { let z = k; } } \
           return 42; }",
        42,
    );
}

#[test]
fn flat_matches_ast_payload_free_enum_returned() {
    // A payload-free enum is its `i32` discriminant at a call boundary as in a signature.
    assert_parity(
        "enum E { A, B }\n\
         fn f() -> E { let e = E::B; return e; }\n\
         fn main() -> i32 { let e = f(); match e { E::B => { return 42; } _ => { return 0; } } }",
        42,
    );
}

#[test]
fn flat_matches_ast_topology_as_a_value() {
    // A topology used as a value is its dispatch id, the `i32` the checker types it as: bound
    // by an unannotated `let` through a folded `comptime` `if`, and compared.
    assert_parity(
        "fn main() -> i32 { let t = if comptime true { Topology::CPU } else { Topology::GPU }; \
           if t == Topology::CPU { return 42; } return 0; }",
        42,
    );
}

#[test]
fn flat_matches_ast_print_in_value_position() {
    // `let status = print!(..)` prints and binds a status nothing reads; the output is what
    // both paths agree on.
    assert_output_parity(
        "fn main() -> i32 { let x = 7; let status : i32 = print!(\"x=\", x, \"|\"); \
           let more : i32 = println!(\"done\"); return 0; }",
    );
}

#[test]
fn flat_matches_ast_matmul_with_run_time_extents() {
    // `a @ b` over `[?, ?]` operands: the result buffer takes its extents off the operands.
    // Both the value form and the assignment into an owned destination; a matmul consumes its
    // operands, so each gets its own identity. a = [[1, 2], [4, 2]]: c = a, then d = c, so
    // c[1][0] * 10 + c[1][1] = 42, twice, minus 42.
    assert_parity(
        "fn build(n : i32, m : i32) -> Tensor<f32, [?, ?]> { \
           let mut t = Tensor<f32, [?, ?]>::new([n, m]); \
           t[0][0] = 0.0; t[0][1] = 0.0; t[1][0] = 0.0; t[1][1] = 0.0; return t; }\n\
         fn main() -> i32 { let mut a = build(2, 2); a[0][0] = 1.0; a[0][1] = 2.0; \
           a[1][0] = 4.0; a[1][1] = 2.0; \
           let mut b = build(2, 2); b[0][0] = 1.0; b[1][1] = 1.0; \
           let c = a @ b; let x = (c[1][0] * 10.0 + c[1][1]) as i32; \
           let mut b2 = build(2, 2); b2[0][0] = 1.0; b2[1][1] = 1.0; \
           let mut d = Tensor<f32, [?, ?]>::new([2, 2]); d = c @ b2; \
           return x + (d[1][0] * 10.0 + d[1][1]) as i32 - 42; }",
        42,
    );
}

#[test]
fn flat_matches_ast_print_of_a_string_local() {
    // A string bound to a local is a pointer value; printing it goes through `print_str`, as
    // a literal does.
    assert_output_parity(
        "fn main() -> i32 { let s = \"hi|\"; let st : i32 = print!(s); print!(s); \
           println!(\"\"); return 0; }",
    );
}

#[test]
fn flat_matches_ast_construction_behind_an_unsafe_block() {
    // A construction as the tail of an `unsafe` block already sits in a slot; the `let` binds
    // that slot rather than spilling its address as a value. 40 + 2.
    assert_parity(
        "struct W { a : i32, b : i32 }\n\
         fn mk() -> W { let w = unsafe { W { a : 40, b : 2 } }; return w; }\n\
         fn main() -> i32 { let w = mk(); return w.a + w.b; }",
        42,
    );
}

#[test]
fn flat_matches_ast_reborrow_of_a_raw_pointer() {
    // `&mut *ptr` on a raw pointer is the pointer itself, returned as a reference and read
    // through: the `Vec::as_mut_slice` shape.
    assert_parity(
        "extern \"C\" { fn vx_vec_alloc(elem_size: i64, cap: i64) -> *mut i8; }\n\
         struct Buf { data : *mut i32 }\n\
         fn first(b : &mut Buf) -> &mut i32 { let ptr : *mut i32 = b.data; \
           return unsafe { &mut *ptr }; }\n\
         fn main() -> i32 { let p : *mut i32 = unsafe { vx_vec_alloc(4, 4) }; \
           let mut b = Buf { data : p }; unsafe { b.data[0] = 42; } \
           let r = first(&mut b); return *r; }",
        42,
    );
}

#[test]
fn flat_prints_a_string_value() {
    // A `String` value prints its C string through `print_str`. Flat-only: the AST path types
    // a print argument by its MLIR spelling and has no case for the struct.
    let flat = flat_llvm(
        "extern \"C\" { fn vx_string_from_c_str(c_str : *const i8) -> *mut i8; \
           fn vx_string_as_c_str(ptr : *mut i8) -> *const i8; \
           fn vx_string_free_c_str(ptr : *const i8) -> i32; }\n\
         struct String { ptr : *mut i8 }\n\
         fn main() -> i32 { let s = String { ptr : unsafe { vx_string_from_c_str(\"hi|\") } }; \
           print!(s); println!(\"\"); return 0; }",
    )
    .expect("flat path lowers this printing program");
    assert_eq!(run_output(&flat), normalize("hi|\n"));
}

#[test]
fn flat_matches_ast_comptime_if_statement_with_a_reachability_predicate() {
    // A statement-position `comptime` `if` on a `Reachable` predicate: the checker decides it
    // and empties the losing branch, and only the survivor lowers; the predicate itself has no
    // lowering on either path. Both branches give 42, so the answer does not depend on the
    // machine's reachability.
    assert_parity(
        "fn main() -> i32 { let mut r = 0; \
           if comptime Reachable<Topology::CPU, Topology::GPU> { r = 42; } else { r = 42; } \
           return r; }",
        42,
    );
}

#[test]
fn flat_matches_ast_transfer_of_a_scalar() {
    // A scalar has no bytes to move between spaces, so its transfer is the value itself:
    // both paths write the `vx.transfer` op, and lowering folds it to its operand. The value
    // stays `Pinned`, so it is printed rather than computed with.
    assert_output_parity(
        "fn main() -> i32 { \
           let g = spawn on(Topology::GPU) { 40 }; \
           let h = transfer(g, Memory::CPU_DRAM); \
           print(h); \
           return 0; }",
    );
}

#[test]
fn flat_matches_ast_placement_query_folded_to_its_answer() {
    // `x.topology()` is a fact of `x`'s type and the checker decides the comparison: neither
    // path ever sees the query, only the `true` it became.
    assert_parity(
        "fn main() -> i32 { let g = spawn on(Topology::GPU) { 1 }; \
           if g.topology() == Some(Topology::GPU) { return 42; } return 7; }",
        42,
    );
}

#[test]
fn flat_matches_ast_tensor_store_element_coercion() {
    // A default-`f32` float literal stored into a non-`f32` tensor is coerced to the element type at
    // the store (`bf16` -> `arith.truncf`), matching the AST's `coerce_type` before its `memref.store`
    // — without it the store is ill-typed. Read the stored values back via `as f32` and sum so the
    // (correctly truncated) value is observable: 4 × 1.5 = 6.0.
    assert_output_parity(
        "fn main() -> i32 { let mut a = Tensor<bf16>([4]); for i in 0..4 { a[i] = 1.5; } \
         let mut s = 0.0f32; for i in 0..4 { s = s + (a[i] as f32); } print(s); return 0; }",
    );
}

#[test]
fn flat_matches_ast_enum_construct_and_match() {
    // A payload-free (C-like) enum: `Dir::East` constructs the discriminant ordinal (a bare `i32`),
    // an enum-typed param passes it, and `match d { Dir::.. => .. }` is an eq-compare chain over the
    // ordinal (#227). `East`=2, `West`=3 -> `code` returns 3, 4 -> 3 + 4 = 7.
    assert_parity(
        "enum Dir { North, South, East, West }\n\
         fn code(d: Dir) -> i32 { let mut r = 0; \
           match d { Dir::North => { r = 1; } Dir::South => { r = 2; } \
                     Dir::East => { r = 3; } Dir::West => { r = 4; } } return r; }\n\
         fn main() -> i32 { return code(Dir::East) + code(Dir::West); }",
        7,
    );
}

#[test]
fn flat_matches_ast_enum_match_wildcard() {
    // The wildcard arm is the unconditional default: `Color::Green` matches neither listed arm, so
    // it falls through to `_ => 99`.
    assert_parity(
        "enum Color { Red, Green, Blue }\n\
         fn pick(c: Color) -> i32 { let mut r = 0; \
           match c { Color::Red => { r = 10; } Color::Blue => { r = 30; } _ => { r = 99; } } return r; }\n\
         fn main() -> i32 { return pick(Color::Green); }",
        99,
    );
}

/// A `match` used as a value evaluates to the arm that matched.
///
/// The AST lowering used to branch through the arms and then return a hard-coded
/// `arith.constant 0` as the match's value, so every value-producing match evaluated to zero --
/// silently, with the wrong value reaching the process exit code. The flat path declines `Match`
/// and falls back to this oracle, so nothing else was in a position to catch it.
///
/// Asserted through the AST path alone, since the flat path has no lowering to compare against.
#[test]
fn a_value_match_evaluates_to_the_arm_that_matched() {
    let src = "enum Color { Red, Green, Blue }\n\
               fn pick(c: Color) -> i32 { let x = match c { Color::Red => { 1 } \
                 Color::Green => { 42 } Color::Blue => { 3 } }; return x; }\n\
               fn main() -> i32 { return pick(Color::Green); }";
    assert_eq!(ast_exit_code(src), 42, "the Green arm's value, not zero");

    // Each arm, so the answer tracks the scrutinee rather than happening to equal one arm.
    for (variant, want) in [("Red", 1), ("Green", 42), ("Blue", 3)] {
        let src = format!(
            "enum Color {{ Red, Green, Blue }}\n\
             fn pick(c: Color) -> i32 {{ let x = match c {{ Color::Red => {{ 1 }} \
               Color::Green => {{ 42 }} Color::Blue => {{ 3 }} }}; return x; }}\n\
             fn main() -> i32 {{ return pick(Color::{variant}); }}"
        );
        assert_eq!(ast_exit_code(&src), want, "arm {variant}");
    }
}

/// An integer `match` in value position, selected by a wildcard arm.
#[test]
fn a_value_match_over_integers_takes_its_wildcard() {
    assert_eq!(
        ast_exit_code(
            "fn main() -> i32 { let n = 7; let x = match n { 1 => { 10 } 2 => { 20 } \
             _ => { 99 } }; return x; }"
        ),
        99
    );
}

#[test]
fn flat_matches_ast_sizeof_value() {
    // `sizeof<T>()` folds to a compile-time `i64` constant of `T`'s byte size, matching the AST
    // codegen (#228). `sizeof<f64>()` = 8, `sizeof<i8>()` = 1 -> 9.
    assert_parity(
        "fn main() -> i32 { let s: i64 = sizeof<f64>(); let t: i64 = sizeof<i8>(); \
         return (s + t) as i32; }",
        9,
    );
}

#[test]
fn flat_matches_ast_comptime_block() {
    // A `comptime { .. }` block lowers transparently — its `sizeof` folds to a constant, and its
    // `assert` now emits a runtime check whose condition the checker already decided, so it
    // holds and the block still has no observable effect: `main` returns 0 (#228). Matches the
    // AST codegen, which lowers the block the same way.
    assert_parity(
        "fn main() -> i32 { comptime { let s: i64 = sizeof<f64>(); assert(s == 8); } return 0; }",
        0,
    );
}

#[test]
fn flat_matches_ast_return_widening_coercion() {
    // `return 7` — a default-`i32` literal — from an `-> i64` function coerces to `i64` at the
    // return (`arith.extsi`), matching the AST's `coerce_type`. Without it the `func.return` type
    // contradicts the signature. `wide()` = 7.
    assert_parity(
        "fn wide() -> i64 { return 7; } fn main() -> i32 { return wide() as i32; }",
        7,
    );
}

#[test]
fn flat_matches_ast_borrow_tensor_print() {
    // `print(&t)` where `t` is a tensor (`npu_lowering_execution.vx` shape, #230): a tensor is a
    // memref — already a reference — so the borrow `&t` is transparent (yields the tensor itself),
    // matching the AST codegen. The `printMemref` dump (shape/strides/data) matches the oracle.
    assert_output_parity(
        "fn main() -> i32 { let mut a = Tensor<f32>([2]); a[0] = 1.0; a[1] = 2.0; \
         print(&a); return 0; }",
    );
}

#[test]
fn flat_matches_ast_nested_value_if() {
    // A value-`if` whose then-branch is itself a value-`if` (`expr_assignment.vx` shape, #229): the
    // inner if lowers through `lower_expr`'s `Expr::If` arm (its result type inferred from the branch
    // value) into the outer slot. `x=25` -> `25>10` -> `25>20` -> 100.
    assert_parity(
        "fn nested(x: i32) -> i32 { let val: i32 = if x > 10 { if x > 20 { 100 } else { 50 } } \
         else { 0 }; return val; }\n\
         fn main() -> i32 { return nested(25); }",
        100,
    );
}

#[test]
fn flat_matches_ast_value_if_implicit_return() {
    // A value-`if` as a function's implicit return (no annotation): the parser rewrites the trailing
    // `if` to `return if ..`, which lowers through the `Expr::If` arm. `implicit(3)` -> `3>10` false
    // -> `3 + 9` = 12.
    assert_parity(
        "fn implicit(x: i32) -> i32 { if x > 10 { x + 5 } else { x + 9 } }\n\
         fn main() -> i32 { return implicit(3); }",
        12,
    );
}

#[test]
fn flat_matches_ast_value_if_call_argument() {
    // A value-`if` nested as a call argument, in a function with no other control flow (so it lowers
    // in pure-SSA mode — the `if`'s blocks are self-contained). `5 > 2` -> 42.
    assert_parity(
        "fn id(x: i32) -> i32 { x }\n\
         fn main() -> i32 { return id(if 5 > 2 { 42 } else { 7 }); }",
        42,
    );
}

#[test]
fn flat_matches_ast_value_if_else_if_compound_assign() {
    // An `else if` chain as a compound-assign RHS: the `else` branch's trailing value is itself an
    // `if`, lowered through the `Expr::If` arm into the outer slot. `x=15` -> not `>20`, is `>10` ->
    // 50; `r = 1 + 50` = 51.
    assert_parity(
        "fn main() -> i32 { let x = 15; let mut r = 1; \
         r += if x > 20 { 100 } else if x > 10 { 50 } else { 0 }; return r; }",
        51,
    );
}

#[test]
fn flat_matches_ast_nested_calls() {
    // A call whose argument is itself a call — the flat stream nests `Arg`/`Call`
    // pairs, so each `Call` must consume exactly its own trailing args.
    assert_parity(
        "fn inc(x: i32) -> i32 { return x + 1; }\n\
         fn dbl(x: i32) -> i32 { return x + x; }\n\
         fn main() -> i32 { return dbl(inc(9)); }",
        20,
    );
}

#[test]
fn flat_matches_ast_call_from_control_flow() {
    // Bricks 1+2 together: a helper called from inside an `if` in `main`, its
    // argument read from a slot-backed local.
    assert_parity(
        "fn sq(x: i32) -> i32 { return x * x; }\n\
         fn main() -> i32 { let n = 5; let mut r = 0; if n > 0 { r = sq(n); } return r; }",
        25,
    );
}

#[test]
fn flat_matches_ast_struct_field_sum() {
    // A struct built in place, its fields read back and summed — exercises the
    // aggregate `llvm.alloca` + GEP + `llvm.load`/`store` field path.
    assert_parity(
        "struct Point { x: i32, y: i32 }\n\
         fn main() -> i32 { let p = Point { x: 3, y: 4 }; return p.x + p.y; }",
        7,
    );
}

#[test]
fn flat_matches_ast_struct_return() {
    // A function returns a struct by value (#215): the callee builds it in a slot, loads the
    // `!llvm.struct` and returns it; the caller spills the returned value to a slot and reads fields.
    assert_parity(
        "struct P { x: i32, y: i32 }\n\
         fn mk() -> P { return P { x: 3, y: 4 }; }\n\
         fn main() -> i32 { let p = mk(); return p.x + p.y; }",
        7,
    );
}

#[test]
fn flat_matches_ast_struct_return_used_directly() {
    // Return a struct and read a field off the call result without an intermediate binding.
    assert_parity(
        "struct P { x: i32, y: i32 }\n\
         fn mk() -> P { return P { x: 10, y: 5 }; }\n\
         fn main() -> i32 { let p = mk(); return p.x - p.y; }",
        5,
    );
}

#[test]
fn flat_matches_ast_struct_field_in_control_flow() {
    // Bricks 1+3 together: struct fields feed an `if` condition and the returned
    // value (a struct slot and a scalar slot coexist in the same function).
    assert_parity(
        "struct Box { lo: i32, hi: i32 }\n\
         fn main() -> i32 { let b = Box { lo: 2, hi: 9 }; let mut r = 0; if b.lo < b.hi { r = b.hi - b.lo; } return r; }",
        7,
    );
}

#[test]
fn flat_matches_ast_tensor_element_read() {
    // The minimal self-contained tensor program: allocate a tensor, fill it with
    // scalar-element stores, and read one element back (no reduction/cast, so the
    // result is an i32 exit code). Exercises the tensor-type side table +
    // memref.alloc/store/load through the real JIT.
    assert_parity(
        "fn main() -> i32 { let mut q = Tensor<i32>([4]); q[0] = 5; q[1] = 6; q[2] = 7; \
         q[3] = 8; return q[2]; }",
        7,
    );
}

#[test]
fn flat_matches_ast_tensor_row_element_read() {
    // A rank-2 tensor: fill it, then read an element through a row sub-view
    // (`q[1][2]`) — exercises `memref.reinterpret_cast` (row) + `memref.load`
    // through the strided row, flat-vs-AST.
    assert_parity(
        "fn main() -> i32 { let mut q = Tensor<i32>([2, 3]); q[0][0] = 1; q[0][1] = 2; \
         q[0][2] = 3; q[1][0] = 4; q[1][1] = 5; q[1][2] = 6; return q[1][2]; }",
        6,
    );
}

#[test]
fn flat_matches_ast_tensor_param_passed_to_helper() {
    // A tensor built in `main`, passed by value to a helper with a tensor param
    // (a `memref` in the signature) that indexes it. Exercises tensor params +
    // tensor call arguments, flat-vs-AST.
    assert_parity(
        "fn get(q: Tensor<i32, [4]>, i: i32) -> i32 { return q[i]; }\n\
         fn main() -> i32 { let mut q = Tensor<i32>([4]); q[0] = 5; q[1] = 6; q[2] = 7; \
         q[3] = 8; return get(q, 2); }",
        7,
    );
}

#[test]
fn flat_matches_ast_tensor_row_sum_reduction() {
    // A float `sum` over a static-sized row (`q[0]`, a reinterpret_cast sub-view)
    // fed into a compare so the exit code is an i32: sum([1,2,3,4]) = 10 > 9 sets
    // r = 1. Exercises reinterpret_cast + vector.load + vector.reduction through
    // the real JIT, flat-vs-AST. (The AST oracle can only reduce a static-sized
    // slice, so the reduction is over a row, not the whole dynamic tensor.)
    assert_parity(
        "fn main() -> i32 { let mut q = Tensor<f32>([2, 4]); q[0][0] = 1.0; q[0][1] = 2.0; \
         q[0][2] = 3.0; q[0][3] = 4.0; let mut r = 0; if sum(q[0]) > 9.0 { r = 1; } return r; }",
        1,
    );
}

#[test]
fn flat_matches_ast_tensor_elementwise_row_store() {
    // The FlashAttention write-path shape: an elementwise scalar-broadcast multiply
    // over a row (`q[0] * 2.0`) stored back into a row (`o[0] = …`), then an element
    // read + compare so the exit is an i32. o[0] = [2,4,6,8]; o[0][1] = 4 > 3.5 -> 1.
    // Exercises vector.load/broadcast + arith.mulf + vector.store, flat-vs-AST.
    assert_parity(
        "fn main() -> i32 { let mut q = Tensor<f32>([2, 4]); q[0][0] = 1.0; q[0][1] = 2.0; \
         q[0][2] = 3.0; q[0][3] = 4.0; let mut o = Tensor<f32>([2, 4]); o[0] = q[0] * 2.0; \
         let mut r = 0; if o[0][1] > 3.5 { r = 1; } return r; }",
        1,
    );
}

#[test]
fn flat_matches_ast_flashattention_write_path() {
    // Capstone: the FlashAttention inner write path composed end to end through the
    // flat path -- `o[0] = v[0] * (dot(q[0], k[0]) * scale)`. dot([1,2,3,4],[1,1,1,1])
    // = 10; * 0.5 = 5; v[0] * 5 = [10,10,10,10]; o[0][0] = 10 > 9 -> r = 1. Exercises
    // reduction (dot) + scalar multiply + elementwise broadcast + row store + read,
    // all together, flat-vs-AST.
    assert_parity(
        "fn main() -> i32 { \
           let mut q = Tensor<f32>([1, 4]); q[0][0] = 1.0; q[0][1] = 2.0; q[0][2] = 3.0; q[0][3] = 4.0; \
           let mut k = Tensor<f32>([1, 4]); k[0][0] = 1.0; k[0][1] = 1.0; k[0][2] = 1.0; k[0][3] = 1.0; \
           let mut v = Tensor<f32>([1, 4]); v[0][0] = 2.0; v[0][1] = 2.0; v[0][2] = 2.0; v[0][3] = 2.0; \
           let mut o = Tensor<f32>([1, 4]); \
           let scale = 0.5; \
           o[0] = v[0] * (dot(q[0], k[0]) * scale); \
           let mut r = 0; if o[0][0] > 9.0 { r = 1; } return r; }",
        1,
    );
}

#[test]
fn flat_matches_ast_tensor_transfer() {
    // `transfer(q, Memory::NPU_HBM)` moves (copies) the tensor to a memory space;
    // the vx→standard lowering makes it an alloc + memref.copy. Read an element of
    // the copy back for an i32 exit: o[0][2] = q[0][2] = 3.0 > 2.5 -> r = 1.
    // Exercises vx.transfer, flat-vs-AST.
    assert_parity(
        "fn main() -> i32 { let mut q = Tensor<f32>([2, 4]); q[0][0] = 1.0; q[0][1] = 2.0; \
         q[0][2] = 3.0; q[0][3] = 4.0; let o = transfer(q, Memory::NPU_HBM); let mut r = 0; \
         if o[0][2] > 2.5 { r = 1; } return r; }",
        1,
    );
}

#[test]
fn flat_matches_ast_tensor_print_output() {
    // Print a filled tensor and compare the JIT *stdout* (not the exit code) of the
    // flat path against the AST path -- the printMemrefF32 dump (shape/strides/data)
    // must match after normalizing the non-deterministic base pointer.
    assert_output_parity(
        "fn main() -> i32 { let mut q = Tensor<f32>([2, 2]); q[0][0] = 1.0; q[0][1] = 2.0; \
         q[1][0] = 3.0; q[1][1] = 4.0; print(q); return 0; }",
    );
}

#[test]
fn flat_matches_ast_integer_negation() {
    // `-x` on an integer (#214): the emitter now lowers `Neg` (`0 - x`), so it JIT-matches the AST.
    assert_parity(
        "fn main() -> i32 { let a = 5; let b = -a; return b + 12; }",
        7,
    );
}

#[test]
fn flat_matches_ast_float_negation() {
    // `-x` on a float now lowers (`arith.negf`, #214); the negated value is printed for stdout parity.
    assert_output_parity("fn main() -> i32 { let x = 3.0f32; print(-x); return 0; }");
}

#[test]
fn flat_matches_ast_scalar_casts() {
    // Scalar `as` casts now lower + emit through the flat path (#214): a chained widen/narrow, and an
    // int->float->int round trip, both JIT-match the AST oracle.
    assert_parity("fn main() -> i32 { let a = 7; return a as i64 as i32; }", 7);
    assert_parity(
        "fn main() -> i32 { let a = 100; let f = a as f32; return f as i32; }",
        100,
    );
}

/// Compile a "library" module to a serialized `.vxlib` interface (frozen registry + flat-HIR bodies).
fn build_lib_interface(path: &str, src: &str) -> Vec<u8> {
    let mut prog = parse(src);
    prog.module_path = path.into();
    let mut mods = vec![prog];
    let symbol_map = vxc::resolver::build_symbol_map(&mods);
    mods[0].resolve_names(&symbol_map, &[]);
    vxc::pipeline::emit_module_interface(&mods).expect("emit module interface")
}

/// The stdlib-decoupling endgame in miniature (#220 stage 4): a program links an imported function's
/// **body from a precompiled `.vxlib` artifact**, with the library's source *never parsed* in the
/// consumer compile. Proves the whole mechanism end to end through the flat path — producer (harvest +
/// serialize) → deserialize → registry merge → flat codegen links `body_of` → JIT.
#[test]
fn program_links_a_function_body_from_a_vxlib_artifact() {
    use vxc::syntax::{ElementType, Function, Topology, Type};

    // 1. Producer: a library module -> a `.vxlib` interface (bytes). This is the only place its
    //    source is ever seen; the consumer below works purely from these bytes.
    let lib_bytes = build_lib_interface(
        "crate::mathlib",
        "fn double(x: i32) -> i32 { return x * 2; }",
    );

    // 2. Consumer: a program that CALLS `double`, compiled with no access to the library's AST.
    let mut app = parse("fn main() -> i32 { return double(21); }");
    app.module_path = "crate::app".into();
    let mut mods = vec![app];
    let symbol_map = vxc::resolver::build_symbol_map(&mods);
    mods[0].resolve_names(&symbol_map, &[]);

    // Fold the precompiled library interface into the app's frozen registry (no parse of the lib).
    let mut registry = vxc::pipeline::build_frozen_registry(&mods).expect("app registry");
    let lib = vxc::metadata::deserialize_registry_interface(&lib_bytes).expect("deserialize lib");
    registry.merge_from(lib);
    let session = std::sync::Arc::new(GlobalSession::with_registry(1, registry));

    // Lower the app's own functions -- `double(21)` resolves via the merged `fn_sigs`.
    let mut lowered = Vec::new();
    for f in &mods[0].functions {
        let mut worker = LocalWorkerState::new(session.clone());
        lower_function_to_hir(f, &mut worker).expect("app fn lowers to flat HIR");
        lowered.push(worker);
    }

    // Pull `double`'s body from the artifact and give it a signature-only `Function` to emit against.
    let double_gid = session
        .registry
        .fn_sigs
        .get(&vxc::symbol::Symbol::from("double"))
        .expect("double resolves from the merged interface")
        .gid;
    let body = session
        .registry
        .body_of(double_gid)
        .expect("double's body came from the .vxlib artifact")
        .clone();
    let synth = Function {
        is_unsafe: false,
        name: body.name.clone(),
        generics: vec![],
        params: body
            .params
            .iter()
            .enumerate()
            .map(|(i, t)| {
                (
                    vxc::symbol::Symbol::from(format!("a{i}").as_str()),
                    t.clone(),
                )
            })
            .collect(),
        topology: Topology::CPU,
        return_type: body.ret_ty.clone(),
        requires: vec![],
        ensures: vec![],
        where_transfers: vec![],
        body: vec![],
        doc_comment: None,
    };
    assert_eq!(synth.return_type, Type::Scalar(ElementType::I32));

    // Emit one module: the app's `main` + the imported `double` (body from the artifact).
    let mut funcs: Vec<(&Function, &[_], &[_])> = mods[0]
        .functions
        .iter()
        .zip(&lowered)
        .map(|(f, w)| {
            (
                f,
                w.local_hir_stream.as_slice(),
                w.local_type_stream.as_slice(),
            )
        })
        .collect();
    funcs.push((&synth, body.hir.as_slice(), body.types.as_slice()));

    let mlir = vxc::codegen::flat::emit_module_mlir(
        &funcs,
        &session.registry,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        vxc::config::Schedule::Parallel,
    )
    .expect("flat codegen emits the linked module");
    let context = make_context();
    let mut module = melior::ir::Module::parse(&context, &mlir).expect("linked flat MLIR parses");
    lower_to_llvm(&context, &mut module).expect("flat lower_to_llvm");
    assert_eq!(
        exit_code(&module.as_operation().to_string()),
        42,
        "double(21) linked from the .vxlib artifact returns 42"
    );
}

/// Phase 2 (#265 step 7 / #219), runnable through the **real driver**: a consumer compiled with
/// `unsafe fn` crosses a `--link-interface` boundary. The consumer never sees the library's body,
/// so the refusal can only come from the signature in the artifact -- which is the whole reason the
/// flag rides on `FnSig`. Two libraries are built, one `unsafe fn` and one not, and the same call
/// is made against each: refused for the first, accepted for the second. Asserting only the refusal
/// would pass equally if the check refused every imported call.
#[test]
fn driver_link_interface_refuses_an_unguarded_unsafe_import() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("vx_link_unsafe_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (lib, vxlib) = (dir.join("unsafelib.vx"), dir.join("unsafelib.vxlib"));
    let (safelib, safevxlib) = (dir.join("safelib.vx"), dir.join("safelib.vxlib"));
    let app = dir.join("app.vx");
    std::fs::write(&lib, "unsafe fn double(x : i32) -> i32 { return x * 2; }\n").unwrap();
    std::fs::write(&safelib, "fn double(x : i32) -> i32 { return x * 2; }\n").unwrap();
    std::fs::write(&app, "fn main() -> i32 { return double(21); }\n").unwrap();

    let vxc = env!("CARGO_BIN_EXE_vxc");
    let emit = Command::new(vxc)
        .args([
            "--emit-interface",
            lib.to_str().unwrap(),
            "-o",
            vxlib.to_str().unwrap(),
        ])
        .output()
        .expect("run vxc --emit-interface");
    assert!(
        emit.status.success(),
        "emit-interface failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );

    let emit_safe = Command::new(vxc)
        .args([
            "--emit-interface",
            safelib.to_str().unwrap(),
            "-o",
            safevxlib.to_str().unwrap(),
        ])
        .output()
        .expect("run vxc --emit-interface");
    assert!(
        emit_safe.status.success(),
        "emit-interface failed for the safe library"
    );

    let check_against = |artifact: &std::path::Path| -> String {
        let out = Command::new(vxc)
            .args([
                "--link-interface",
                artifact.to_str().unwrap(),
                app.to_str().unwrap(),
                "--action",
                "print-ast",
            ])
            .stdout(std::process::Stdio::null())
            .output()
            .expect("run vxc --link-interface");
        String::from_utf8_lossy(&out.stderr).into_owned()
    };

    let refused = check_against(&vxlib);
    assert!(
        refused.contains("E5001") && refused.contains("double"),
        "an unguarded call to an imported `unsafe fn` must be refused from the artifact's \
         signature alone, got:\n{refused}"
    );

    let accepted = check_against(&safevxlib);
    assert!(
        !accepted.contains("E5001"),
        "the same call against a safe library must not be refused -- otherwise the check \
         above proves only that imported calls are rejected, got:\n{accepted}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `--link-interface` — the library source never passed on the command line — resolves the imported
/// call from the merged registry (frontend) and links its flat-HIR body from the artifact (codegen),
/// then JITs to the expected value. Productionizes the stage-4 mechanism above through `vxc` itself.
#[test]
fn driver_link_interface_runs_a_scalar_import() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("vx_link_run_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (lib, app, vxlib) = (
        dir.join("mathlib.vx"),
        dir.join("app.vx"),
        dir.join("mathlib.vxlib"),
    );
    std::fs::write(&lib, "fn double(x : i32) -> i32 { return x * 2; }\n").unwrap();
    std::fs::write(&app, "fn main() -> i32 { return double(21); }\n").unwrap();

    let vxc = env!("CARGO_BIN_EXE_vxc");
    let emit = Command::new(vxc)
        .args([
            "--emit-interface",
            lib.to_str().unwrap(),
            "-o",
            vxlib.to_str().unwrap(),
        ])
        .output()
        .expect("run vxc --emit-interface");
    assert!(
        emit.status.success(),
        "emit-interface failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );

    // The consumer compile is given only the app + the artifact — never `mathlib.vx`.
    let run = Command::new(vxc)
        .args([
            "--link-interface",
            vxlib.to_str().unwrap(),
            app.to_str().unwrap(),
            "--run",
        ])
        .env("RUST_BACKTRACE", "1")
        .output()
        .expect("run vxc --link-interface --run");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        out.contains("code: 42"),
        "expected the JIT to return 42 from the linked import, got:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The AST codegen cannot link a `--link-interface` import (it has no AST for the imported function),
/// so a consumer whose body falls outside the flat subset must fail with a **clean diagnostic** — the
/// frontend check (incl. cross-module provenance) still passes, only codegen is blocked. Regression
/// guard for the fixed AST-fallback ICE ("Function … not found").
#[test]
fn driver_link_interface_declines_cleanly_outside_flat_subset() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("vx_link_decline_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (lib, app, vxlib) = (
        dir.join("reflib.vx"),
        dir.join("refapp.vx"),
        dir.join("reflib.vxlib"),
    );
    std::fs::write(&lib, "fn double(x : i32) -> i32 { return x * 2; }\n").unwrap();
    // `main` allocates a tensor whose extent is a local (`Tensor<f32, [n, n]>`), which the flat
    // path does not model and so declines, while also calling the imported `double`. The
    // decline forces the AST path, which has no AST for the imported body, so the driver must
    // emit a clean diagnostic rather than ICE. (A data-carrying enum from a call, the construct
    // this test used before, now lowers on the flat path.)
    std::fs::write(
        &app,
        "fn main() -> i32 { let n = 3; let mut a = Tensor<f32, [n, n]>::uninit(); \
           a[0][0] = 1.0; return double(10) + (a[0][0] as i32); }\n",
    )
    .unwrap();

    let vxc = env!("CARGO_BIN_EXE_vxc");
    assert!(Command::new(vxc)
        .args([
            "--emit-interface",
            lib.to_str().unwrap(),
            "-o",
            vxlib.to_str().unwrap(),
        ])
        .output()
        .expect("emit")
        .status
        .success());

    let run = Command::new(vxc)
        .args([
            "--link-interface",
            vxlib.to_str().unwrap(),
            app.to_str().unwrap(),
            "--run",
        ])
        .output()
        .expect("run");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        !run.status.success(),
        "linking an import into an out-of-flat-subset program must fail, got success:\n{out}"
    );
    assert!(
        out.contains("flat-coverage") || out.contains("outside the flat-codegen subset"),
        "expected a clean flat-coverage error, got:\n{out}"
    );
    assert!(
        !out.contains("Internal Error") && !out.contains("panicked"),
        "must be a clean error, not an ICE, got:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// #230 cross-module scalar references: the borrow-checker showcase `fn pick(a : &i32, b : &i32) ->
/// &i32` — the signature rustc cannot compile without a lifetime annotation — defined in a `.vxlib`
/// and *called across the module boundary* with the library source absent. `main` borrows two locals
/// (`&x`, `&y`), passes them, and derefs the returned reference. The whole program lowers on the flat
/// path (address-taken scalars become `llvm.alloca` slots, `pick` links from the artifact) and JITs.
/// `pick` returns `b`, so `*r == 20`.
#[test]
fn driver_import_runs_cross_module_scalar_references() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("vx_import_scalar_ref_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (lib, app, vxlib) = (
        dir.join("reflib.vx"),
        dir.join("refapp.vx"),
        dir.join("reflib.vxlib"),
    );
    std::fs::write(&lib, "fn pick(a : &i32, b : &i32) -> &i32 { return b; }\n").unwrap();
    std::fs::write(
        &app,
        "import reflib;\nfn main() -> i32 { let x = 10; let y = 20; let r = pick(&x, &y); return *r; }\n",
    )
    .unwrap();

    let vxc = env!("CARGO_BIN_EXE_vxc");
    assert!(Command::new(vxc)
        .args([
            "--emit-interface",
            lib.to_str().unwrap(),
            "-o",
            vxlib.to_str().unwrap(),
        ])
        .output()
        .expect("emit")
        .status
        .success());
    // Delete the library source: the reference-returning function must resolve purely from the artifact.
    std::fs::remove_file(&lib).unwrap();

    let run = Command::new(vxc)
        .args([app.to_str().unwrap(), "--run"])
        .env("VX_STD_PATH", dir.to_str().unwrap())
        .env("RUST_BACKTRACE", "1")
        .output()
        .expect("run");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        out.contains("code: 20"),
        "expected 20 from the cross-module `pick(&x, &y)` returning `b`, got:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The #219 auto-load flip: `import mathlib;` resolves to a sibling `mathlib.vxlib` and merges its
/// interface **automatically — no `--link-interface` flag** — with the library source absent. The
/// consumer links + JITs the imported body. This is the ergonomic form of the linking above.
#[test]
fn driver_import_auto_resolves_a_vxlib_artifact() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("vx_import_auto_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (lib, app, vxlib) = (
        dir.join("mathlib.vx"),
        dir.join("imapp.vx"),
        dir.join("mathlib.vxlib"),
    );
    std::fs::write(&lib, "fn double(x : i32) -> i32 { return x * 2; }\n").unwrap();
    std::fs::write(
        &app,
        "import mathlib;\nfn main() -> i32 { return double(21); }\n",
    )
    .unwrap();

    let vxc = env!("CARGO_BIN_EXE_vxc");
    assert!(Command::new(vxc)
        .args([
            "--emit-interface",
            lib.to_str().unwrap(),
            "-o",
            vxlib.to_str().unwrap(),
        ])
        .output()
        .expect("emit")
        .status
        .success());
    // Delete the library source: the import must resolve *purely* to the artifact.
    std::fs::remove_file(&lib).unwrap();

    // No --link-interface flag; VX_STD_PATH makes `import mathlib;` resolve to `<dir>/mathlib.vxlib`.
    let run = Command::new(vxc)
        .args([app.to_str().unwrap(), "--run"])
        .env("VX_STD_PATH", dir.to_str().unwrap())
        .env("RUST_BACKTRACE", "1")
        .output()
        .expect("run");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        out.contains("code: 42"),
        "expected 42 from the auto-loaded `import`, got:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// #219 imported struct: a consumer uses a struct defined only in a `.vxlib` — receives it from an
/// imported function and reads a field — with the library source absent. The field type resolves from
/// the registry's serialized `structs` table (member access off the AST env), the imported body links,
/// and it JITs. `origin() -> Point { x: 7, .. }`, `main -> p.x == 7`.
#[test]
fn driver_import_uses_a_struct_from_a_vxlib() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("vx_import_struct_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (lib, app, vxlib) = (
        dir.join("shapes.vx"),
        dir.join("shapeapp.vx"),
        dir.join("shapes.vxlib"),
    );
    std::fs::write(
        &lib,
        "struct Point { x : i32, y : i32 }\nfn origin() -> Point { return Point { x : 7, y : 9 }; }\n",
    )
    .unwrap();
    std::fs::write(
        &app,
        "import shapes;\nfn main() -> i32 { let p = origin(); return p.x; }\n",
    )
    .unwrap();

    let vxc = env!("CARGO_BIN_EXE_vxc");
    assert!(Command::new(vxc)
        .args([
            "--emit-interface",
            lib.to_str().unwrap(),
            "-o",
            vxlib.to_str().unwrap(),
        ])
        .output()
        .expect("emit")
        .status
        .success());
    // Delete the library source: the struct type + its field must resolve purely from the artifact.
    std::fs::remove_file(&lib).unwrap();

    let run = Command::new(vxc)
        .args([app.to_str().unwrap(), "--run"])
        .env("VX_STD_PATH", dir.to_str().unwrap())
        .env("RUST_BACKTRACE", "1")
        .output()
        .expect("run");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        out.contains("code: 7"),
        "expected 7 (Point.x from the imported struct), got:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// #219 imported struct construction: a consumer *builds* a struct defined only in a `.vxlib`
/// (`Point { x: 5, y: 9 }`) and reads a field — with the library source absent. The struct literal's
/// field checking resolves from the registry's `structs` table (off the AST env), and it JITs.
#[test]
fn driver_import_constructs_a_struct_from_a_vxlib() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("vx_import_ctor_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (lib, app, vxlib) = (
        dir.join("shapes.vx"),
        dir.join("ctorapp.vx"),
        dir.join("shapes.vxlib"),
    );
    std::fs::write(&lib, "struct Point { x : i32, y : i32 }\n").unwrap();
    std::fs::write(
        &app,
        "import shapes;\nfn main() -> i32 { let p = Point { x : 5, y : 9 }; return p.x; }\n",
    )
    .unwrap();

    let vxc = env!("CARGO_BIN_EXE_vxc");
    assert!(Command::new(vxc)
        .args([
            "--emit-interface",
            lib.to_str().unwrap(),
            "-o",
            vxlib.to_str().unwrap(),
        ])
        .output()
        .expect("emit")
        .status
        .success());
    // Delete the source: the struct type + its fields must resolve purely from the artifact.
    std::fs::remove_file(&lib).unwrap();

    let run = Command::new(vxc)
        .args([app.to_str().unwrap(), "--run"])
        .env("VX_STD_PATH", dir.to_str().unwrap())
        .env("RUST_BACKTRACE", "1")
        .output()
        .expect("run");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        out.contains("code: 5"),
        "expected 5 (constructed imported Point.x), got:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn flat_matches_ast_string_value_pointer_arg() {
    // A string literal in *value* position (#231): bound to a local, then passed as an `!llvm.ptr`
    // argument to a pointer-typed parameter — the shape of `vx_stdout_write(msg, 14)`. The callee
    // ignores the pointer and returns the length, so the exit code is observable without a deref or a
    // runtime symbol (self-contained). Exercises `StringConst` + pointer param + pointer arg.
    assert_parity(
        "fn take(p: *const u8, n: i32) -> i32 { return n; }\n\
         fn main() -> i32 { let msg = \"Hello!\\n\"; let r = take(msg, 14); return r; }",
        14,
    );
}

#[test]
fn flat_matches_ast_pointer_return_and_memory_slot() {
    // The FFI pointer ABI (#235): a pointer-*returning* function (like an FFI allocator), a pointer
    // local that must survive across basic blocks (the `if` forces the memory model, so the pointer
    // lives in an `llvm.alloca` slot), and a pointer argument read back from that slot. All Vx-defined,
    // so no external symbol is linked — the exit code alone proves the pointer threaded through.
    assert_parity(
        "fn make(x: *const u8) -> *const u8 { return x; }\n\
         fn take(p: *const u8) -> i32 { return 7; }\n\
         fn main() -> i32 {\n\
         let s = \"hi\";\n\
         let p = make(s);\n\
         let mut r = 0;\n\
         if r == 0 { r = take(p); }\n\
         return r;\n\
         }",
        7,
    );
}

#[test]
fn flat_matches_ast_corpus_ffi_stdio_strings() {
    // The #231 corpus driver: string *values* bound to locals and passed to the `vx_stdout_write`/
    // `vx_stderr_write` FFI externs (`!llvm.ptr` arguments), plus print-position strings (#225). Its
    // JIT stdout must match the AST oracle byte for byte.
    assert_output_parity(&corpus("ffi_stdio.vx"));
}

#[test]
fn flat_matches_ast_corpus_ffi_option() {
    // The #235 corpus driver: FFI externs returning/taking `*mut i8` (`!llvm.ptr`), a pointer local
    // stored across an `if` (memory-mode pointer slot), and a `Bool`-returning extern as the `if`
    // condition. Its JIT stdout must match the AST oracle.
    assert_output_parity(&corpus("ffi_option.vx"));
}

#[test]
fn flat_infers_int_literal_argument_to_param_type() {
    // #240: an unsuffixed literal argument adopts its parameter's type — `42` is born `i64` at the
    // call to `wants_i64`, so the flat `func.call` operand is `i64` and matches the callee. No
    // implicit coercion is involved; the literal simply infers to the checking position's type.
    assert_parity(
        "fn wants_i64(n: i64) -> i64 { return n; }\n\
         fn main() -> i32 { return wants_i64(42) as i32; }",
        42,
    );
}

#[test]
fn flat_explicit_as_cast_widens_int_argument() {
    // #240: Vx has no implicit numeric conversion, so a non-literal `i32` value passed to an `i64`
    // parameter is a type error unless the programmer writes an explicit `as`. The `AsCast` lowers
    // identically on both backends; parity with the AST oracle confirms no divergence.
    assert_parity(
        "fn wants_i64(n: i64) -> i64 { return n; }\n\
         fn main() -> i32 { let x = 7; let r = wants_i64(x as i64); return r as i32; }",
        7,
    );
}

#[test]
fn flat_explicit_as_cast_widens_float_argument() {
    // #240: an `f32` local widened to an `f64` parameter with an explicit `as` (`arith.extf`), the
    // float analogue of the integer case. The cast value flows through and truncates to the exit code.
    assert_parity(
        "fn wants_f64(x: f64) -> f64 { return x; }\n\
         fn main() -> i32 { let a: f32 = 2.0; return wants_f64(a as f64) as i32; }",
        2,
    );
}

#[test]
fn flat_coerces_binary_op_and_comparison_operands() {
    // #238 (flat-lowerer coercion): a wider-typed local combined with default-`i32` literals in a
    // binary op (`i + 10`), a comparison (`i < 3`), and a compound assign — all must coerce the
    // literal to the operand's type. Previously the flat path emitted `arith.addi(i64, i32)` and
    // failed verification. Loop vars are `i64` in the flat path, so this is safe here.
    assert_parity(
        "fn main() -> i32 {\n\
         let mut i: i64 = 0;\n\
         if i < 3 { i = i + 10; }\n\
         i += 5;\n\
         return i as i32;\n\
         }",
        15,
    );
}

#[test]
fn flat_coerces_let_annotation_slot_and_assignment() {
    // #238: an annotated `let` sizes its slot from the annotation and coerces the initializer, so a
    // later assignment of a matching-width value fits (`let mut r: i64 = 0; r = 5`). Before, the slot
    // took the initializer's default `i32` and a wider store did not fit.
    assert_parity(
        "fn main() -> i32 { let mut r: i64 = 0; r = 5; return r as i32; }",
        5,
    );
}

#[test]
fn flat_coerces_value_if_branch_to_result_type() {
    // #238: a value-position `if` whose branches carry differently-typed literals must reconcile both
    // to the result (slot) type (`let v: i64 = if c { 1 } else { 2 }`).
    assert_parity(
        "fn main() -> i32 {\n\
         let c = 1;\n\
         let v: i64 = if c > 0 { 7 } else { 9 };\n\
         return v as i32;\n\
         }",
        7,
    );
}

#[test]
fn flat_lowers_short_circuit_logical_ops() {
    // #239: `&&` / `||` lower to the AST's short-circuit branch skeleton. Exercise both operators as
    // an `if` condition, across truth combinations, so the flat path's result matches the oracle.
    // 3>0 && 7<10 -> true (returns 1).
    assert_parity(
        "fn main() -> i32 { let a = 3; let b = 7; if a > 0 && b < 10 { return 1; } return 0; }",
        1,
    );
    // 3>0 && 7<5 -> false (falls through to 0).
    assert_parity(
        "fn main() -> i32 { let a = 3; let b = 7; if a > 0 && b < 5 { return 1; } return 0; }",
        0,
    );
    // false || true -> true (returns 1).
    assert_parity(
        "fn main() -> i32 { let a = 3; let b = 7; if a < 0 || b > 0 { return 1; } return 0; }",
        1,
    );
    // false || false -> false.
    assert_parity(
        "fn main() -> i32 { let a = 3; let b = 7; if a < 0 || b < 0 { return 1; } return 0; }",
        0,
    );
}

#[test]
fn flat_lowers_logical_op_returning_bool() {
    // #239: a straight-line function whose only "control flow" is the logical op — the memory model
    // is forced by `body_has_control_flow` detecting the `&&` in the returned expression.
    assert_parity(
        "fn in_range(x: i32) -> bool { return x > 0 && x < 100; }\n\
         fn main() -> i32 { if in_range(50) && !in_range(-1) { return 42; } return 0; }",
        42,
    );
}

#[test]
fn flat_lowers_nested_logical_ops() {
    // #239: nested `&&`/`||` chain, bound to a local then branched on — exercises multiple result
    // slots and merge blocks in one function.
    assert_parity(
        "fn main() -> i32 {\n\
         let a = 5; let b = 0; let c = 9;\n\
         let ok = a > 0 && (b > 0 || c > 0);\n\
         if ok { return 7; } return 0;\n\
         }",
        7,
    );
}

/// Flat-only exit-code assertion, for a construct the AST oracle can't JIT (so there is no parity
/// to assert). The AST codegen emits a value array literal as `tensor.from_elements` + `tensor.extract`
/// that *fails MLIR verification* at lowering — so array programs never JIT through `--legacy-codegen`
/// (they only pass the emit-mlir FileCheck tests). The flat path lowers them correctly (a memref
/// buffer + element stores + `TensorIndex` loads), so it is the reference here: we assert the flat
/// exit code directly rather than compare to a broken oracle (#239).
fn assert_flat_exit(src: &str, expected: i32) {
    assert_eq!(
        flat_exit_code(src),
        Some(expected),
        "flat path exit code for `{src}`"
    );
}

#[test]
fn flat_lowers_value_array_literal() {
    // A value array `[…]` bound to a local and indexed. The AST path fails MLIR verification on this
    // (see `assert_flat_exit`), so the flat path is validated standalone.
    // Sum in a loop: 10+20+30+40+50 = 150.
    assert_flat_exit(
        "fn main() -> i32 {\n\
         let arr = [10, 20, 30, 40, 50];\n\
         let mut total = 0;\n\
         for i in 0..5 { total += arr[i]; }\n\
         return total;\n\
         }",
        150,
    );
    // Direct constant index.
    assert_flat_exit(
        "fn main() -> i32 { let arr = [7, 8, 9]; return arr[1]; }",
        8,
    );
    // A mutable array: store through an element place, then read it back.
    assert_flat_exit(
        "fn main() -> i32 { let mut arr = [1, 2, 3]; arr[0] = 40; return arr[0] + arr[2]; }",
        43,
    );
}

/// Read a corpus program from `tests/backend/pass/`. The `RUN`/`CHECK`/`EXPECT`
/// and license lines are `//` comments the parser ignores.
fn corpus(name: &str) -> String {
    std::fs::read_to_string(format!("tests/backend/pass/{name}"))
        .unwrap_or_else(|e| panic!("read {name}: {e}"))
}

#[test]
fn flat_matches_ast_corpus_slice_reductions() {
    // A real corpus program end to end through the flat path: tensor alloc +
    // scalar-element stores + typed row bindings + dot/sum/max/min reductions + a
    // `for`-loop scalar oracle + scalar-element stores of the results + `print(o)`.
    // The printed output must match the AST oracle.
    assert_output_parity(&corpus("slice_reductions.vx"));
}

#[test]
fn flat_matches_ast_corpus_tensor_view_2d() {
    // Two views over foreign memory, read by element and multiplied. The descriptor the flat
    // path builds over the pointer has to address the same bytes the oracle's does.
    assert_output_parity(&corpus("tensor_view_2d.vx"));
    // A view with run-time extents: its `memref<?x?xf32>` reads its sizes out of the
    // descriptor, so this is the one that checks them -- the row offset of `a[1][2]` is the
    // column count, and `extent(1)` is it directly. (The descriptor's strides are read only
    // by a consumer that takes the dynamic memref whole, which nothing on the flat path does
    // yet; they are row-major, as the oracle's.)
    assert_parity(
        "extern \"C\" { fn vx_alloc_f32(n : i32) -> *mut f32; fn vx_free_f32(p : *mut f32, n : i32) -> void; }\n\
         fn view(p : *mut f32, r : i32, c : i32) -> Tensor<f32, [?, ?]> { \
           let v = unsafe { tensor_view_2d(p, r, c) }; return v; }\n\
         fn main() -> i32 { let p : *mut f32 = unsafe { vx_alloc_f32(6) }; \
           unsafe { for i in 0..6 { p[i] = (i + 1) as f32; } } \
           let a = view(p, 2, 3); let x = a[1][2]; let w = a.extent(1); \
           unsafe { vx_free_f32(p, 6); } return (x * 10.0) as i32 + w; }",
        63,
    );
}

#[test]
fn flat_matches_ast_corpus_linear_attention() {
    // An attention-corpus program (no softmax/exp): tensor allocs, `for` loops,
    // and `print`. Its printed output must match the AST oracle through the flat
    // path.
    assert_output_parity(&corpus("linear_attention.vx"));
}

/// A minimal self-contained `Vec<T>` (the shape of `stdlib/std/vec.vx`, minus googletest): a struct
/// with a raw-pointer field, a generic `impl` allocating through the Rust byte allocator, and
/// `push`/`get`/`len`. Exercises the whole `#242` surface — a pointer-field aggregate, field access
/// through a `self` pointer, raw-pointer indexing, construction, and a generic static call.
/// A self-contained `Vec` + `VecIter` + `Option` (the shape of `stdlib/std/vec.vx`) exercising the
/// whole `for x in v.iter()` surface: iterator construction, `next()` returning a data-carrying
/// `Option`, deref-through-a-pointer-field (`(*self.vec).len`), raw-pointer indexing, and the
/// for-over-iterator loop. Validated *standalone* on the flat path: the AST oracle can't monomorphize
/// this inline generic enum/iterator (it errors `generic type reached codegen`), so it can't be a
/// differential reference here — the stdlib form is validated flat-vs-oracle by the corpus sweep. (#242)
const ITER_MINI: &str = r#"
extern "C" {
  fn vx_vec_alloc(elem_size: i64, cap: i64) -> *mut i8;
  fn vx_vec_grow(ptr: *mut i8, old_cap: i64, new_cap: i64, elem_size: i64) -> *mut i8;
}
struct Vec<T> { data: *mut T, len: i32, capacity: i32 }
enum Option<T> { Some(T), None }
struct VecIter<T> { vec: *const Vec<T>, current: i32 }
impl<T> Vec<T> {
  fn with_capacity(capacity: i32) -> Vec<T> {
    let ptr: *mut T = unsafe { vx_vec_alloc(sizeof<T>(), capacity as i64) };
    return Vec<T> { data: ptr, len: 0, capacity: capacity };
  }
  fn new() -> Vec<T> { return Vec<T>::with_capacity(2); }
  fn push(self: &mut Vec<T>, val: T) -> i32 {
    if self.len == self.capacity {
      let mut nc: i32 = 4;
      if self.capacity > 0 { nc = self.capacity * 2; }
      let p: *mut i8 = self.data;
      self.data = unsafe { vx_vec_grow(p, self.capacity as i64, nc as i64, sizeof<T>()) };
      self.capacity = nc;
    }
    unsafe { self.data[self.len] = val; }
    self.len = self.len + 1;
    return 0;
  }
  fn iter(self: &Vec<T>) -> VecIter<T> { return VecIter<T> { vec: self, current: 0 }; }
}
impl<T> VecIter<T> {
  fn next(self: &mut VecIter<T>) -> Option<T> {
    let vlen: i32 = unsafe { (*self.vec).len };
    let mut ret = Option<T>::None;
    if self.current < vlen {
      let val: T = unsafe { (*self.vec).data[self.current] };
      self.current = self.current + 1;
      ret = Option<T>::Some(val);
    }
    return ret;
  }
}
"#;

#[test]
fn flat_lowers_for_over_iterator_sum() {
    // `for x in v.iter() { s = s + x; }` on the flat path: the for-over-iterator sugar over
    // `next()` + `match Some/None`. `10 + 20 + 30 = 60`. Standalone (see `ITER_MINI`). (#242)
    assert_flat_exit(
        &format!(
            "{ITER_MINI}\nfn main() -> i32 {{ let mut v = Vec<i32>::new(); v.push(10); v.push(20); \
             v.push(30); let mut s: i32 = 0; for x in v.iter() {{ s = s + x; }} return s; }}"
        ),
        60,
    );
}

#[test]
fn flat_lowers_for_over_iterator_count() {
    // A `for x in v.iter()` whose body ignores the element (`count = count + 1`) — exercises the loop
    // dispatch + `None`-terminated exit without a payload use. `3`. (#242)
    assert_flat_exit(
        &format!(
            "{ITER_MINI}\nfn main() -> i32 {{ let mut v = Vec<i32>::new(); v.push(10); v.push(20); \
             v.push(30); let mut c: i32 = 0; for x in v.iter() {{ c = c + 1; }} return c; }}"
        ),
        3,
    );
}

const VEC_MINI: &str = r#"
extern "C" {
  fn vx_vec_alloc(elem_size: i64, cap: i64) -> *mut i8;
  fn vx_vec_grow(ptr: *mut i8, old_cap: i64, new_cap: i64, elem_size: i64) -> *mut i8;
  fn vx_vec_bounds_check(index: i64, len: i64) -> i32;
}
struct Vec<T> { data: *mut T, len: i32, capacity: i32 }
impl<T> Vec<T> {
  fn with_capacity(capacity: i32) -> Vec<T> {
    let ptr: *mut T = unsafe { vx_vec_alloc(sizeof<T>(), capacity as i64) };
    return Vec<T> { data: ptr, len: 0, capacity: capacity };
  }
  fn new() -> Vec<T> { return Vec<T>::with_capacity(2); }
  fn push(self: &mut Vec<T>, val: T) -> i32 {
    if self.len == self.capacity {
      let mut new_cap: i32 = 4;
      if self.capacity > 0 { new_cap = self.capacity * 2; }
      let ptr: *mut i8 = self.data;
      self.data = unsafe { vx_vec_grow(ptr, self.capacity as i64, new_cap as i64, sizeof<T>()) };
      self.capacity = new_cap;
    }
    unsafe { self.data[self.len] = val; }
    self.len = self.len + 1;
    return 0;
  }
  fn get(self: &Vec<T>, index: i32) -> T {
    unsafe { vx_vec_bounds_check(index as i64, self.len as i64); }
    return unsafe { self.data[index] };
  }
  fn len(self: &Vec<T>) -> i32 { return self.len; }
}
"#;

#[test]
fn flat_matches_ast_vec_push_get_len() {
    // The core `Vec<T>` surface (#242) end to end through the flat path: `Vec<i32>::new()` (a
    // generic static call returning an aggregate by value), three `push`es (the third grows the
    // buffer through `vx_vec_grow`), field access through the `&mut self`/`&self` pointer, raw-pointer
    // element stores/loads (`self.data[i]`), and `&v` aggregate borrows at the rewritten method calls.
    // `10 + 30 + 3 = 43`.
    assert_parity(
        &format!(
            "{VEC_MINI}\nfn main() -> i32 {{ let mut v: Vec<i32> = Vec<i32>::new(); v.push(10); \
             v.push(20); v.push(30); return v.get(0) + v.get(2) + v.len(); }}"
        ),
        43,
    );
}

#[test]
fn flat_matches_ast_data_enum_construct_and_match() {
    // A data-carrying enum (`Option<i32>`, #242): construct `Some(30)` as a `{ i32 tag, i32 payload }`
    // aggregate, then a statement-`match` loads the tag to dispatch and binds the payload in the
    // `Some` arm. `30`. (The enum is named `Option` so the AST oracle's `Option<`-prefixed payload
    // handling matches the flat path's synthesized layout.)
    assert_parity(
        "enum Option<T> { Some(T), None }\n\
         fn main() -> i32 { let o = Option<i32>::Some(30); let mut r = 0; \
         match o { Option<i32>::Some(v) => { r = v; } Option<i32>::None => { r = -1; } } return r; }",
        30,
    );
}

#[test]
fn flat_matches_ast_data_enum_none_arm() {
    // The `None` arm of the same surface: `None` stores only the tag, and the match dispatches to the
    // `None` arm (tag 1). `7`.
    assert_parity(
        "enum Option<T> { Some(T), None }\n\
         fn main() -> i32 { let o = Option<i32>::None; let mut r = 0; \
         match o { Option<i32>::Some(v) => { r = v; } Option<i32>::None => { r = 7; } } return r; }",
        7,
    );
}

#[test]
fn flat_matches_ast_pointer_dereference() {
    // A raw-pointer dereference store `*p = 42` then read `let v = *p` (`Box`'s heap cell, #242),
    // each lowered as `p[0]` through `PtrIndex`/`PtrStore`. Now a differential target: the oracle's
    // deref was fixed (the store was silently dropped in `AssignStmt`; the read defaulted to `f32`).
    assert_parity(
        "extern \"C\" { fn malloc(size: i64) -> *mut i8; }\n\
         fn main() -> i32 { let p: *mut i32 = unsafe { malloc(4) }; unsafe { *p = 42; } \
         let v: i32 = unsafe { *p }; return v; }",
        42,
    );
}

#[test]
fn flat_matches_ast_vec_of_vec() {
    // A nested container `Vec<Vec<i32>>` (#242): the element type is itself an aggregate, so `push`
    // passes a `Vec<i32>` by value (a `!llvm.struct` param), `self.data[i] = val` stores the whole
    // struct through the raw pointer, `get` loads it back by value, and `sizeof<Vec<i32>>()` sizes the
    // outer buffer. `22 + 55 + 3 = 80`.
    //
    // Now a normal `assert_parity` target: the oracle's `sizeof<struct>` was fixed to the real layout
    // size (was a hardcoded `8`, which under-allocated the outer buffer and made the element stores UB
    // — see journal Entry 66/67), so both paths size the buffer to 16 and agree deterministically.
    assert_parity(
        &format!(
            "{VEC_MINI}\nfn main() -> i32 {{ let mut outer = Vec<Vec<i32>>::new(); \
             let mut a = Vec<i32>::new(); a.push(11); a.push(22); outer.push(a); \
             let mut b = Vec<i32>::new(); b.push(33); b.push(44); b.push(55); outer.push(b); \
             let r0 = outer.get(0); let r1 = outer.get(1); \
             return r0.get(1) + r1.get(2) + r1.len(); }}"
        ),
        80,
    );
}

#[test]
fn flat_matches_ast_vec_loop_push_sum() {
    // A `for`-loop that pushes 0..10 (repeatedly growing the buffer past the initial capacity of 2)
    // then a second loop summing every element back via `get` — stressing the grow path and the
    // raw-pointer read/write across many indices. `0+1+...+9 = 45`.
    assert_parity(
        &format!(
            "{VEC_MINI}\nfn main() -> i32 {{ let mut v: Vec<i32> = Vec<i32>::new(); \
             for i in 0..10 {{ v.push(i); }} let mut s = 0; \
             for i in 0..v.len() {{ s = s + v.get(i); }} return s; }}"
        ),
        45,
    );
}

#[test]
fn flat_matches_ast_function_pointer() {
    // A bare function name used as a value (`FuncConst`) and called through the pointer parameter
    // (`CallIndirect`): `apply(square, 6) + apply(add3, 10) = 36 + 13 = 49`. Two distinct functions are
    // dispatched through the same `f : fn(i32)->i32` parameter, so the indirect call really selects at
    // runtime rather than being a devirtualized direct call. (#242)
    assert_parity(
        "fn square(x: i32) -> i32 { return x * x; } \
         fn add3(x: i32) -> i32 { return x + 3; } \
         fn apply(f: fn(i32)->i32, v: i32) -> i32 { return f(v); } \
         fn main() -> i32 { return apply(square, 6) + apply(add3, 10); }",
        49,
    );
}

#[test]
fn flat_matches_ast_nested_aggregate_field_method() {
    // A by-value nested-aggregate struct field (`Outer { inner: Inner, .. }`): constructing it stores
    // the whole `Inner` value, `self.inner.sum()` takes the field's address (`FieldAddr`) as the method
    // receiver, and `self.inner.a` GEPs through it. `(3 + 4) + 7 = 14`. (#242)
    assert_parity(
        "struct Inner { a: i32, b: i32 } \
         struct Outer { inner: Inner, x: i32 } \
         impl Inner { fn sum(self: &Inner) -> i32 { return self.a + self.b; } } \
         impl Outer { fn total(self: &Outer) -> i32 { return self.inner.sum() + self.x; } } \
         fn main() -> i32 { let i = Inner { a: 3, b: 4 }; \
             let o = Outer { inner: i, x: 7 }; return o.total(); }",
        14,
    );
}

#[test]
fn flat_matches_ast_function_pointer_struct_field() {
    // A function-pointer struct field (`Holder { f: fn(i32)->i32, .. }`): a bare function stored into
    // the field (a `FuncConst` pointer), loaded back, and called through (`CallIndirect`). `sq(6) = 36`.
    // This is the shape of an iterator adapter carrying its mapping function. (#242)
    assert_parity(
        "struct Holder { f: fn(i32)->i32, x: i32 } \
         fn sq(n: i32) -> i32 { return n * n; } \
         fn call_it(h: Holder, v: i32) -> i32 { let fp = h.f; return fp(v); } \
         fn main() -> i32 { let h = Holder { f: sq, x: 5 }; return call_it(h, 6); }",
        36,
    );
}

#[test]
fn flat_matches_ast_matmul_non_square() {
    // `[2, 3] @ [3, 4]` is `[2, 4]`. The flattener gave the result the LEFT operand's shape
    // until Vx#390 -- which only shows on rectangular operands, since a square pair makes the
    // two coincide. Nothing compared the paths on a matmul at all before this: the flat
    // emitter declined, so it was never a differential target, so the wrong result type behind
    // the decline had nothing looking at it.
    //
    // `c[1][3]` is 4*4 + 5*8 + 6*12 = 128, the same value gpu_matmul_roles.vx expects.
    assert_parity(
        "fn main() -> i32 { let mut a = Tensor<f32>([2, 3]); \
         let mut b = Tensor<f32>([3, 4]); \
         for i in 0..2 { for j in 0..3 { a[i][j] = (i * 3 + j + 1) as f32; } } \
         for i in 0..3 { for j in 0..4 { b[i][j] = (i * 4 + j + 1) as f32; } } \
         let c = a @ b; return c[1][3] as i32; }",
        128,
    );
}

#[test]
fn flat_matches_ast_matmul_half_stores_half() {
    // A half matmul accumulates and STORES half. The flattener widened the result to f32 the
    // way it does for elementwise arithmetic until Vx#390, which would have given the product
    // an f32 buffer.
    //
    // f16 represents integers exactly only up to 2048, so 1024 + 1025 is the smallest sum that
    // tells the two storages apart: 2048 in half, 2049 in f32. Both operands are exact in half,
    // so the difference is the accumulator's own type and nothing else. Offset by 2038 to keep
    // both outcomes inside the 8-bit exit code and away from zero -- 10 for half, 11 for f32.
    assert_parity(
        "fn main() -> i32 { let mut a = Tensor<f16>([2, 2]); \
         let mut b = Tensor<f16>([2, 2]); \
         a[0][0] = 1024.0; a[0][1] = 1025.0; a[1][0] = 0.0; a[1][1] = 0.0; \
         b[0][0] = 1.0; b[0][1] = 0.0; b[1][0] = 1.0; b[1][1] = 0.0; \
         let c = a @ b; return (c[0][0] - 2038.0) as i32; }",
        10,
    );
}

#[test]
fn flat_carries_subspace_scheduling_metadata() {
    // P0-1: the sub-space scheduler's output (`space`/`within`/`granule`/`capacity`/`scope` + the
    // bump-allocated `offset`/`slots`) must survive on the *flat* path, not just under
    // `--legacy-codegen`. SMEM's granule is 16 KB; a 128x128 f32 tile is exactly 4 granules (65536 B),
    // so the two tiles land at offset 0 and offset 65536, 4 slots each — the same values the AST path
    // assigns (verified byte-identical against `--legacy-codegen`). Mirrors `subspace_schedule.vx`.
    let src = "\
        Memory GPU_HBM { capacity: 40 GiB, bandwidth: 3 TB/s } \
        Memory SMEM { within: Memory::GPU_HBM, capacity: 228 KiB, granule: 16 KiB, scope: sm } \
        fn main() -> i32 { \
            let a = Tensor<f32>([128, 128]); \
            let b = Tensor<f32>([128, 128]); \
            let sa = transfer(a, Memory::SMEM); \
            let sb = transfer(b, Memory::SMEM); \
            return 0; \
        }";
    let mlir = flat_module_mlir(src).expect("subspace program lowers on the flat path");
    // Descriptor attributes are present on the flat path.
    for needle in [
        "space = \"SMEM\"",
        "within = \"GPU_HBM\"",
        "granule = 16384 : i64",
        "capacity = 233472 : i64",
        "scope = \"sm\"",
    ] {
        assert!(
            mlir.contains(needle),
            "flat vx.transfer missing `{needle}`\n{mlir}"
        );
    }
    // The bump allocator assigned distinct, granule-rounded offsets (0 then 65536), 4 slots each.
    assert!(
        mlir.contains("offset = 0 : i64"),
        "missing offset 0\n{mlir}"
    );
    assert!(
        mlir.contains("offset = 65536 : i64"),
        "missing bumped offset 65536\n{mlir}"
    );
    assert_eq!(
        mlir.matches("slots = 4 : i64").count(),
        2,
        "expected 4 slots on each of the two transfers\n{mlir}"
    );
}

/// #273: `&scalar_local` passed to a *reference-returning* call, whose reference result binds a local that
/// is then dereferenced — the return-provenance example (`pick` returns `b`, so `*r` is `y`). The flat path
/// already lowers this (the #275/#278 reference substrate covers `&x` call-args, a reference return, and
/// `*r`); the gap was the AST *oracle*, which coerced a mutable scalar's `memref<i32>` descriptor to the
/// `&i32` parameter's `!llvm.ptr` via a bitcast that failed LLVM translation. `coerce_type` now extracts the
/// memref's aligned pointer (as #278 does), so both paths run. `x` is mutable and mutated after the call to
/// exercise the memref path; `pick` returns `b`, so the later `x = 99` does not change the result (20).
#[test]
fn flat_runs_a_reference_returning_call_over_scalar_locals() {
    assert_parity(
        "fn pick(a : &i32, b : &i32) -> &i32 { return b; }\n\
         fn main() -> i32 { let mut x = 10; let y = 20; let r = pick(&x, &y); x = 99; return *r; }",
        20,
    );
}

/// #273 companion: the call returns the *first* reference argument (`return a`), so `*r` observes the
/// mutation through the same storage — `let mut x = 10; …; x = 99; return *r` yields 99. Proves the
/// extracted aligned pointer aliases `x`'s real memref cell across the call boundary, on both paths.
#[test]
fn flat_reference_returning_call_aliases_the_mutated_local() {
    assert_parity(
        "fn pick(a : &i32, b : &i32) -> &i32 { return a; }\n\
         fn main() -> i32 { let mut x = 10; let y = 20; let r = pick(&x, &y); x = 99; return *r; }",
        99,
    );
}

/// The parallel pipeline's own MLIR, JIT-ed and checked against the AST oracle (#311).
///
/// `flat_llvm` above drives the flat emitter through a hand-assembled single-program lowering.
/// This drives the *pipeline*: real files on disk, parsed in parallel, macro-expanded, name-
/// resolved, frozen into a registry, type-checked and lowered per function on a rayon pool,
/// reconciled at the dedup barrier, SIMD-patched, and only then emitted. That is the orchestration
/// whose scaling is being measured, and until it produced an artifact there was nothing to check it
/// against. Parity with the AST oracle is what makes a speedup measured on it a statement about
/// compiling a program rather than about running a frontend.
///
/// Two modules, so the phases that only exist because compilation is per-module — the parallel
/// parse, the routing of results back to their module, the cross-worker reconciliation — are all on
/// the path. `main` calls into both.
#[test]
fn pipeline_emits_mlir_that_matches_the_ast_oracle() {
    use std::io::Write;

    let a = "fn add(x: i32, y: i32) -> i32 { return x + y; }\n\
             fn tri(n: i32) -> i32 { let mut s = 0; for i in 0..n { s = s + i; } return s; }\n";
    let b = "fn twice(n: i32) -> i32 { return n * 2; }\n\
             fn main() -> i32 { return add(twice(tri(5)), 4); }\n";

    let dir = std::env::temp_dir().join(format!("vx_pipe_mlir_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut paths = Vec::new();
    for (name, src) in [("a.vx", a), ("b.vx", b)] {
        let p = dir.join(name);
        std::fs::File::create(&p)
            .unwrap()
            .write_all(src.as_bytes())
            .unwrap();
        paths.push(p.to_string_lossy().to_string());
    }

    let text = vxc::pipeline::compile_pipeline_mlir(&paths)
        .expect("pipeline")
        .expect("the flat emitter covers this corpus");
    let _ = std::fs::remove_dir_all(&dir);

    let context = make_context();
    let mut module = melior::ir::Module::parse(&context, &text)
        .unwrap_or_else(|| panic!("pipeline MLIR does not parse:\n{text}"));
    lower_to_llvm(&context, &mut module).expect("pipeline lower_to_llvm");

    // tri(5) = 0+1+2+3+4 = 10; twice -> 20; add(20, 4) = 24.
    assert_eq!(exit_code(&module.as_operation().to_string()), 24);
    assert_eq!(ast_exit_code(&format!("{a}{b}")), 24, "oracle agrees");
}

/// Vx#395: `closure as ||->T` panicked the checker's GID recheck (`env.functions` stores a
/// `(return, .., params)` tuple, not a `Type::Function`), and the oracle then built the fat
/// pointer `{fn, env}` while every call site extracts `{env, fn}` — a jump into the environment.
/// The corpus program asserts through both call shapes (a local fat pointer and a returned one),
/// so it is executed here, not just emitted.
#[test]
fn closure_fat_ptr_program_runs_through_the_oracle() {
    let vxc = env!("CARGO_BIN_EXE_vxc");
    let run = std::process::Command::new(vxc)
        .args([
            "tests/frontend/pass/closure_fat_ptr.vx",
            "--action",
            "run-jit",
        ])
        .output()
        .expect("run vxc");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(!out.contains("panicked"), "internal panic:\n{out}");
    assert!(
        !out.contains("non-zero code"),
        "the program's own asserts failed:\n{out}"
    );
    assert!(run.status.success(), "vxc failed:\n{out}");
}

#[test]
fn flat_matches_ast_tensor_returning_callee() {
    // A callee returning a statically shaped tensor had no spelling on the flat path (the
    // whole family behind Vx#383's return-type buckets): the signature, the call result, and
    // the `Ret` all needed the memref form, and `return a + b` returns a vector register that
    // must be spilled into the buffer the signature promises.
    //
    // add(a, b)[2] = 3.0 + 30.0 = 33.
    assert_parity(
        "fn add(a : Tensor<f32, [4]>, b : Tensor<f32, [4]>) -> Tensor<f32, [4]> { \
           return a + b; \
         } \
         fn main() -> i32 { \
           let mut a = Tensor<f32>([4]); \
           let mut b = Tensor<f32>([4]); \
           for i in 0..4 { a[i] = ((i + 1) * 1) as f32; b[i] = ((i + 1) * 10) as f32; } \
           let c = add(a, b); \
           return c[2] as i32; }",
        33,
    );
}

/// Vx#398 family 1: `v[0]` on a `Vec` reached the AST index lowering as a struct base and
/// emitted `memref.load` on it -- the whole macro_vec set failed to compile on the default
/// path. The checker now rewrites container index reads to the container's own `get`, one
/// construct for both backends; the program's own asserts check the values.
#[test]
fn vec_index_sugar_program_runs() {
    let vxc = env!("CARGO_BIN_EXE_vxc");
    let run = std::process::Command::new(vxc)
        .args([
            "tests/frontend/pass/macro_vec_single.vx",
            "--action",
            "run-jit",
        ])
        .output()
        .expect("run vxc");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(!out.contains("panicked"), "internal panic:\n{out}");
    assert!(
        !out.contains("non-zero code"),
        "the program's own asserts failed:\n{out}"
    );
    assert!(run.status.success(), "vxc failed:\n{out}");
}

/// `t.extent(k)` is the runtime extent of dimension k. Nothing lowered the construct on
/// either backend (the AST path panicked "Cannot resolve member access shape", Vx#398);
/// both now emit `memref.dim` as one construct.
#[test]
fn flat_matches_ast_shape_query() {
    assert_parity(
        "fn main() -> i32 { let mut t = Tensor<f32>([3, 4]); \
         t[0][0] = 1.0; \
         return t.extent(1); }",
        4,
    );
}

/// Vx#401: a generic instantiated at two tensor shapes got one monomorph, and the second
/// call was emitted against the first one's signature -- a `memref<4x5xf32>` passed to a
/// function declaring `memref<2x3xf32>`, which the debug assertion caught as invalid MLIR.
/// Extents are part of a tensor's identity, so the mangled name carries them. `y[3][4]`
/// reads a position only the [4, 5] shape has, and 1.0 + 7.0 is the exit code.
#[test]
fn a_generic_over_two_tensor_shapes_gets_two_monomorphs() {
    let dir = std::env::temp_dir().join("vx_mono_shapes");
    let _ = std::fs::create_dir_all(&dir);
    let src = dir.join("mono_shapes.vx");
    std::fs::write(
        &src,
        "fn ident<T>(v : T) -> T {\n  return v;\n}\n\
         fn main() -> i32 {\n\
         \x20 let mut a : Tensor<f32, [2, 3]> = Tensor<f32>([2, 3]);\n\
         \x20 let mut b : Tensor<f32, [4, 5]> = Tensor<f32>([4, 5]);\n\
         \x20 for i in 0..2 { for j in 0..3 { a[i][j] = 1.0; } }\n\
         \x20 for i in 0..4 { for j in 0..5 { b[i][j] = 7.0; } }\n\
         \x20 let x = ident(a);\n\
         \x20 let y = ident(b);\n\
         \x20 return (x[1][2] + y[3][4]) as i32;\n}\n",
    )
    .unwrap();

    let run = std::process::Command::new(env!("CARGO_BIN_EXE_vxc"))
        .args([src.to_str().unwrap(), "--action", "run-jit"])
        .output()
        .expect("run vxc");
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!out.contains("panicked"), "internal panic:\n{out}");
    assert!(
        out.contains("exited with code: 8"),
        "expected 1.0 + 7.0 through two distinct monomorphs, got:\n{out}"
    );
}
