//===- driver.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Compiler driver orchestrating file I/O, parsing, semantic analysis, and code generation.
//
//===----------------------------------------------------------------------===//

use crate::diagnostic::DiagnosticLevel;
use crate::syntax::MacroExpander;
use crate::syntax_printer::AstPrinter;
use clap::{Parser, ValueEnum};
use codegen::MeliorGenerator;
use melior::ir::operation::OperationLike;
use std::path::PathBuf;

use crate::hir::{GlobalAstEnv, TypeChecker};
use crate::module_loader::ModuleLoader;
use crate::session::{GlobalSession, LocalWorkerState};

use crate::codegen;
#[derive(Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum Action {
    /// Only run the lexer and parser
    ParseOnly,
    /// Parse and typecheck the program, then print the AST
    PrintAst,
    /// Parse, typecheck, and emit the MLIR representation
    EmitMlir,
    /// Parse, typecheck, emit MLIR, and execute via the JIT Engine
    RunJit,
    /// Compile to an object file
    EmitObj,
    /// Parse, typecheck, emit MLIR, lower to LLVM dialect, and translate to LLVM IR
    EmitLlvm,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct DriverOptions {
    /// Action to perform
    #[arg(short = 'a', long = "action", value_enum, default_value_t = Action::RunJit)]
    #[arg(overrides_with_all = ["compile", "parse_only", "print_ast", "emit_mlir", "emit_llvm", "run_jit"])]
    pub action: Action,

    /// Output file
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    /// Compile to object file (alias for --action emit-obj)
    #[arg(short = 'c', overrides_with = "action")]
    pub compile: bool,

    /// Parse only (alias for --action parse-only)
    #[arg(short = 'p', long = "parse-only", overrides_with = "action")]
    pub parse_only: bool,

    /// Print AST (alias for --action print-ast)
    #[arg(long = "print-ast", overrides_with = "action")]
    pub print_ast: bool,

    /// Emit MLIR (alias for --action emit-mlir)
    #[arg(long = "emit-mlir", overrides_with = "action")]
    pub emit_mlir: bool,

    /// Emit LLVM IR (alias for --action emit-llvm)
    #[arg(long = "emit-llvm", overrides_with = "action")]
    pub emit_llvm: bool,

    /// Run JIT (alias for --action run-jit)
    #[arg(long = "run", overrides_with = "action")]
    pub run_jit: bool,

    /// Emit MLIR/LLVM backend diagnostics
    #[arg(long = "emit-backend-diagnostics")]
    pub emit_backend_diagnostics: bool,

    /// Discharge per-seam boundary obligations at cross-device transfers (assert
    /// pre-scan + z3 checks). Off by default; requires z3 on PATH (fails open if absent).
    #[arg(long = "verify-seams")]
    pub verify_seams: bool,

    /// Disable Vx optimizations
    #[arg(long = "disable-vx-optimizations")]
    pub disable_vx_optimizations: bool,

    /// Disable MLIR optimizations
    #[arg(long = "disable-mlir-optimizations")]
    pub disable_mlir_optimizations: bool,

    /// Disable LLVM optimizations
    #[arg(long = "disable-llvm-optimizations")]
    pub disable_llvm_optimizations: bool,

    /// Optimization level (-O0 to -O3)
    #[arg(short = 'O', num_args = 0..=1, default_missing_value = "3", default_value_t = 0)]
    pub opt_level: u8,

    /// Input source files
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,

    /// Specify the language of the input file (e.g. mlir, vx)
    #[arg(short = 'x', long = "language")]
    pub language: Option<String>,

    /// Pass an argument to a specific backend tool (e.g., -X mlir=--pass-pipeline=...)
    #[arg(short = 'X')]
    pub tool_args: Vec<String>,

    /// Arguments to pass to the running program
    #[arg(last = true)]
    pub program_args: Vec<String>,
}

pub struct CompilerDriver {
    pub options: DriverOptions,
}

impl CompilerDriver {
    pub fn new(mut options: DriverOptions) -> Self {
        // Handle alias flags
        if options.parse_only {
            options.action = Action::ParseOnly;
        } else if options.print_ast {
            options.action = Action::PrintAst;
        } else if options.emit_mlir {
            options.action = Action::EmitMlir;
        } else if options.emit_llvm {
            options.action = Action::EmitLlvm;
        } else if options.run_jit {
            options.action = Action::RunJit;
        } else if options.compile {
            options.action = Action::EmitObj;
        }

        Self { options }
    }

    pub fn execute(&self) -> Result<(), String> {
        if self.options.inputs.is_empty() {
            return Err("No input files provided".to_string());
        }

        let main_file = &self.options.inputs[0];
        let filename = main_file.to_string_lossy().to_string();

        let language = self
            .options
            .language
            .clone()
            .unwrap_or_else(|| "vx".to_string());

        let mut mlir_args: Vec<String> = Vec::new();
        for arg in &self.options.tool_args {
            if let Some(mlir_arg) = arg.strip_prefix("mlir=") {
                mlir_args.push(mlir_arg.to_string());
            }
        }

        if language == "mlir" {
            return self.execute_mlir_pipeline(main_file, &mlir_args);
        }

        self.execute_vx_pipeline(main_file, &filename, &mlir_args)
    }

    fn execute_mlir_pipeline(
        &self,
        main_file: &std::path::Path,
        mlir_args: &[String],
    ) -> Result<(), String> {
        let mlir_src = std::fs::read_to_string(main_file).map_err(|e| e.to_string())?;

        if self.options.action == Action::EmitMlir || self.options.action == Action::EmitLlvm {
            let optimized_mlir = apply_mlir_opt(&mlir_src, mlir_args, main_file)?;
            if self.options.action == Action::EmitLlvm {
                let llvm_ir = translate_to_llvm_ir(&optimized_mlir, main_file)?;
                println!("{}", llvm_ir);
            } else {
                println!("{}", optimized_mlir);
            }
            return Ok(());
        }

        if self.options.action == Action::RunJit {
            let mut args = vec![self.options.inputs[0].to_string_lossy().into_owned()];
            args.extend(self.options.program_args.clone());
            let out = crate::jit::execute_mlir(
                &mlir_src,
                args,
                self.options.opt_level,
                self.options.disable_llvm_optimizations,
            )
            .map_err(|e| e.to_string())?;
            println!("{}", out);
            return Ok(());
        }

        Err(format!(
            "Action {:?} is not supported for MLIR inputs",
            self.options.action
        ))
    }

    fn execute_vx_pipeline(
        &self,
        main_file: &std::path::Path,
        filename: &str,
        mlir_args: &[String],
    ) -> Result<(), String> {
        let mut program_arr = self.load_and_expand(filename)?;

        if self.options.action == Action::ParseOnly {
            return self.handle_parse_only(&program_arr, filename);
        }

        let (mut main_ast, mut other_asts) =
            self.prepare_semantic_analysis(&mut program_arr, filename)?;
        self.run_semantic_analysis(&mut main_ast, &mut other_asts, filename)?;

        if self.options.action == Action::PrintAst {
            AstPrinter::print_program(&main_ast, &mut std::io::stdout()).unwrap();
            return Ok(());
        }

        self.run_codegen(main_ast, other_asts, filename, main_file, mlir_args)
    }

    fn load_and_expand(&self, filename: &str) -> Result<Vec<crate::syntax::Program>, String> {
        let mut loader = ModuleLoader::new();
        if let Err(e) = loader.load_main(filename) {
            return Err(format!("Frontend failed to parse '{}': {}", filename, e));
        }
        let mut program_arr = loader.into_programs();

        let mut global_macros = std::collections::HashMap::new();
        for m in &program_arr {
            for mac in &m.macros {
                global_macros.insert(mac.name.clone(), mac.rules.clone());
            }
        }
        let mut expander = MacroExpander::new(&global_macros);
        for m in &mut program_arr {
            if let Err(e) = expander.expand_module(m) {
                return Err(format!("Macro expansion failed: {}", e));
            }
        }
        Ok(program_arr)
    }

    fn handle_parse_only(
        &self,
        program_arr: &[crate::syntax::Program],
        filename: &str,
    ) -> Result<(), String> {
        let ast = program_arr
            .iter()
            .find(|p| {
                p.module_path == <std::string::String as Clone>::clone(&filename.to_string()).into()
            })
            .unwrap();
        println!("{:#?}", ast);
        Ok(())
    }

    fn prepare_semantic_analysis(
        &self,
        program_arr: &mut Vec<crate::syntax::Program>,
        filename: &str,
    ) -> Result<
        (
            crate::syntax::Program,
            std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Program>,
        ),
        String,
    > {
        let syntax_idx = program_arr
            .iter()
            .position(|p| {
                p.module_path == <std::string::String as Clone>::clone(&filename.to_string()).into()
            })
            .unwrap();
        let ast = program_arr.remove(syntax_idx);

        let mut module_syntaxes = std::collections::HashMap::new();
        for mut p in program_arr.drain(..) {
            p.functions.retain(|f| f.generics.is_empty());
            module_syntaxes.insert(p.module_path.clone(), p);
        }
        Ok((ast, module_syntaxes))
    }

    fn run_semantic_analysis(
        &self,
        ast: &mut crate::syntax::Program,
        other_asts: &mut std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Program>,
        filename: &str,
    ) -> Result<(), String> {
        let global_session = std::sync::Arc::new(GlobalSession::new(1));

        let cloned_ast_sig = ast.clone_signature();
        let mut env_modules: Vec<&crate::syntax::Program> = other_asts.values().collect();
        env_modules.push(&cloned_ast_sig);
        let env = GlobalAstEnv::build_from_refs(&env_modules);

        let mut worker = LocalWorkerState::new(global_session.clone());
        let mut checker = TypeChecker::new(&env, &mut worker);
        checker.verify_seams = self.options.verify_seams;

        for f in &mut ast.functions {
            checker.check_function(f);
        }
        for i in &mut ast.impls {
            for f in &mut i.methods {
                checker.check_function(f);
            }
        }

        let has_errors = checker
            .errors
            .iter()
            .any(|d| d.level == DiagnosticLevel::Error);

        // Print warnings first (so they appear before errors)
        for diag in checker.errors.iter() {
            if diag.level == DiagnosticLevel::Warning {
                eprintln!("{}", diag);
            }
        }

        // Eval metric M1: per-seam proof cost discharged during this compile. The
        // one-time solver startup is reported separately from the marginal per-seam
        // solving time (the persistent solver is spawned once and reused).
        if checker.seam_checks > 0 {
            eprintln!(
                "[seam] {} obligation(s): solver init {:.3} ms (once) + {:.3} ms solving \
                 total = {:.4} ms/seam marginal",
                checker.seam_checks,
                checker.solver_init_time.as_secs_f64() * 1e3,
                checker.seam_check_time.as_secs_f64() * 1e3,
                checker.seam_check_time.as_secs_f64() * 1e3 / checker.seam_checks as f64,
            );
        }

        if has_errors {
            let error_count = checker.errors.error_count();
            for diag in checker.errors.iter() {
                if diag.level == DiagnosticLevel::Error {
                    eprintln!("{}", diag);
                }
            }
            return Err(format!(
                "Semantic check failed on '{}': {} error(s) emitted",
                filename, error_count
            ));
        }

        let mut orig_functions = ast.functions.clone();
        orig_functions.retain(|f| f.generics.is_empty());

        let mut new_functions: Vec<_> = checker
            .monomorphized_functions
            .into_iter()
            .map(|(f, _)| f)
            .collect();
        new_functions.extend(orig_functions);
        ast.functions = new_functions;
        ast.structs.extend(checker.generated_structs);

        Ok(())
    }

    fn run_codegen(
        &self,
        monomorphized_ast: crate::syntax::Program,
        module_syntaxes: std::collections::HashMap<crate::symbol::Symbol, crate::syntax::Program>,
        filename: &str,
        main_file: &std::path::Path,
        mlir_args: &[String],
    ) -> Result<(), String> {
        codegen::register_vx_passes();

        let registry = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&registry);
        melior::utility::register_all_passes();
        let context = melior::Context::new();
        let emit_diagnostics = self.options.emit_backend_diagnostics;
        context.attach_diagnostic_handler(move |diagnostic| {
            if emit_diagnostics {
                eprintln!("{}", diagnostic);
            }
            true
        });
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        codegen::register_vx_dialect(&context);

        let mut codegen = MeliorGenerator::new(&context, monomorphized_ast.module_path.to_string());
        codegen
            .generate(&monomorphized_ast, &module_syntaxes)
            .map_err(|e| format!("Codegen Error: {:?}", e))?;
        let mut module = codegen.into_module();

        if !module.as_operation().verify() {
            return Err(format!("MLIR verification failed for {}", filename));
        }

        let llvm_lower = matches!(
            self.options.action,
            Action::EmitLlvm | Action::RunJit | Action::EmitObj
        );
        let pipeline_str = get_optimization_pipeline(
            self.options.opt_level,
            llvm_lower,
            self.options.disable_mlir_optimizations,
        );

        let pass_manager = melior::pass::PassManager::new(&context);
        pass_manager.enable_verifier(true);
        if let Err(e) = melior::utility::parse_pass_pipeline(
            pass_manager.as_operation_pass_manager(),
            &pipeline_str,
        ) {
            return Err(format!("Failed to parse MLIR pass pipeline: {}", e));
        }

        // Apply custom CLI MLIR args if provided
        for arg in mlir_args {
            if let Some(custom_pipeline) = arg.strip_prefix("--pass-pipeline=") {
                if let Err(e) = melior::utility::parse_pass_pipeline(
                    pass_manager.as_operation_pass_manager(),
                    custom_pipeline,
                ) {
                    return Err(format!("Failed to parse custom pass pipeline: {}", e));
                }
            }
        }

        if let Err(e) = pass_manager.run(&mut module) {
            return Err(format!("MLIR passes failed for {}: {}", filename, e));
        }

        match self.options.action {
            Action::EmitMlir | Action::EmitLlvm => {
                let mlir_str = format!("{}", module.as_operation());
                println!("{}", mlir_str);
            }
            Action::RunJit => {
                let mlir_str = format!("{}", module.as_operation());
                let mut args = vec![self.options.inputs[0].to_string_lossy().into_owned()];
                args.extend(self.options.program_args.clone());
                let out = crate::jit::execute_mlir(
                    &mlir_str,
                    args,
                    self.options.opt_level,
                    self.options.disable_llvm_optimizations,
                )
                .map_err(|e| e.to_string())?;
                println!("{}", out);
            }
            Action::EmitObj => {
                let current_dir = std::env::current_dir().unwrap();
                let vx_std_core = format!(
                    "{}/target/debug/{}vx_std_core{}",
                    current_dir.display(),
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                );
                let libnpu = format!(
                    "{}/target/jit/{}npu_shared{}",
                    current_dir.display(),
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                );

                let mlir_c_runner =
                    format!("libmlir_c_runner_utils{}", std::env::consts::DLL_SUFFIX);
                let mlir_runner = format!("libmlir_runner_utils{}", std::env::consts::DLL_SUFFIX);
                let mut shared_libs = vec![
                    mlir_c_runner.clone(),
                    mlir_runner.clone(),
                    vx_std_core.clone(),
                ];
                if cfg!(target_os = "macos") {
                    shared_libs.push(libnpu.clone());
                }

                let shared_libs_refs: Vec<&str> = shared_libs.iter().map(|s| s.as_ref()).collect();

                let engine = melior::ExecutionEngine::new(
                    &module,
                    self.options.opt_level as usize,
                    &shared_libs_refs,
                    true,
                    true,
                );

                let output_path = self.options.output.clone().unwrap_or_else(|| {
                    let mut p = main_file.to_path_buf();
                    p.set_extension("o");
                    p
                });

                engine.dump_to_object_file(output_path.to_str().unwrap());
            }
            _ => {}
        }

        Ok(())
    }
}

fn get_optimization_pipeline(
    opt_level: u8,
    llvm_lower: bool,
    disable_mlir_optimizations: bool,
) -> String {
    let mut passes = vec![];

    if opt_level > 0 || llvm_lower {
        passes.push("convert-vx-to-standard".to_string());
    }

    if opt_level > 0 && !disable_mlir_optimizations {
        passes.push("canonicalize".to_string());
        passes.push("cse".to_string());
        passes.push(
            "func.func(affine-loop-fusion,affine-loop-tile,affine-loop-unroll,affine-scalrep)"
                .to_string(),
        );
        passes.push("lower-affine".to_string());
        passes.push("canonicalize".to_string());
        passes.push("cse".to_string());
    }

    if llvm_lower {
        passes.push("vx-to-llvm".to_string());
        passes.push("func.func(convert-linalg-to-loops,lower-affine)".to_string());
        passes.push("convert-scf-to-cf".to_string());
        passes.push("expand-strided-metadata".to_string());
        passes.push("finalize-memref-to-llvm".to_string());
        passes.push("convert-vector-to-llvm".to_string());
        passes.push("convert-func-to-llvm".to_string());
        passes.push("convert-index-to-llvm".to_string());
        passes.push("convert-math-to-llvm".to_string());
        passes.push("convert-math-to-libm".to_string());
        passes.push("convert-cf-to-llvm".to_string());
        passes.push("convert-arith-to-llvm".to_string());
        passes.push("reconcile-unrealized-casts".to_string());
    }

    format!("builtin.module({})", passes.join(","))
}

extern "C" {
    fn run_vx_opt(
        argc: std::os::raw::c_int,
        argv: *const *const std::os::raw::c_char,
    ) -> std::os::raw::c_int;
}

pub fn apply_mlir_opt(
    mlir_src: &str,
    mlir_args: &[String],
    _main_file: &std::path::Path,
) -> Result<String, String> {
    if mlir_args.is_empty() {
        return Ok(mlir_src.to_string());
    }

    let mut temp_in = tempfile::Builder::new()
        .prefix("vx_opt_in_")
        .suffix(".mlir")
        .tempfile()
        .map_err(|e| e.to_string())?;

    let temp_out = tempfile::Builder::new()
        .prefix("vx_opt_out_")
        .suffix(".mlir")
        .tempfile()
        .map_err(|e| e.to_string())?;

    std::io::Write::write_all(temp_in.as_file_mut(), mlir_src.as_bytes())
        .map_err(|e| e.to_string())?;

    let mut args = vec![
        "vx-opt".to_string(),
        temp_in.path().to_string_lossy().into_owned(),
        "-o".to_string(),
        temp_out.path().to_string_lossy().into_owned(),
    ];
    args.extend_from_slice(mlir_args);

    let c_args: Vec<std::ffi::CString> = args
        .into_iter()
        .map(|a| std::ffi::CString::new(a).unwrap())
        .collect();
    let c_ptrs: Vec<*const std::os::raw::c_char> = c_args.iter().map(|a| a.as_ptr()).collect();

    let status = unsafe { run_vx_opt(c_ptrs.len() as std::os::raw::c_int, c_ptrs.as_ptr()) };

    if status != 0 {
        return Err("vx-opt failed. Check stderr for details.".to_string());
    }

    let out_str = std::fs::read_to_string(temp_out.path()).unwrap_or_default();
    Ok(out_str)
}

pub fn translate_to_llvm_ir(
    mlir_src: &str,
    _main_file: &std::path::Path,
) -> Result<String, String> {
    use std::io::Write;
    let mut child = std::process::Command::new("mlir-translate")
        .arg("--mlir-to-llvmir")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn mlir-translate: {}", e))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(mlir_src.as_bytes())
            .map_err(|e| e.to_string())?;
    }

    let output = child.wait_with_output().map_err(|e| e.to_string())?;

    if !output.status.success() {
        return Err(format!(
            "mlir-translate failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}
