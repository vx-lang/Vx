use clap::{Parser, ValueEnum};
use melior::ir::operation::OperationLike;
use std::path::PathBuf;

use crate::module_loader::ModuleLoader;
use crate::sema::{GlobalAstEnv, TypeChecker};
use crate::session::{GlobalSession, LocalWorkerState};

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
    pub action: Action,

    /// Output file
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    /// Compile to object file (alias for --action emit-obj)
    #[arg(short = 'c')]
    pub compile: bool,

    /// Parse only (alias for --action parse-only)
    #[arg(short = 'p', long = "parse-only")]
    pub parse_only: bool,

    /// Print AST (alias for --action print-ast)
    #[arg(long = "print-ast")]
    pub print_ast: bool,

    /// Emit MLIR (alias for --action emit-mlir)
    #[arg(long = "emit-mlir")]
    pub emit_mlir: bool,

    /// Emit LLVM IR (alias for --action emit-llvm)
    #[arg(long = "emit-llvm")]
    pub emit_llvm: bool,

    /// Run JIT (alias for --action run-jit)
    #[arg(long = "run")]
    pub run_jit: bool,

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

        // For simplicity, we process the first input as the main file,
        // just like the old main.rs behavior.
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
            let mlir_src = std::fs::read_to_string(main_file).map_err(|e| e.to_string())?;

            if self.options.action == Action::EmitMlir || self.options.action == Action::EmitLlvm {
                let optimized_mlir = apply_mlir_opt(&mlir_src, &mlir_args, main_file)?;
                if self.options.action == Action::EmitLlvm {
                    let llvm_ir = translate_to_llvm_ir(&optimized_mlir, main_file)?;
                    println!("{}", llvm_ir);
                } else {
                    println!("{}", optimized_mlir);
                }
                return Ok(());
            }

            if self.options.action == Action::RunJit {
                let out = crate::jit::execute_mlir(&mlir_src, mlir_args, self.options.opt_level)
                    .map_err(|e| e.to_string())?;
                println!("{}", out);
                return Ok(());
            }

            return Err(format!(
                "Action {:?} is not supported for MLIR inputs",
                self.options.action
            ));
        }

        let mut loader = ModuleLoader::new();
        let mut program_arr = loader
            .load_main(&filename)
            .map_err(|e| format!("Frontend failed to parse '{}': {}", filename, e))?;

        if self.options.action == Action::ParseOnly {
            let ast = program_arr
                .iter()
                .find(|p| p.module_path == filename)
                .unwrap();
            println!("{:#?}", ast);
            return Ok(());
        }

        let ast_idx = program_arr
            .iter()
            .position(|p| p.module_path == filename)
            .unwrap();
        let mut ast = program_arr.remove(ast_idx);

        if self.options.action == Action::PrintAst {
            crate::ast_printer::AstPrinter::print_program(&ast);
        }

        let global_session = std::sync::Arc::new(GlobalSession::new(1));
        let mut all_programs = program_arr.clone();
        all_programs.push(ast.clone());

        let env = GlobalAstEnv::build(&all_programs);
        let mut worker = LocalWorkerState::new(global_session.clone());
        let mut checker = TypeChecker::new(&env, &mut worker);

        for f in &mut ast.functions {
            checker.check_function(f);
        }

        if !checker.errors.is_empty() {
            let mut err_msg = format!(
                "Semantic check failed on '{}':
",
                filename
            );
            for err in checker.errors {
                err_msg.push_str(&format!(
                    "  {:?}
",
                    err
                ));
            }
            return Err(err_msg);
        }

        if self.options.action == Action::PrintAst {
            return Ok(());
        }

        let mut monomorphized_ast = ast;
        let mut orig_functions = monomorphized_ast.functions;
        orig_functions.retain(|f| f.generics.is_empty());

        let mut new_functions: Vec<_> = checker
            .monomorphized_functions
            .into_iter()
            .map(|(f, _)| f)
            .collect();
        new_functions.extend(orig_functions);
        monomorphized_ast.functions = new_functions;

        let mut module_asts = std::collections::HashMap::new();
        for mut p in program_arr {
            p.functions.retain(|f| f.generics.is_empty());
            module_asts.insert(p.module_path.clone(), p);
        }

        crate::codegen::register_vx_passes();

        let registry = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&registry);
        melior::utility::register_all_passes();
        let context = melior::Context::new();
        context.attach_diagnostic_handler(|diagnostic| {
            eprintln!("{}", diagnostic);
            true
        });
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        crate::codegen::register_vx_dialect(&context);

        let mut codegen = crate::codegen::MeliorGenerator::new(&context);
        codegen.generate(&monomorphized_ast, &module_asts);
        let mut module = codegen.into_module();

        if !module.as_operation().verify() {
            eprintln!("Warning: MLIR Verification failed for {}", filename);
        }

        let llvm_lower = matches!(
            self.options.action,
            Action::EmitLlvm | Action::RunJit | Action::EmitObj
        );
        let pipeline_str = get_optimization_pipeline(self.options.opt_level, llvm_lower);

        let pass_manager = melior::pass::PassManager::new(&context);
        if let Err(e) = melior::utility::parse_pass_pipeline(
            pass_manager.as_operation_pass_manager(),
            &pipeline_str,
        ) {
            return Err(format!("Failed to parse MLIR pass pipeline: {}", e));
        }

        // Apply custom CLI MLIR args if provided
        for arg in &mlir_args {
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
            eprintln!("Warning: MLIR passes failed: {}", e);
        }

        match self.options.action {
            Action::EmitMlir | Action::EmitLlvm => {
                let mlir_str = format!("{}", module.as_operation());
                println!("{}", mlir_str);
            }
            Action::RunJit => {
                let mlir_str = format!("{}", module.as_operation());
                let out = crate::jit::execute_mlir(&mlir_str, vec![], self.options.opt_level)
                    .map_err(|e| e.to_string())?;
                println!("{}", out);
            }
            Action::EmitObj => {
                let current_dir = std::env::current_dir().unwrap();
                let vx_std_core = format!(
                    "{}/target/debug/libvx_std_core.dylib",
                    current_dir.display()
                );
                let libnpu = format!("{}/target/jit/libnpu_shared.dylib", current_dir.display());

                let shared_libs = [
                    "/opt/homebrew/opt/llvm/lib/libmlir_c_runner_utils.dylib",
                    "/opt/homebrew/opt/llvm/lib/libmlir_runner_utils.dylib",
                    &vx_std_core,
                    &libnpu,
                ];

                let engine = melior::ExecutionEngine::new(
                    &module,
                    self.options.opt_level as usize,
                    &shared_libs,
                    true,
                    true,
                );

                let output_path = self.options.output.clone().unwrap_or_else(|| {
                    let mut p = main_file.clone();
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

fn get_optimization_pipeline(opt_level: u8, llvm_lower: bool) -> String {
    let mut passes = vec![];

    if opt_level > 0 || llvm_lower {
        passes.push("convert-vx-to-standard".to_string());
    }

    if opt_level > 0 {
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
        passes.push("convert-scf-to-cf".to_string());
        passes.push("expand-strided-metadata".to_string());
        passes.push("finalize-memref-to-llvm".to_string());
        passes.push("convert-vector-to-llvm".to_string());
        passes.push("convert-func-to-llvm".to_string());
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
    main_file: &std::path::Path,
) -> Result<String, String> {
    if mlir_args.is_empty() {
        return Ok(mlir_src.to_string());
    }
    let temp_in = format!(
        "{}_temp_in.mlir",
        main_file.file_name().unwrap().to_string_lossy()
    );
    let temp_out = format!(
        "{}_temp_out.mlir",
        main_file.file_name().unwrap().to_string_lossy()
    );

    let mut file = std::fs::File::create(&temp_in).unwrap();
    std::io::Write::write_all(&mut file, mlir_src.as_bytes()).unwrap();

    let mut args = vec![
        "vx-opt".to_string(),
        temp_in.clone(),
        "-o".to_string(),
        temp_out.clone(),
    ];
    args.extend_from_slice(mlir_args);

    let c_args: Vec<std::ffi::CString> = args
        .into_iter()
        .map(|a| std::ffi::CString::new(a).unwrap())
        .collect();
    let c_ptrs: Vec<*const std::os::raw::c_char> = c_args.iter().map(|a| a.as_ptr()).collect();

    let status = unsafe { run_vx_opt(c_ptrs.len() as std::os::raw::c_int, c_ptrs.as_ptr()) };

    let _ = std::fs::remove_file(&temp_in);
    if status != 0 {
        let _ = std::fs::remove_file(&temp_out);
        return Err("vx-opt failed. Check stderr for details.".to_string());
    }

    let out_str = std::fs::read_to_string(&temp_out).unwrap_or_default();
    let _ = std::fs::remove_file(&temp_out);
    Ok(out_str)
}

pub fn translate_to_llvm_ir(mlir_src: &str, main_file: &std::path::Path) -> Result<String, String> {
    let temp_mlir = format!(
        "{}_temp_llvm.mlir",
        main_file.file_name().unwrap().to_string_lossy()
    );
    let mut file = std::fs::File::create(&temp_mlir).unwrap();
    std::io::Write::write_all(&mut file, mlir_src.as_bytes()).unwrap();

    let mut cmd = std::process::Command::new("/opt/homebrew/opt/llvm/bin/mlir-translate");
    cmd.arg("--mlir-to-llvmir");
    let mlir_translate_out = cmd.arg(&temp_mlir).output().map_err(|e| e.to_string())?;

    let _ = std::fs::remove_file(&temp_mlir);
    if !mlir_translate_out.status.success() {
        return Err(format!(
            "mlir-translate failed:\n{}",
            String::from_utf8_lossy(&mlir_translate_out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&mlir_translate_out.stdout).to_string())
}
