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

    /// Run JIT (alias for --action run-jit)
    #[arg(long = "run")]
    pub run_jit: bool,

    /// Input source files
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,

    /// Use the legacy code generator (non-melior)
    #[arg(long = "use-legacy", hide = true)]
    pub use_legacy: bool,
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

        let use_melior = !self.options.use_legacy;

        match self.options.action {
            Action::EmitMlir => {
                if use_melior {
                    let registry = melior::dialect::DialectRegistry::new();
                    melior::utility::register_all_dialects(&registry);
                    let context = melior::Context::new();
                    context.append_dialect_registry(&registry);
                    context.load_all_available_dialects();
                    crate::melior_codegen::register_vx_dialect(&context);

                    let mut codegen = crate::melior_codegen::MeliorGenerator::new(&context);
                    codegen.generate(&monomorphized_ast, &module_asts);
                    let mut module = codegen.into_module();

                    let vx_pm = melior::pass::PassManager::new(&context);
                    unsafe {
                        crate::melior_codegen::addVxLoweringPass(vx_pm.to_raw());
                    }
                    if let Err(e) = vx_pm.run(&mut module) {
                        eprintln!("Failed to lower Vx dialect: {}", e);
                    }

                    if !module.as_operation().verify() {
                        eprintln!("Warning: MLIR Verification failed for {}", filename);
                    }
                    println!("{}", module.as_operation());
                } else {
                    let mut codegen = crate::codegen::MlirGenerator::new();
                    let mlir_str = codegen.generate(&monomorphized_ast, &module_asts);
                    println!("{}", mlir_str);
                }
            }
            Action::RunJit => {
                if use_melior {
                    let registry = melior::dialect::DialectRegistry::new();
                    melior::utility::register_all_dialects(&registry);
                    let context = melior::Context::new();
                    context.append_dialect_registry(&registry);
                    context.load_all_available_dialects();
                    crate::melior_codegen::register_vx_dialect(&context);

                    let mut codegen = crate::melior_codegen::MeliorGenerator::new(&context);
                    codegen.generate(&monomorphized_ast, &module_asts);
                    let mut module = codegen.into_module();

                    if !module.as_operation().verify() {
                        return Err("MLIR Module Verification Failed".to_string());
                    }

                    crate::melior_codegen::lower_to_llvm(&context, &mut module)
                        .map_err(|e| format!("Failed to lower to LLVM: {}", e))?;

                    let mlir_str = format!("{}", module.as_operation());
                    let out = crate::jit::execute_mlir(&mlir_str).map_err(|e| e.to_string())?;
                    println!("{}", out);
                } else {
                    let mut codegen = crate::codegen::MlirGenerator::new();
                    let mlir_str = codegen.generate(&monomorphized_ast, &module_asts);
                    let out = crate::jit::execute_mlir(&mlir_str).map_err(|e| e.to_string())?;
                    println!("{}", out);
                }
            }
            Action::EmitObj => {
                if use_melior {
                    let registry = melior::dialect::DialectRegistry::new();
                    melior::utility::register_all_dialects(&registry);
                    let context = melior::Context::new();
                    context.append_dialect_registry(&registry);
                    context.load_all_available_dialects();

                    let mut codegen = crate::melior_codegen::MeliorGenerator::new(&context);
                    codegen.generate(&monomorphized_ast, &module_asts);
                    let mut module = codegen.into_module();

                    if !module.as_operation().verify() {
                        return Err(format!(
                            "MLIR Verification failed:
{}",
                            module.as_operation()
                        ));
                    }

                    crate::melior_codegen::lower_to_llvm(&context, &mut module)?;

                    let current_dir = std::env::current_dir().unwrap();
                    let vx_std_core = format!(
                        "{}/target/debug/libvx_std_core.dylib",
                        current_dir.display()
                    );
                    let libnpu =
                        format!("{}/target/jit/libnpu_shared.dylib", current_dir.display());

                    let shared_libs = [
                        "/opt/homebrew/opt/llvm/lib/libmlir_c_runner_utils.dylib",
                        "/opt/homebrew/opt/llvm/lib/libmlir_runner_utils.dylib",
                        &vx_std_core,
                        &libnpu,
                    ];

                    let engine = melior::ExecutionEngine::new(&module, 2, &shared_libs, true, true);

                    let output_path = self.options.output.clone().unwrap_or_else(|| {
                        let mut p = main_file.clone();
                        p.set_extension("o");
                        p
                    });

                    engine.dump_to_object_file(output_path.to_str().unwrap());
                } else {
                    return Err(
                        "Object file emission is only supported with the Melior backend."
                            .to_string(),
                    );
                }
            }
            _ => {}
        }

        Ok(())
    }
}
