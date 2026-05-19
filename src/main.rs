use melior::ir::operation::OperationLike;
use std::env;
use std::fs;
use vxc::lexer::Lexer;

fn main() {
    println!("Vx Compiler (vxc) - Bootstrap Phase (Rust)");
    println!("============================================");

    let args: Vec<String> = env::args().collect();
    let mut parse_only = false;
    let mut emit_mlir = false;
    let mut run_jit = false;
    let mut use_melior = true; // Default to the new Melior backend
    let mut filename = "";

    let mut print_ast = false;

    for arg in args.iter().skip(1) {
        if arg == "-p" {
            parse_only = true;
        } else if arg == "--print-ast" {
            print_ast = true;
        } else if arg == "--emit-mlir" {
            emit_mlir = true;
        } else if arg == "--run" {
            run_jit = true;
        } else if arg == "--use-melior" {
            use_melior = true;
        } else if arg == "--use-legacy" {
            use_melior = false;
        } else {
            filename = arg;
        }
    }

    if !filename.is_empty() {
        let source = match fs::read_to_string(filename) {
            Ok(src) => src,
            Err(e) => {
                eprintln!("Failed to open file: {} - {}", filename, e);
                return;
            }
        };

        let mut lexer = Lexer::new(&source);
        let _tokens = lexer.tokenize();

        if parse_only {
            let mut loader = vxc::module_loader::ModuleLoader::new();
            match loader.load_main(filename) {
                Ok(programs) => {
                    let ast = programs.iter().find(|p| p.module_path == filename).unwrap();
                    if print_ast {
                        vxc::ast_printer::AstPrinter::print_program(ast);
                    } else {
                        println!("{:#?}", ast);
                    }
                }
                Err(e) => eprintln!("Parse Error: {}", e),
            }
        } else {
            let mut loader = vxc::module_loader::ModuleLoader::new();
            match loader.load_main(filename) {
                Ok(mut program_arr) => {
                    let ast_idx = program_arr
                        .iter()
                        .position(|p| p.module_path == filename)
                        .unwrap();
                    let mut ast = program_arr.remove(ast_idx);

                    if print_ast {
                        vxc::ast_printer::AstPrinter::print_program(&ast);
                    }
                    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));

                    // We need all programs for the global env
                    let mut all_programs = program_arr.clone();
                    all_programs.push(ast.clone());

                    let env = vxc::sema::GlobalAstEnv::build(&all_programs);
                    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
                    let mut checker = vxc::sema::TypeChecker::new(&env, &mut worker);
                    for f in &mut ast.functions {
                        checker.check_function(f);
                    }

                    if checker.errors.is_empty() {
                        let monomorphized_ast = ast;
                        let mut module_asts = std::collections::HashMap::new();
                        for p in program_arr {
                            module_asts.insert(p.module_path.clone(), p);
                        }

                        if emit_mlir {
                            if use_melior {
                                let registry = melior::dialect::DialectRegistry::new();
                                melior::utility::register_all_dialects(&registry);
                                let context = melior::Context::new();
                                context.append_dialect_registry(&registry);
                                context.load_all_available_dialects();
                                let mut codegen =
                                    vxc::melior_codegen::MeliorGenerator::new(&context);
                                codegen.generate(&monomorphized_ast, &module_asts);
                                let module = codegen.into_module();
                                if module.as_operation().verify() {
                                    println!("{}", module.as_operation());
                                } else {
                                    eprintln!("MLIR Verification failed:");
                                    println!("{}", module.as_operation());
                                }
                            } else {
                                let mut codegen = vxc::codegen::MlirGenerator::new();
                                let mlir_str = codegen.generate(&monomorphized_ast, &module_asts);
                                println!("{}", mlir_str);
                            }
                        } else if run_jit {
                            if use_melior {
                                let registry = melior::dialect::DialectRegistry::new();
                                melior::utility::register_all_dialects(&registry);
                                let context = melior::Context::new();
                                context.append_dialect_registry(&registry);
                                context.load_all_available_dialects();
                                let mut codegen =
                                    vxc::melior_codegen::MeliorGenerator::new(&context);
                                codegen.generate(&monomorphized_ast, &module_asts);
                                let module = codegen.into_module();
                                if module.as_operation().verify() {
                                    let mlir_str = format!("{}", module.as_operation());
                                    match vxc::jit::execute_mlir(&mlir_str) {
                                        Ok(output) => println!("{}", output),
                                        Err(e) => eprintln!("Execution Error: {}", e),
                                    }
                                } else {
                                    eprintln!("MLIR Verification failed:");
                                    println!("{}", module.as_operation());
                                }
                            } else {
                                let mut codegen = vxc::codegen::MlirGenerator::new();
                                let mlir_str = codegen.generate(&monomorphized_ast, &module_asts);
                                match vxc::jit::execute_mlir(&mlir_str) {
                                    Ok(output) => println!("{}", output),
                                    Err(e) => eprintln!("Execution Error: {}", e),
                                }
                            }
                        }
                    } else {
                        eprintln!("Semantic Errors:");
                        for err in checker.errors {
                            eprintln!(" - {}", err);
                        }
                    }
                }
                Err(e) => eprintln!("Parse Error: {}", e),
            }
        }
    } else {
        println!("Usage: vxc [-p] [--print-ast] [--emit-mlir] [--run] <source_file.vx>");
    }
}
