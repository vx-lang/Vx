use std::env;
use std::fs;
use vxc::lexer::Lexer;
use vxc::parser;
use vxc::sema;

fn main() {
    println!("Vx Compiler (vxc) - Bootstrap Phase (Rust)");
    println!("============================================");

    let args: Vec<String> = env::args().collect();
    let mut parse_only = false;
    let mut emit_mlir = false;
    let mut run_jit = false;
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
        let tokens = lexer.tokenize();

        if parse_only {
            let mut parser = parser::Parser::new(tokens, &source);
            match parser.parse() {
                Ok(ast) => {
                    if print_ast {
                        vxc::ast_printer::AstPrinter::print_program(&ast);
                    } else {
                        println!("{:#?}", ast);
                    }
                }
                Err(e) => eprintln!("Parse Error: {}", e),
            }
        } else {
            let mut parser = parser::Parser::new(tokens, &source);
            match parser.parse() {
                Ok(mut ast) => {
                    if print_ast {
                        vxc::ast_printer::AstPrinter::print_program(&ast);
                    }
                    let mut checker = sema::TypeChecker::new();
                    match checker.check_program(&mut ast) {
                        Ok((monomorphized_ast, module_asts)) => {
                            if emit_mlir {
                                let mut codegen = vxc::codegen::MlirGenerator::new();
                                let mlir_str = codegen.generate(&monomorphized_ast, &module_asts);
                                println!("{}", mlir_str);
                            } else if run_jit {
                                let mut codegen = vxc::codegen::MlirGenerator::new();
                                let mlir_str = codegen.generate(&monomorphized_ast, &module_asts);
                                match vxc::jit::execute_mlir(&mlir_str) {
                                    Ok(output) => println!("{}", output),
                                    Err(e) => eprintln!("Execution Error: {}", e),
                                }
                            }
                        }
                        Err(errs) => {
                            eprintln!("Semantic Errors:");
                            for err in errs {
                                eprintln!(" - {}", err);
                            }
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
