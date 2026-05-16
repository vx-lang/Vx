use std::env;
use std::fs;
use vxc::lexer::Lexer;
use vxc::parser;
use vxc::sema;

fn main() {
    println!("=====================================");
    println!("       Vx Benchmark Runner           ");
    println!("=====================================\n");

    let mut benchmarks_dir = env::current_dir().unwrap();
    benchmarks_dir.push("benchmarks");

    if !benchmarks_dir.exists() {
        eprintln!("Error: 'benchmarks' directory not found.");
        return;
    }

    let entries = match fs::read_dir(&benchmarks_dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Error reading benchmarks directory: {}", err);
            return;
        }
    };

    let mut results = Vec::new();

    let re_main = regex::Regex::new(r"fn\s+main\s*\(\)\s*(->\s*[^{]+)?\s*\{").unwrap();
    let re_time = regex::Regex::new(r"\[([0-9]+\.[0-9]+)\]").unwrap();

    for entry in entries {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_file() {
            let file_name = path.file_name().unwrap().to_str().unwrap();

            // Skip non-.vx files
            if !file_name.ends_with(".vx") {
                continue;
            }

            print!("▶ Benchmarking {:<30} ", file_name);
            use std::io::Write;
            std::io::stdout().flush().unwrap();

            let source = fs::read_to_string(&path).unwrap();

            // Inject the harness!
            let new_source = re_main.replace(&source, |caps: &regex::Captures| {
                let ret_type = caps.get(1).map_or("", |m| m.as_str());
                format!("fn __user_main() {} {{", ret_type)
            });

            let mut harness_code = String::new();
            if !source.contains("vx_get_time") {
                harness_code.push_str(
                    "
extern {
    fn vx_get_time() -> f32;
}
",
                );
            }
            if !source.contains("vx_print_float") {
                harness_code.push_str(
                    "
extern {
    fn vx_print_float(val: f32) -> i32;
}
",
                );
            }

            harness_code.push_str(
                "
fn main() -> i32 {
    unsafe {
        let __start = vx_get_time();
        let _ = __user_main();
        let __end = vx_get_time();
        let _ = vx_print_float(__end - __start);
    }
    return 0;
}
",
            );
            let final_source = format!("{}\n{}", new_source, harness_code);

            // Run the compiler
            let mut lexer = Lexer::new(&final_source);
            let tokens = lexer.tokenize();
            let mut parser = parser::Parser::new(tokens, &final_source);

            match parser.parse() {
                Ok(mut ast) => {
                    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
                    let program_arr = [ast.clone()];
                    let env = vxc::sema::GlobalAstEnv::build(&program_arr);
                    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
                    let mut checker = vxc::sema::TypeChecker::new(&env, &mut worker);
                    for f in &mut ast.functions { checker.check_function(f); }
                    if checker.errors.is_empty() {
                        let monomorphized_ast = ast;
                        let module_asts = std::collections::HashMap::new();
                        match Ok((monomorphized_ast, module_asts)) {
                        Ok((monomorphized_ast, module_asts)) => {
                            let mut codegen = vxc::codegen::MlirGenerator::new();
                            let mlir_str = codegen.generate(&monomorphized_ast, &module_asts);
                            match vxc::jit::execute_mlir(&mlir_str) {
                                Ok(output) => {
                                    // Parse the output to find the float time like [0.125]
                                    if let Some(caps) = re_time.captures(&output) {
                                        let time_f = caps[1].parse::<f32>().unwrap();
                                        println!("{:.4}s", time_f);
                                        results.push((file_name.to_string(), time_f));
                                    } else {
                                        println!("(no timing output)");
                                    }
                                }
                                Err(e) => {
                                    println!("FAILED");
                                    eprintln!("Execution Error: {}", e);
                                }
                            }
                        }
                        Err(errs) => {
                            println!("FAILED");
                            eprintln!("Semantic Errors:");
                            for err in errs {
                                eprintln!(" - {}", err);
                            }
                            eprintln!("Final Source:\n{}", final_source);
                        }
                    }
                }
                Err(e) => {
                    println!("FAILED");
                    eprintln!("Parse Error: {}", e);
                }
            }
        }
    }

    println!("\n=====================================");
    println!("             Summary                 ");
    println!("=====================================");
    for (name, time) in &results {
        println!("{:<30} {:.4}s", name, time);
    }
}
