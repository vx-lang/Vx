mod vx_generator;

use vx_generator::{FunctionBuilder, ModuleBuilder, StructBuilder};
use vxc::lexer::Lexer;
use vxc::parser::Parser;
use vxc::sema::TypeChecker;

#[test]
#[ignore]
fn test_stress_broad_ast_core_saturation() -> Result<(), String> {
    let mut module = ModuleBuilder::new();

    // Generate a file containing 1,000 completely independent functions
    // performing basic mathematical operations.
    let num_functions = 1000;

    // To ensure the types exist, we will first create a basic type definition
    // or just use builtins. We will use `i32` builtins.
    for i in 0..num_functions {
        let mut func = FunctionBuilder::new(&format!("compute_{}", i));
        func.add_arg("a", "i32");
        func.add_arg("b", "i32");
        func.set_return_type("i32");

        // Return a * b + i
        func.add_statement(&format!("return a * b + {};", i));
        module.add_function(func);
    }

    let input = module.build();

    // 1. Lexing
    let mut lexer = Lexer::new(&input);
    let tokens = lexer.tokenize();
    if tokens.is_empty() {
        return Err("Tokens should not be empty".into());
    }

    // 2. Parsing
    let mut parser = Parser::new(&tokens, &input);
    let mut ast = parser
        .parse()
        .map_err(|e| format!("Failed to parse the massive AST: {:?}", e))?;
    if ast.functions.len() != num_functions {
        return Err("Should have parsed 1000 functions".into());
    }

    // 3. Semantic Analysis
    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let program_arr = [ast.clone()];
    let env = vxc::sema::GlobalAstEnv::build(&program_arr);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut type_checker = TypeChecker::new(&env, &mut worker);
    for f in &mut ast.functions {
        type_checker.check_function(f);
    }
    if !type_checker.errors.is_empty() {
        return Err(format!(
            "Semantic analysis failed on broad AST: {:?}",
            type_checker.errors
        ));
    }
    Ok(())
}

#[test]
#[ignore]
fn test_stress_deep_control_flow_nesting() -> Result<(), String> {
    let mut module = ModuleBuilder::new();

    let mut func = FunctionBuilder::new("deeply_nested");
    func.set_return_type("i32");

    func.add_statement("let mut sum = 0;");

    let depth = 50;

    for i in 0..depth {
        func.add_statement(&format!("let mut i_{} = 0;", i));
        func.add_statement("loop {");
        func.add_statement(&format!("if i_{} == 2 {{ break; }}", i));
    }

    func.add_statement("sum = sum + 1;");

    for i in (0..depth).rev() {
        func.add_statement(&format!("i_{} = i_{} + 1;", i, i));
        func.add_statement("}"); // Close loop
    }

    func.add_statement("return sum;");

    module.add_function(func);

    let input = module.build();

    // 1. Lexing
    let mut lexer = Lexer::new(&input);
    let tokens = lexer.tokenize();

    // 2. Parsing
    let mut parser = Parser::new(&tokens, &input);
    let mut ast = parser
        .parse()
        .map_err(|e| format!("Failed to parse deeply nested control flow: {:?}", e))?;

    // 3. Semantic Analysis
    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let program_arr = [ast.clone()];
    let env = vxc::sema::GlobalAstEnv::build(&program_arr);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut type_checker = TypeChecker::new(&env, &mut worker);
    for f in &mut ast.functions {
        type_checker.check_function(f);
    }
    if !type_checker.errors.is_empty() {
        return Err(format!(
            "Semantic analysis failed on deeply nested AST: {:?}",
            type_checker.errors
        ));
    }
    Ok(())
}

#[test]
#[ignore]
fn test_stress_massive_struct_definitions() -> Result<(), String> {
    let mut module = ModuleBuilder::new();
    let num_structs = 500;

    for i in 0..num_structs {
        let mut st = StructBuilder::new(&format!("ModelLayer{}", i));
        st.add_field("weights", "Tensor<f32>");
        st.add_field("bias", "Tensor<f32>");
        module.add_struct(st);
    }

    let input = module.build();
    let mut lexer = Lexer::new(&input);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(&tokens, &input);
    let ast = parser
        .parse()
        .map_err(|e| format!("Failed to parse massive struct definitions: {:?}", e))?;
    if ast.structs.len() != num_structs {
        return Err(format!("Should have parsed {} structs", num_structs));
    }
    Ok(())
}
