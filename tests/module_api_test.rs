//===- module_api_test.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file tests the compiler's programmatic Module API.
// It verifies that external Rust code can successfully inject custom ASTs,
// register synthetic functions, and manipulate the module registry bypassing
// the standard text-based parser.
//
//===----------------------------------------------------------------------===//
use vxc::parse_module;

#[test]
fn test_parse_module_api() -> Result<(), String> {
    let source = "
        fn hello_world() -> Tensor {
            let x = 42;
        }
    ";
    let module = parse_module(source);
    if module.is_err() {
        return Err(format!("Parse failed: {:?}", module.err()));
    }
    let module = module.unwrap();
    if module.functions.len() != 1 {
        return Err(format!(
            "Assertion failed: {} != {}",
            module.functions.len(),
            1
        ));
    }
    if module.functions[0].name != "hello_world" {
        return Err(format!(
            "Assertion failed: {} != {}",
            module.functions[0].name, "hello_world"
        ));
    }

    Ok(())
}

use vxc::ast::VxModule;

#[test]
fn test_ak_module_add_function() -> Result<(), String> {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::test".to_string(),
        externs: vec![],
        structs: vec![],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        macros: vec![],
        functions: vec![],
    };

    // The 'pub' keyword is automatically stripped by our From<&str> implementation
    module.add(vxc::parser::Parser::parse_fn_from_str(
        "pub fn foo() -> i64 { return 10; }",
    ));

    if module.functions.len() != 1 {
        return Err(format!(
            "Assertion failed: {} != {}",
            module.functions.len(),
            1
        ));
    }
    if module.functions[0].name != "foo" {
        return Err(format!(
            "Assertion failed: {} != {}",
            module.functions[0].name, "foo"
        ));
    }

    Ok(())
}
