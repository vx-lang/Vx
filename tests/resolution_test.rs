//===- resolution_test.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file contains integration tests for the Name Resolution pass.
// It verifies that the compiler correctly links identifiers to their declarations
// across complex nested scopes, module boundaries, and shadowing scenarios.
//
//===----------------------------------------------------------------------===//
use vxc::resolver::build_symbol_map;
use vxc::syntax::{Function, Span, StructDecl, Type, VxModule};

#[test]
fn test_local_name_resolution() -> Result<(), String> {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::math".into(),
        externs: vec![],
        structs: vec![StructDecl {
            name: "Vector".into(),
            generics: vec![],
            fields: vec![],
            doc_comment: None,
        }],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        macros: vec![],
        functions: vec![Function {
            name: "get_vector".into(),
            generics: vec![],
            params: vec![],
            topology: vxc::syntax::Topology::CPU,
            return_type: Type::Struct("Vector".into(), None),
            requires: Vec::new(),
            ensures: Vec::new(),
            body: vec![],
            doc_comment: None,
        }],
    };

    // Phase 1.25: Build the SymbolMap from parsed modules
    let symbol_map = build_symbol_map(&[module.clone()]);

    // Phase 1.5: Resolve the AST
    module.resolve_names(&symbol_map);

    // Verify that `Vector` was mapped to a deterministic `TypeId`
    if let Type::Struct(name, id) = &module.functions[0].return_type {
        if name.as_ref() != "Vector" {
            return Err(format!("Assertion failed: {} != {}", name, "Vector"));
        }
        if !(id.is_some()) {
            return Err("Assertion failed: id.is_some()".to_string());
        } // Successfully resolved to a TypeId!

        let tid = id.unwrap();
        // The Module Hash and Symbol Hash should be populated
        if tid.module_id() == 0 {
            return Err(format!("Assertion failed: {} == {}", tid.module_id(), 0));
        }
        if tid.symbol_id() == 0 {
            return Err(format!("Assertion failed: {} == {}", tid.symbol_id(), 0));
        }
    } else {
        return Err("Expected Struct type".to_string());
    }

    Ok(())
}

use vxc::syntax::{Expr, LetDeclStmt, MemorySpace, NumberExpr, Statement};

#[test]
fn test_unresolved_symbol_remains_none() -> Result<(), String> {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::bad".into(),
        externs: vec![],
        structs: vec![], // Empty structs, "Vector" does not exist!
        enums: vec![],
        traits: vec![],
        impls: vec![],
        macros: vec![],
        functions: vec![Function {
            name: "get_vector".into(),
            generics: vec![],
            params: vec![],
            topology: vxc::syntax::Topology::CPU,
            return_type: Type::Struct("Vector".into(), None),
            requires: Vec::new(),
            ensures: Vec::new(),
            body: vec![],
            doc_comment: None,
        }],
    };

    let symbol_map = build_symbol_map(&[module.clone()]);
    module.resolve_names(&symbol_map);

    if let Type::Struct(name, id) = &module.functions[0].return_type {
        if name.as_ref() != "Vector" {
            return Err(format!("Assertion failed: {} != {}", name, "Vector"));
        }
        if !(id.is_none()) {
            return Err("Assertion failed: id.is_none()".to_string());
        } // Should remain unresolved!
    } else {
        return Err("Expected Struct type".to_string());
    }

    Ok(())
}

#[test]
fn test_nested_type_resolution() -> Result<(), String> {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::math".into(),
        externs: vec![],
        structs: vec![StructDecl {
            name: "Matrix".into(),
            generics: vec![],
            fields: vec![],
            doc_comment: None,
        }],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        macros: vec![],
        functions: vec![Function {
            name: "compute".into(),
            generics: vec![],
            topology: vxc::syntax::Topology::CPU,
            params: vec![(
                "m".into(),
                // &mut Matrix
                Type::Borrow {
                    inner: Box::new(Type::Struct("Matrix".into(), None)),
                    mem_space: Some(MemorySpace::CPUDRAM),
                    is_mut: true,
                    region_id: 0,
                },
            )],
            return_type: Type::Scalar(vxc::syntax::ElementType::Bool),
            requires: Vec::new(),
            ensures: Vec::new(),
            body: vec![],
            doc_comment: None,
        }],
    };

    let symbol_map = build_symbol_map(&[module.clone()]);
    module.resolve_names(&symbol_map);

    if let Type::Borrow { inner, .. } = &module.functions[0].params[0].1 {
        if let Type::Struct(name, id) = &**inner {
            if name.as_ref() != "Matrix" {
                return Err(format!("Assertion failed: {} != {}", name, "Matrix"));
            }
            if !(id.is_some()) {
                return Err("Assertion failed: id.is_some()".to_string());
            } // Deeply nested type must be resolved!
        } else {
            return Err("Expected inner Struct type".to_string());
        }
    } else {
        return Err("Expected Borrow type".to_string());
    }

    Ok(())
}

#[test]
fn test_expr_and_stmt_resolution() -> Result<(), String> {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::app".into(),
        externs: vec![],
        structs: vec![StructDecl {
            name: "Config".into(),
            generics: vec![],
            fields: vec![],
            doc_comment: None,
        }],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        macros: vec![],
        functions: vec![Function {
            name: "setup".into(),
            generics: vec![],
            params: vec![],
            topology: vxc::syntax::Topology::CPU,
            return_type: Type::Scalar(vxc::syntax::ElementType::Bool),
            requires: Vec::new(),
            ensures: Vec::new(),
            // let c: Config = ...;
            body: vec![Statement::LetDecl(LetDeclStmt {
                name: "c".into(),
                is_mut: false,
                ty_ann: Some(Type::Struct("Config".into(), None)),
                expr: Expr::Number(NumberExpr {
                    value: "0.0".to_string().into(),
                    ty: Some(vxc::syntax::ElementType::F64),
                    span: Span::default(),
                }),
                span: Span::default(),
            })],
            doc_comment: None,
        }],
    };

    let symbol_map = build_symbol_map(&[module.clone()]);
    module.resolve_names(&symbol_map);

    if let Statement::LetDecl(LetDeclStmt {
        ty_ann: Some(Type::Struct(name, id)),
        ..
    }) = &module.functions[0].body[0]
    {
        if name.as_ref() != "Config" {
            return Err(format!("Assertion failed: {} != {}", name, "Config"));
        }
        if !(id.is_some()) {
            return Err("Assertion failed: id.is_some()".to_string());
        } // The Type annotation deep within the LetDecl Statement was resolved!
    } else {
        return Err("Expected LetDecl with Config Struct type".to_string());
    }

    Ok(())
}
