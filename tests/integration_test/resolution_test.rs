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

/// Parse a module from source and stamp its module path (as the driver does after parsing).
fn parse_module(path: &str, src: &str) -> VxModule {
    let mut lexer = vxc::lexer::Lexer::new(src);
    let tokens = lexer.tokenize();
    let mut parser = vxc::parser::Parser::new(&tokens, src);
    let mut program = parser.parse().expect("parse failed");
    program.module_path = path.into();
    program
}

fn param_ty(m: &VxModule, fn_idx: usize) -> Type {
    m.functions[fn_idx].params[0].1.clone()
}

fn sym(s: &str) -> vxc::symbol::Symbol {
    s.into()
}

/// #194 acceptance: a *qualified* cross-module reference `A::Foo` (Foo defined in module A, used in
/// module B) is attached the **defining** module's GID -- word 0 = A's module hash -- and matches
/// A's own `Foo` GID exactly. Exercises the full path: the parser producing a `::`-qualified nominal
/// and `resolve_names` routing it to the defining module via the cross-module symbol map.
#[test]
fn cross_module_qualified_reference_resolves_to_defining_module() {
    let module_a = parse_module("A", "struct Foo { x: i32 }");
    let mut module_b = parse_module("B", "fn use_foo(f: A::Foo) -> i32 { return 0; }");

    let symbol_map = build_symbol_map(&[module_a.clone(), module_b.clone()]);
    module_b.resolve_names(&symbol_map);

    let a_foo = symbol_map[&sym("A")][&sym("Foo")];
    match param_ty(&module_b, 0) {
        Type::Struct(name, Some(id)) => {
            assert_eq!(
                name.as_ref(),
                "A::Foo",
                "qualified path kept as the nominal name"
            );
            assert_eq!(id, a_foo, "resolves to A's Foo GID, not None or a local");
            assert_eq!(
                id.module_id(),
                vxc::hash::compute_module_hash("A"),
                "word 0 is the *defining* module's hash"
            );
        }
        other => panic!("expected resolved Struct, got {other:?}"),
    }
}

/// A qualified cross-module reference must **not** be shadowed by a same-named local type. Module B
/// defines its own `Foo`, but `A::Foo` still resolves to A's `Foo` (distinct GID).
#[test]
fn qualified_reference_is_not_shadowed_by_local_same_name() {
    let module_a = parse_module("A", "struct Foo { x: i32 }");
    let mut module_b = parse_module(
        "B",
        "struct Foo { y: f32 }\nfn use_foo(f: A::Foo) -> i32 { return 0; }",
    );

    let symbol_map = build_symbol_map(&[module_a.clone(), module_b.clone()]);
    module_b.resolve_names(&symbol_map);

    let a_foo = symbol_map[&sym("A")][&sym("Foo")];
    let b_foo = symbol_map[&sym("B")][&sym("Foo")];
    assert_ne!(a_foo, b_foo, "distinct modules -> distinct GIDs");

    match param_ty(&module_b, 0) {
        Type::Struct(_, Some(id)) => {
            assert_eq!(id, a_foo, "A::Foo resolves to A's Foo, not B's local Foo");
            assert_ne!(id, b_foo);
        }
        other => panic!("expected resolved Struct, got {other:?}"),
    }
}

/// An `import a::Foo;` brings the *unqualified* name `Foo` into scope from module `a`; a plain `Foo`
/// reference then resolves to `a`'s GID (when the current module has no local `Foo`).
#[test]
fn imported_unqualified_name_resolves_cross_module() {
    let module_a = parse_module("crate::a", "struct Foo { x: i32 }");
    let mut module_b = parse_module(
        "crate::b",
        "import crate::a::Foo;\nfn use_foo(f: Foo) -> i32 { return 0; }",
    );

    let symbol_map = build_symbol_map(&[module_a.clone(), module_b.clone()]);
    module_b.resolve_names(&symbol_map);

    let a_foo = symbol_map[&sym("crate::a")][&sym("Foo")];
    match param_ty(&module_b, 0) {
        Type::Struct(name, Some(id)) => {
            assert_eq!(name.as_ref(), "Foo");
            assert_eq!(id, a_foo, "imported Foo resolves to crate::a's Foo");
            assert_eq!(id.module_id(), vxc::hash::compute_module_hash("crate::a"));
        }
        other => panic!("expected resolved Struct, got {other:?}"),
    }
}

#[test]
fn test_local_name_resolution() -> Result<(), String> {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::math".into(),
        topologies: vec![],
        memories: vec![],
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
            where_transfers: Vec::new(),
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
        topologies: vec![],
        memories: vec![],
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
            where_transfers: Vec::new(),
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
        topologies: vec![],
        memories: vec![],
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
            where_transfers: Vec::new(),
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
        topologies: vec![],
        memories: vec![],
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
            where_transfers: Vec::new(),
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
