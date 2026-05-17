use vxc::ast::{Function, Span, StructDecl, Type, VxModule};
use vxc::resolver::build_symbol_map;

#[test]
fn test_local_name_resolution() {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::math".to_string(),
        externs: vec![],
        structs: vec![StructDecl {
            name: "Vector".to_string(),
            generics: vec![],
            fields: vec![],
        }],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        functions: vec![Function {
            name: "get_vector".to_string(),
            generics: vec![],
            params: vec![],
            return_type: Type::Struct("Vector".to_string(), None),
            body: vec![],
        }],
    };

    // Phase 1.25: Build the SymbolMap from parsed modules
    let symbol_map = build_symbol_map(&[module.clone()]);

    // Phase 1.5: Resolve the AST
    module.resolve_names(&symbol_map);

    // Verify that `Vector` was mapped to a deterministic `TypeId`
    if let Type::Struct(name, id) = &module.functions[0].return_type {
        assert_eq!(name, "Vector");
        assert!(id.is_some()); // Successfully resolved to a TypeId!

        let tid = id.unwrap();
        // The Module Hash and Symbol Hash should be populated
        assert_ne!(tid.module_id(), 0);
        assert_ne!(tid.symbol_id(), 0);
    } else {
        panic!("Expected Struct type");
    }
}

use vxc::ast::{Expr, MemorySpace, Statement};

#[test]
fn test_unresolved_symbol_remains_none() {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::bad".to_string(),
        externs: vec![],
        structs: vec![], // Empty structs, "Vector" does not exist!
        enums: vec![],
        traits: vec![],
        impls: vec![],
        functions: vec![Function {
            name: "get_vector".to_string(),
            generics: vec![],
            params: vec![],
            return_type: Type::Struct("Vector".to_string(), None),
            body: vec![],
        }],
    };

    let symbol_map = build_symbol_map(&[module.clone()]);
    module.resolve_names(&symbol_map);

    if let Type::Struct(name, id) = &module.functions[0].return_type {
        assert_eq!(name, "Vector");
        assert!(id.is_none()); // Should remain unresolved!
    } else {
        panic!("Expected Struct type");
    }
}

#[test]
fn test_nested_type_resolution() {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::math".to_string(),
        externs: vec![],
        structs: vec![StructDecl {
            name: "Matrix".to_string(),
            generics: vec![],
            fields: vec![],
        }],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        functions: vec![Function {
            name: "compute".to_string(),
            generics: vec![],
            params: vec![(
                "m".to_string(),
                // &mut Matrix
                Type::Borrow(
                    Box::new(Type::Struct("Matrix".to_string(), None)),
                    Some(MemorySpace::HostDRAM),
                    true,
                ),
            )],
            return_type: Type::Scalar(vxc::ast::ElementType::Bool),
            body: vec![],
        }],
    };

    let symbol_map = build_symbol_map(&[module.clone()]);
    module.resolve_names(&symbol_map);

    if let Type::Borrow(inner, _, _) = &module.functions[0].params[0].1 {
        if let Type::Struct(name, id) = &**inner {
            assert_eq!(name, "Matrix");
            assert!(id.is_some()); // Deeply nested type must be resolved!
        } else {
            panic!("Expected inner Struct type");
        }
    } else {
        panic!("Expected Borrow type");
    }
}

#[test]
fn test_expr_and_stmt_resolution() {
    let mut module = VxModule {
        imports: Vec::new(),
        module_path: "core::app".to_string(),
        externs: vec![],
        structs: vec![StructDecl {
            name: "Config".to_string(),
            generics: vec![],
            fields: vec![],
        }],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        functions: vec![Function {
            name: "setup".to_string(),
            generics: vec![],
            params: vec![],
            return_type: Type::Scalar(vxc::ast::ElementType::Bool),
            // let c: Config = ...;
            body: vec![Statement::LetDecl(
                "c".to_string(),
                false,
                Some(Type::Struct("Config".to_string(), None)),
                Expr::Number(
                    "0.0".to_string(),
                    Some(vxc::ast::ElementType::F64),
                    Span::default(),
                ),
                Span::default(),
            )],
        }],
    };

    let symbol_map = build_symbol_map(&[module.clone()]);
    module.resolve_names(&symbol_map);

    if let Statement::LetDecl(_, _, Some(Type::Struct(name, id)), _, _) =
        &module.functions[0].body[0]
    {
        assert_eq!(name, "Config");
        assert!(id.is_some()); // The Type annotation deep within the LetDecl Statement was resolved!
    } else {
        panic!("Expected LetDecl with Config Struct type");
    }
}
