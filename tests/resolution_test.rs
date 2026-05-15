use vxc::ast::{VxModule, StructDecl, Type, Function, Span};
use vxc::resolver::build_symbol_map;

#[test]
fn test_local_name_resolution() {
    let mut module = VxModule {
        module_path: "core::math".to_string(),
        externs: vec![],
        structs: vec![StructDecl {
            name: "Vector".to_string(),
            type_params: vec![],
            fields: vec![],
            span: Span::default(),
        }],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        functions: vec![Function {
            name: "get_vector".to_string(),
            params: vec![],
            ret_type: Some(Type::Struct("Vector".to_string(), None)),
            body: vec![],
            is_unsafe: false,
            span: Span::default(),
        }],
    };

    // Phase 1.25: Build the SymbolMap from parsed modules
    let symbol_map = build_symbol_map(&[module.clone()]);

    // Phase 1.5: Resolve the AST
    module.resolve_names(&symbol_map);

    // Verify that `Vector` was mapped to a deterministic `TypeId`
    if let Some(Type::Struct(name, id)) = &module.functions[0].ret_type {
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
