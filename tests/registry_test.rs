use vxc::gid::TypeId;
use vxc::hash::{compute_module_hash, DefPath};
use vxc::registry::{ImmutableGlobalRegistry, TypeDefinition};

#[test]
fn test_valid_acyclic_registry() {
    let mod_hash = compute_module_hash("core::test");
    let struct_a_hash = DefPath::Named("A".to_string()).compute_symbol_hash();
    let struct_b_hash = DefPath::Named("B".to_string()).compute_symbol_hash();

    let id_a = TypeId::new(mod_hash, struct_a_hash, 0, 0);
    let id_b = TypeId::new(mod_hash, struct_b_hash, 0, 0);

    let def_a = TypeDefinition {
        id: id_a,
        name: "A".to_string(),
        size_bytes: 4,
        align_bytes: 4,
        by_value_dependencies: vec![], // A has no dependencies
    };

    let def_b = TypeDefinition {
        id: id_b,
        name: "B".to_string(),
        size_bytes: 8,
        align_bytes: 4,
        by_value_dependencies: vec![id_a], // B depends on A by-value
    };

    let result = ImmutableGlobalRegistry::build_and_validate(vec![def_a, def_b]);
    assert!(result.is_ok());
    let registry = result.unwrap();
    assert_eq!(registry.layouts.len(), 2);
    
    // Check that module index works
    let mod_index = registry.module_indices.get(&mod_hash).unwrap();
    assert_eq!(mod_index.get("A"), Some(&id_a));
}

#[test]
fn test_invalid_cyclic_registry() {
    let mod_hash = compute_module_hash("core::test");
    let struct_a_hash = DefPath::Named("A".to_string()).compute_symbol_hash();
    let struct_b_hash = DefPath::Named("B".to_string()).compute_symbol_hash();

    let id_a = TypeId::new(mod_hash, struct_a_hash, 0, 0);
    let id_b = TypeId::new(mod_hash, struct_b_hash, 0, 0);

    let def_a = TypeDefinition {
        id: id_a,
        name: "A".to_string(),
        size_bytes: 8,
        align_bytes: 8,
        by_value_dependencies: vec![id_b], // A depends on B
    };

    let def_b = TypeDefinition {
        id: id_b,
        name: "B".to_string(),
        size_bytes: 8,
        align_bytes: 8,
        by_value_dependencies: vec![id_a], // B depends on A (Cycle!)
    };

    let result = ImmutableGlobalRegistry::build_and_validate(vec![def_a, def_b]);
    assert!(result.is_err());
    let err = match result { Err(e) => e, _ => panic!() };
    assert!(err.contains("Infinite-sized recursive layout detected"));
}
