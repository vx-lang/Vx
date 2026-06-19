//===- registry_test.rs - Vx Compiler --------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file tests the Module Registry's dependency resolution logic.
// It verifies that the registry can correctly topologically sort module imports,
// detect cyclic dependencies, and gracefully report errors when cyclical imports
// are encountered.
//
//===----------------------------------------------------------------------===//
use vxc::gid::TypeId;
use vxc::hash::{compute_module_hash, DefPath};
use vxc::registry::{ImmutableGlobalRegistry, TypeDefinition};

#[test]
fn test_valid_acyclic_registry() -> Result<(), String> {
    let mod_hash = compute_module_hash("core::test");
    let struct_a_hash = DefPath::Named("A").compute_symbol_hash();
    let struct_b_hash = DefPath::Named("B").compute_symbol_hash();

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
    if result.is_err() {
        return Err("Assertion failed: result.is_ok()".to_string());
    }
    let registry = result.unwrap();
    if registry.layouts.len() != 2 {
        return Err(format!(
            "Assertion failed: {} != {}",
            registry.layouts.len(),
            2
        ));
    }

    // Check that module index works
    let mod_index = registry.module_indices.get(&mod_hash).unwrap();
    if mod_index.get("A") != Some(&id_a) {
        return Err(format!(
            "Assertion failed: {:?} != {:?}",
            mod_index.get("A"),
            Some(&id_a)
        ));
    }

    Ok(())
}

#[test]
fn test_invalid_cyclic_registry() -> Result<(), String> {
    let mod_hash = compute_module_hash("core::test");
    let struct_a_hash = DefPath::Named("A").compute_symbol_hash();
    let struct_b_hash = DefPath::Named("B").compute_symbol_hash();

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
    if result.is_ok() {
        return Err("Assertion failed: result.is_err()".to_string());
    }
    let err = match result {
        Err(e) => e,
        _ => return Err("Unexpected match".to_string()),
    };
    if !(err.contains("Infinite-sized recursive layout detected")) {
        return Err(
            "Assertion failed: err.contains(\"Infinite-sized recursive layout detected\")"
                .to_string(),
        );
    }

    Ok(())
}
