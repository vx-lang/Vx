use std::collections::HashMap;
use petgraph::graph::DiGraph;
use petgraph::algo::toposort;

use crate::gid::TypeId;

/// Structural layout definition of a nominal type.
#[derive(Debug, Clone)]
pub struct TypeDefinition {
    pub id: TypeId,
    pub name: String,
    pub size_bytes: usize,
    pub align_bytes: usize,
    /// Dependencies represent the types embedded directly (by-value) within this type.
    /// Used to detect infinite-sized recursive structs.
    pub by_value_dependencies: Vec<TypeId>,
}

/// The globally frozen type registry for parallel compilation phases.
pub struct ImmutableGlobalRegistry {
    pub layouts: HashMap<TypeId, TypeDefinition>,
    pub module_indices: HashMap<u64, HashMap<String, TypeId>>,
}

impl ImmutableGlobalRegistry {
    /// Builds and validates the registry from a collection of local module thread maps.
    /// Runs a fast cycle-detection pass to ensure no infinite-sized recursive layouts exist.
    pub fn build_and_validate(definitions: Vec<TypeDefinition>) -> Result<Self, String> {
        let mut layouts = HashMap::new();
        let mut module_indices: HashMap<u64, HashMap<String, TypeId>> = HashMap::new();
        
        let mut graph = DiGraph::<TypeId, ()>::new();
        let mut node_map = HashMap::new();

        // 1. Register all layouts and build the node map
        for def in &definitions {
            layouts.insert(def.id, def.clone());
            
            let mod_id = def.id.module_id();
            module_indices.entry(mod_id).or_default().insert(def.name.clone(), def.id);
            
            let node_idx = graph.add_node(def.id);
            node_map.insert(def.id, node_idx);
        }

        // 2. Add dependency edges
        for def in &definitions {
            let source_idx = *node_map.get(&def.id).unwrap();
            for dep in &def.by_value_dependencies {
                if let Some(target_idx) = node_map.get(dep) {
                    graph.add_edge(source_idx, *target_idx, ());
                } else {
                    return Err(format!("Unresolved by-value dependency: {:?}", dep));
                }
            }
        }

        // 3. Cycle Detection
        // `toposort` returns an error if a cycle exists in a directed graph.
        if let Err(cycle_err) = toposort(&graph, None) {
            let cyclic_type_id = graph[cycle_err.node_id()];
            if let Some(cyclic_def) = layouts.get(&cyclic_type_id) {
                return Err(format!(
                    "Infinite-sized recursive layout detected in struct '{}'. Recursive fields must be wrapped in a Box or Pointer.",
                    cyclic_def.name
                ));
            }
            return Err("Infinite-sized recursive layout detected.".to_string());
        }

        Ok(Self {
            layouts,
            module_indices,
        })
    }
}
