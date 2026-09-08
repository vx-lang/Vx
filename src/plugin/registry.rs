//===- registry.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file manages the dynamic loading and registration of Akar hardware plugins.
// It allows the Vx compiler to discover and interface with external, cycle-accurate
// hardware models (like NPUs or TPUs) at runtime using C-ABI shared libraries.
//
//===----------------------------------------------------------------------===//
use super::hardware_trait::{TopologyID, VxHardwarePlugin};
use std::collections::HashMap;
use std::sync::Arc;

/// A registry for managing hardware-specific compiler plugins.
pub struct PluginRegistry {
    plugins: HashMap<TopologyID, Arc<dyn VxHardwarePlugin + Send + Sync>>,
}

impl PluginRegistry {
    /// Initializes a new PluginRegistry.
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Registers a plugin for a specific topology.
    /// If a plugin for the same topology already exists, it is replaced.
    pub fn register(&mut self, plugin: Arc<dyn VxHardwarePlugin + Send + Sync>) {
        self.plugins.insert(plugin.target_topology(), plugin);
    }

    /// Fetches a plugin reference for the given topology ID.
    pub fn get(&self, topology_id: TopologyID) -> Option<Arc<dyn VxHardwarePlugin + Send + Sync>> {
        self.plugins.get(&topology_id).cloned()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::hardware_trait::{mlir, PluginError, TensorLayout};
    use std::borrow::Cow;

    struct MockNPUPlugin;

    impl VxHardwarePlugin for MockNPUPlugin {
        fn plugin_name(&self) -> Cow<'static, str> {
            Cow::Borrowed("MockNPU")
        }
        fn target_topology(&self) -> TopologyID {
            42
        }
        fn preferred_tensor_layout(&self) -> TensorLayout {
            TensorLayout::Contiguous
        }
        fn required_alignment(&self) -> usize {
            64
        }
        fn lower_to_binary(&self, module: mlir::Module) -> Result<Vec<u8>, PluginError> {
            Ok(module.text.into_bytes())
        }
    }

    #[test]
    fn test_plugin_registry_registration() {
        let mut registry = PluginRegistry::new();
        let plugin = std::sync::Arc::new(MockNPUPlugin);
        registry.register(plugin);

        let retrieved = registry
            .get(42)
            .expect("Plugin should be registered under topology 42");
        assert_eq!(retrieved.plugin_name(), "MockNPU");
        assert_eq!(retrieved.target_topology(), 42);
        assert_eq!(retrieved.required_alignment(), 64);

        let binary = retrieved
            .lower_to_binary(mlir::Module {
                text: "mock".to_string(),
            })
            .unwrap();
        assert_eq!(binary, b"mock");

        assert!(
            registry.get(99).is_none(),
            "Unregistered topology should return None"
        );
    }
}
