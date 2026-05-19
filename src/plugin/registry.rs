//===- registry.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
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

pub struct PluginRegistry {
    plugins: HashMap<TopologyID, Box<dyn VxHardwarePlugin>>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    pub fn register(&mut self, plugin: Box<dyn VxHardwarePlugin>) {
        self.plugins.insert(plugin.target_topology(), plugin);
    }

    pub fn get(&self, topology_id: TopologyID) -> Option<&dyn VxHardwarePlugin> {
        self.plugins.get(&topology_id).map(|p| p.as_ref())
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}
