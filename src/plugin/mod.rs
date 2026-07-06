//===- mod.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file serves as the root module for the hardware plugin subsystem.
// It aggregates the plugin registry, hardware traits, and specific plugin
// implementations, providing a unified API for the compiler to query and utilize
// heterogeneous compute resources.
//
//===----------------------------------------------------------------------===//

#[cfg(target_os = "macos")]
pub mod apple_npe;

pub mod hardware_trait;
pub mod registry;

pub use hardware_trait::VxHardwarePlugin;
pub use registry::PluginRegistry;

use hardware_trait::TopologyID;
use std::sync::{Arc, LazyLock};

/// The compiler's built-in hardware plugins, keyed by topology dispatch id
/// (`arch::topology_dispatch_id`). Static — plugins are compiled in, not registered per
/// program — so, unlike the topology registry, this needs no per-compilation reset.
static BUILTIN_PLUGINS: LazyLock<PluginRegistry> = LazyLock::new(|| {
    let mut registry = PluginRegistry::new();
    #[cfg(target_os = "macos")]
    registry.register(Arc::new(apple_npe::AppleNPEPlugin));
    registry
});

/// The hardware plugin that claims `topology_id` (a dispatch id from
/// `arch::topology_dispatch_id`), if one is registered. This is the compiler's
/// topology → plugin selection point.
pub fn plugin_for(topology_id: TopologyID) -> Option<Arc<dyn VxHardwarePlugin + Send + Sync>> {
    BUILTIN_PLUGINS.get(topology_id)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn apple_npe_is_selected_by_ane_dispatch_id() {
        // The plugin must be keyed by the same dispatch id codegen stamps on `vx.spawn`, so
        // the topology → plugin lookup in the spawn lowering actually finds it.
        let ane_id = crate::arch::topology_dispatch_id(&crate::syntax::Topology::ANE) as TopologyID;
        let plugin = plugin_for(ane_id).expect("ANE plugin should be registered on macOS");
        assert_eq!(plugin.plugin_name(), "Apple_NPE_v1");
        assert_eq!(plugin.target_topology(), ane_id);

        // A topology no plugin claims (CPU) selects nothing.
        let cpu_id = crate::arch::topology_dispatch_id(&crate::syntax::Topology::CPU) as TopologyID;
        assert!(plugin_for(cpu_id).is_none());
    }
}
