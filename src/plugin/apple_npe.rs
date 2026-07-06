//===- apple_npe.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file contains the plugin implementation for the Apple Neural Engine (ANE).
// It acts as a bridge between the Vx compiler's dispatch system and the proprietary
// execution models required to schedule and execute tensor workloads on Apple
// Silicon hardware.
//
//===----------------------------------------------------------------------===//
use super::hardware_trait::{mlir, PluginError, TensorLayout, TopologyID, VxHardwarePlugin};
use std::borrow::Cow;

pub struct AppleNPEPlugin;

impl VxHardwarePlugin for AppleNPEPlugin {
    fn plugin_name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Apple_NPE_v1")
    }

    fn target_topology(&self) -> TopologyID {
        // The ANE's runtime dispatch id, from the single source of truth in `arch`, so the
        // plugin is keyed by the same number codegen stamps on `vx.spawn topology(N)`.
        crate::arch::topology_dispatch_id(&crate::syntax::Topology::ANE) as TopologyID
    }

    fn preferred_tensor_layout(&self) -> TensorLayout {
        TensorLayout::Contiguous
    }

    fn required_alignment(&self) -> usize {
        16 // Float32 alignment
    }

    fn lower_to_binary(&self, module: mlir::Module) -> Result<Vec<u8>, PluginError> {
        // Placeholder: a real backend would invoke the ANE/CoreML compiler (as `build.rs`
        // does for the fixed-shape primitives) and return a compiled model. Until the trait
        // is ported onto real melior types, this passes the serialized module bytes through
        // unchanged rather than fabricating a "compiled" artifact. See issue #171.
        Ok(module.text.into_bytes())
    }
}
