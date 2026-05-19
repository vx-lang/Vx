//===- hardware_trait.rs - Vx Compiler -------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file defines the core traits and interfaces for hardware accelerators.
// It establishes the standard contract that all external accelerator plugins must
// implement to interact with the Vx compiler's dispatch and memory management
// systems.
//
//===----------------------------------------------------------------------===//
// Mock MLIR types since Vx currently uses string-based codegen instead of a Rust MLIR crate.
pub mod mlir {
    pub struct Operation {
        pub text: String,
    }

    pub struct PassManager;

    pub struct Module {
        pub text: String,
    }
}

pub type TopologyID = u32;

pub enum TensorLayout {
    NHWC,
    NCHW,
    Contiguous,
}

pub trait VxHardwarePlugin: Send + Sync {
    /// The vendor's identifier
    fn plugin_name(&self) -> &str;

    /// The topology this plugin claims responsibility for
    fn target_topology(&self) -> TopologyID;

    /// Buffer Layout Constraints
    fn preferred_tensor_layout(&self) -> TensorLayout;
    fn required_alignment(&self) -> usize;

    /// Verification Contract
    fn is_op_supported(&self, _op: &mlir::Operation) -> bool {
        true
    }

    /// Compile-Time Escape Hatch (Metadata Annotation)
    fn annotate_operation(&self, _op: &mut mlir::Operation) {}

    /// The Pass Pipeline
    fn register_passes(&self, _pass_manager: &mut mlir::PassManager) {}

    /// Final Lowering
    /// Takes the optimized MLIR module and emits the final hardware-specific payload.
    fn lower_to_binary(&self, module: mlir::Module) -> Result<Vec<u8>, String>;
}
