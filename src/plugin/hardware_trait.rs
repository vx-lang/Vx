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
use std::borrow::Cow;
use std::fmt;

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

#[derive(Debug)]
pub enum PluginError {
    LoweringFailed(String),
    UnsupportedOperation(String),
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PluginError::LoweringFailed(msg) => write!(f, "Lowering failed: {}", msg),
            PluginError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
        }
    }
}

impl std::error::Error for PluginError {}

pub trait VxHardwarePlugin: Send + Sync {
    /// The vendor's identifier. Cow allows for static or dynamic strings.
    fn plugin_name(&self) -> Cow<'static, str>;

    /// The topology this plugin claims responsibility for
    fn target_topology(&self) -> TopologyID;

    /// Buffer Layout Constraints
    fn preferred_tensor_layout(&self) -> TensorLayout;
    fn required_alignment(&self) -> usize;

    /// Verification Contract: Returns true if the hardware can execute this specific operation.
    /// Default implementation accepts all ops.
    fn is_op_supported(&self, _op: &mlir::Operation) -> bool {
        true
    }

    /// Compile-Time Escape Hatch: Allows the plugin to attach hardware-specific metadata
    /// or attributes to the operation before the lowering phase.
    fn annotate_operation(&self, _op: &mut mlir::Operation) {}

    /// The Pass Pipeline
    fn register_passes(&self, _pass_manager: &mut mlir::PassManager) {}

    /// Final Lowering
    /// Takes the optimized MLIR module and emits the final hardware-specific payload.
    fn lower_to_binary(&self, module: mlir::Module) -> Result<Vec<u8>, PluginError>;
}
