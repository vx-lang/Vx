use super::hardware_trait::{mlir, TensorLayout, TopologyID, VxHardwarePlugin};

pub struct AppleNPEPlugin;

impl VxHardwarePlugin for AppleNPEPlugin {
    fn plugin_name(&self) -> &str {
        "Apple_NPE_v1"
    }

    fn target_topology(&self) -> TopologyID {
        // Assume ANE has topology ID 3 based on src/codegen.rs Topology matching
        3
    }

    fn preferred_tensor_layout(&self) -> TensorLayout {
        TensorLayout::Contiguous
    }

    fn required_alignment(&self) -> usize {
        16 // Float32 alignment
    }

    fn lower_to_binary(&self, module: mlir::Module) -> Result<Vec<u8>, String> {
        // In a real implementation, this would invoke the Apple Neural Engine compiler
        // or emit a compiled model. Since we use MLIR string emission, we simply
        // return the string bytes so `codegen.rs` can embed it.
        Ok(module.text.into_bytes())
    }
}
