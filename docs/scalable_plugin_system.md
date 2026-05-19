# Vx Architecture Specification: Heterogeneous Plugin System

## 1. Objective

To provide a zero-overhead, plug-and-play architecture that allows hardware vendors (NPUs, GPUs, custom ASICs) to integrate their silicon into the Vx language. This architecture bypasses opaque runtime FFI calls in favor of **Compile-Time MLIR Pass Injection** and a **Minimal Runtime Dispatch C-ABI**.

## 2. Phase 1: The Compile-Time Plugin Contract (Rust)

The compiler must expose a trait that hardware vendors implement. This trait is loaded dynamically or compiled directly into `vxc`. It allows the vendor to intercept the MLIR graph, verify support, inject custom metadata, and lower the graph to their proprietary binary format.

### File: `src/plugin/hardware_trait.rs`

```rust
use mlir::{Operation, PassManager, Module};
use std::collections::HashMap;

/// Unique identifier for a hardware topology (e.g., Topology::NPU[0])
pub type TopologyID = u32;

pub trait VxHardwarePlugin: Send + Sync {
    /// 1. Identity
    /// The vendor's identifier, e.g., "MemWise_v1"
    fn plugin_name(&self) -> &str;

    /// The topology this plugin claims responsibility for
    fn target_topology(&self) -> TopologyID;

    /// 1.5. Buffer Layout Constraints
    /// Different hardware accelerators have strict alignment and layout requirements.
    /// The Vx compiler queries these to pre-format data before lowering.
    fn preferred_tensor_layout(&self) -> TensorLayout; // e.g., NHWC, NCHW
    fn required_alignment(&self) -> usize; // e.g., 64 bytes

    /// 2. Verification Contract
    /// Vx asks the plugin if it natively supports this MLIR operation.
    /// If false, Vx's frontend will attempt to decompose it (e.g., GeLU -> Erf/Add/Mul).
    fn is_op_supported(&self, op: &Operation) -> bool;

    /// 3. Compile-Time Escape Hatch (Metadata Annotation)
    /// Allows the vendor to attach proprietary MLIR Attributes to operations
    /// before standard passes run (e.g., pinning a tensor to SRAM).
    /// Default is a no-op.
    fn annotate_operation(&self, op: &mut Operation) {}

    /// 4. The Pass Pipeline
    /// The vendor injects their specific MLIR passes (e.g., lowering `linalg`
    /// to their custom hardware dialect, fusion, and register allocation).
    fn register_passes(&self, pass_manager: &mut PassManager);

    /// 5. Final Lowering
    /// Takes the optimized MLIR module and emits the final hardware-specific payload.
    /// Returns the raw binary (command buffer, ELF, or PE) to be executed at runtime.
    fn lower_to_binary(&self, module: Module) -> Result<Vec<u8>, String>;
}

```

### Compiler Integration (`src/driver.rs`)

When `vxc` runs, it holds a registry of these plugins.

```rust
// Inside the vxc compile loop:
for (topology, plugin) in &self.plugin_registry {
    if let Some(mut block) = extract_topology_block(&mlir_module, *topology) {
        // 1. Let vendor annotate
        block.walk_mut(|op| plugin.annotate_operation(op));

        // 2. Setup passes
        let mut pm = PassManager::new();
        plugin.register_passes(&mut pm);

        // 3. Run optimization
        pm.run(&mut block).expect("Vendor pass failed");

        // 4. Extract binary payload
        let binary_payload = plugin.lower_to_binary(block)?;
        self.embed_payload_in_executable(*topology, binary_payload);
    }
}

```

## 3. Phase 2: The Runtime Dispatch Contract (C-ABI)

The vendor must provide a minimal runtime shim (a `.so`, `.dylib`, or `.dll`) that the Vx application links against. Because the Vx compiler enforces type safety, memory affinity (`Pinned<T>`), and liveness, the runtime is stripped of complex safety checks.

### File: `include/vx_hardware_runtime.h`

```c
#ifndef VX_HARDWARE_RUNTIME_H
#define VX_HARDWARE_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// 1. Memory Transfer Operations
/// Allocates memory in the target topology's address space and optionally copies data.
/// Fulfills Vx `.to_device(Topology)` requests.
void* vx_plugin_alloc_and_transfer(size_t bytes, void* host_ptr, uint32_t topology_id);

/// 2. Asynchronous Execution
/// Takes the compiled binary payload embedded by the compiler and dispatches it.
/// Note: The `device_args` unpacks MLIR MemRefs (typically a struct of base pointer,
/// aligned pointer, offset, sizes, and strides) mapping directly to the MLIR ABI.
/// Returns a Future/Event ID immediately (Non-blocking).
uint64_t vx_plugin_dispatch_async(const void* binary_payload, size_t payload_size, void** device_args);

/// 3. Synchronization
/// Blocks the host CPU until the specific execution completes.
/// Called when a `Verified<T>` or `Pinned<T>` is explicitly read by the host.
void vx_plugin_await_future(uint64_t future_id);

/// 4. Memory Lifecycle & Teardown
/// Device to Host read-back (fulfills `.to_host()`).
int32_t vx_plugin_transfer_device_to_host(void* device_ptr, void* host_ptr, size_t bytes);

/// Frees memory allocated on the device.
void vx_plugin_free(void* device_ptr, uint32_t topology_id);

/// Frees the future/event ID after await completes.
void vx_plugin_release_future(uint64_t future_id);

/// 5. The Configuration & Lifecycle Escape Hatch (ioctl-style)
/// Allows the host to initialize drivers, teardown, or send custom configurations.
int32_t vx_plugin_control(uint32_t opcode, void* payload);

// Standard Opcodes
#define VX_CTRL_INIT_DEVICE  0x01
#define VX_CTRL_SHUTDOWN     0x02
#define VX_CTRL_GET_DEVICE_COUNT 0x03 // Returns number of valid Topology::NPU[i] indices
#define VX_CTRL_VENDOR_BASE  0x1000 // Vendors start their custom opcodes here

#ifdef __cplusplus
}
#endif

#endif // VX_HARDWARE_RUNTIME_H

```

## 4. Phase 3: Language-Level Integration (Vx Frontend)

With the compiler and runtime contracts defined, you can expose this safely in the Vx language.

### File: `tests/npu_integration.vx`

```rust
// 1. Get the hardware reference
let npu = Topology::NPU[0];

// 2. Runtime Configuration Escape Hatch
// The string payload is serialized and sent to `vx_plugin_control`
// with a vendor-specific opcode before compute begins.
npu.configure("clock_freq=max; power_profile=compute");

// 3. Define the data
let A = Tensor::ones([1024, 1024]);
let B = Tensor::ones([1024, 1024]);

// 4. Memory Transfer (Calls `vx_plugin_alloc_and_transfer`)
// .to_device() converts standard Tensors into strictly typed Pinned<Tensor>
let pinned_A: Pinned<Tensor, npu> = A.to_device(npu);
let pinned_B: Pinned<Tensor, npu> = B.to_device(npu);

// 5. Execution Block
// Compiler groups this into an MLIR region, passes it to the vendor plugin,
// gets the binary, and emits a call to `vx_plugin_dispatch_async`.
let result_future = spawn on(npu) {
    // Standard library emits `linalg.matmul` with memory space mappings
    return tensor::matmul(pinned_A, pinned_B);
};

// 6. Synchronization (Calls `vx_plugin_await_future` + `.to_host()`)
let host_result = result_future.await.to_host();

```

## 5. Disaggregated Runtimes and Multi-Device Support

For disaggregated runtimes (like VLLM) utilizing multiple kinds of devices simultaneously, users can invoke the compiler separately for each device architecture. The minimal C-ABI (`VX_CTRL_GET_DEVICE_COUNT`) allows the runtime to discover how many distinct `Topology::NPU[i]` instances of a single architecture are available.

If complex multi-device pooling across distinct vendors is needed within a single process, the application acts as the orchestrator by instantiating the runtime plugins independently.

## 6. Recommended Implementation Steps

To start coding this into `vxc` today, follow this sequence to avoid getting blocked. Use Apples NPE for your reference implementation.

1. **Implement the Trait (`hardware_trait.rs`):** Define the `VxHardwarePlugin` Rust trait in your compiler tree.
1. **Build a "Dummy" Plugin:** Create a mock CPU backend plugin (`DummyCPUPlugin`) that implements the trait but just lowers MLIR `linalg` to standard LLVM IR. This acts as your test harness to ensure the plugin registry and pass manager injection works.
1. **Draft the C Header:** Add the `vx_hardware_runtime.h` to your project and implement a mock C library for it.
1. **Wire the IR Emission:** Update your `src/codegen.rs` so that `spawn on` regions are wrapped in `scf.execute_region`, and inside those blocks, ensure standard math calls map to `linalg` MLIR operations.
1. **Wire the Runtime:** Have your compiler emit LLVM IR calls to the C-ABI (`vx_plugin_dispatch_async`) when it encounters the `spawn on` block boundaries.
