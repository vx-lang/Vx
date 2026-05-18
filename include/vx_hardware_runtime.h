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

/// 2.5. Temporary Flat Execution (Pending Issue #31 MLIR Migration)
uint64_t vx_plugin_dispatch_async_flat(float* xout, float* x, float* w, int n, int d);

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
