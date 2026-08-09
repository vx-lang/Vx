#ifndef VX_HARDWARE_RUNTIME_H
#define VX_HARDWARE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <string.h> /* vx_payload_field: strlen, memcmp */

#ifdef __cplusplus
extern "C" {
#endif

/// 1. Memory Transfer Operations
/// Allocates memory in the target topology's address space and optionally
/// copies data. Fulfills Vx `.to_device(Topology)` requests.
void *vx_plugin_alloc_and_transfer(size_t bytes, void *host_ptr,
                                   uint32_t topology_id);

/// Per-argument ABI tag layout, produced by abiTagForType/LaunchOpLowering in
/// src/dialect/VxLowering.cpp.
///
///   bits [7:0]    kind -- 0 for a memref descriptor, otherwise the scalar's
///                 own type code (the values below)
///   bits [15:8]   element type code, memref arguments only
///   bits [23:16]  rank, memref arguments only
///
/// A consumer that only reconstructs the calling convention must mask with
/// VX_ABI_KIND: the low byte is the original encoding, so tag == 0 still means
/// "pass a pointer" and the libffi mapping is unchanged.
///
/// Rank and element type are what let a runtime interpret the descriptor it was
/// handed. Without them a plugin can only guess, which is why the Apple path
/// hardcodes float and rank 2. Shapes need not be transmitted separately --
/// they live in the descriptor, and rank is what makes them readable.
#define VX_ABI_KIND(tag) ((tag) & 0xFF)
#define VX_ABI_ELEM(tag) (((tag) >> 8) & 0xFF)
#define VX_ABI_RANK(tag) (((tag) >> 16) & 0xFF)
#define VX_ABI_MEMREF_TAG(elem, rank)                                          \
  (((int32_t)(rank) << 16) | ((int32_t)(elem) << 8))

enum {
  VX_ABI_KIND_MEMREF = 0,
  VX_ABI_KIND_I1 = 1,
  VX_ABI_KIND_I8 = 2,
  VX_ABI_KIND_I16 = 3,
  VX_ABI_KIND_I32 = 4,
  VX_ABI_KIND_I64 = 5,
  VX_ABI_KIND_F32 = 6,
  VX_ABI_KIND_F64 = 7
};

/// Element type codes. The first eight coincide with the scalar kinds above;
/// f16 and bf16 exist only as element types today, since a by-value half has no
/// libffi type.
enum {
  VX_DTYPE_UNKNOWN = 0,
  VX_DTYPE_I1 = 1,
  VX_DTYPE_I8 = 2,
  VX_DTYPE_I16 = 3,
  VX_DTYPE_I32 = 4,
  VX_DTYPE_I64 = 5,
  VX_DTYPE_F32 = 6,
  VX_DTYPE_F64 = 7,
  VX_DTYPE_F16 = 8,
  VX_DTYPE_BF16 = 9
};

static inline size_t vx_dtype_bytes(int32_t code) {
  switch (code) {
  case VX_DTYPE_I1:
  case VX_DTYPE_I8:
    return 1;
  case VX_DTYPE_I16:
  case VX_DTYPE_F16:
  case VX_DTYPE_BF16:
    return 2;
  case VX_DTYPE_I32:
  case VX_DTYPE_F32:
    return 4;
  case VX_DTYPE_I64:
  case VX_DTYPE_F64:
    return 8;
  default:
    return 0;
  }
}

static inline const char *vx_dtype_name(int32_t code) {
  switch (code) {
  case VX_DTYPE_I1:
    return "i1";
  case VX_DTYPE_I8:
    return "i8";
  case VX_DTYPE_I16:
    return "i16";
  case VX_DTYPE_I32:
    return "i32";
  case VX_DTYPE_I64:
    return "i64";
  case VX_DTYPE_F32:
    return "f32";
  case VX_DTYPE_F64:
    return "f64";
  case VX_DTYPE_F16:
    return "f16";
  case VX_DTYPE_BF16:
    return "bf16";
  default:
    return "?";
  }
}

/// A ranked memref descriptor is {allocated, aligned, offset, sizes[rank],
/// strides[rank]}. Sizes therefore begin after two pointers and the offset.
static inline const int64_t *vx_memref_sizes(const void *desc) {
  return (const int64_t *)((const char *)desc + 2 * sizeof(void *) +
                           sizeof(int64_t));
}

static inline const int64_t *vx_memref_strides(const void *desc, int32_t rank) {
  return vx_memref_sizes(desc) + rank;
}

/// Look up a `key=` entry in a dispatch payload.
///
/// The payload is a NUL-separated blob, bounded by the `payload_size` argument
/// of vx_plugin_dispatch_async: the first entry is the kernel name, and any
/// further entries are `key=value`. Reading the payload as a `const char *`
/// therefore still yields the kernel name, so a consumer that ignores this
/// function behaves as it always did.
///
/// `key` includes the `=` (e.g. "kind="). Returns a pointer to the value, still
/// within the blob and NUL-terminated, or NULL when absent. A zero size means a
/// producer that predates the extension: report absence rather than reading a
/// length that was never written.
///
/// The one entry defined today is `kind=`, naming the operation the kernel
/// computes ("matmul") so a plugin can route it to a vendor library instead of
/// guessing from buffer shapes -- which cannot be done for square operands,
/// where every operand assignment conforms. Absent means unclassified, which is
/// not an error: the kernel takes the ordinary path.
static inline const char *
vx_payload_field(const void *payload, size_t payload_size, const char *key) {
  if (!payload || payload_size == 0 || !key) {
    return NULL;
  }

  const char *base = (const char *)payload;
  size_t key_len = strlen(key);
  size_t pos = 0;

  /* Skip the kernel name, then walk the remaining NUL-terminated entries. */
  while (pos < payload_size && base[pos] != '\0') {
    ++pos;
  }
  ++pos;

  while (pos < payload_size) {
    const char *entry = base + pos;
    size_t remaining = payload_size - pos;
    size_t len = 0;
    while (len < remaining && entry[len] != '\0') {
      ++len;
    }
    if (len == remaining) {
      /* Unterminated: refuse rather than read past the blob. */
      return NULL;
    }
    if (len > key_len && memcmp(entry, key, key_len) == 0) {
      return entry + key_len;
    }
    pos += len + 1;
  }

  return NULL;
}

/// 2. Asynchronous Execution
/// Takes the compiled binary payload embedded by the compiler and dispatches
/// it. `device_args[i]` points to the value of the i-th C-interface argument
/// and `arg_tags[i]` is its Vx ABI type tag (see the layout above); together
/// they let the runtime reconstruct the platform calling convention via libffi.
/// Returns a Future/Event ID immediately (Non-blocking).
uint64_t vx_plugin_dispatch_async(const void *binary_payload,
                                  size_t payload_size, void **device_args,
                                  const int32_t *arg_tags, int64_t num_args);

/// 2.5. Temporary Flat Execution (Pending Issue #31 MLIR Migration)
uint64_t vx_plugin_dispatch_async_flat(float *xout, float *x, float *w, int n,
                                       int d);

/// 3. Synchronization
/// Blocks the host CPU until the specific execution completes.
/// Called when a `Verified<T>` or `Pinned<T>` is explicitly read by the host.
void vx_plugin_await_future(uint64_t future_id);

/// 4. Memory Lifecycle & Teardown
/// Device to Host read-back (fulfills `.to_host()`).
int32_t vx_plugin_transfer_device_to_host(void *device_ptr, void *host_ptr,
                                          size_t bytes);

/// Frees memory allocated on the device.
void vx_plugin_free(void *device_ptr, uint32_t topology_id);

/// Frees the future/event ID after await completes.
void vx_plugin_release_future(uint64_t future_id);

/// 5. The Configuration & Lifecycle Escape Hatch (ioctl-style)
/// Allows the host to initialize drivers, teardown, or send custom
/// configurations.
int32_t vx_plugin_control(uint32_t opcode, void *payload);

// Standard Opcodes
#define VX_CTRL_INIT_DEVICE 0x01
#define VX_CTRL_SHUTDOWN 0x02
#define VX_CTRL_GET_DEVICE_COUNT                                               \
  0x03 // Returns number of valid Topology::NPU[i] indices
#define VX_CTRL_VENDOR_BASE 0x1000 // Vendors start their custom opcodes here

#ifdef __cplusplus
}
#endif

#endif // VX_HARDWARE_RUNTIME_H
