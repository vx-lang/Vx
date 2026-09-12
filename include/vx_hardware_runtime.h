#ifndef VX_HARDWARE_RUNTIME_H
#define VX_HARDWARE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h> /* vx_payload_topology: strtol */
#include <string.h> /* vx_payload_field: strlen, memcmp */

#ifdef __cplusplus
extern "C" {
#endif

/// What the program's machine model claimed about host access to the space a
/// transfer is staging into, passed as the `space_access` argument below.
///
/// The runtime cannot work this out for itself, and the compiler cannot decide
/// it alone either. `managed:` describes the machine the program was *declared*
/// against, while whether a device is actually present is known only here --
/// the same staged program is meant to run on a box with one and a box without.
/// So the model's claim travels down and the backend that knows which machine
/// this is compares the two.
typedef enum {
  /// The caller does not know. Nothing is checked. The remote worker passes
  /// this because the mode is not carried on the wire yet.
  VX_SPACE_ACCESS_UNKNOWN = 0,
  /// Declared `managed: explicit`: the model says movement in and out needs an
  /// explicit transfer, so device memory is what the program expects.
  VX_SPACE_ACCESS_EXPLICIT = 1,
  /// Declared `managed: cached`, or declaring nothing. The model says the host
  /// may read this space. Staging it as device memory contradicts that, and the
  /// contradiction surfaces as a host load of a device pointer somewhere later.
  VX_SPACE_ACCESS_HOST_READABLE = 2
} vx_space_access;

/// 1. Memory Transfer Operations
/// Allocates memory in the target topology's address space and optionally
/// copies data. Fulfills Vx `.to_device(Topology)` requests.
///
/// `space_access` is a `vx_space_access`. A backend that hands out memory the
/// host cannot read must refuse, or at least complain, when it is told the
/// program believes otherwise -- see the note in cuda_dispatch.cpp.
void *vx_plugin_alloc_and_transfer(size_t bytes, void *host_ptr,
                                   uint32_t topology_id,
                                   uint32_t space_access);

/// Per-argument ABI tag layout, produced by abiTagForType/LaunchOpLowering in
/// src/dialect/VxLowering.cpp.
///
///   bits [7:0]    kind -- 0 for a memref descriptor, otherwise the scalar's
///                 own type code (the values below)
///   bits [15:8]   element type code, memref arguments only
///   bits [23:16]  rank, memref arguments only
///   bit  [24]     slot: the argument is storage holding a descriptor, and the
///                 element type and rank above describe that descriptor
///
/// A consumer that only reconstructs the calling convention must mask with
/// VX_ABI_KIND: the low byte is the original encoding, so tag == 0 still means
/// "pass a pointer" and the libffi mapping is unchanged.
///
/// Rank and element type are what let a runtime interpret the descriptor it was
/// handed. Without them a plugin can only guess, which is why the Apple path
/// hardcodes float and rank 2. Shapes need not be transmitted separately --
/// they live in the descriptor, and rank is what makes them readable.
///
/// The slot bit distinguishes the two things a memref argument can be. An
/// ordinary buffer argument is read and written in place. A slot -- MLIR
/// `memref<memref<...>>` -- is where a kernel publishes a result it allocated
/// itself, as `c = a @ b` does. Taken verbatim a slot is a rank-0 memref with
/// no scalar element type, which is byte-identical to the opaque-pointer
/// encoding; a plugin substituting for the kernel would have no way to tell it
/// must allocate and write a descriptor there, nor of what shape. Use
/// vx_memref_aligned() to reach the descriptor a slot holds.
#define VX_ABI_KIND(tag) ((tag) & 0xFF)
#define VX_ABI_ELEM(tag) (((tag) >> 8) & 0xFF)
#define VX_ABI_RANK(tag) (((tag) >> 16) & 0xFF)
#define VX_ABI_IS_SLOT(tag) (((tag) >> 24) & 1)
#define VX_ABI_SLOT_BIT (1 << 24)
#define VX_ABI_MEMREF_TAG(elem, rank)                                          \
  (((int32_t)(rank) << 16) | ((int32_t)(elem) << 8))
#define VX_ABI_SLOT_TAG(elem, rank)                                            \
  (VX_ABI_SLOT_BIT | VX_ABI_MEMREF_TAG(elem, rank))

/// The largest rank a consumer must be able to hold.
///
/// The tag's rank field is eight bits wide, so 255 is what it can *encode*;
/// this is what a runtime is required to *accept*. The distinction matters
/// where rank arrives from outside the process and then indexes an array --
/// runtime/vx_wire.h decodes a rank off a socket -- because an unchecked value
/// there writes past whatever it indexes. Bounding it once, here, means the
/// check reads the same in every consumer instead of each inventing a limit.
///
/// Eight is chosen against what the compiler emits (rank 1 and 2 today) with
/// room for the batched and blocked layouts a serving program would add.
/// Raising it is a recompile; a memref that exceeds it is refused, never
/// truncated.
#define VX_ABI_MAX_RANK 8

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

/// The data pointer a descriptor points at. For a slot argument
/// (VX_ABI_IS_SLOT) this is the descriptor the slot holds, not element data.
///
/// This is the *base*: the first element sits `offset` elements further on, and
/// a view into a larger buffer has a non-zero offset. Use vx_memref_data() to
/// get the element the descriptor's index [0,...] refers to.
static inline void *vx_memref_aligned(const void *desc) {
  return ((void *const *)desc)[1];
}

/// Elements between the aligned base and the first element of this view.
static inline int64_t vx_memref_offset(const void *desc) {
  return *(const int64_t *)((const char *)desc + 2 * sizeof(void *));
}

/// Where a descriptor's elements actually start.
static inline void *vx_memref_data(const void *desc, int32_t dtype) {
  return (char *)vx_memref_aligned(desc) +
         vx_memref_offset(desc) * (int64_t)vx_dtype_bytes(dtype);
}

/// Write a contiguous row-major descriptor for `data` into `desc`.
///
/// The counterpart of the reads above, for a plugin that computed a result
/// itself and must hand it back the way the outlined kernel would have: the
/// kernel's `memref.alloc` + `memref.store` through the slot becomes an
/// allocation plus this descriptor. Strides are derived row-major because that
/// is the layout the compiler's own allocation has.
static inline void vx_memref_write_desc(void *desc, void *data, int32_t rank,
                                        const int64_t *sizes) {
  void **ptrs = (void **)desc;
  int64_t *fields;
  int64_t stride;
  int32_t d;

  ptrs[0] = data; /* allocated */
  ptrs[1] = data; /* aligned   */
  fields = (int64_t *)((char *)desc + 2 * sizeof(void *));
  fields[0] = 0; /* offset */

  for (d = 0; d < rank; ++d) {
    fields[1 + d] = sizes[d];
  }
  stride = 1;
  for (d = rank - 1; d >= 0; --d) {
    fields[1 + rank + d] = stride;
    stride *= sizes[d];
  }
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

/// Topology id bands, mirroring `topology_dispatch_id` in src/arch.rs. An id is
/// a kind plus a device index, so `Topology::GPU[1]` is 501.
///
/// A plugin needs the *index* -- which of the devices it serves this launch is
/// for -- and decoding the band by hand in each backend would put the same
/// arithmetic in every one of them, to drift the first time a band moves. Hence
/// one decoder here, next to the producer it has to agree with.
#define VX_TOPO_CPU 0
#define VX_TOPO_NPU_BASE 100
#define VX_TOPO_ACCCORE_BASE 200
#define VX_TOPO_AMX 300
#define VX_TOPO_ANE 400
#define VX_TOPO_GPU_BASE 500
// Network memory, as a `vx.transfer` target. Distinct bands rather than the
// shared 300 they used to answer, which was `VX_TOPO_AMX` -- a transfer into a
// peer's memory arrived indistinguishable from a request for Apple's matrix
// coprocessor, and the two network spaces were indistinguishable from each
// other (#348).
#define VX_TOPO_NIC_BASE 800
#define VX_TOPO_REMOTE_BASE 900
// Names declared in a machine file (`Topology DecodeWorker { .. }`) hash into
// 1000..1999, and topology slices into 2000..2999. A name is the identity; the
// hash is a shortcut, which is why `toponame=` in the payload is what a plugin
// resolving a worker to an endpoint should key on.
#define VX_TOPO_NAMED_BASE 1000
#define VX_TOPO_SLICE_BASE 2000
#define VX_TOPO_BAND 100

/// The device ordinal within `base`'s band, or -1 when the id is not in it.
///
/// Returning -1 rather than 0 is the point: "not a GPU id" and "GPU 0" are
/// different answers, and a plugin that conflated them would set the current
/// device from a topology meant for other hardware.
static inline int vx_topology_device_index(int32_t topology_id, int32_t base) {
  if (topology_id < base || topology_id >= base + VX_TOPO_BAND) {
    return -1;
  }
  return (int)(topology_id - base);
}

/// The `topo=` payload entry as an integer, or -1 when absent. Absence means a
/// producer predating the entry, not device 0.
static inline int32_t vx_payload_topology(const void *payload,
                                          size_t payload_size) {
  const char *value = vx_payload_field(payload, payload_size, "topo=");
  if (!value || value[0] == '\0') {
    return -1;
  }
  return (int32_t)strtol(value, NULL, 10);
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
/// Device to Host read-back (fulfills `.to_host()`). `topology_id` names the
/// device being read from, for the same reason the allocation side takes one:
/// a backend is not required to be able to infer a device from an address.
int32_t vx_plugin_transfer_device_to_host(void *device_ptr, void *host_ptr,
                                          size_t bytes, uint32_t topology_id);

/// Device to device. Allocates in `dst_topology_id`'s memory, copies `bytes`
/// there from `src_device_ptr`, and returns the new pointer; the source is left
/// alone and is the caller's to free.
///
/// This is the edge a disaggregated placement is made of (#347): the KV cache
/// prefill produced on one device has to reach the device that decodes from it,
/// and no other entry point can express a movement whose two endpoints are both
/// devices. A backend with a single device answers it with a copy, so a program
/// that names the movement stays correct on a machine that does not need it.
void *vx_plugin_transfer_peer(void *src_device_ptr, uint32_t src_topology_id,
                              uint32_t dst_topology_id, size_t bytes);

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
