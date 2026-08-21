//===- cuda_dispatch.cpp - CUDA backend for the Vx plugin ABI ---*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Provider of the vx_plugin_* ABI (include/vx_hardware_runtime.h) for NVIDIA
// GPUs. Device memory is real device memory, and a kernel the compiler
// classified as a matrix multiply runs as a cuBLAS GEMM instead of as the
// outlined loop nest.
//
// Two facts make the routing possible, both of them stated by the compiler
// rather than inferred here (#325): `kind=matmul` says the region is a GEMM,
// and `roles=a:i,b:j,out:k` says which argument is which. Shapes cannot answer
// the second question -- for square operands every assignment of A and B
// conforms, and getting it backwards yields a plausible matrix of wrong
// numbers rather than an error. runtime/vx_dispatch_plan.h does the decoding
// and is tested without a GPU; this file is the part that needs one.
//
// A region that is *not* a recognised GEMM now runs on the device too, from a
// kernel of its own: the compiler compiles every self-contained device region
// to PTX and carries the image in the dispatch payload (#251), and
// `run_device_image` below loads and launches it. cuBLAS still wins the matmul
// -- it is measured faster than anything emitted here (#321) -- so that route
// is tried first and this one is what used to be a refusal.
//
// That refusal was not a performance outcome, which is what makes this a
// correctness fix rather than an optimisation. The old claim was that failing
// to route "costs performance and never correctness", and it held only while
// the operands were host memory. `transfer(x, Memory::GPU_HBM)` is the
// construct that ends that: after one, the pointers are device pointers, and an
// unrouted kernel is handed them and dereferences them on the CPU.
// tests/backend/pass/flash_attention_placed.vx is exactly that shape, and it
// passes in CI because CI has no GPU -- the transfer is a no-op and the
// pointers stay host pointers. On an A100 it segmentation-faulted inside the
// outlined kernel.
//
// What remains unroutable is a region whose result is a slot the kernel
// allocates through: there is no descriptor on this side to launch against.
// That is refused, still with a diagnostic, and is the open half of #251.
//
// Anything with no device image at all still runs on the host through libffi,
// exactly as the portable shim would.
//
// Scope: operands are staged to the device per dispatch and the result staged
// back. Keeping tensors resident across dispatches is what `.to_device()` and
// the placement analysis exist for, and is the next milestone (#321); the
// allocation entry points below already return device pointers, so a resident
// buffer arrives here as one and is used in place.
//
//===----------------------------------------------------------------------===//

#include "vx_device_pool.h"
#include "vx_dispatch_plan.h"
#include "vx_host_call.h"
#include "vx_kernel_launch.h"
#include "vx_remote_routing.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

#include <cublas_v2.h>
// The driver API, for loading a module the compiler emitted. The runtime API
// has no equivalent before CUDA 12's `cudaLibraryLoadData`, and mixing the two
// is ordinary: driver calls act on the current context, which the runtime
// creates lazily -- `run_device_image` forces it with the usual `cudaFree(0)`.
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// Diagnostics are opt-in: the backend test harness compares a program's stdout
// against its `// EXPECT:` lines, so an unconditional banner would fail every
// placed test. Everything below writes to stderr in any case.
bool verbose() {
  static const bool on = [] {
    const char *v = getenv("VX_DISPATCH_VERBOSE");
    return v && v[0] != '\0' && strcmp(v, "0") != 0;
  }();
  return on;
}

// A GPU that reports an error mid-GEMM has produced no result, and the buffer
// it was to write is either untouched or half written. Continuing would hand
// the program numbers it cannot distinguish from an answer, so stop.
#define VX_CUDA_CHECK(expr)                                                    \
  do {                                                                         \
    cudaError_t vx_err_ = (expr);                                              \
    if (vx_err_ != cudaSuccess) {                                              \
      fprintf(stderr, "[Vx CUDA] FATAL: %s failed: %s\n", #expr,               \
              cudaGetErrorString(vx_err_));                                    \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define VX_CUBLAS_CHECK(expr)                                                  \
  do {                                                                         \
    cublasStatus_t vx_st_ = (expr);                                            \
    if (vx_st_ != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "[Vx CUDA] FATAL: %s failed with cuBLAS status %d\n",    \
              #expr, (int)vx_st_);                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

/// Whether a usable device exists, decided once.
///
/// A build with the CUDA backend compiled in can still run on a machine with no
/// GPU -- the same binary, a different box. That is a reason to run kernels on
/// the host, not to fail, so it is reported once and then taken as read.
bool cuda_available() {
  static const bool available = [] {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
      fprintf(stderr,
              "[Vx CUDA] no CUDA device (%s); kernels will run on the host\n",
              err == cudaSuccess ? "none present" : cudaGetErrorString(err));
      return false;
    }
    return true;
  }();
  return available;
}

/// Select the device this launch targets, and report whether it exists.
///
/// The topology id says which GPU the program asked for -- `Topology::GPU[1]`
/// is 501 -- and honouring it is the whole of running prefill on one device and
/// decode on another. A launch that names a device this machine does not have
/// is a mistake worth stopping for: silently running it on device 0 would
/// produce a correct-looking answer from the wrong half of a disaggregated run.
///
/// An absent or out-of-band id leaves the current device alone, which is what a
/// producer predating the entry means and what a non-GPU topology means.
/// `why` names what the device was selected *for*, and is not decoration.
///
/// This line used to be `[Vx CUDA] device 1` whoever asked, which made an
/// allocation indistinguishable from a dispatch in the trace -- and the
/// disaggregated demo's "did it use two GPUs?" check counts these lines. A
/// short run puts every iteration in prefill, so decode never executes, and the
/// only device-1 lines are the decode replica's weight allocations. The check
/// saw thirteen of them, reported two devices, and the token comparison agreed
/// because it was the same computation run twice on one GPU. Both halves of the
/// evidence passed for a run that never disaggregated anything.
bool select_device(int32_t topology_id, const char *why) {
  int index = vx_topology_device_index(topology_id, VX_TOPO_GPU_BASE);
  if (index < 0) {
    return true;
  }

  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || index >= count) {
    fprintf(stderr,
            "[Vx CUDA] FATAL: launch targets GPU %d, but this machine has %d\n",
            index, count);
    abort();
  }

  VX_CUDA_CHECK(cudaSetDevice(index));
  if (verbose()) {
    fprintf(stderr, "[Vx CUDA] device %d %s\n", index, why);
  }
  return true;
}

/// One handle per device, created on first use of that device.
///
/// A cuBLAS handle belongs to whichever device was current when `cublasCreate`
/// ran, and using it after `cudaSetDevice` has moved elsewhere does not fail --
/// it keeps computing in the original device's context. One handle for the
/// process was therefore correct exactly as long as there was one device. With
/// two, a dispatch to GPU 1 would call `select_device(501)`, print
/// `[Vx CUDA] device 1`, and hand the GEMM to device 0: the arithmetic in the
/// wrong place and every observable saying otherwise (#346).
///
/// Keyed by device index rather than topology id. The id is the program's name
/// for a device and the index is the driver's, and it is the driver that owns a
/// handle; two topologies that resolved to one device must share one.
///
/// Read the current device rather than taking it as an argument, so this cannot
/// disagree with the `cudaSetDevice` that `select_device` already performed --
/// the handle is created in the same context the work will run in, by
/// construction.
cublasHandle_t cublas_handle() {
  static std::unordered_map<int, cublasHandle_t> handles;

  int device = 0;
  VX_CUDA_CHECK(cudaGetDevice(&device));

  auto it = handles.find(device);
  if (it != handles.end()) {
    return it->second;
  }

  cublasHandle_t h = nullptr;
  VX_CUBLAS_CHECK(cublasCreate(&h));
  handles.emplace(device, h);
  return h;
}

/// True when the pointer is already device-resident, so staging it would copy
/// memory that is where it needs to be. Host pointers are the common case
/// today; `.to_device()` produces the other.
bool is_device_ptr(const void *ptr) {
  if (!ptr) {
    return false;
  }
  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err != cudaSuccess) {
    // An unregistered host pointer is reported as an error on some versions;
    // clear it so the next real error is not attributed to this call.
    cudaGetLastError();
    return false;
  }
  return attrs.type == cudaMemoryTypeDevice ||
         attrs.type == cudaMemoryTypeManaged;
}

cudaDataType_t cuda_dtype(int32_t dtype) {
  switch (dtype) {
  case VX_DTYPE_F32:
    return CUDA_R_32F;
  case VX_DTYPE_F64:
    return CUDA_R_64F;
  case VX_DTYPE_F16:
    return CUDA_R_16F;
  case VX_DTYPE_BF16:
    return CUDA_R_16BF;
  default:
    // vx_gemm_dtype_supported() gates every caller.
    fprintf(stderr, "[Vx CUDA] FATAL: unsupported GEMM element type %d\n",
            (int)dtype);
    abort();
  }
}

/// Per-dispatch device allocations, cached between dispatches.
///
/// A dispatch stages its operands and frees them again, and a program runs the
/// same shapes in a loop -- so the driver sees the same `cudaMalloc` and
/// `cudaFree` over and over. On an A100 that was ~5.7 ms of an 11.2 ms 2048^2
/// GEMM, 51%, against 1.06 ms for the cuBLAS call itself.
///
/// Only the staging buffers go through here, not `vx_plugin_transfer`: that one
/// is a placement the program asked for and lives until it is freed, whereas
/// these are scratch that exists for one dispatch. A remote worker stages its
/// operands the same way, so this is a fleet-side cost too, not only a local
/// one.
///
/// `VX_CUDA_POOL_MAX_MB=0` turns the cache off and restores the previous
/// behaviour exactly, which is the first thing to try if a dispatch ever
/// returns memory that looks like someone else's.
static vx::DevicePool<void *(*)(size_t), void (*)(void *, size_t)> &
device_pool() {
  static auto raw_alloc = [](size_t bytes) -> void * {
    void *p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
      cudaGetLastError();
      return nullptr;
    }
    return p;
  };
  static auto raw_free = [](void *p, size_t) { cudaFree(p); };
  static vx::DevicePool<void *(*)(size_t), void (*)(void *, size_t)> pool(
      raw_alloc, raw_free, vx::vx_pool_capacity_bytes());
  return pool;
}

/// A device-side operand, staged from host memory or borrowed in place.
///
/// Movable and not copyable, deliberately: a copy would leave two objects each
/// believing it owns the allocation, and the second destructor would free a
/// pointer the first already freed. Returning one by value is the normal case
/// here, and named-return elision is permitted rather than guaranteed.
struct DeviceBuffer {
  void *ptr = nullptr;
  int64_t row_stride = 0; // in elements
  bool owned = false;
  size_t bytes = 0; // what to hand back to the pool; 0 when borrowed

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : ptr(other.ptr), row_stride(other.row_stride), owned(other.owned),
        bytes(other.bytes) {
    other.ptr = nullptr;
    other.owned = false;
    other.bytes = 0;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      if (owned && ptr) {
        device_pool().release(ptr, bytes);
      }
      ptr = other.ptr;
      row_stride = other.row_stride;
      owned = other.owned;
      bytes = other.bytes;
      other.ptr = nullptr;
      other.owned = false;
      other.bytes = 0;
    }
    return *this;
  }

  ~DeviceBuffer() {
    if (owned && ptr) {
      device_pool().release(ptr, bytes);
    }
  }
};

/// Make `rows x cols` elements available on the device. A device pointer is
/// used where it lies; a host one is copied into a packed buffer, so the device
/// row stride is `cols` whatever the host's padding was.
DeviceBuffer stage(const void *host_data, int64_t rows, int64_t cols,
                   int64_t row_stride, size_t elem_bytes) {
  DeviceBuffer buf;

  // An operand that lives on another machine cannot be staged from here.
  //
  // Falling back to the local path is safe exactly while every operand is
  // local. Once a dispatch has been routed to a worker, its results live there,
  // and a *later* dispatch that declines to route -- for any reason, including
  // a wire failure -- reaches this function holding a handle. The address is
  // minted non-canonical so the fault is immediate rather than silent, and it
  // is: a SIGSEGV inside cudaMemcpy2D, three frames below anything that names
  // the problem. This says it instead. It is still fatal, because there is no
  // correct way to continue, but the message identifies which side the memory
  // is on rather than leaving a backtrace through the driver.
  if (vx_remote_addr_is_handle((uint64_t)(uintptr_t)host_data)) {
    fprintf(stderr,
            "[Vx CUDA] FATAL: operand %p is a remote handle (worker %u), so "
            "this dispatch cannot run locally.\n"
            "          A dispatch declined to route while its operands were "
            "already on a worker (#348).\n",
            host_data, vx_remote_addr_worker((uint64_t)(uintptr_t)host_data));
    abort();
  }

  if (is_device_ptr(host_data)) {
    buf.ptr = const_cast<void *>(host_data);
    buf.row_stride = row_stride;
    buf.owned = false;
    return buf;
  }

  size_t width = (size_t)cols * elem_bytes;
  buf.bytes = (size_t)rows * width;
  buf.ptr = device_pool().acquire(buf.bytes);
  if (!buf.ptr) {
    fprintf(stderr,
            "[Vx CUDA] FATAL: out of device memory staging %zu bytes "
            "(%lld x %lld).\n",
            buf.bytes, (long long)rows, (long long)cols);
    abort();
  }
  buf.owned = true;
  buf.row_stride = cols;
  VX_CUDA_CHECK(cudaMemcpy2D(buf.ptr, width, host_data,
                             (size_t)row_stride * elem_bytes, width,
                             (size_t)rows, cudaMemcpyHostToDevice));
  return buf;
}

/// C = A * B on the device, all three row-major.
///
/// cuBLAS is column-major, and a row-major buffer read as column-major is its
/// own transpose. So computing C^T = B^T * A^T in cuBLAS terms -- swapping the
/// operands and the m/n extents, with no transpose flags -- leaves C correct
/// when read back row-major, and needs no repacking of any operand.
void gemm_on_device(int32_t dtype, int64_t m, int64_t n, int64_t k,
                    const DeviceBuffer &a, const DeviceBuffer &b, void *c_ptr,
                    int64_t c_row_stride) {
  cublasHandle_t handle = cublas_handle();
  int mi = (int)m, ni = (int)n, ki = (int)k;
  int lda = (int)a.row_stride, ldb = (int)b.row_stride, ldc = (int)c_row_stride;

  if (dtype == VX_DTYPE_F32) {
    const float alpha = 1.0f, beta = 0.0f;
    VX_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ni, mi, ki,
                                &alpha, (const float *)b.ptr, ldb,
                                (const float *)a.ptr, lda, &beta,
                                (float *)c_ptr, ldc));
    return;
  }

  if (dtype == VX_DTYPE_F64) {
    const double alpha = 1.0, beta = 0.0;
    VX_CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ni, mi, ki,
                                &alpha, (const double *)b.ptr, ldb,
                                (const double *)a.ptr, lda, &beta,
                                (double *)c_ptr, ldc));
    return;
  }

  // f16 and bf16 accumulate in f32: the products of a 16-bit GEMM overflow and
  // lose precision far sooner than the inputs do, and f32 accumulation is what
  // the outlined kernel this stands in for would effectively do.
  const float alpha = 1.0f, beta = 0.0f;
  cudaDataType_t ct = cuda_dtype(dtype);
  VX_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, ni, mi, ki,
                               &alpha, b.ptr, ct, ldb, a.ptr, ct, lda, &beta,
                               c_ptr, ct, ldc, CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT));
}

/// Run a decoded plan. Returns false if it could not be run at all, in which
/// case the caller falls back to the outlined kernel.
bool run_gemm(const vx_gemm_plan &plan) {
  if (!cuda_available()) {
    return false;
  }

  size_t esz = vx_dtype_bytes(plan.dtype);
  if (esz == 0) {
    return false;
  }

  if (verbose()) {
    fprintf(stderr, "[Vx CUDA] GEMM %lldx%lldx%lld %s -> %s\n",
            (long long)plan.m, (long long)plan.n, (long long)plan.k,
            vx_dtype_name(plan.dtype),
            plan.out_kind == VX_GEMM_OUT_SLOT ? "slot" : "buffer");
  }

  DeviceBuffer a = stage(plan.a_data, plan.m, plan.k, plan.a_row_stride, esz);
  DeviceBuffer b = stage(plan.b_data, plan.k, plan.n, plan.b_row_stride, esz);

  // The result lands in device memory first whatever its destination: writing
  // straight into host memory would need it to be pinned, which nothing here
  // guarantees.
  DeviceBuffer c;
  bool out_is_device =
      plan.out_kind == VX_GEMM_OUT_BUFFER && is_device_ptr(plan.out_data);
  if (out_is_device) {
    c.ptr = plan.out_data;
    c.row_stride = plan.out_row_stride;
    c.owned = false;
  } else {
    VX_CUDA_CHECK(cudaMalloc(&c.ptr, (size_t)plan.m * (size_t)plan.n * esz));
    c.owned = true;
    c.row_stride = plan.n;
  }

  gemm_on_device(plan.dtype, plan.m, plan.n, plan.k, a, b, c.ptr, c.row_stride);
  VX_CUDA_CHECK(cudaDeviceSynchronize());

  if (out_is_device) {
    return true;
  }

  size_t width = (size_t)plan.n * esz;
  if (plan.out_kind == VX_GEMM_OUT_SLOT) {
    // The kernel would have allocated its result and stored the descriptor
    // through the slot; standing in for it means doing both. The allocation
    // outlives this call by design -- the caller loads the descriptor out of
    // the slot and returns it -- and is freed on the same terms as the
    // kernel's own, which is to say by the program's allocator at exit.
    void *result = malloc((size_t)plan.m * width);
    if (!result) {
      fprintf(stderr, "[Vx CUDA] FATAL: out of memory for a %lldx%lld result\n",
              (long long)plan.m, (long long)plan.n);
      abort();
    }
    VX_CUDA_CHECK(cudaMemcpy(result, c.ptr, (size_t)plan.m * width,
                             cudaMemcpyDeviceToHost));
    vx_gemm_publish_slot(&plan, result);
    return true;
  }

  VX_CUDA_CHECK(cudaMemcpy2D(plan.out_data, (size_t)plan.out_row_stride * esz,
                             c.ptr, width, width, (size_t)plan.m,
                             cudaMemcpyDeviceToHost));
  return true;
}

} // namespace

extern "C" {

void *vx_plugin_alloc_and_transfer(size_t bytes, void *host_ptr,
                                   uint32_t topology_id) {
  /* Asked first, because a topology the manifest names is not this machine's
     to allocate in however capable this one is. */
  void *remote = nullptr;
  if (vx_routing_try_alloc(bytes, host_ptr, topology_id, &remote)) {
    return remote;
  }

  // A topology that names no device is host memory. The way home lowers to
  // exactly this call -- null source, topology 0 -- and its contract is
  // "allocate host memory and stop" (src/dialect/VxLowering.cpp). This path
  // used to fall through to `cudaMalloc` whenever a device existed, so the
  // "host" side of a fetch was device memory, the fetch quietly became a
  // device-to-device copy (unified addressing obliges), and the program
  // faulted the first time it looked at a result it had transferred back.
  // Invisible on a device-less box, which is the only place the way home had
  // run before Vx#377.
  if (!cuda_available() ||
      vx_topology_device_index((int32_t)topology_id, VX_TOPO_GPU_BASE) < 0) {
    void *ptr = malloc(bytes);
    if (ptr && host_ptr) {
      memcpy(ptr, host_ptr, bytes);
    }
    return ptr;
  }

  // The topology is which device's memory the caller asked for, and honouring
  // it is the whole meaning of the parameter. It was discarded, so a buffer
  // "staged onto GPU 1" landed on whichever device happened to be current --
  // accidentally right while there was one GPU, and wrong in the first
  // configuration where the answer mattered (#346).
  select_device((int32_t)topology_id, "stage");

  void *device_ptr = nullptr;
  VX_CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
  if (host_ptr) {
    VX_CUDA_CHECK(
        cudaMemcpy(device_ptr, host_ptr, bytes, cudaMemcpyHostToDevice));
  }
  return device_ptr;
}

void *vx_plugin_transfer_peer(void *src_device_ptr, uint32_t src_topology_id,
                              uint32_t dst_topology_id, size_t bytes) {
  void *routed = nullptr;
  if (vx_routing_try_peer(src_device_ptr, src_topology_id, dst_topology_id,
                          bytes, &routed)) {
    return routed;
  }
  // No device, or a buffer that never reached one: both topologies name the
  // same host memory, so the movement is a copy. Keeping this path means a
  // disaggregated program is still correct on a machine with no GPUs, which is
  // the property the rest of this ABI already has.
  if (!cuda_available() || !is_device_ptr(src_device_ptr)) {
    vx_routing_refuse_handle("a peer handoff", src_device_ptr, src_topology_id);
    void *dst = malloc(bytes);
    if (dst && src_device_ptr) {
      memcpy(dst, src_device_ptr, bytes);
    }
    return dst;
  }

  int src =
      vx_topology_device_index((int32_t)src_topology_id, VX_TOPO_GPU_BASE);
  int dst =
      vx_topology_device_index((int32_t)dst_topology_id, VX_TOPO_GPU_BASE);
  if (src < 0 || dst < 0) {
    // This entry point exists to move between two GPUs. A topology that is not
    // one has no device index to copy between, and guessing which device was
    // meant is exactly the class of silence this change removes.
    fprintf(stderr,
            "[Vx CUDA] FATAL: peer transfer between topologies %u and %u, "
            "which are not both GPUs\n",
            src_topology_id, dst_topology_id);
    abort();
  }

  // Allocate in the destination's memory, then copy into it. `cudaMemcpyPeer`
  // needs no peer access enabled and no P2P path to exist: where NVLink or a
  // PCIe switch route allows a direct copy it takes it, and otherwise it stages
  // through the host itself. One call is therefore correct on an NVLink pod and
  // on a PCIe-only one, which matters because a rented pod does not say which
  // it gave you.
  select_device((int32_t)dst_topology_id, "peer");
  void *dst_ptr = nullptr;
  VX_CUDA_CHECK(cudaMalloc(&dst_ptr, bytes));
  VX_CUDA_CHECK(cudaMemcpyPeer(dst_ptr, dst, src_device_ptr, src, bytes));

  if (verbose()) {
    fprintf(stderr, "[Vx CUDA] peer %d -> %d, %zu bytes\n", src, dst, bytes);
  }
  return dst_ptr;
}

/// Run the kernel the payload carries, on this device.
///
/// The compiler compiles every self-contained device region to PTX and puts it
/// in the dispatch payload (#251), so a region no vendor library can stand in
/// for is no longer a region this backend has to refuse. `kind=matmul` still
/// goes to cuBLAS -- it is measured faster than anything emitted here, and this
/// is only reached once that route has declined.
///
/// Returns false when there is no image or it cannot be used, which leaves the
/// caller's existing refusal in place rather than replacing one diagnostic with
/// a worse one.
bool run_device_image(const void *payload, size_t payload_size,
                      void **device_args, const int32_t *arg_tags,
                      int64_t num_args) {
  const char *image = vx_payload_field(payload, payload_size, "image=");
  const char *kernel_name = static_cast<const char *>(payload);
  if (!image || !*image || !cuda_available()) {
    return false;
  }

  /* A module belongs to a context, and this process may serve more than one
     device, so a loaded function is only reusable on the device it was loaded
     for. Keyed by both, because keying by name alone hands GPU 1 a function
     belonging to GPU 0's context -- which fails at launch rather than
     silently, but fails a long way from the cause. */
  static std::unordered_map<std::string, CUfunction> loaded;

  int device = 0;
  cudaGetDevice(&device);
  const std::string key =
      std::string(kernel_name) + "@" + std::to_string(device);

  CUfunction fn = nullptr;
  auto found = loaded.find(key);
  if (found != loaded.end()) {
    fn = found->second;
  } else {
    /* The driver API needs a current context. The runtime API creates the
       primary one lazily, and this is the idiom that forces it: without it
       cuModuleLoadData returns CUDA_ERROR_INVALID_CONTEXT on a dispatch that
       happens to be the first CUDA call of the process. */
    cudaFree(nullptr);

    CUmodule mod = nullptr;
    /* PTX, so the driver JITs it here -- once per process per device, which is
       why this is cached rather than done per dispatch. The cost buys an image
       that needs no `ptxas` anywhere and stays loadable on a newer device. */
    CUresult rc = cuModuleLoadData(&mod, image);
    if (rc != CUDA_SUCCESS) {
      const char *name = nullptr;
      cuGetErrorName(rc, &name);
      fprintf(stderr,
              "[Vx CUDA] FATAL: the device image for %s did not load: %s.\n"
              "          It was compiled for %s. Two known causes: an "
              "unresolved `__nv_` symbol\n"
              "          (libdevice was not linked -- set VX_LIBDEVICE), or "
              "static `.shared`\n"
              "          storage over the 48 KiB ceiling -- a cooperative "
              "kernel's SMEM tiles\n"
              "          must fit it (Vx#379; sum the .with_memory tensors' "
              "bytes).\n",
              kernel_name, name ? name : "?",
              vx_payload_field(payload, payload_size, "chip=")
                  ? vx_payload_field(payload, payload_size, "chip=")
                  : "sm_80");
      abort();
    }
    rc = cuModuleGetFunction(&fn, mod, kernel_name);
    if (rc != CUDA_SUCCESS) {
      fprintf(stderr,
              "[Vx CUDA] FATAL: the device image loaded but has no entry "
              "named %s.\n",
              kernel_name);
      abort();
    }
    loaded.emplace(key, fn);
  }

  vx_launch_params params;
  if (!vx_launch_build_params(device_args, arg_tags, num_args, &params)) {
    /* Not fatal: a slot output is the one argument shape that cannot be
       marshalled, and that is a known gap rather than a broken program. The
       caller's refusal says so. */
    if (verbose()) {
      fprintf(stderr,
              "[Vx CUDA] %s has an argument this path cannot marshal (a "
              "publication slot); not launching\n",
              kernel_name);
    }
    return false;
  }

  /* The signature is right here in the image, so there is no reason to launch
     on faith. `cuLaunchKernel` cannot check a parameter array against the
     kernel it launches: too few, and the kernel reads parameters that were
     never written, which is a wrong answer rather than an error. */
  const int declared = vx_launch_entry_param_count(image, kernel_name);
  if (declared != params.count) {
    fprintf(stderr,
            "[Vx CUDA] FATAL: %s declares %d parameters and its arguments "
            "produced %d.\n"
            "          Launching would read past what was supplied. The "
            "compiler's idea of\n"
            "          this kernel's signature and the runtime's have "
            "diverged (#251).\n",
            kernel_name, declared, params.count);
    abort();
  }

  /* One thread by default. Nothing in a serial region indexes by thread --
     the outliner produces a serial loop nest -- so a wider launch would run
     the whole computation once per thread over the same output and race.

     `launch=` changes that: it is the trip count of a loop the frontend
     proved disjoint and the device pipeline grid-strided (#251). A strided
     kernel is correct under ANY configuration -- each thread walks
     gtid, gtid+stride, ... -- so the field is a sizing hint with a
     correctness floor. One iteration per thread up to a block of 128, then
     enough blocks to cover the rest; the grid stride absorbs any remainder
     and any cap. */
  unsigned grid = 1, block = 1;
  if (const char *launch =
          vx_payload_field(payload, payload_size, "launch=")) {
    char *rest = nullptr;
    long trip = strtol(launch, &rest, 10);
    if (rest && *rest == ',') {
      /* Two-level "B,T" (Vx#379): B is a block count -- the block loop
         strides by gridDim, so capping it is correct, not lossy -- and T is
         the block shape the compiler suggested. */
      long t = strtol(rest + 1, nullptr, 10);
      if (trip > 0 && t > 0) {
        grid = trip > 4096 ? 4096u : (unsigned)trip;
        block = t > 1024 ? 1024u : (unsigned)t;
      }
    } else if (trip > 1) {
      block = trip < 128 ? (unsigned)trip : 128u;
      unsigned long need = ((unsigned long)trip + block - 1) / block;
      grid = need > 4096 ? 4096u : (unsigned)need;
    }
  }
  /* Device-side timing, on request. Wall clock cannot see this kernel any
     more: at 128 blocks the flash sweep's whole device time is ~50 ms under
     ~300 ms of rented-pod host jitter, and one outlier fit a NEGATIVE
     per-K slope. CUDA events are stamped by the device, so they are immune
     to everything the host does between enqueue and sync. */
  const bool time_kernel = getenv("VX_TIME_KERNEL") != nullptr;
  cudaEvent_t t0 = nullptr, t1 = nullptr;
  if (time_kernel) {
    /* The occupancy the DRIVER computed for this function as it will actually
       launch -- not what ptxas said offline. The 108-vs-109-block cliff on the
       A100 said one block per SM; this is the number that says whether that is
       the driver's decision or contention, without a profiler (rented pods
       refuse the performance counters: ERR_NVGPUCTRPERM). */
    int resident = 0;
    if (cuOccupancyMaxActiveBlocksPerMultiprocessor(&resident, fn, (int)block,
                                                    0) == CUDA_SUCCESS) {
      fprintf(stderr,
              "[Vx CUDA] %s occupancy: %d block(s) of %u threads per SM\n",
              kernel_name, resident, block);
    }
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
  }
  CUresult rc = cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr,
                               params.params, nullptr);
  if (rc != CUDA_SUCCESS) {
    const char *name = nullptr;
    cuGetErrorName(rc, &name);
    fprintf(stderr, "[Vx CUDA] FATAL: launching %s failed: %s\n", kernel_name,
            name ? name : "?");
    abort();
  }
  const cudaError_t sync = cudaDeviceSynchronize();
  if (sync != cudaSuccess) {
    fprintf(stderr, "[Vx CUDA] FATAL: %s faulted on the device: %s\n",
            kernel_name, cudaGetErrorString(sync));
    abort();
  }

  if (time_kernel) {
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    /* Always printed when asked for, independent of verbose(): asking to time
       the kernel IS asking for this line, and a harness greps for it. */
    fprintf(stderr, "[Vx CUDA] %s device time %.3f ms (%ux%u threads)\n",
            kernel_name, ms, grid, block);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
  }

  if (verbose()) {
    fprintf(stderr,
            "[Vx CUDA] %s ran on GPU %d from its own image "
            "(%d params, %ux%u threads)\n",
            kernel_name, device, params.count, grid, block);
  }
  return true;
}

uint64_t vx_plugin_dispatch_async(const void *binary_payload,
                                  size_t payload_size, void **device_args,
                                  const int32_t *arg_tags, int64_t num_args) {
  const char *kernel_name = static_cast<const char *>(binary_payload);

  if (vx_routing_try_dispatch(binary_payload, payload_size, device_args,
                              arg_tags, num_args)) {
    return 1;
  }

  vx_gemm_plan plan;
  if (vx_gemm_plan_decode(binary_payload, payload_size, device_args, arg_tags,
                          num_args, &plan)) {
    if (cuda_available()) {
      select_device(vx_payload_topology(binary_payload, payload_size),
                    "dispatch");
    }
    if (run_gemm(plan)) {
      return 1;
    }
  } else if (vx_payload_field(binary_payload, payload_size, "image=")) {
    /* Not a matmul, but the compiler emitted a kernel for it. This is the case
       the header of this file said had to run on the host and could not. */
    if (cuda_available()) {
      select_device(vx_payload_topology(binary_payload, payload_size),
                    "dispatch");
    }
    if (run_device_image(binary_payload, payload_size, device_args, arg_tags,
                         num_args)) {
      return 1;
    }
  } else if (verbose()) {
    const char *kind = vx_payload_field(binary_payload, payload_size, "kind=");
    fprintf(stderr, "[Vx CUDA] %s (kind=%s) not routed; running on the host\n",
            kernel_name, kind ? kind : "<unclassified>");
  }

  // A kernel that is not routed runs here, on the host. That is safe exactly
  // while its operands are host memory -- and `transfer(x, Memory::GPU_HBM)`
  // is precisely the construct that makes them not be.
  //
  // The header of this file claims a refusal to route "costs performance and
  // never correctness". That is true for a program whose tensors live in host
  // memory and false for one that declares a memory hierarchy and moves into
  // it. tests/backend/pass/flash_attention_placed.vx does the second: it
  // transfers Q, K, V and O into GPU_HBM and then runs a fused online-softmax
  // loop, which is not a GEMM and is not classified as one. On a machine
  // without this plugin the transfer is a no-op, the pointers stay host
  // pointers, and the test passes -- which is why it passes in CI. On a machine
  // with a GPU the loop dereferences device memory and the process dies inside
  // the outlined kernel, three frames below anything that names the reason.
  //
  // Copying the operands back, running, and copying them out again would make
  // the fallback honest, and is the right fix. Until then this refuses in a way
  // that says what happened, because a segmentation fault inside
  // vx_npu_kernel_0 does not.
  for (int64_t i = 0; i < num_args; ++i) {
    if (VX_ABI_KIND(arg_tags[i]) != VX_ABI_KIND_MEMREF || !device_args[i]) {
      continue;
    }
    const void *desc = *(const void **)device_args[i];
    if (!desc) {
      continue;
    }
    if (VX_ABI_IS_SLOT(arg_tags[i])) {
      desc = (const void *)vx_memref_aligned(desc);
      if (!desc) {
        continue;
      }
    }
    const void *data = vx_memref_data(desc, (int32_t)VX_ABI_ELEM(arg_tags[i]));
    if (data && is_device_ptr(data)) {
      fprintf(stderr,
              "[Vx CUDA] FATAL: %s was not routed, so it would run on the "
              "host,\n"
              "          but argument %lld at %p is in device memory -- a "
              "`transfer`\n"
              "          into a device memory space put it there. The host "
              "cannot read it.\n"
              "          Either the kernel needs a device implementation "
              "(#251), or its\n"
              "          operands must stay in host memory.\n",
              kernel_name, (long long)i, data);
      abort();
    }
  }

  vx_host_refuse_coop(binary_payload, payload_size, kernel_name, "CUDA");
  void *kernel = vx_host_kernel_symbol(kernel_name);
  if (!kernel) {
    fprintf(stderr, "[Vx CUDA] FATAL: outlined kernel %s not found\n",
            kernel_name);
    abort();
  }
  if (!vx_host_call_kernel(kernel, device_args, arg_tags, num_args)) {
    fprintf(stderr, "[Vx CUDA] FATAL: could not build a call for %s\n",
            kernel_name);
    abort();
  }
  return 1;
}

uint64_t vx_plugin_dispatch_async_flat(float *xout, float *x, float *w, int n,
                                       int d) {
  // The flat path's matvec: xout[i] = sum_j w[i * n + j] * x[j] (#31). Read
  // column-major, the row-major w is its own transpose, so one transposed GEMV
  // computes it with no repacking.
  if (!cuda_available() || n <= 0 || d <= 0) {
    for (int i = 0; i < d; ++i) {
      float acc = 0.0f;
      for (int j = 0; j < n; ++j) {
        acc += w[i * n + j] * x[j];
      }
      xout[i] = acc;
    }
    return 1;
  }

  const float alpha = 1.0f, beta = 0.0f;
  DeviceBuffer dw = stage(w, d, n, n, sizeof(float));
  DeviceBuffer dx = stage(x, 1, n, n, sizeof(float));

  DeviceBuffer dy;
  bool out_is_device = is_device_ptr(xout);
  if (out_is_device) {
    dy.ptr = xout;
  } else {
    VX_CUDA_CHECK(cudaMalloc(&dy.ptr, (size_t)d * sizeof(float)));
    dy.owned = true;
  }

  VX_CUBLAS_CHECK(cublasSgemv(cublas_handle(), CUBLAS_OP_T, n, d, &alpha,
                              (const float *)dw.ptr, n, (const float *)dx.ptr,
                              1, &beta, (float *)dy.ptr, 1));
  VX_CUDA_CHECK(cudaDeviceSynchronize());

  if (!out_is_device) {
    VX_CUDA_CHECK(cudaMemcpy(xout, dy.ptr, (size_t)d * sizeof(float),
                             cudaMemcpyDeviceToHost));
  }
  return 1;
}

void vx_plugin_await_future(uint64_t future_id) {
  // Dispatch synchronizes before returning, so the result is ready already.
  // Overlapping it with host work is what the future ID is for and is part of
  // the same milestone as device residency (#321).
  (void)future_id;
}

int32_t vx_plugin_transfer_device_to_host(void *device_ptr, void *host_ptr,
                                          size_t bytes, uint32_t topology_id) {
  if (vx_routing_try_fetch(device_ptr, host_ptr, bytes, topology_id)) {
    return 1;
  }
  // Before `is_device_ptr`, which answers "no" for a handle -- the driver has
  // never seen that address -- and would send this to the memcpy below.
  vx_routing_refuse_handle("a read-back", device_ptr, topology_id);
  if (is_device_ptr(device_ptr)) {
    // Unified addressing lets the driver infer the device from the pointer, so
    // this would mostly work without the parameter. It names the device anyway,
    // because the ABI is the contract a non-CUDA backend implements and nothing
    // guarantees that backend can infer anything from an address (#346).
    select_device((int32_t)topology_id, "fetch");
    VX_CUDA_CHECK(
        cudaMemcpy(host_ptr, device_ptr, bytes, cudaMemcpyDeviceToHost));
  } else {
    memcpy(host_ptr, device_ptr, bytes);
  }
  return 1;
}

void vx_plugin_free(void *device_ptr, uint32_t topology_id) {
  if (!device_ptr) {
    return;
  }
  if (vx_routing_try_free(device_ptr, topology_id)) {
    return;
  }
  vx_routing_refuse_handle("a free", device_ptr, topology_id);
  if (is_device_ptr(device_ptr)) {
    select_device((int32_t)topology_id, "free");
    cudaFree(device_ptr);
  } else {
    free(device_ptr);
  }
}

void vx_plugin_release_future(uint64_t future_id) { (void)future_id; }

int32_t vx_plugin_control(uint32_t opcode, void *payload) {
  (void)payload;
  switch (opcode) {
  case VX_CTRL_GET_DEVICE_COUNT: {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
      return 0;
    }
    return count;
  }
  case VX_CTRL_INIT_DEVICE:
    return cuda_available() ? 1 : 0;
  case VX_CTRL_SHUTDOWN:
    if (cuda_available()) {
      cudaDeviceReset();
    }
    return 1;
  default:
    return 0;
  }
}

} // extern "C"
