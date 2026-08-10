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
// Anything not recognised runs on the host through libffi, exactly as the
// portable shim would. A refusal to route therefore costs performance and
// never correctness, which is what makes it safe for the decoder to be strict.
//
// Scope: operands are staged to the device per dispatch and the result staged
// back. Keeping tensors resident across dispatches is what `.to_device()` and
// the placement analysis exist for, and is the next milestone (#321); the
// allocation entry points below already return device pointers, so a resident
// buffer arrives here as one and is used in place.
//
//===----------------------------------------------------------------------===//

#include "vx_dispatch_plan.h"
#include "vx_host_call.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include <cublas_v2.h>
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
bool select_device(int32_t topology_id) {
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
    fprintf(stderr, "[Vx CUDA] device %d\n", index);
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

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : ptr(other.ptr), row_stride(other.row_stride), owned(other.owned) {
    other.ptr = nullptr;
    other.owned = false;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      if (owned && ptr) {
        cudaFree(ptr);
      }
      ptr = other.ptr;
      row_stride = other.row_stride;
      owned = other.owned;
      other.ptr = nullptr;
      other.owned = false;
    }
    return *this;
  }

  ~DeviceBuffer() {
    if (owned && ptr) {
      cudaFree(ptr);
    }
  }
};

/// Make `rows x cols` elements available on the device. A device pointer is
/// used where it lies; a host one is copied into a packed buffer, so the device
/// row stride is `cols` whatever the host's padding was.
DeviceBuffer stage(const void *host_data, int64_t rows, int64_t cols,
                   int64_t row_stride, size_t elem_bytes) {
  DeviceBuffer buf;
  if (is_device_ptr(host_data)) {
    buf.ptr = const_cast<void *>(host_data);
    buf.row_stride = row_stride;
    buf.owned = false;
    return buf;
  }

  size_t width = (size_t)cols * elem_bytes;
  VX_CUDA_CHECK(cudaMalloc(&buf.ptr, (size_t)rows * width));
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
  if (!cuda_available()) {
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
  select_device((int32_t)topology_id);

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
  // No device, or a buffer that never reached one: both topologies name the
  // same host memory, so the movement is a copy. Keeping this path means a
  // disaggregated program is still correct on a machine with no GPUs, which is
  // the property the rest of this ABI already has.
  if (!cuda_available() || !is_device_ptr(src_device_ptr)) {
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
  select_device((int32_t)dst_topology_id);
  void *dst_ptr = nullptr;
  VX_CUDA_CHECK(cudaMalloc(&dst_ptr, bytes));
  VX_CUDA_CHECK(cudaMemcpyPeer(dst_ptr, dst, src_device_ptr, src, bytes));

  if (verbose()) {
    fprintf(stderr, "[Vx CUDA] peer %d -> %d, %zu bytes\n", src, dst, bytes);
  }
  return dst_ptr;
}

uint64_t vx_plugin_dispatch_async(const void *binary_payload,
                                  size_t payload_size, void **device_args,
                                  const int32_t *arg_tags, int64_t num_args) {
  const char *kernel_name = static_cast<const char *>(binary_payload);

  vx_gemm_plan plan;
  if (vx_gemm_plan_decode(binary_payload, payload_size, device_args, arg_tags,
                          num_args, &plan)) {
    if (cuda_available()) {
      select_device(vx_payload_topology(binary_payload, payload_size));
    }
    if (run_gemm(plan)) {
      return 1;
    }
  } else if (verbose()) {
    const char *kind = vx_payload_field(binary_payload, payload_size, "kind=");
    fprintf(stderr, "[Vx CUDA] %s (kind=%s) not routed; running on the host\n",
            kernel_name, kind ? kind : "<unclassified>");
  }

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
  if (is_device_ptr(device_ptr)) {
    // Unified addressing lets the driver infer the device from the pointer, so
    // this would mostly work without the parameter. It names the device anyway,
    // because the ABI is the contract a non-CUDA backend implements and nothing
    // guarantees that backend can infer anything from an address (#346).
    select_device((int32_t)topology_id);
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
  if (is_device_ptr(device_ptr)) {
    select_device((int32_t)topology_id);
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
