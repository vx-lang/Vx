// vx_flash_cudnn.cpp - the same vx_flash_fwd_f16_hd64 ABI, backed by cuDNN's
// fused flash SDPA through cudnn-frontend (header-only, v1.x graph API).
//
// A second provider for Vx's kind=attention route (Vx#378): the runtime
// dlopens whatever VX_FLASH_LIB names and calls one exported symbol, so which
// vendor computes the softmax(QK^T)V is a deployment decision, not a compiler
// one. Against the FlashAttention-2 shim (vx_flash_shim.cu) this trades a
// from-source build of another project's kernels for a library that ships
// with every CUDA installation and is tuned per-arch by NVIDIA.
//
// The graph (build_operation_graph + heuristics + plan compile) costs
// milliseconds, so plans are cached by shape; the steady-state call is one
// graph->execute. The scale is baked into the plan (it is part of the cache
// key), matching how the compiler emits it: a constant in the region or a
// captured scalar that is fixed for a given program point.
//
// Layout contract, identical to the FA-2 shim: Q/K/V/O are (b, s, h, d)
// row-major contiguous -- which for b=1, h=1 is exactly a Vx [S, D] tensor.
// In cuDNN's (B, H, S, D) dim order that memory is strides
// {H*S*D, D, H*D, 1}. `lse` is accepted and ignored: this ABI's consumers
// are inference dispatches, and cuDNN only materializes stats when asked.
//
// Build (cuDNN 8.9.3+ at runtime; tested against 9.8.0 on an A100):
//   g++ -O3 -std=c++17 -fPIC -shared \
//     -I $CUDNN_FRONTEND/include -I /usr/local/cuda/include \
//     vx_flash_cudnn.cpp -o libvx_flash_cudnn.so \
//     -L /usr/local/cuda/lib64 -lcudnn -lcudart

#include <cudnn_frontend.h>

#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>

namespace fe = cudnn_frontend;

namespace {

constexpr fe::graph::Tensor_attributes::uid_t Q_UID = 1;
constexpr fe::graph::Tensor_attributes::uid_t K_UID = 2;
constexpr fe::graph::Tensor_attributes::uid_t V_UID = 3;
constexpr fe::graph::Tensor_attributes::uid_t O_UID = 4;

struct Plan {
  std::shared_ptr<fe::graph::Graph> graph;
  void *workspace = nullptr;
  int64_t workspace_size = 0;
};

cudnnHandle_t handle() {
  static cudnnHandle_t h = [] {
    cudnnHandle_t hh = nullptr;
    cudnnCreate(&hh);
    return hh;
  }();
  return h;
}

Plan *plan_for(int b, int h, int sq, int sk, float scale) {
  static std::map<std::tuple<int, int, int, int, float>, std::unique_ptr<Plan>>
      cache;
  auto key = std::make_tuple(b, h, sq, sk, scale);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second.get();

  const int64_t B = b, H = h, SQ = sq, SK = sk, D = 64;

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  auto Q = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("Q")
                             .set_uid(Q_UID)
                             .set_dim({B, H, SQ, D})
                             .set_stride({H * SQ * D, D, H * D, 1}));
  auto K = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("K")
                             .set_uid(K_UID)
                             .set_dim({B, H, SK, D})
                             .set_stride({H * SK * D, D, H * D, 1}));
  auto V = graph->tensor(fe::graph::Tensor_attributes()
                             .set_name("V")
                             .set_uid(V_UID)
                             .set_dim({B, H, SK, D})
                             .set_stride({H * SK * D, D, H * D, 1}));

  auto sdpa_options = fe::graph::SDPA_attributes()
                          .set_name("vx_flash")
                          .set_generate_stats(false)
                          .set_attn_scale(scale);

  auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);
  (void)Stats;
  O->set_output(true)
      .set_dim({B, H, SQ, D})
      .set_stride({H * SQ * D, D, H * D, 1})
      .set_uid(O_UID);

  if (!graph->build(handle(), {fe::HeurMode_t::A}).is_good())
    return nullptr;

  auto p = std::make_unique<Plan>();
  p->graph = graph;
  if (!graph->get_workspace_size(p->workspace_size).is_good())
    return nullptr;
  if (p->workspace_size > 0 &&
      cudaMalloc(&p->workspace, (size_t)p->workspace_size) != cudaSuccess)
    return nullptr;

  Plan *ret = p.get();
  cache[key] = std::move(p);
  return ret;
}

} // namespace

extern "C" int vx_flash_fwd_f16_hd64(void *q, void *k, void *v, void *o,
                                     void *lse, int b, int h, int sq, int sk,
                                     float scale, cudaStream_t stream) {
  (void)lse; // inference-only ABI; cuDNN materializes no stats unasked
  Plan *p = plan_for(b, h, sq, sk, scale);
  if (!p)
    return 1;
  cudnnSetStream(handle(), stream);
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> pack = {
      {Q_UID, q}, {K_UID, k}, {V_UID, v}, {O_UID, o}};
  return p->graph->execute(handle(), pack, p->workspace).is_good() ? 0 : 1;
}
