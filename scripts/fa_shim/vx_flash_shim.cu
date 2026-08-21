// vx_flash_shim.cu - call FlashAttention-2's forward kernel with no torch anywhere.
//
// The params struct is plain data (flash.h); this fills exactly the fields
// flash_api.cpp's set_params_fprop + mha_fwd fill for the non-causal,
// no-dropout, fixed-length, fp16, hdim<=64 case, and calls the sm_80
// instantiation directly. Layout contract: Q/K/V/O are (b, seqlen, h, d)
// row-major contiguous, which for b=1,h=1 is exactly a Vx [S, D] tensor.
#include "namespace_config.h"
#include <cutlass/numeric_types.h>
#include "flash.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cmath>

extern "C" int vx_flash_fwd_f16_hd64(void *q, void *k, void *v, void *o,
                                     void *lse /* b*h*sq floats, device */,
                                     int b, int h, int sq, int sk, float scale,
                                     cudaStream_t stream) {
  using FLASH_NAMESPACE::Flash_fwd_params;
  Flash_fwd_params p;
  memset(&p, 0, sizeof(p));
  const int d = 64;
  p.is_bf16 = false;
  p.q_ptr = q; p.k_ptr = k; p.v_ptr = v; p.o_ptr = o;
  p.q_row_stride = (int64_t)h * d;
  p.k_row_stride = (int64_t)h * d;
  p.v_row_stride = (int64_t)h * d;
  p.o_row_stride = (int64_t)h * d;
  p.q_head_stride = d; p.k_head_stride = d; p.v_head_stride = d; p.o_head_stride = d;
  p.q_batch_stride = (int64_t)sq * h * d;
  p.k_batch_stride = (int64_t)sk * h * d;
  p.v_batch_stride = (int64_t)sk * h * d;
  p.o_batch_stride = (int64_t)sq * h * d;
  p.cu_seqlens_q = nullptr; p.cu_seqlens_k = nullptr; p.seqused_k = nullptr;
  p.p_ptr = nullptr;
  p.softmax_lse_ptr = lse;
  p.b = b; p.h = h; p.h_k = h; p.h_h_k_ratio = 1;
  p.seqlen_q = sq; p.seqlen_k = sk;
  p.seqlen_q_rounded = (sq + 127) / 128 * 128;
  p.seqlen_k_rounded = (sk + 127) / 128 * 128;
  p.d = d; p.d_rounded = d;
  p.scale_softmax = scale;
  p.scale_softmax_log2 = scale * (float)M_LOG2E;
  p.softcap = 0.0f;
  p.p_dropout = 1.0f;                 // stored as KEEP probability
  p.p_dropout_in_uint8_t = 255;
  p.rp_dropout = 1.0f;
  p.scale_softmax_rp_dropout = scale;
  p.is_causal = false;
  p.window_size_left = -1; p.window_size_right = -1;
  p.is_seqlens_k_cumulative = true;
  p.num_splits = 0;
  p.alibi_slopes_ptr = nullptr;
  p.unpadded_lse = false;
  p.seqlenq_ngroups_swapped = false;
  FLASH_NAMESPACE::run_mha_fwd_<cutlass::half_t, 64, false>(p, stream);
  return (int)cudaPeekAtLastError();
}
