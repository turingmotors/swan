#ifndef CONTEXT_HPP_
#define CONTEXT_HPP_

#include <string>

#include "tensor.hpp"

namespace swan {

struct Context {
  // Input
  Tensor1d input; // [dim]

  // Attention
  Tensor2dRMS attn_norm; // [layer, dim]
  Tensor2dRMS attn_wqx;  // [layer, dim]
  Tensor2dRMS attn_wkx;  // [layer, dim]
  Tensor2dRMS attn_wvx;  // [layer, dim]
  Tensor2dRMS attn_q_r;  // [layer, dim]
  Tensor2dRMS attn_k_r;  // [layer, dim]
  Tensor2dQKSM attn_qk;  // [layer, seq_len]
  Tensor2dQKSM attn_sm;  // [layer, seq_len]
  Tensor2dRMS attn_val;  // [layer, dim]
  Tensor2dRMS attn_out;  // [layer, dim]
  Tensor2dRMS attn_res;  // [layer, dim]

  // FFN
  Tensor2dRMS ffn_norm; // [layer, dim]
  Tensor2dFFNC ffn_w1x; // [layer, ffn_dim]
  Tensor2dFFNC ffn_w3x; // [layer, ffn_dim]
  Tensor2dFFNC ffn_act; // [layer, ffn_dim]
  Tensor2dFFNC ffn_dot; // [layer, ffn_dim]
  Tensor2dRMS ffn_out;  // [layer, dim]
  Tensor2dRMS ffn_res;  // [layer, dim]

  // Output
  // Tensor1d final_norm;    // [dim]
  // Tensor1dLogits logits;  // [vocab_size]

  // Cache
  // Tensor3dCache k_cache;  // [layer, seq_len, dim]
  // Tensor3dCache v_cache;  // [layer, seq_len, dim]
};

void DumpContext(std::string prefix, const Context& ctx, int n_layers);

} // namespace swan

#endif // CONTEXT_HPP_
