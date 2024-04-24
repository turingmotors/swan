#ifndef WEIGHT_HPP_
#define WEIGHT_HPP_

#include <fstream>
#include <string>

#include "tensor.hpp"

namespace swan {

struct Weights {
  // Dummy variable for alignment
  char dummy[28];

  // Attention
  Tensor2dRMS rms_att_w; // [n_layers, dim]
  Tensor3dAttn attn_wq;  // [n_layers, dim, dim]
  Tensor3dAttn attn_wk;  // [n_layers, dim, dim]
  Tensor3dAttn attn_wv;  // [n_layers, dim, dim]
  Tensor3dAttn attn_wo;  // [n_layers, dim, dim]

  // FFN
  Tensor2dRMS rms_ffn_w; // [n_layers, dim]
  Tensor3dFFNA ffn_w1;   // [n_layers, ffn_dim, dim]
  Tensor3dFFNB ffn_w2;   // [n_layers, dim, ffn_dim]
  Tensor3dFFNA ffn_w3;   // [n_layers, ffn_dim, dim]

  // Final rmsnorm
  Tensor1d rms_final; // [dim]

  // freq_cis for RoPE relatively positional embeddings
  Tensor2dSinCos cos_table; // [seq_len, (dim/n_heads)/2]
  Tensor2dSinCos sin_table; // [seq_len, (dim/n_heads)/2]
};

void LoadWeights(Weights& w, Tensor2dTok& tok, std::ifstream& fs);

} // namespace swan

#endif // WEIGHT_HPP_
