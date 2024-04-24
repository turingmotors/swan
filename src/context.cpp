#include "context.hpp"

#include <fstream>

namespace swan {

void DumpTensor1d(std::string file, const Tensor1d& tensor) {
  std::ofstream fs(file);
  for (int i = 0; i < kDim; ++i) {
    fs << tensor[i] << std::endl;
  }
  fs.close();
}

void DumpTensor1dFFNB(std::string file, const Tensor1dFFNB& tensor) {
  std::ofstream fs(file);
  for (int i = 0; i < kFFNDim; ++i) {
    fs << tensor[i] << std::endl;
  }
  fs.close();
}

// Dump the context to files.
// The files are named as prefix + field_name.
void DumpContext(std::string prefix, const Context& ctx, int n_layers) {
  DumpTensor1d(prefix + "input", ctx.input);
  for (int layer = 0; layer < n_layers; ++layer) {
    DumpTensor1d(prefix + std::to_string(layer) + "_attn_norm",
                 ctx.attn_norm[layer]);
    DumpTensor1d(prefix + std::to_string(layer) + "_attn_wqx",
                 ctx.attn_wqx[layer]);
    DumpTensor1d(prefix + std::to_string(layer) + "_attn_wkx",
                 ctx.attn_wkx[layer]);
    DumpTensor1d(prefix + std::to_string(layer) + "_attn_wvx",
                 ctx.attn_wvx[layer]);
    DumpTensor1d(prefix + std::to_string(layer) + "_attn_q_rope",
                 ctx.attn_q_r[layer]);
    DumpTensor1d(prefix + std::to_string(layer) + "_attn_k_rope",
                 ctx.attn_k_r[layer]);
    DumpTensor1d(prefix + std::to_string(layer) + "_attn_val",
                 ctx.attn_val[layer]);
    DumpTensor1d(prefix + std::to_string(layer) + "_attn_out",
                 ctx.attn_out[layer]);
    DumpTensor1d(prefix + std::to_string(layer) + "_attn_res",
                 ctx.attn_res[layer]);
    DumpTensor1d(prefix + std::to_string(layer) + "_ffn_norm",
                 ctx.ffn_norm[layer]);
    DumpTensor1dFFNB(prefix + std::to_string(layer) + "_ffn_w1x",
                     ctx.ffn_w1x[layer]);
    DumpTensor1dFFNB(prefix + std::to_string(layer) + "_ffn_w3x",
                     ctx.ffn_w3x[layer]);
    DumpTensor1dFFNB(prefix + std::to_string(layer) + "_ffn_act",
                     ctx.ffn_act[layer]);
    DumpTensor1dFFNB(prefix + std::to_string(layer) + "_ffn_dot",
                     ctx.ffn_dot[layer]);
    DumpTensor1d(prefix + std::to_string(layer) + "_ffn_out",
                 ctx.ffn_out[layer]);
    DumpTensor1d(prefix + std::to_string(layer) + "_ffn_res",
                 ctx.ffn_res[layer]);
  }
  // DumpTensor1d(prefix + "final_norm", ctx.final_norm);
  // DumpTensor1d(prefix + "logits", ctx.logits);
}

} // namespace swan
