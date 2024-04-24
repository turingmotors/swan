#include "weight.hpp"

namespace swan {

// Implement for initializing the tensor from the file.
// Template specialization for 1D tensor.
template <typename T, size_t Size>
void InitTensor(T (&tensor)[Size], std::ifstream& fs) {
  fs.read(reinterpret_cast<char*>(tensor), Size * sizeof(T));
}

// Implement for initializing the tensor from the file.
// Template specialization for 2D tensor.
template <typename T, size_t Outer, size_t Inner>
void InitTensor(T (&tensor)[Outer][Inner], std::ifstream& fs) {
  for (size_t outer = 0; outer < Outer; outer++) {
    InitTensor(tensor[outer], fs);
  }
}

// Implement for initializing the tensor from the file.
// Template specialization for 3D tensor.
template <typename T, size_t Outer, size_t Middle, size_t Inner>
void InitTensor(T (&tensor)[Outer][Middle][Inner], std::ifstream& fs) {
  for (size_t outer = 0; outer < Outer; outer++) {
    InitTensor(tensor[outer], fs);
  }
}

// Implement for initializing the tensor from the file.
void LoadWeights(Weights& w, Tensor2dTok& tok_emb_table, std::ifstream& fs) {
  fs.read(w.dummy, sizeof(w.dummy)); // Dummy variable for alignment
  InitTensor(tok_emb_table, fs);     // [kVocabSize, kDim]
  InitTensor(w.rms_att_w, fs);       // [kNumLayers, kDim]
  InitTensor(w.attn_wq, fs);         // [kNumLayers, kDim, kDim]
  InitTensor(w.attn_wk, fs);         // [kNumLayers, kDim, kDim]
  InitTensor(w.attn_wv, fs);         // [kNumLayers, kDim, kDim]
  InitTensor(w.attn_wo, fs);         // [kNumLayers, kDim, kDim]
  InitTensor(w.rms_ffn_w, fs);       // [kNumLayers, kDim]
  InitTensor(w.ffn_w1, fs);          // [kNumLayers, kFFNDim, kDim]
  InitTensor(w.ffn_w2, fs);          // [kNumLayers, kDim, kFFNDim]
  InitTensor(w.ffn_w3, fs);          // [kNumLayers, kFFNDim, kDim]
  InitTensor(w.rms_final, fs);       // [kDim]
  InitTensor(w.cos_table, fs);       // [kSeqLen, kSinCosTable]
  InitTensor(w.sin_table, fs);       // [kSeqLen, kSinCosTable]
}

} // namespace swan
