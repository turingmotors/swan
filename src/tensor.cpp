#include "tensor.hpp"

#include <cmath>
#include <fstream>

namespace swan {

/* ---------------------------------  /
              Copy Tensor
/  --------------------------------- */

// Copies the contents from the source 1D tensor to the destination 1D tensor.
void CopyTensor1d(Tensor1d& dst, const Tensor1d& src) {
  for (size_t i = 0; i < kDim; i++) {
    dst[i] = src[i];
  }
}

// Copies the contents from the source 2D tensor to the destination 2D tensor.
void CopyTensor2d(Tensor2d& dst, const Tensor2d& src) {
  for (size_t i = 0; i < kDim; i++) {
    CopyTensor1d(dst[i], src[i]);
  }
}

// Copies the contents from the source 3D tensor to the destination 3D tensor.
void CopyTensor3d(Tensor3d& dst, const Tensor3d& src) {
  for (size_t i = 0; i < kDim; i++) {
    CopyTensor2d(dst[i], src[i]);
  }
}

/* ---------------------------------  /
      Basic Arithmetic Operations
/  --------------------------------- */

// Add a scalar to each element of the input tensor.
void Add(Tensor1d& out, const Tensor1d& in, float a) {
  for (size_t i = 0; i < kDim; i++) {
    out[i] = in[i] + a;
  }
}

// Subtract a scalar from each element of the input tensor.
void Sub(Tensor1d& out, const Tensor1d& in, float a) {
  for (size_t i = 0; i < kDim; i++) {
    out[i] = in[i] - a;
  }
}

// Multiply each element of the input tensor by a scalar.
void Mul(Tensor1dQKSM& out, const Tensor1dQKSM& in, float a) {
  for (size_t i = 0; i < kSeqLen; i++) {
    out[i] = in[i] * a;
  }
}

// Divide each element of the input tensor by a scalar.
void Div(Tensor1d& out, const Tensor1d& in, float a) {
  for (size_t i = 0; i < kDim; i++) {
    out[i] = in[i] / a;
  }
}

// Add each element of the first input tensor to the corresponding element of
// the second input tensor.
void Add(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs) {
  for (size_t i = 0; i < kDim; ++i) {
    out[i] = lhs[i] + rhs[i];
  }
}

// Subtract each element of the second input tensor from the corresponding
// element of the first input tensor.
void Sub(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs) {
  for (size_t i = 0; i < kDim; ++i) {
    out[i] = lhs[i] - rhs[i];
  }
}

// Multiply each element of the first input tensor by the corresponding element
// of the second input tensor.
void Mul(Tensor1dFFNB& out, const Tensor1dFFNB& lhs, const Tensor1dFFNB& rhs) {
  for (size_t i = 0; i < kFFNDim; ++i) {
    out[i] = lhs[i] * rhs[i];
  }
}

// Divide each element of the first input tensor by the corresponding element of
// the second input tensor.
void Div(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs) {
  for (size_t i = 0; i < kDim; ++i) {
    out[i] = lhs[i] / rhs[i];
  }
}

/* ---------------------------------  /
           Matrix Operations
/  --------------------------------- */

// Compute the inner product of two input tensors.
// out = Sum_i (lhs[i] . rhs[i])
float InnerProduct(const Tensor1d& lhs, const Tensor1d& rhs) {
  float sum = 0;
  for (size_t i = 0; i < kDim; ++i) {
    sum += lhs[i] * rhs[i];
  }
  return sum;
}

// Compute the matrix multiplication of two input tensors.
// Tensor1d [dim] . Tensor2dAttn [dim, dim] = Tensor1d [dim]
// out[i] = w[i,j] . in[j]
void Matmul(Tensor1d& out, const Tensor1d& in, const Tensor2dAttn& w) {
  for (size_t i = 0; i < kDim; ++i) {
    float sum = 0;
    for (size_t j = 0; j < kDim; j++) {
      sum += w[i][j] * in[j];
    }
    out[i] = sum;
  }
}

// Compute the matrix multiplication of two input tensors.
// Tensor1dFFNB [ffn_dim] . Tensor2dFFNA [ffn_dim, dim] = Tensor1dFFNB [ffn_dim]
// out[i] = w[i,j] . in[j]
void Matmul(Tensor1dFFNB& out, const Tensor1d& in, const Tensor2dFFNA& w) {
  for (size_t i = 0; i < kFFNDim; ++i) {
    float sum = 0;
    for (size_t j = 0; j < kDim; j++) {
      sum += w[i][j] * in[j];
    }
    out[i] = sum;
  }
}

// Compute the matrix multiplication of two input tensors.
// Tensor1d [dim] . Tensor2dFFNB [dim, ffn_dim] = Tensor1dFFNB [ffn_dim]
// out[i] = w[i,j] . in[j]
void Matmul(Tensor1d& out, const Tensor1dFFNB& in, const Tensor2dFFNB& w) {
  for (size_t i = 0; i < kDim; ++i) {
    float sum = 0;
    for (size_t j = 0; j < kFFNDim; j++) {
      sum += w[i][j] * in[j];
    }
    out[i] = sum;
  }
}

// Compute the matrix multiplication of two input tensors.
// Tensor1d [dim] . Tensor2dTok [vocab_size, dim] = Tensor1dLogits [vocab_size]
// out[i] = w[i,j] . in[j]
void MutmulVocab(Tensor1dLogits& out, const Tensor1d& in, const Tensor2dTok& w) {
  float x[kDim];
  float w_vec[kDim];
  float in_tmp[kDim];
  for (size_t i = 0; i < kVocabSize; ++i) {
    float sum = 0;
    for (size_t j = 0; j < kDim; j++) {
      w_vec[j] = w[i][j];
      in_tmp[j] = in[j];
    }
    for (size_t j = 0; j < kDim; j++) {
      x[j] = w_vec[j] * in_tmp[j];
    }
    for (size_t j = 0; j < kDim; j++) {
      sum += x[j];
    }
    out[i] = sum;
  }
}

// Compute the matrix multiplication of two input tensors.
// Tensor1dQKSM [seq_len] . Tensor2dCache [seq_len, dim] = Tensor1dQKSM [seq_len]
// out[i] = w[i,j] . in[j]
// i = out_begin..out_end
// j = j_begin..j_end
void MutmulRanged(Tensor1dQKSM& out, const Tensor1d& in, const Tensor2dCache& w,
                  int out_begin, int out_end, int in_begin, int in_end) {
  for (size_t i = out_begin; i < out_end; ++i) {
    float sum = 0;
    for (size_t j = in_begin; j < in_end; ++j) {
      sum += w[i][j] * in[j];
    }
    out[i] = sum;
  }
}

// Compute the matrix multiplication of two input tensors.
// Tensor1dQKSM [seq_len] . Tensor2dCache [seq_len, dim] = Tensor1dQKSM [seq_len]
// out[i] = w[j,i] . in[j] <- transpose w
// i = out_begin..out_end
// j = j_begin..j_end
void MutmulRangedTranspose(Tensor1d& out, const Tensor1dQKSM& in, const Tensor2dCache& w,
                           int i_begin, int i_end, int j_begin, int j_end) {
  for (size_t i = i_begin; i < i_end; ++i) {
    float sum = 0;
    for (size_t j = j_begin; j < j_end; ++j) {
      // reverse the order of w's index (w[j][i] -> w[i][j])
      sum += w[j][i] * in[j];
    }
    out[i] = sum;
  }
}

/* ---------------------------------  /
         Activation Functions
/  --------------------------------- */

inline float ReLU(float x) {
  return x > 0 ? x : 0;
}

// Apply the ReLU activation function to each element of the input tensor.
// out[i] = max(0, in[i])
void ReLU(Tensor1d& out, const Tensor1d& in) {
  for (size_t i = 0; i < kDim; ++i) {
    out[i] = ReLU(in[i]);
  }
}

inline float SiLU(float x) {
  return x * (1.0 / (1.0 + std::exp(-x)));
}

// Apply the SiLU activation function to each element of the input tensor.
// out[i] = in[i] * (1 / (1 + exp(-in[i])))
void SiLU(Tensor1dFFNB& out, const Tensor1dFFNB& in) {
  for (size_t i = 0; i < kFFNDim; ++i) {
    out[i] = SiLU(in[i]);
  }
}

/* ---------------------------------  /
      Normalization Operations
/  --------------------------------- */

// Apply the RMS normalization to the input tensor.
// norm = 1 / sum_i..N (in[i]^2) / N
// out[i] = x[i] * norm * w[i]
void RMSNorm(Tensor1d& out, const Tensor1d& in, const Tensor1d& w) {
  // 1. Summation of Square
  float sum = 0.0;
  for (size_t i = 0; i < kDim; i++) {
#pragma HLS UNROLL
    sum += in[i] * in[i];
  }

  // 2. Normalize Factor
  //    Add small number to avoid "zero dividing error"
  constexpr float eps = 1e-5;
  const float norm = 1 / std::sqrt(sum / kDim + eps);

  // 3. Normalize and Scale with Weight
  for (size_t i = 0; i < kDim; i++) {
    out[i] = in[i] * norm * w[i];
  }
}

// Apply the softmax function to the input tensor.
// out[i] = exp(in[i]) / sum(exp(in[i]))
void Softmax(Tensor1dQKSM& out, const Tensor1dQKSM& in, int in_max_idx) {
  if (in_max_idx == -1) {
    in_max_idx = kSeqLen;
  }

  // 1. Get Max
  float max_val = in[0];
  for (int i = 1; i < in_max_idx; i++) {
    if (in[i] > max_val) {
      max_val = in[i];
    }
  }

  // 2. Exp and Sum
  float sum = 0;
  for (int i = 0; i < in_max_idx; i++) {
    out[i] = std::exp(in[i] - max_val);
    sum += out[i];
  }

  // 3. Normalize
  for (int i = 0; i < in_max_idx; i++) {
    out[i] /= sum;
  }
}

// Apply the softmax function to the input tensor.
// out[i] = exp(in[i]) / sum(exp(in[i]))
void Softmax(Tensor1dLogits& out, const Tensor1dLogits& in, int in_max_idx) {
  if (in_max_idx == -1) {
    in_max_idx = kVocabSize;
  }

  // 1. Get Max
  float max_val = in[0];
  for (int i = 1; i < in_max_idx; i++) {
    if (in[i] > max_val) {
      max_val = in[i];
    }
  }

  // 2. Exp and Sum
  float sum = 0;
  for (int i = 0; i < in_max_idx; i++) {
    out[i] = std::exp(in[i] - max_val);
    sum += out[i];
  }

  // 3. Normalize
  for (int i = 0; i < in_max_idx; i++) {
    out[i] /= sum;
  }
}

/* ---------------------------------  /
      Argmax and Max Operations
/  --------------------------------- */

// Return the index and value of the maximum value in the input tensor.
// max_i = argmax_i (in[i])
// max_val = max_i (in[i])
std::pair<int, float> FindMaxIndexAndValue(const Tensor1d& in) {
  std::pair<int, float> max(0, in[0]);
  for (int i = 1; i < kDim; i++) {
    if (in[i] > max.second) {
      max.first = i;
      max.second = in[i];
    }
  }
  return max;
}

// Return the sum of the input tensor.
// max_val = max_i (in[i])
float Max(const Tensor1d& in) {
  float max_val = in[0];
  for (int i = 1; i < kDim; i++) {
    if (in[i] > max_val) {
      max_val = in[i];
    }
  }
  return max_val;
}

// Return the index of the maximum value in the input tensor.
// max_i = argmax_i (in[i])
int Argmax(const Tensor1dLogits& in) {
  int max_i = 0;
  float max_value = in[0];
  for (size_t i = 1; i < kVocabSize; ++i)
    if (in[i] > max_value) {
      max_i = i;
      max_value = in[i];
    }
  return max_i;
}

/* ---------------------------------  /
      RoPE: Position Encoding
/  --------------------------------- */

// Apply the rotary position encoding to the input tensor.
// q_out[i] = q_in[i] * cos_vec[i] - q_in[i+1] * sin_vec[i]
// q_out[i+1] = q_in[i] * sin_vec[i] + q_in[i+1] * cos_vec[i]
// k_out[i] = k_in[i] * cos_vec[i] - k_in[i+1] * sin_vec[i]
// k_out[i+1] = k_in[i] * sin_vec[i] + k_in[i+1] * cos_vec[i]
void RoPE(Tensor1d& q_out, Tensor1d& k_out, const Tensor1d& q_in, const Tensor1d& k_in,
          const Tensor1dSinCos& cos_vec, const Tensor1dSinCos& sin_vec, int head_begin,
          int head_dim) {
  for (int i = 0; i < kHalvedHeadDim; ++i) {
    int i0 = head_begin + i * 2 + 0;
    int i1 = head_begin + i * 2 + 1;

    float q0 = q_in[i0];
    float q1 = q_in[i1];

    float k0 = k_in[i0];
    float k1 = k_in[i1];

    float cos = cos_vec[i];
    float sin = sin_vec[i];

    q_out[i0] = q0 * cos - q1 * sin;
    q_out[i1] = q0 * sin + q1 * cos;

    k_out[i0] = k0 * cos - k1 * sin;
    k_out[i1] = k0 * sin + k1 * cos;
  }
}

} // namespace swan
