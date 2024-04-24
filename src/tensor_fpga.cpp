#include "tensor_fpga.hpp"

#ifndef USE_CPU_ONLY

namespace swan {

/* ---------------------------------  /
      Basic Arithmetic Operations
/  --------------------------------- */

// Add a scalar to each element of the input tensor.
void AddFPGA(Tensor1d& out, const Tensor1d& in, float a, cl::CommandQueue q,
             cl::Kernel kernel_add, float* ptr_a, float* ptr_b,
             float* ptr_result, cl::Buffer buffer_a, cl::Buffer buffer_b,
             cl::Buffer buffer_result) {
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kDim; i++) {
    ptr_b[i] = a;
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_add.setArg(3, kDim);
  q.enqueueTask(kernel_add);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kDim; i++) {
    out[i] = ptr_result[i];
  }
}

// Multiply each element of the input tensor by a scalar.
void MulFPGA(Tensor1dQKSM& out, const Tensor1dQKSM& in, float a,
             cl::CommandQueue q, cl::Kernel kernel_mul, float* ptr_a,
             float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kSeqLen; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kSeqLen; i++) {
    ptr_b[i] = a;
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_mul.setArg(3, kSeqLen);
  q.enqueueTask(kernel_mul);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kSeqLen; i++) {
    out[i] = ptr_result[i];
  }
}

// Add each element of the first input tensor to the corresponding element of
// the second input tensor.
void AddFPGA(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs,
             cl::CommandQueue q, cl::Kernel kernel_add, float* ptr_a,
             float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = lhs[i];
  }
  for (int i = 0; i < kDim; i++) {
    ptr_b[i] = rhs[i];
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_add.setArg(3, kDim);
  q.enqueueTask(kernel_add);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kDim; i++) {
    out[i] = ptr_result[i];
  }
}

// Multiply each element of the first input tensor by the corresponding element
// of the second input tensor.
void MulFPGA(Tensor1dFFNB& out, const Tensor1dFFNB& lhs,
             const Tensor1dFFNB& rhs, cl::CommandQueue q, cl::Kernel kernel_mul,
             float* ptr_a, float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kFFNDim; i++) {
    ptr_a[i] = lhs[i];
  }
  for (int i = 0; i < kFFNDim; i++) {
    ptr_b[i] = rhs[i];
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_mul.setArg(3, kFFNDim);
  q.enqueueTask(kernel_mul);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kFFNDim; i++) {
    out[i] = ptr_result[i];
  }
}

/* ---------------------------------  /
           Matrix Operations
/  --------------------------------- */

// Compute the matrix multiplication of two input tensors.
// Tensor1d [dim] . Tensor2dAttn [dim, dim] = Tensor1d [dim]
// out[i] = w[i,j] . in[j]
void MatmulFPGA(Tensor1d& out, const Tensor1d& in, const Tensor2dAttn& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kDim; i++) {
    for (int j = 0; j < kDim; j++) {
      ptr_b[i * kDim + j] = w[i][j];
    }
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_matmul.setArg(3, kDim);
  kernel_matmul.setArg(4, kDim);
  q.enqueueTask(kernel_matmul);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kDim; i++) {
    out[i] = ptr_result[i];
  }
}

// Compute the matrix multiplication of two input tensors.
// Tensor1dFFNB [ffn_dim] . Tensor2dFFNA [ffn_dim, dim] = Tensor1dFFNB [ffn_dim]
// out[i] = w[i,j] . in[j]
void MatmulFPGA(Tensor1dFFNB& out, const Tensor1d& in, const Tensor2dFFNA& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kFFNDim; i++) {
    for (int j = 0; j < kDim; j++) {
      ptr_b[i * kDim + j] = w[i][j];
    }
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_matmul.setArg(3, kDim);
  kernel_matmul.setArg(4, kFFNDim);
  q.enqueueTask(kernel_matmul);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kFFNDim; i++) {
    out[i] = ptr_result[i];
  }
}

// Compute the matrix multiplication of two input tensors.
// Tensor1d [dim] . Tensor2dFFNB [dim, ffn_dim] = Tensor1dFFNB [ffn_dim]
// out[i] = w[i,j] . in[j]
void MatmulFPGA(Tensor1d& out, const Tensor1dFFNB& in, const Tensor2dFFNB& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kFFNDim; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kDim; i++) {
    for (int j = 0; j < kFFNDim; j++) {
      ptr_b[i * kFFNDim + j] = w[i][j];
    }
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_matmul.setArg(3, kFFNDim);
  kernel_matmul.setArg(4, kDim);
  q.enqueueTask(kernel_matmul);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kDim; i++) {
    out[i] = ptr_result[i];
  }
}

/* ---------------------------------  /
      Normalization Operations
/  --------------------------------- */

// Apply the RMS normalization to the input tensor.
// norm = 1 / sum_i..N (in[i]^2) / N
// out[i] = x[i] * norm * w[i]
void RMSNormFPGA(Tensor1d& out, const Tensor1d& in, const Tensor1d& w,
                 cl::CommandQueue q, cl::Kernel kernel_rmsnorm, float* ptr_a,
                 float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                 cl::Buffer buffer_b, cl::Buffer buffer_result) {
  for (int i = 0; i < kDim; i++) {
    ptr_a[i] = in[i];
  }
  for (int i = 0; i < kDim; i++) {
    ptr_b[i] = w[i];
  }
  q.enqueueMigrateMemObjects({buffer_a, buffer_b}, 0);
  kernel_rmsnorm.setArg(3, kDim);
  q.enqueueTask(kernel_rmsnorm);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < kDim; i++) {
    out[i] = ptr_result[i];
  }
}

// Apply the softmax function to the input tensor.
// out[i] = exp(in[i]) / sum(exp(in[i]))
void SoftmaxFPGA(Tensor1dQKSM& out, const Tensor1dQKSM& in, int in_max_idx,
                 cl::CommandQueue q, cl::Kernel kernel_softmax, float* ptr_a,
                 float* ptr_result, cl::Buffer buffer_a,
                 cl::Buffer buffer_result) {
  if (in_max_idx == -1) {
    in_max_idx = kSeqLen;
  }

  for (int i = 0; i < in_max_idx; i++) {
    ptr_a[i] = in[i];
  }
  q.enqueueMigrateMemObjects({buffer_a}, 0);
  kernel_softmax.setArg(2, in_max_idx);
  q.enqueueTask(kernel_softmax);
  q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();
  for (int i = 0; i < in_max_idx; i++) {
    out[i] = ptr_result[i];
  }
}

/* ---------------------------------  /
      RoPE: Position Encoding
/  --------------------------------- */

// Apply the rotary position encoding to the input tensor.
// q_out[i] = q_in[i] * cos_vec[i] - q_in[i+1] * sin_vec[i]
// q_out[i+1] = q_in[i] * sin_vec[i] + q_in[i+1] * cos_vec[i]
// k_out[i] = k_in[i] * cos_vec[i] - k_in[i+1] * sin_vec[i]
// k_out[i+1] = k_in[i] * sin_vec[i] + k_in[i+1] * cos_vec[i]
void RoPEFPGA(Tensor1d& q_out, Tensor1d& k_out, const Tensor1d& q_in,
              const Tensor1d& k_in, const Tensor1dSinCos& cos_vec,
              const Tensor1dSinCos& sin_vec, int head_begin, int head_dim,
              cl::CommandQueue q, cl::Kernel kernel_rope, float* ptr_a,
              float* ptr_b, float* ptr_c, float* ptr_d, float* ptr_result,
              float* ptr_result2, cl::Buffer buffer_a, cl::Buffer buffer_b,
              cl::Buffer buffer_c, cl::Buffer buffer_d,
              cl::Buffer buffer_result, cl::Buffer buffer_result2) {

  for (int i = 0; i < 288; i++) {
    ptr_a[i] = q_in[i];
    ptr_b[i] = k_in[i];
  }

  for (int i = 0; i < 24; i++) {
    ptr_c[i] = cos_vec[i];
    ptr_d[i] = sin_vec[i];
  }

  q.enqueueMigrateMemObjects({buffer_a, buffer_b, buffer_c, buffer_d}, 0);
  kernel_rope.setArg(6, head_begin);
  q.enqueueTask(kernel_rope);
  q.enqueueMigrateMemObjects({buffer_result, buffer_result2},
                             CL_MIGRATE_MEM_OBJECT_HOST);
  q.finish();

  for (int i = 0; i < 288; i++) {
    q_out[i] = ptr_result[i];
    k_out[i] = ptr_result2[i];
  }
}

} // namespace swan

#endif // USE_CPU_ONLY
