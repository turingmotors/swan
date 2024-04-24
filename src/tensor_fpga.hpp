#ifndef TENSOR_FPGA_HPP_
#define TENSOR_FPGA_HPP_

#ifndef USE_CPU_ONLY

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>

#include "tensor.hpp"

namespace swan {

void AddFPGA(Tensor1d& out, const Tensor1d& in, float a, cl::CommandQueue q,
             cl::Kernel kernel_add, float* ptr_a, float* ptr_b,
             float* ptr_result, cl::Buffer buffer_a, cl::Buffer buffer_b,
             cl::Buffer buffer_result);
void MulFPGA(Tensor1dQKSM& out, const Tensor1dQKSM& in, float a,
             cl::CommandQueue q, cl::Kernel kernel_mul, float* ptr_a,
             float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result);

void AddFPGA(Tensor1d& out, const Tensor1d& lhs, const Tensor1d& rhs,
             cl::CommandQueue q, cl::Kernel kernel_add, float* ptr_a,
             float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result);
void MulFPGA(Tensor1dFFNB& out, const Tensor1dFFNB& lhs,
             const Tensor1dFFNB& rhs, cl::CommandQueue q, cl::Kernel kernel_mul,
             float* ptr_a, float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
             cl::Buffer buffer_b, cl::Buffer buffer_result);

void MatmulFPGA(Tensor1d& out, const Tensor1d& in, const Tensor2dAttn& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result);
void MatmulFPGA(Tensor1dFFNB& out, const Tensor1d& in, const Tensor2dFFNA& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result);
void MatmulFPGA(Tensor1d& out, const Tensor1dFFNB& in, const Tensor2dFFNB& w,
                cl::CommandQueue q, cl::Kernel kernel_matmul, float* ptr_a,
                float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                cl::Buffer buffer_b, cl::Buffer buffer_result);

void RMSNormFPGA(Tensor1d& out, const Tensor1d& in, const Tensor1d& w,
                 cl::CommandQueue q, cl::Kernel kernel_rmsnorm, float* ptr_a,
                 float* ptr_b, float* ptr_result, cl::Buffer buffer_a,
                 cl::Buffer buffer_b, cl::Buffer buffer_result);
void SoftmaxFPGA(Tensor1dQKSM& out, const Tensor1dQKSM& in, int max_pos,
                 cl::CommandQueue q, cl::Kernel kernel_softmax, float* ptr_a,
                 float* ptr_result, cl::Buffer buffer_a,
                 cl::Buffer buffer_result);

void RoPEFPGA(Tensor1d& q_out, Tensor1d& k_out, const Tensor1d& q_in,
              const Tensor1d& k_in, const Tensor1dSinCos& cos_vec,
              const Tensor1dSinCos& sin_vec, int head_begin, int head_size,
              cl::CommandQueue q, cl::Kernel kernel_rope, float* ptr_a,
              float* ptr_b, float* ptr_c, float* ptr_d, float* ptr_result,
              float* ptr_result2, cl::Buffer buffer_a, cl::Buffer buffer_b,
              cl::Buffer buffer_c, cl::Buffer buffer_d,
              cl::Buffer buffer_result, cl::Buffer buffer_result2);

} // namespace swan

#endif // USE_CPU_ONLY

#endif // TENSOR_FPGA_HPP_
