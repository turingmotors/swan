#ifndef DECODE_HPP_
#define DECODE_HPP_

#include "context.hpp"
#include "weight.hpp"

#ifndef USE_CPU_ONLY
#include "tensor_fpga.hpp"

#include <CL/cl2.hpp>
#endif // USE_CPU_ONLY

namespace swan {

void Decode(int tok, int pos, const Tensor1d& ctx_input,
            Tensor3dCache& ctx_k_cache, Tensor3dCache& ctx_v_cache,
            Tensor1d& ctx_final_norm, const Weights& w
#ifndef USE_CPU_ONLY
            ,
            cl::CommandQueue q, cl::Kernel kernel_matmul, cl::Kernel kernel_mul,
            cl::Kernel kernel_rmsnorm, cl::Kernel kernel_softmax,
            cl::Kernel kernel_add, cl::Kernel kernel_rope, float* ptr_a,
            float* ptr_b, float* ptr_c, float* ptr_d, float* ptr_result,
            float* ptr_result2, cl::Buffer buffer_a, cl::Buffer buffer_b,
            cl::Buffer buffer_c, cl::Buffer buffer_d, cl::Buffer buffer_result,
            cl::Buffer buffer_result2
#endif // USE_CPU_ONLY
);

} // namespace swan

#endif // DECODE_HPP_
