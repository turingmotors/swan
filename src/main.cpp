#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>

#include "context.hpp"
#include "decode.hpp"
#include "vocab.hpp"
#include "weight.hpp"

#ifndef USE_CPU_ONLY
#include <stdlib.h>

#include <CL/cl2.hpp>

#define OCL_CHECK(error, call)                                             \
  call;                                                                    \
  if (error != CL_SUCCESS) {                                               \
    printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, \
           __LINE__, error);                                               \
    exit(EXIT_FAILURE);                                                    \
  }

static const int MAX_DATA_SIZE = 1024;

template <typename T>
struct aligned_allocator {
  using value_type = T;
  T* allocate(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num) { free(p); }
};
#endif // USE_CPU_ONLY

// Command line arguments.
struct Args {
  std::string weight_path = "./model/stories15M.bin";
  std::string vocab_path = "./model/tokenizer.bin";
  uint64_t max_seq = 256;
  float temp = 0.5;
  bool color = false;
  bool print_softmax = false;
  bool log = false;
  bool help = false;
};

// Parse the command line arguments.
void ParseArgument(int argc, char* argv[], Args& args) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--weight_path") == 0 && i + 1 < argc) {
      args.weight_path = argv[++i];
    } else if (std::strcmp(argv[i], "--vocab_path") == 0 && i + 1 < argc) {
      args.vocab_path = argv[++i];
    } else if (std::strcmp(argv[i], "--max_seq") == 0 && i + 1 < argc) {
      args.max_seq = std::stoull(argv[++i]);
    } else if (std::strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
      args.temp = std::stof(argv[++i]);
    } else if (std::strcmp(argv[i], "--color") == 0) {
      args.color = true;
    } else if (std::strcmp(argv[i], "--print_softmax") == 0) {
      args.print_softmax = true;
    } else if (std::strcmp(argv[i], "--log") == 0) {
      args.log = true;
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      args.help = true;
    } else {
      std::cerr << "[ERROR] Unknown Option: " << argv[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

// Random Sampling
int SelectFromLogits(const swan::Tensor1dLogits& prob_dist) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  const int vocab_size = swan::kVocabSize;
  float rand = dis(gen);

  float cdf = 0.0;
  for (size_t i = 0; i < vocab_size; ++i) {
    cdf += prob_dist[i];
    if (rand < cdf) {
      return i;
    }
  }

  // in case of rounding errors
  return vocab_size - 1;
}

int main(int argc, char* argv[]) {
  // 1. Parse arguments.
  Args args;
  ParseArgument(argc, argv, args);

  if (args.help) {
    std::cout << "Usage: " << argv[0] << " [options]" << std::endl
              << "Options:" << std::endl
              << "  --weight_path   : Weight file path" << std::endl
              << "  --vocab_path    : Tokenizer file path" << std::endl
              << "  --max_seq       : Maximum sequence length" << std::endl
              << "  --temp          : Temperature for sampling" << std::endl
              << "  --color         : Enable color output" << std::endl
              << "  --log           : Enable log output" << std::endl
              << "  --help, -h      : Show this help message" << std::endl;
    return 0;
  }

  // 2. Print hyper parameters.
  std::cout << "Hyper Parameters" << std::endl
            << "  dim       : " << swan::kDim << std::endl
            << "  ffn_dim   : " << swan::kFFNDim << std::endl
            << "  n_layers  : " << swan::kNumLayers << std::endl
            << "  n_heads   : " << swan::kNumHeads << std::endl
            << "  n_kv_heads: " << swan::kNumKVHeads << std::endl
            << "  vocab_size: " << swan::kVocabSize << std::endl
            << "  seq_len   : " << swan::kSeqLen << std::endl;

  // 3. Load model parameters.
  std::ifstream weight_fs(args.weight_path, std::ios::in | std::ios::binary);
  if (!weight_fs) {
    std::cout << "Failed to open: " << args.weight_path << std::endl;
    return EXIT_FAILURE;
  }
  static swan::Weights weights;
  static swan::Tensor2dTok tok_emb_table; // [vocab_size, dim]
  swan::LoadWeights(weights, tok_emb_table, weight_fs);
  weight_fs.close();

  // 4. Load vocabrary.
  std::ifstream vocab_fs(args.vocab_path, std::ios::in | std::ios::binary);
  if (!vocab_fs) {
    std::cout << "Failed to open: " << args.vocab_path << std::endl;
    return EXIT_FAILURE;
  }
  static swan::Vocab vocab;
  const int vocab_size = swan::kVocabSize;
  swan::ResizeVocab(vocab, vocab_size);
  swan::LoadVocab(vocab, vocab_fs);
  vocab_fs.close();

#ifndef USE_CPU_ONLY
  // 5. OpenCL Settings
  std::string xclbinFilename = "./binary_container_1.bin";

  size_t size_in_bytes = MAX_DATA_SIZE * sizeof(float);

  std::vector<cl::Device> devices;
  cl_int err;
  cl::Context context;
  cl::CommandQueue q;
  cl::Kernel kernel_matmul;
  cl::Kernel kernel_mul;
  cl::Kernel kernel_rmsnorm;
  cl::Kernel kernel_softmax;
  cl::Kernel kernel_add;
  cl::Kernel kernel_rope;
  cl::Program program;
  std::vector<cl::Platform> platforms;
  bool found_device = false;

  // traversing all Platforms To find Xilinx Platform and targeted
  // Device in Xilinx Platform
  cl::Platform::get(&platforms);
  for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
    cl::Platform platform = platforms[i];
    std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
    if (platformName == "Xilinx") {
      devices.clear();
      platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
      if (devices.size()) {
        found_device = true;
        break;
      }
    }
  }
  if (found_device == false) {
    std::cout << "Error: Unable to find Target Device " << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "INFO: Reading " << xclbinFilename << std::endl;
  FILE* fp;
  if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
    printf("ERROR: %s xclbin not available please build\n",
           xclbinFilename.c_str());
    exit(EXIT_FAILURE);
  }
  // Load xclbin
  std::cout << "Loading: '" << xclbinFilename << "'\n";
  std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
  bin_file.seekg(0, bin_file.end);
  unsigned nb = bin_file.tellg();
  bin_file.seekg(0, bin_file.beg);
  char* buf = new char[nb];
  bin_file.read(buf, nb);

  // Creating Program from Binary File
  cl::Program::Binaries bins;
  bins.push_back({buf, nb});
  bool valid_device = false;
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err,
              context = cl::Context(device, nullptr, nullptr, nullptr, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                        CL_QUEUE_PROFILING_ENABLE, &err));
    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(context, {device}, bins, nullptr, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      OCL_CHECK(err,
                kernel_matmul = cl::Kernel(program, "kernel_matmul", &err));
      std::cout << "load : kernel_matmul" << std::endl;
      OCL_CHECK(err, kernel_mul = cl::Kernel(program, "kernel_mul", &err));
      std::cout << "load : kernel_mul" << std::endl;
      OCL_CHECK(err,
                kernel_rmsnorm = cl::Kernel(program, "kernel_rmsnorm", &err));
      std::cout << "load : kernel_rmsnorm" << std::endl;
      OCL_CHECK(err,
                kernel_softmax = cl::Kernel(program, "kernel_softmax", &err));
      std::cout << "load : kernel_softmax" << std::endl;
      OCL_CHECK(err, kernel_add = cl::Kernel(program, "kernel_add", &err));
      std::cout << "load : kernel_add" << std::endl;
      OCL_CHECK(err, kernel_rope = cl::Kernel(program, "kernel_rope", &err));
      std::cout << "load : kernel_rope" << std::endl;
      valid_device = true;
      break; // we break because we found a valid device
    }
  }
  if (!valid_device) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }

  // These commands will allocate memory on the Device. The cl::Buffer objects
  // can be used to reference the memory locations on the device.
  OCL_CHECK(err, cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, size_in_bytes,
                                     NULL, &err));
  OCL_CHECK(err,
            cl::Buffer buffer_b(context, CL_MEM_READ_ONLY,
                                size_in_bytes * size_in_bytes, NULL, &err));
  OCL_CHECK(err, cl::Buffer buffer_c(context, CL_MEM_READ_ONLY, size_in_bytes,
                                     NULL, &err));
  OCL_CHECK(err, cl::Buffer buffer_d(context, CL_MEM_READ_ONLY, size_in_bytes,
                                     NULL, &err));
  OCL_CHECK(err, cl::Buffer buffer_result(context, CL_MEM_WRITE_ONLY,
                                          size_in_bytes, NULL, &err));
  OCL_CHECK(err, cl::Buffer buffer_result2(context, CL_MEM_WRITE_ONLY,
                                           size_in_bytes, NULL, &err));

  // set the kernel Arguments
  OCL_CHECK(err, err = kernel_rmsnorm.setArg(0, buffer_a));
  OCL_CHECK(err, err = kernel_rmsnorm.setArg(1, buffer_b));
  OCL_CHECK(err, err = kernel_rmsnorm.setArg(2, buffer_result));

  OCL_CHECK(err, err = kernel_matmul.setArg(0, buffer_a));
  OCL_CHECK(err, err = kernel_matmul.setArg(1, buffer_b));
  OCL_CHECK(err, err = kernel_matmul.setArg(2, buffer_result));
  OCL_CHECK(err, err = kernel_matmul.setArg(3, swan::kDim));
  OCL_CHECK(err, err = kernel_matmul.setArg(4, swan::kDim));

  OCL_CHECK(err, err = kernel_mul.setArg(0, buffer_a));
  OCL_CHECK(err, err = kernel_mul.setArg(1, buffer_b));
  OCL_CHECK(err, err = kernel_mul.setArg(2, buffer_result));

  OCL_CHECK(err, err = kernel_add.setArg(0, buffer_a));
  OCL_CHECK(err, err = kernel_add.setArg(1, buffer_b));
  OCL_CHECK(err, err = kernel_add.setArg(2, buffer_result));

  OCL_CHECK(err, err = kernel_softmax.setArg(0, buffer_a));
  OCL_CHECK(err, err = kernel_softmax.setArg(1, buffer_result));

  OCL_CHECK(err, err = kernel_rope.setArg(0, buffer_a));
  OCL_CHECK(err, err = kernel_rope.setArg(1, buffer_b));
  OCL_CHECK(err, err = kernel_rope.setArg(2, buffer_c));
  OCL_CHECK(err, err = kernel_rope.setArg(3, buffer_d));
  OCL_CHECK(err, err = kernel_rope.setArg(4, buffer_result));
  OCL_CHECK(err, err = kernel_rope.setArg(5, buffer_result2));
  OCL_CHECK(err, err = kernel_rope.setArg(6, 1));

  // We then need to map our OpenCL buffer5 to get the pointers
  float* ptr_a;
  float* ptr_b;
  float* ptr_c;
  float* ptr_d;
  float* ptr_result;
  float* ptr_result2;
  OCL_CHECK(err, ptr_a = (float*)q.enqueueMapBuffer(
                     buffer_a, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes, NULL,
                     NULL, &err));
  OCL_CHECK(err, ptr_b = (float*)q.enqueueMapBuffer(
                     buffer_b, CL_TRUE, CL_MAP_WRITE, 0,
                     size_in_bytes * size_in_bytes, NULL, NULL, &err));
  OCL_CHECK(err, ptr_c = (float*)q.enqueueMapBuffer(
                     buffer_c, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes, NULL,
                     NULL, &err));
  OCL_CHECK(err, ptr_d = (float*)q.enqueueMapBuffer(
                     buffer_d, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes, NULL,
                     NULL, &err));
  OCL_CHECK(err, ptr_result = (float*)q.enqueueMapBuffer(
                     buffer_result, CL_TRUE, CL_MAP_READ, 0, size_in_bytes,
                     NULL, NULL, &err));
  OCL_CHECK(err, ptr_result2 = (float*)q.enqueueMapBuffer(
                     buffer_result2, CL_TRUE, CL_MAP_READ, 0, size_in_bytes,
                     NULL, NULL, &err));
#endif // USE_CPU_ONLY

  // 6. Decode
  static swan::Context ctx;
  swan::Tensor1d ctx_input;
  static swan::Tensor3dCache ctx_k_cache;
  static swan::Tensor3dCache ctx_v_cache;
  swan::Tensor1dLogits ctx_logits;
  swan::Tensor1d ctx_final_norm;

  clock_t start_clk = clock();

  int next;
  int token = 1; // BOS (Begin of Sequence)

  for (int pos = 0; pos < args.max_seq; ++pos) {

    // 6-1. Load the context input and decode the next token.
    swan::CopyTensor1d(ctx_input, tok_emb_table[token]);
    swan::Decode(token, pos, ctx_input, ctx_k_cache, ctx_v_cache,
                 ctx_final_norm, weights
#ifndef USE_CPU_ONLY
                 ,
                 q, kernel_matmul, kernel_mul, kernel_rmsnorm, kernel_softmax,
                 kernel_add, kernel_rope, ptr_a, ptr_b, ptr_c, ptr_d,
                 ptr_result, ptr_result2, buffer_a, buffer_b, buffer_c,
                 buffer_d, buffer_result, buffer_result2
#endif // USE_CPU_ONLY
    );

    // 6-2. Calculate the logits and softmax.
    swan::MutmulVocab(ctx_logits, ctx_final_norm, tok_emb_table);

    if (args.print_softmax) {
      printf("\nSoftmax\n <- ");
      for (int i = 0; i <= pos; ++i)
        printf("%5.4f, ", ctx.attn_qk[0][i]);
      printf("\n -> ");
      for (int i = 0; i <= pos; ++i)
        printf("%5.4f, ", ctx.attn_sm[0][i]);
      printf("\n");
    }

    // 6-3. Sampling the next token.
    if (args.temp < 1e-5) {
      next = swan::Argmax(ctx_logits);
    } else {
      for (int q = 0; q < vocab_size; ++q) {
        ctx_logits[q] /= args.temp;
      }
      swan::Softmax(ctx_logits, ctx_logits);
      next = SelectFromLogits(ctx_logits);
    }

    args.color ? printf("\e[31m%s\e[0m", vocab.dict.at(next).data())
               : printf("%s", vocab.dict.at(next).data());
    std::cout << std::flush;

    // Dump the contexts.
    if (args.log) {
      DumpContext("log/" + std::to_string(pos) + "_", ctx, swan::kNumLayers);
    }

    token = next;
  }
  std::cout << "\n";

  // 7. Print the time and speed.
  clock_t end_clk = clock();
  double decode_time = (double)(end_clk - start_clk) / CLOCKS_PER_SEC;
  std::cout << "Time : " << decode_time << "[s]" << std::endl
            << "Speed: " << args.max_seq / decode_time << "[tok/s]"
            << std::endl;

#ifndef USE_CPU_ONLY
  // 8. Flush OpenCL Device Memory
  OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_a, ptr_a));
  OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_b, ptr_b));
  OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_result, ptr_result));
  OCL_CHECK(err, err = q.finish());
#endif // USE_CPU_ONLY

  return 0;
}
