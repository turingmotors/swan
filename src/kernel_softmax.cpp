#ifndef USE_CPU_ONLY

#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

#define MAX_DATA_SIZE 256

static void load_vec(float* i_vec, hls::stream<float>& inStream, int vec_size) {
mem_rd:
  for (int i = 0; i < vec_size; i++) {
    inStream << i_vec[i];
  }
}

static void compute_softmax(hls::stream<float>& in_stream,
                            hls::stream<float>& out_stream, int vec_size) {

  int in_max_idx = vec_size;
  float vec_local_1[MAX_DATA_SIZE];
  float vec_local_2[MAX_DATA_SIZE];

  if (vec_size == -1) {
    in_max_idx = 256;
  }

  for (int i = 0; i < in_max_idx; i++) {
    vec_local_1[i] = in_stream.read();
  }

  // 1. Get Max
  float max_val = vec_local_1[0];
  for (int i = 1; i < in_max_idx; i++) {
    if (vec_local_1[i] > max_val) {
      max_val = vec_local_1[i];
    }
  }

  // 2. Exp and Sum
  float sum = 0;
  for (int i = 0; i < in_max_idx; i++) {
    vec_local_2[i] = std::exp(vec_local_1[i] - max_val);
    sum += vec_local_2[i];
  }

  // 3. Normalize
  for (int i = 0; i < in_max_idx; i++) {
    vec_local_2[i] /= sum;
  }

  for (int i = 0; i < in_max_idx; i++) {
    out_stream << vec_local_2[i];
  }
}

static void store_result(float* out, hls::stream<float>& out_stream,
                         int vec_size) {
mem_wr:
  for (int i = 0; i < vec_size; i++) {
    out[i] = out_stream.read();
  }
}

extern "C" {
void kernel_softmax(float* i_vec, float* o_vec, int vec_size) {
#pragma HLS INTERFACE m_axi port = i_vec bundle = gmem0
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0

  static hls::stream<float> vec_stream("vec_stream");
  static hls::stream<float> out_stream("out_stream");

#pragma HLS dataflow
  load_vec(i_vec, vec_stream, vec_size);
  compute_softmax(vec_stream, out_stream, vec_size);
  store_result(o_vec, out_stream, vec_size);
}
}

#endif // USE_CPU_ONLY
