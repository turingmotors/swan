#ifndef USE_CPU_ONLY

#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

static void load_vec(float* i_vec, hls::stream<float>& inStream, int vec_size) {
mem_rd:
  for (int i = 0; i < vec_size; i++) {
    inStream << i_vec[i];
  }
}

static void compute_rmsnorm(hls::stream<float>& in1_stream,
                            hls::stream<float>& in2_stream,
                            hls::stream<float>& out_stream) {

  float vec_local_1[288];
  float vec_local_2[288];
  float sum_local = 0;
  for (int i = 0; i < 288; i++) {
    vec_local_1[i] = in1_stream.read();
  }
  for (int i = 0; i < 288; i++) {
    vec_local_2[i] = in2_stream.read();
  }

  for (int i = 0; i < 288; i++) {
    sum_local += vec_local_1[i] * vec_local_1[i];
  }

  constexpr float eps = 1e-5;
  const float norm = 1 / std::sqrt(sum_local / 288 + eps);

  for (size_t i = 0; i < 288; i++) {
    out_stream << vec_local_1[i] * norm * vec_local_2[i];
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
void kernel_rmsnorm(float* i_vec_1, float* i_vec_2, float* o_vec,
                    int vec_size) {
#pragma HLS INTERFACE m_axi port = i_vec_1 bundle = gmem0
#pragma HLS INTERFACE m_axi port = i_vec_2 bundle = gmem1
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0

  static hls::stream<float> vec_stream_1("vec_stream_1");
  static hls::stream<float> vec_stream_2("mat_stream_2");
  static hls::stream<float> out_stream("out_stream");

#pragma HLS dataflow
  load_vec(i_vec_1, vec_stream_1, vec_size);
  load_vec(i_vec_2, vec_stream_2, vec_size);
  compute_rmsnorm(vec_stream_1, vec_stream_2, out_stream);
  store_result(o_vec, out_stream, vec_size);
}
}

#endif // USE_CPU_ONLY
