#ifndef USE_CPU_ONLY

#include <hls_stream.h>
#include <stdint.h>

static void load_vec(float* i_vec, hls::stream<float>& inStream, int vec_size) {
mem_rd:
  for (int i = 0; i < vec_size; i++) {
    inStream << i_vec[i];
  }
}

static void compute_mul(hls::stream<float>& in1_stream,
                        hls::stream<float>& in2_stream,
                        hls::stream<float>& out_stream, int vec_size) {

  for (int i = 0; i < vec_size; i++) {
    out_stream << in1_stream.read() * in2_stream.read();
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
void kernel_mul(float* i_vec_1, float* i_vec_2, float* o_vec, int vec_size) {
#pragma HLS INTERFACE m_axi port = i_vec_1 bundle = gmem0
#pragma HLS INTERFACE m_axi port = i_vec_2 bundle = gmem1
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0

  static hls::stream<float> vec_stream_1("vec_stream_1");
  static hls::stream<float> vec_stream_2("vec_stream_2");
  static hls::stream<float> out_stream("out_stream");

#pragma HLS dataflow
  load_vec(i_vec_1, vec_stream_1, vec_size);
  load_vec(i_vec_2, vec_stream_2, vec_size);
  compute_mul(vec_stream_1, vec_stream_2, out_stream, vec_size);
  store_result(o_vec, out_stream, vec_size);
}
}

#endif // USE_CPU_ONLY
