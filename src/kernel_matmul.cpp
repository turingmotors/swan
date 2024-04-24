#ifndef USE_CPU_ONLY

#include <hls_stream.h>
#include <stdint.h>

#define MAX_DATA_SIZE 1024

static void load_vec(float* i_vec, hls::stream<float>& inStream, int vec_size) {
mem_rd:
  for (int i = 0; i < vec_size; i++) {
    inStream << i_vec[i];
  }
}

static void load_mat(float* i_mat, hls::stream<float>& inStream, int vec_size,
                     int col_size) {
mem_rd:
  for (int i = 0; i < col_size; i++) {
    for (int j = 0; j < vec_size; j++) {
      inStream << i_mat[vec_size * i + j];
    }
  }
}

static void compute_matmul(hls::stream<float>& in1_stream,
                           hls::stream<float>& in2_stream,
                           hls::stream<float>& out_stream, int vec_size,
                           int col_size) {

  float vec_local[MAX_DATA_SIZE];
  float sum_local = 0;
  for (int i = 0; i < vec_size; i++) {
    vec_local[i] = in1_stream.read();
  }
execute:
  for (int i = 0; i < col_size; i++) {
    for (int j = 0; j < vec_size; j++) {
#pragma HLS UNROLL
      sum_local += vec_local[j] * in2_stream.read();
    }
    out_stream << sum_local;
    sum_local = 0;
  }
}

static void store_result(float* out, hls::stream<float>& out_stream,
                         int col_size) {
mem_wr:
  for (int i = 0; i < col_size; i++) {
    out[i] = out_stream.read();
  }
}

extern "C" {
void kernel_matmul(float* i_vec, float* i_mat, float* o_vec, int vec_size,
                   int col_size) {
#pragma HLS INTERFACE m_axi port = i_vec bundle = gmem0
#pragma HLS INTERFACE m_axi port = i_mat bundle = gmem1
#pragma HLS INTERFACE m_axi port = o_vec bundle = gmem0

  static hls::stream<float> vec_stream("vec_stream");
  static hls::stream<float> mat_stream("mat_stream");
  static hls::stream<float> out_stream("out_stream");

#pragma HLS dataflow
  load_vec(i_vec, vec_stream, vec_size);
  load_mat(i_mat, mat_stream, vec_size, col_size);
  compute_matmul(vec_stream, mat_stream, out_stream, vec_size, col_size);
  store_result(o_vec, out_stream, col_size);
}
}

#endif // USE_CPU_ONLY