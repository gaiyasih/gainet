#ifndef GAINET_CPU_BLAS_H_
#define GAINET_CPU_BLAS_H_

#include "gainet.h"

namespace gainet {

void cpuTensorAdd(float* out, float* in1, float* in2, 
                  uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w);
void cpuTensorMul(float* out, float* in1, float* in2, 
                  uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w);
void cpuTensorDiagonal(float *out, float *in1, uint32_t in1_n, uint32_t in1_c);
void cpuTensorTransfer(float *out, float *in1, uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w);
void cpuTensorRot90Left(float *out, float *in1, uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w);
void cpuTensorRot180Left(float *out, float *in1, uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w);
void cpuTensorpPowf(float *out, float *in1, float scale, uint32_t size);

// nchw -> c/(n*h*w)
void cpuTensorAvg(float *out, float *in1, uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w);

//     k              k
//  ---------     ---------
//  |       |     |       |
// m|       |    n|       |
//  |       |     |       |
//  ---------     ---------
void cpuHymm(float *out, float *in, float *weight, uint32_t m, uint32_t k, uint32_t n);
void cpuConvolutionForward(float *out, float *in, float *weight, float *bias,
                           int stride_h, int stride_w, int groups, 
                           uint32_t in_n, uint32_t in_c, uint32_t in_h, uint32_t in_w,
                           uint32_t out_c, uint32_t weight_h, uint32_t weight_w);
void cpuConvolutionBackward(float *gradient_out, float *gradient_in, float *weight, 
                            int stride_h, int stride_w, int groups, 
                            uint32_t in_n, uint32_t in_c, uint32_t in_h, uint32_t in_w,
                            uint32_t out_c, uint32_t weight_h, uint32_t weight_w);
void cpuConvolutionweightGradient(float *weight_gradient, float *in, float *gradient_in, 
                                  int stride_h, int stride_w, int groups, 
                                  uint32_t in_n, uint32_t in_c, uint32_t in_h, uint32_t in_w, 
                                  uint32_t out_c, uint32_t weight_h, uint32_t weight_w);
void cpuConvolutionBiasGradient(float *bias_gradient, float *gradient_in, 
                                uint32_t gradient_in_n, uint32_t gradient_in_c, uint32_t gradient_in_h, uint32_t gradient_in_w);
void cpuConnectedForward(float *out, float *in, float *weight, float *bias,
                         uint32_t in_n, uint32_t in_c,
                         uint32_t out_c);
void cpuConnectedBackward(float *gradient_out, float *gradient_in, float *weight, float *bias,
                          uint32_t in_n, uint32_t in_c,
                          uint32_t out_c);
void cpuConnectedweightGradient(float *weight_gradient, float *in, float *gradient_in, 
                                uint32_t in_n, uint32_t in_c,
                                uint32_t out_c);
void cpuConnectedweightGradient(float *bias_gradient, float *gradient_in, 
                                uint32_t in_n, uint32_t out_c);
void cpuConcat(float *out, float *in1, float *in2, uint32_t n, uint32_t in1_c, uint32_t in2_c, uint32_t h, uint32_t w);
void cpuDeConcat(float *out1, float *out2, float *in1, uint32_t n, uint32_t in1_c, uint32_t in2_c, uint32_t h, uint32_t w);
void cpuRand(float *out, float min, float max, uint32_t size);
void cpuRandperm(uint32_t* out, uint32_t size);
void cpuZeroPadForward(float *out, float *in1, 
                       uint32_t pad_h, uint32_t pad_w,
                       uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w);
void cpuZeroPadBackward(float *gradient_out, float *gradient_in, 
                        uint32_t pad_h, uint32_t pad_w,
                        uint32_t in_n, uint32_t in_c, uint32_t in_h, uint32_t in_w);
void cpuUpSample(float *out, float *in1,
                 uint32_t scale, 
                 uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w);
void cpuDownSample(float *out, float *in1,
                   uint32_t scale,
                   uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w);

} // namespace gainet

#endif // GAINET_CPU_BLAS_H_