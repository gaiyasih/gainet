#ifndef GAINET_GAI_TYPE_H_
#define GAINET_GAI_TYPE_H_

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>

#define B1 0.9
#define B2 0.999
#define EPSILON 0.00001
#define BATCH_NORM_ROLLING 0.9

namespace gainet {

struct CONVOLUTION
{
    float *in;
    float *out;
    float *gradient_in;
    float *gradient_out;
    float *weight;
    int stride_h;
    int stride_w;
    uint32_t n;
    int in_c;
    int in_h;
    int in_w;
    uint32_t in_s;
    int out_c;
    int out_h;
    int out_w;
    uint32_t out_s;
    uint32_t kernel_out_c;
    int kernel_in_c;
    int kernel_h;
    int kernel_w;
    float *bias;
};

struct CONNECTED
{
    float *in;
    float *out;
    float *gradient_in;
    float *gradient_out;
    float *weight;
    uint32_t n;
    int in_c;
    uint32_t in_s;
    int out_c;
    uint32_t out_s;
    uint32_t kernel_out_c;
    int kernel_in_c;
    float *bias;
};

struct ZEROPAD
{
    float *in;
    float *out;
    float *gradient_in;
    float *gradient_out;
    int pad_h;
    int pad_w;
    uint32_t n;
    int in_c;
    int in_h;
    int in_w;
    uint32_t in_s;
    int out_c;
    int out_h;
    int out_w;
    uint32_t out_s;
};

struct UPSAMPLE
{
    float *in;
    float *out;
    float *gradient_in;
    float *gradient_out;
    int scale;
    uint32_t n;
    int in_c;
    int in_h;
    int in_w;
    uint32_t in_s;
    int out_c;
    int out_h;
    int out_w;
    uint32_t out_s;
};

struct DOWNSAMPLE
{
    float *in;
    float *out;
    float *gradient_in;
    float *gradient_out;
    int scale;
    uint32_t n;
    int in_c;
    int in_h;
    int in_w;
    uint32_t in_s;
    int out_c;
    int out_h;
    int out_w;
    uint32_t out_s;
};

struct MAXPOOL
{
    float *in;
    float *out;
    float *gradient_in;
    float *gradient_out;
    int stride_h;
    int stride_w;
    uint32_t n;
    int in_c;
    int in_h;
    int in_w;
    uint32_t in_s;
    int out_c;
    int out_h;
    int out_w;
    uint32_t out_s;
    int kernel_h;
    int kernel_w;
};

struct AVGPOOL
{
    float *in;
    float *out;
    float *gradient_in;
    float *gradient_out;
    int stride_h;
    int stride_w;
    uint32_t n;
    int in_c;
    int in_h;
    int in_w;
    uint32_t in_s;
    int out_c;
    int out_h;
    int out_w;
    uint32_t out_s;
    int kernel_h;
    int kernel_w;
};

struct ADAM
{
    uint32_t size;
    float *gradient;
    float *gradient_m;
    float *gradient_v;
    float *gradient_mt;
    float *gradient_vt;
    float lr;
    float l2;
    float b1 = B1;
    float b2 = B2;
    float epsilon = EPSILON;
};

struct BATCHNORMALIZATION
{
    float batch_norm_rolling = BATCH_NORM_ROLLING;
    float *gamma;
    float *beta;
    float *mean;
    float *var;
    float *rolling_mean;
    float *rolling_var;
    float epsilon = EPSILON;
};

struct ACTIVATION
{
    std::string activation;
    float activation_param;
    float *in;
    float *out;
    float *gradient_in;
    float *gradient_out;
    uint32_t size;
};

} // namespace gainet

#endif // GAINET_GAI_TYPE_H_
