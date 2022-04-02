#include "cpu_blas.h"
#include <immintrin.h>
#include <omp.h>
#include <math.h>
#include <random>
#include <cstring>

#define AVXSIZE 8
#define CPUBLOCK 16

namespace gainet {

static inline float sum(float *in1, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    float out = 0;
    __m256 loadData1;
    __m256 sumData = _mm256_setzero_ps();
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        sumData = _mm256_add_ps(sumData, loadData1);
        in1 += nBlockWidth;
    }
    sumData = _mm256_hadd_ps(sumData, sumData);
    sumData = _mm256_hadd_ps(sumData, sumData);
    out += sumData[0];
    out += sumData[4];
    for (uint32_t i = 0; i < cntRem; ++i) {
        out += in1[i];
    }

    return out;
}

static inline void add(float *out, float *in1, float *in2, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        ResultData = _mm256_add_ps(loadData1, loadData2);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] + in2[i];
    }
}

static inline void sub(float *out, float *in1, float *in2, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        ResultData = _mm256_sub_ps(loadData1, loadData2);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] - in2[i];
    }
}

static inline void mul(float *out, float *in1, float *in2, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        ResultData = _mm256_mul_ps(loadData1, loadData2);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] * in2[i];
    }
}

static inline void div(float *out, float *in1, float *in2, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        ResultData = _mm256_div_ps(loadData1, loadData2);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] / in2[i];
    }
}

static inline void addadd(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_add_ps(loadData1, loadData2);
        ResultData = _mm256_add_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] + in2[i] + in3[i];
    }
}

static inline void addsub(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_add_ps(loadData1, loadData2);
        ResultData = _mm256_sub_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] + in2[i] - in3[i];
    }
}

static inline void addmul(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_add_ps(loadData1, loadData2);
        ResultData = _mm256_mul_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = (in1[i] + in2[i]) * in3[i];
    }
}

static inline void adddiv(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_add_ps(loadData1, loadData2);
        ResultData = _mm256_div_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = (in1[i] + in2[i]) / in3[i];
    }
}

static inline void subsub(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_sub_ps(loadData1, loadData2);
        ResultData = _mm256_sub_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] - in2[i] - in3[i];
    }
}

static inline void subadd(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_sub_ps(loadData1, loadData2);
        ResultData = _mm256_add_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] - in2[i] + in3[i];
    }
}

static inline void submul(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_sub_ps(loadData1, loadData2);
        ResultData = _mm256_mul_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = (in1[i] - in2[i]) * in3[i];
    }
}

static inline void subdiv(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_sub_ps(loadData1, loadData2);
        ResultData = _mm256_div_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = (in1[i] - in2[i]) / in3[i];
    }
}

static inline void muladd(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_mul_ps(loadData1, loadData2);
        ResultData = _mm256_add_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] * in2[i] + in3[i];
    }
}

static inline void mulsub(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_mul_ps(loadData1, loadData2);
        ResultData = _mm256_sub_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] * in2[i] - in3[i];
    }
}

static inline void mulmul(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_mul_ps(loadData1, loadData2);
        ResultData = _mm256_mul_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] * in2[i] * in3[i];
    }
}

static inline void muldiv(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_mul_ps(loadData1, loadData2);
        ResultData = _mm256_div_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = (in1[i] * in2[i]) / in3[i];
    }
}

static inline void divadd(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_div_ps(loadData1, loadData2);
        ResultData = _mm256_add_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = (in1[i] / in2[i]) + in3[i];
    }
}

static inline void divsub(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_div_ps(loadData1, loadData2);
        ResultData = _mm256_sub_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = (in1[i] / in2[i]) - in3[i];
    }
}

static inline void divmul(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_div_ps(loadData1, loadData2);
        ResultData = _mm256_mul_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = (in1[i] / in2[i]) * in3[i];
    }
}

static inline void divdiv(float *out, float *in1, float *in2, float *in3, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        ResultData = _mm256_div_ps(loadData1, loadData2);
        ResultData = _mm256_div_ps(ResultData, loadData3);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = (in1[i] / in2[i]) / in3[i];
    }
}

static inline void addaddadd(float *out, float *in1, float *in2, float *in3, float *in4, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3, loadData4;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        loadData4 = _mm256_loadu_ps(in4);
        ResultData = _mm256_add_ps(loadData1, loadData2);
        ResultData = _mm256_add_ps(ResultData, loadData3);
        ResultData = _mm256_add_ps(ResultData, loadData4);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        in4 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] + in2[i] + in3[i] + in4[i];
    }
}

static inline void addaddsub(float *out, float *in1, float *in2, float *in3, float *in4, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3, loadData4;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        loadData4 = _mm256_loadu_ps(in4);
        ResultData = _mm256_add_ps(loadData1, loadData2);
        ResultData = _mm256_add_ps(ResultData, loadData3);
        ResultData = _mm256_sub_ps(ResultData, loadData4);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        in4 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] + in2[i] + in3[i] - in4[i];
    }
}

static inline void addsubadd(float *out, float *in1, float *in2, float *in3, float *in4, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3, loadData4;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        loadData4 = _mm256_loadu_ps(in4);
        ResultData = _mm256_add_ps(loadData1, loadData2);
        ResultData = _mm256_sub_ps(ResultData, loadData3);
        ResultData = _mm256_add_ps(ResultData, loadData4);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        in4 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] + in2[i] - in3[i] + in4[i];
    }
}

static inline void addsubsub(float *out, float *in1, float *in2, float *in3, float *in4, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3, loadData4;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        loadData4 = _mm256_loadu_ps(in4);
        ResultData = _mm256_add_ps(loadData1, loadData2);
        ResultData = _mm256_sub_ps(ResultData, loadData3);
        ResultData = _mm256_sub_ps(ResultData, loadData4);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        in4 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] + in2[i] - in3[i] - in4[i];
    }
}

static inline void subaddadd(float *out, float *in1, float *in2, float *in3, float *in4, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3, loadData4;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        loadData4 = _mm256_loadu_ps(in4);
        ResultData = _mm256_sub_ps(loadData1, loadData2);
        ResultData = _mm256_add_ps(ResultData, loadData3);
        ResultData = _mm256_add_ps(ResultData, loadData4);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        in4 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] - in2[i] + in3[i] + in4[i];
    }
}

static inline void subaddsub(float *out, float *in1, float *in2, float *in3, float *in4, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3, loadData4;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        loadData4 = _mm256_loadu_ps(in4);
        ResultData = _mm256_sub_ps(loadData1, loadData2);
        ResultData = _mm256_add_ps(ResultData, loadData3);
        ResultData = _mm256_sub_ps(ResultData, loadData4);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        in4 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] - in2[i] + in3[i] - in4[i];
    }
}

static inline void subsubadd(float *out, float *in1, float *in2, float *in3, float *in4, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3, loadData4;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        loadData4 = _mm256_loadu_ps(in4);
        ResultData = _mm256_sub_ps(loadData1, loadData2);
        ResultData = _mm256_sub_ps(ResultData, loadData3);
        ResultData = _mm256_add_ps(ResultData, loadData4);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        in4 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] - in2[i] - in3[i] + in4[i];
    }
}

static inline void subsubsub(float *out, float *in1, float *in2, float *in3, float *in4, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2, loadData3, loadData4;
    __m256 ResultData;
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        loadData3 = _mm256_loadu_ps(in3);
        loadData4 = _mm256_loadu_ps(in4);
        ResultData = _mm256_sub_ps(loadData1, loadData2);
        ResultData = _mm256_sub_ps(ResultData, loadData3);
        ResultData = _mm256_sub_ps(ResultData, loadData4);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
        in3 += nBlockWidth;
        in4 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] - in2[i] - in3[i] - in4[i];
    }
}

static inline void addc(float *out, float *in1, float constant, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2;
    __m256 ResultData;
    loadData2 = _mm256_set1_ps(constant);
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        ResultData = _mm256_add_ps(loadData1, loadData2);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] + constant;
    }
}

static inline void subc(float *out, float *in1, float constant, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2;
    __m256 ResultData;
    loadData2 = _mm256_set1_ps(constant);
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        ResultData = _mm256_sub_ps(loadData1, loadData2);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] - constant;
    }
}

static inline void csub(float *out, float constant, float *in2, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2;
    __m256 ResultData;
    loadData1 = _mm256_set1_ps(constant);
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData2 = _mm256_loadu_ps(in2);
        ResultData = _mm256_sub_ps(loadData1, loadData2);
        _mm256_storeu_ps(out, ResultData);
        in2 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = constant - in2[i];
    }
}

static inline void mulc(float *out, float *in1, float constant, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2;
    __m256 ResultData;
    loadData2 = _mm256_set1_ps(constant);
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        ResultData = _mm256_mul_ps(loadData1, loadData2);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] * constant;
    }
}

static inline void divc(float *out, float *in1, float constant, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2;
    __m256 ResultData;
    loadData2 = _mm256_set1_ps(constant);
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        ResultData = _mm256_div_ps(loadData1, loadData2);
        _mm256_storeu_ps(out, ResultData);
        in1 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = in1[i] / constant;
    }
}

static inline void cdiv(float *out, float constant, float *in2, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    __m256 loadData1, loadData2;
    __m256 ResultData;
    loadData1 = _mm256_set1_ps(constant);
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData2 = _mm256_loadu_ps(in2);
        ResultData = _mm256_div_ps(loadData1, loadData2);
        _mm256_storeu_ps(out, ResultData);
        in2 += nBlockWidth;
        out += nBlockWidth;
    }
    for (uint32_t i = 0; i < cntRem; ++i) {
        out[i] = constant / in2[i];
    }
}

static inline float fmadd(float *in1, float *in2, uint32_t size) 
{
    uint32_t nBlockWidth = AVXSIZE;
    uint32_t cntBlock = size / nBlockWidth;
    uint32_t cntRem = size - cntBlock * nBlockWidth;

    float out = 0;
    __m256 loadData1, loadData2;
    __m256 sumData = _mm256_setzero_ps();
    for (uint32_t i = 0; i < cntBlock; ++i) {
        loadData1 = _mm256_loadu_ps(in1);
        loadData2 = _mm256_loadu_ps(in2);
        sumData = _mm256_fmadd_ps(loadData1, loadData2, sumData);
        in1 += nBlockWidth;
        in2 += nBlockWidth;
    }
    sumData = _mm256_hadd_ps(sumData, sumData);
    sumData = _mm256_hadd_ps(sumData, sumData);
    out += sumData[0];
    out += sumData[4];
    for (uint32_t i = 0; i < cntRem; ++i) {
        out += in1[i] * in2[i];
    }

    return out;
}


void cpuTensorAdd(float* out, float* in1, float* in2, 
                  uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w) 
{
    uint32_t step_nchw = in1_n * in1_c * in1_h * in1_w;
    uint32_t cntBlock = CPUBLOCK;
    uint32_t nBlockWidth = step_nchw / cntBlock;
    uint32_t cntRem = step_nchw - cntBlock * nBlockWidth;
    #pragma omp parallel for
    for (uint32_t bid = 0; bid < cntBlock; bid++) {
        uint32_t idx = bid * nBlockWidth;
        add(out + idx, in1 + idx, in2 + idx, nBlockWidth);
    }
    uint32_t fullBlock = cntBlock * nBlockWidth;
    add(out + fullBlock, in1 + fullBlock, in2 + fullBlock, cntRem);
}

void cpuTensorMul(float* out, float* in1, float* in2, 
                  uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w) 
{
    uint32_t step_nchw = in1_n * in1_c * in1_h * in1_w;
    uint32_t cntBlock = CPUBLOCK;
    uint32_t nBlockWidth = step_nchw / cntBlock;
    uint32_t cntRem = step_nchw - cntBlock * nBlockWidth;
    #pragma omp parallel for
    for (uint32_t bid = 0; bid < cntBlock; bid++) {
        uint32_t idx = bid * nBlockWidth;
        mul(out + idx, in1 + idx, in2 + idx, nBlockWidth);
    }
    uint32_t fullBlock = cntBlock * nBlockWidth;
    mul(out + fullBlock, in1 + fullBlock, in2 + fullBlock, cntRem);
}

void cpuTensorDiagonal(float *out, float *in1, uint32_t in1_n, uint32_t in1_c) 
{
    #pragma omp parallel for
    for (uint32_t n = 0; n < in1_n; ++n) {
        for (uint32_t c = 0; c < in1_c; ++c) {
            out[n * in1_c * in1_c + c * in1_c + c] = in1[n * in1_c + c];
        }
    }
}

void cpuTensorTransfer(float *out, float *in1, uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w) 
{
    uint32_t step_nc = in1_n * in1_c;
    uint32_t step_hw = in1_h * in1_w;
    #pragma omp parallel for
    for (uint32_t nc = 0; nc < step_nc; ++nc) {
        for (uint32_t hw = 0; hw < step_hw; ++hw) {
            uint32_t h, w;
            h = hw / in1_w;
            w = hw - h * in1_w;
            out[nc * step_hw + w * in1_h + h] = in1[nc * step_hw + hw];
        }
    }
}

void cpuTensorRot90Left(float *out, float *in1, uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w) 
{
    uint32_t step_nc = in1_n * in1_c;
    uint32_t step_hw = in1_h * in1_w;
    #pragma omp parallel for
    for (uint32_t nc = 0; nc < step_nc; ++nc) {
        for (uint32_t hw = 0; hw < step_hw; ++hw) {
            uint32_t h, w;
            h = hw / in1_w;
            w = hw - h * in1_w;
            out[nc * step_hw + (in1_w - 1 - w) * in1_h + h] = in1[nc * step_hw + hw];
        }
    }
}

void cpuTensorRot180Left(float *out, float *in1, uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w) 
{
    uint32_t step_nc = in1_n * in1_c;
    uint32_t step_hw = in1_h * in1_w;
    #pragma omp parallel for
    for (uint32_t nc = 0; nc < step_nc; ++nc) {
        for (uint32_t hw = 0; hw < step_hw; ++hw) {
            out[nc * step_hw + step_hw - 1 - hw] = in1[nc * step_hw + hw];
        }
    }
}

void cpuTensorpPowf(float *out, float *in1, float scale, uint32_t size) 
{
    uint32_t cntBlock = CPUBLOCK;
    uint32_t nBlockWidth = size / cntBlock;
    #pragma omp parallel for
    for (uint32_t bid = 0; bid < cntBlock; ++bid) {
        for (uint32_t tid = 0; tid < nBlockWidth; ++tid) {
            uint32_t idx = bid * nBlockWidth + tid;
            out[idx] = powf(in1[idx], scale);
        }
    }
    for (uint32_t idx = cntBlock * nBlockWidth; idx < size; ++idx) {
        out[idx] = powf(in1[idx], scale);
    }
}

void cpuTensorAvg(float *out, float *in1, uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w) 
{
    uint32_t step_nc = in1_n * in1_c;
    uint32_t step_hw = in1_h * in1_w;
    float *tmp_sum = float_new_memory(step_nc);
    #pragma omp parallel for
    for (uint32_t nc = 0; nc < step_nc; ++nc) {
        tmp_sum[nc] = sum(in1 + nc * step_hw, step_hw);
    }
    #pragma omp parallel for
    for (uint32_t c = 0; c < in1_c; ++c) {
        out[c] = 0;
        for (uint32_t n = 0; n < in1_n; ++n) {
            out[c] += tmp_sum[n * in1_c + c];
        }
        out[c] /= (in1_n * in1_h * in1_w);
    }
    float_delete_memory(tmp_sum);
}

void cpuHymm(float *out, float *in, float *weight, uint32_t m, uint32_t k, uint32_t n) 
{
    uint32_t step_mn = m * n;
    #pragma omp parallel for
    for (uint32_t mn = 0; mn < step_mn; ++mn) {
        uint32_t i, j;
        i = mn / n;
        j = mn - i * n;
        out[mn] = fmadd(in + i * k, weight + j * k, k);
    }
}

void cpuConvolutionForward(float *out, float *in, float *weight, float *bias,
                           int stride_h, int stride_w, int groups, 
                           uint32_t in_n, uint32_t in_c, uint32_t in_h, uint32_t in_w,
                           uint32_t out_c, uint32_t weight_h, uint32_t weight_w) 
{
    uint32_t out_h = float(in_h - weight_h) / stride_h + 1;
    uint32_t out_w = float(in_w - weight_w) / stride_w + 1;
    uint32_t group_out_c = out_c / groups;
    uint32_t group_in_c = in_c / groups;
    uint32_t step_out_hw = out_h * out_w;
    uint32_t step_out_chw = out_c * step_out_hw;
    uint32_t step_weight_hw = weight_h * weight_w;
    uint32_t step_weight_chw = group_in_c * step_weight_hw;
    uint32_t step_in_hw = in_h * in_w;
    uint32_t step_in_chw = in_c * step_in_hw;
    uint32_t step_ngroup = in_n * groups;
    #pragma omp parallel for
    for (uint32_t ngroup = 0; ngroup < step_ngroup; ++ngroup) {
        float *tmp = float_new_memory(step_weight_chw);
        int n, group;
        n = ngroup / groups;
        group = ngroup - n * groups;
        for (uint32_t out_hw = 0; out_hw < step_out_hw; ++out_hw) {
            int i, j;
            i = out_hw / out_w;
            j = out_hw - i * out_w;
            for (uint32_t weight_chw = 0; weight_chw < step_weight_chw; ++weight_chw) {
                int k, l, m;
                k = weight_chw / step_weight_hw;
                l = weight_chw % step_weight_hw / weight_w;
                m = weight_chw % step_weight_hw % weight_w;
                tmp[weight_chw] = in[n * step_in_chw + (group * group_in_c + k) * step_in_hw + (i * stride_h + l) * in_w + (j * stride_w + m)];
            }
            for (uint32_t k = 0; k < group_out_c; ++k) {
                out[n * step_out_chw + (group * group_out_c + k) * step_out_hw + i * out_w + j] = fmadd(tmp, weight + (group * group_out_c + k) * step_weight_chw, step_weight_chw) + bias[group * group_out_c + k];
            }
        }
        float_delete_memory(tmp);
    }
}

void cpuConvolutionBackward(float *gradient_out, float *gradient_in, float *weight, 
                            int stride_h, int stride_w, int groups, 
                            uint32_t in_n, uint32_t in_c, uint32_t in_h, uint32_t in_w,
                            uint32_t out_c, uint32_t weight_h, uint32_t weight_w)
{
    uint32_t group_in_c = in_c / groups;
    uint32_t group_out_c = out_c / groups;
    uint32_t step_weight_hw = weight_h * weight_w;
    float *tmp_weight = float_new_memory(out_c * group_in_c * weight_h * weight_w);
    #pragma omp parallel for
    for (uint32_t weight_hw = 0; weight_hw < step_weight_hw; ++weight_hw) {
        for (uint32_t group = 0; group < groups; ++group) {
            for (uint32_t i = 0; i < group_out_c; ++i) {
                for (uint32_t j = 0; j < group_in_c; ++j) {
                    tmp_weight[((group * group_in_c + j) * group_out_c + i) * step_weight_hw + step_weight_hw - 1 - weight_hw] = weight[((group * group_out_c + i) * group_in_c + j) * step_weight_hw + weight_hw];
                }
            }
        }
    }
    
    uint32_t step_pad_h = weight_h - 1;
    uint32_t step_pad_w = weight_w - 1;
    uint32_t out_h = float(in_h - weight_h) / stride_h + 1;
    uint32_t out_w = float(in_w - weight_w) / stride_w + 1;
    uint32_t gradient_in_h = in_h - weight_h + 1 + step_pad_h * 2;
    uint32_t gradient_in_w = in_w - weight_w + 1 + step_pad_w * 2;
    uint32_t gradient_in_hw = gradient_in_h * gradient_in_w;
    uint32_t step_nc = in_n * out_c;
    uint32_t step_hw = out_h * out_w;
    float *tmp_gradient_in = float_new_memory(step_nc * gradient_in_hw);
    #pragma omp parallel for
    for (uint32_t nc = 0; nc < step_nc; ++nc) {
        for (uint32_t hw = 0; hw < step_hw; ++hw) {
            int h, w;
            h = hw / out_w;
            w = hw - h * out_w;
            tmp_gradient_in[nc * gradient_in_hw + (h * stride_h + step_pad_h) * gradient_in_w + (w * stride_w + step_pad_w)] = gradient_in[nc * step_hw + hw];
        }
    }
    
    float *tmp_bias = float_new_memory(in_c);
    cpuConvolutionForward(gradient_out, tmp_gradient_in, tmp_weight, tmp_bias,
                          1, 1, groups, 
                          in_n, out_c, gradient_in_h, gradient_in_w, 
                          in_c, weight_h, weight_w);

    float_delete_memory(tmp_weight);
    float_delete_memory(tmp_gradient_in);
    float_delete_memory(tmp_bias);
}

void cpuConvolutionweightGradient(float *weight_gradient, float *in, float *gradient_in, 
                                  int stride_h, int stride_w, int groups, 
                                  uint32_t in_n, uint32_t in_c, uint32_t in_h, uint32_t in_w, 
                                  uint32_t out_c, uint32_t weight_h, uint32_t weight_w)
{
    uint32_t out_h = float(in_h - weight_h) / stride_h + 1;
    uint32_t out_w = float(in_w - weight_w) / stride_w + 1;
    uint32_t gradient_in_h = in_h - weight_h + 1;
    uint32_t gradient_in_w = in_w - weight_w + 1;
    uint32_t step_gradient_in_hw = gradient_in_h * gradient_in_w;
    uint32_t step_nc = in_n * out_c;
    uint32_t step_hw = out_h * out_w;
    float *tmp_gradient_in;
    if (stride_h > 1 || stride_w > 1) {
        tmp_gradient_in = float_new_memory(step_nc * step_gradient_in_hw);
        #pragma omp parallel for
        for (uint32_t nc = 0; nc < step_nc; ++nc) {
            for (uint32_t hw = 0; hw < step_hw; ++hw) {
                int h, w;
                h = hw / out_w;
                w = hw - h * out_w;
                tmp_gradient_in[nc * step_gradient_in_hw + (h * stride_h) * gradient_in_w + (w * stride_w)] = gradient_in[nc * step_hw + hw];
            }
        }
    } else {
        tmp_gradient_in = gradient_in;
    }
    
    uint32_t group_out_c = out_c / groups;
    uint32_t group_in_c = in_c / groups;
    uint32_t step_gradient_in_chw = out_c * step_gradient_in_hw;
    uint32_t step_weight_hw = weight_h * weight_w;
    uint32_t step_weight_chw = group_in_c * step_weight_hw;
    uint32_t step_in_hw = in_h * in_w;
    uint32_t step_in_chw = in_c * step_in_hw;
    uint32_t step_grouphw = groups * step_weight_hw;
    uint32_t step_ngrouphw = in_n * step_grouphw;
    #pragma omp parallel for
    for (uint32_t ngrouphw = 0; ngrouphw < step_ngrouphw; ++ngrouphw) {
        float *tmp = float_new_memory(step_gradient_in_hw);
        int n, group, h, w;
        n = ngrouphw / step_grouphw;
        group = ngrouphw % step_grouphw / step_weight_hw;
        h = ngrouphw % step_grouphw % step_weight_hw / weight_w;
        w = ngrouphw % step_grouphw % step_weight_hw % weight_w;
        for (uint32_t i = 0; i < group_in_c; ++i) {
            for (uint32_t gradient_in_hw = 0; gradient_in_hw < step_gradient_in_hw; ++gradient_in_hw) {
                int j, k;
                j = gradient_in_hw / gradient_in_w;
                k = gradient_in_hw - j * gradient_in_w;
                tmp[gradient_in_hw] = in[n * step_in_chw + (group * group_in_c + i) * step_in_hw + (h + j) * in_w + (w + k)];
            }
            for (uint32_t l = 0; l < group_out_c; ++l) {
                weight_gradient[(group * group_out_c + l) * step_weight_chw + i * step_weight_hw + h * weight_w + w] += (fmadd(tmp, tmp_gradient_in + n * step_gradient_in_chw + (group * group_out_c + l) * step_gradient_in_hw, step_gradient_in_hw) / in_n);
            }
        }
        float_delete_memory(tmp);
    }
    if (stride_h > 1 || stride_w > 1) {
        float_delete_memory(tmp_gradient_in);
    }
}

void cpuConvolutionBiasGradient(float *bias_gradient, float *gradient_in, 
                                uint32_t gradient_in_n, uint32_t gradient_in_c, uint32_t gradient_in_h, uint32_t gradient_in_w) 
{
    uint32_t step_nc = gradient_in_n * gradient_in_c;
    uint32_t step_hw = gradient_in_h * gradient_in_w;
    float *tmp_sum = float_new_memory(step_nc);
    #pragma omp parallel for
    for (uint32_t nc = 0; nc < step_nc; ++nc) {
        tmp_sum[nc] = sum(gradient_in + nc * step_hw, step_hw);
    }
    #pragma omp parallel for
    for (uint32_t c = 0; c < gradient_in_c; ++c) {
        for (uint32_t n = 0; n < gradient_in_n; ++n) {
            bias_gradient[c] += tmp_sum[n * gradient_in_c + c];
        }
        bias_gradient[c] /= gradient_in_n;
    }
    float_delete_memory(tmp_sum);
}

void cpuConnectedForward(float *out, float *in, float *weight, float *bias,
                         uint32_t in_n, uint32_t in_c,
                         uint32_t out_c) 
{
    uint32_t step_mn = in_n * out_c;
    #pragma omp parallel for
    for (uint32_t mn = 0; mn < step_mn; ++mn) {
        uint32_t m, n;
        m = mn / out_c;
        n = mn - m * out_c;
        out[mn] = fmadd(in + m * in_c, weight + n * in_c, in_c) + bias[n];
    }
}

void cpuConnectedBackward(float *gradient_out, float *gradient_in, float *weight, float *bias,
                          uint32_t in_n, uint32_t in_c,
                          uint32_t out_c) 
{
    float *tmp_weight = float_new_memory(out_c * in_c);
    cpuTensorTransfer(tmp_weight, weight, 1, 1, out_c, in_c);
    cpuHymm(gradient_out, gradient_in, tmp_weight, in_n, out_c, in_c);
    float_delete_memory(tmp_weight);
}

void cpuConnectedweightGradient(float *weight_gradient, float *in, float *gradient_in, 
                                uint32_t in_n, uint32_t in_c,
                                uint32_t out_c)
{
    uint32_t step_nc = in_n * out_c;
    #pragma omp prarllel for
    for (uint32_t nc = 0; nc < step_nc; ++nc) {
        float *tmp = float_new_memory(in_c);
        uint32_t n, c;
        n = nc / out_c;
        c = nc - n * out_c;
        mulc(tmp, in + n * in_c, gradient_in[n * out_c + c] / in_n, in_c);
        add(weight_gradient + c * in_c, weight_gradient + c * in_c, tmp, in_c);
        float_delete_memory(tmp);
    }
}

void cpuConnectedweightGradient(float *bias_gradient, float *gradient_in, 
                                uint32_t in_n, uint32_t out_c) 
{
    #pragma omp parallel for
    for (uint32_t c = 0; c < out_c; ++c) {
        for (uint32_t n = 0; n < in_n; ++n) {
            bias_gradient[c] += gradient_in[n * out_c + c];
        }
        bias_gradient[c] /= in_n;
    }
}

void cpuConcat(float *out, float *in1, float *in2, uint32_t n, uint32_t in1_c, uint32_t in2_c, uint32_t h, uint32_t w) 
{
    uint32_t step_in1_chw = in1_c * h * w;
    uint32_t step_in2_chw = in2_c * h * w;
    for (uint32_t i = 0; i < n; ++i) {
        memcpy(out + i * (step_in1_chw + step_in2_chw), in1 + i * step_in1_chw, step_in1_chw * sizeof(float));
        memcpy(out + i * (step_in1_chw + step_in2_chw) + step_in1_chw, in2 + i * step_in2_chw, step_in2_chw * sizeof(float));
    }
}

void cpuDeConcat(float *out1, float *out2, float *in1, uint32_t n, uint32_t in1_c, uint32_t in2_c, uint32_t h, uint32_t w) 
{
    uint32_t step_in1_chw = in1_c * h * w;
    uint32_t step_in2_chw = in2_c * h * w;
    for (uint32_t i = 0; i < n; ++i) {
        memcpy(out1 + i * step_in1_chw, in1 + i * (step_in1_chw + step_in2_chw), step_in1_chw * sizeof(float));
        memcpy(out2 + i * step_in2_chw, in1 + i * (step_in1_chw + step_in2_chw) + step_in1_chw, step_in2_chw * sizeof(float));
    }
}

void cpuRand(float *out, float min, float max, uint32_t size) 
{
    std::random_device rd;
    std::default_random_engine gen = std::default_random_engine(rd());
    std::uniform_real_distribution<> dis(0, 1);
    for (uint32_t i = 0; i < size; ++i) {
        out[i] = (max - min) * dis(gen) + min;
    }
}

void cpuRandperm(uint32_t* out, uint32_t size) 
{
    std::random_device rd;
    std::default_random_engine gen = std::default_random_engine(rd());
    std::uniform_real_distribution<> dis(0, 1);
    uint32_t tmp_tmp, tmp_1, tmp_2;
    for (uint32_t i = 0; i < size; ++i) {
        out[i] = i;
    }
    for (uint32_t i = 0; i < size; ++i) {
        tmp_1 = (uint32_t)floorf(size * dis(gen));
        tmp_2 = (uint32_t)floorf(size * dis(gen));
        tmp_tmp = out[tmp_1];
        out[tmp_1] = out[tmp_2];
        out[tmp_2] = tmp_tmp;
    }
}

void cpuZeroPadForward(float *out, float *in1, 
                       uint32_t pad_h, uint32_t pad_w,
                       uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w) 
{
    uint32_t out_h = in1_h + 2 * pad_h;
    uint32_t out_w = in1_w + 2 * pad_w;
    uint32_t step_nch = in1_n * in1_c * in1_h;
    uint32_t step_out_hw = out_h * out_w;
    uint32_t step_in1_hw = in1_h * in1_w;
    for (uint32_t nch = 0; nch < step_nch; ++nch) {
        uint32_t i, j;
        i = nch / in1_h;
        j = nch - i * in1_h;
        uint32_t out_shift = i * step_out_hw + (j + pad_h) * out_w + pad_w;
        uint32_t in1_shift = i * step_in1_hw + j * in1_w;
        memcpy(out + out_shift, in1 + in1_shift, in1_w * sizeof(float));
    }
}

void cpuZeroPadBackward(float *gradient_out, float *gradient_in, 
                        uint32_t pad_h, uint32_t pad_w,
                        uint32_t in_n, uint32_t in_c, uint32_t in_h, uint32_t in_w) 
{
    uint32_t out_h = in_h + 2 * pad_h;
    uint32_t out_w = in_w + 2 * pad_w;
    uint32_t step_nch = in_n * in_c * in_h;
    uint32_t step_out_hw = out_h * out_w;
    uint32_t step_in_hw = in_h * in_w;
    for (uint32_t nch = 0; nch < step_nch; ++nch) {
        uint32_t i, j;
        i = nch / in_h;
        j = nch - i * in_h;
        uint32_t out_shift = i * step_out_hw + (j + pad_h) * out_w + pad_w;
        uint32_t in_shift = i * step_in_hw + j * in_w;
        memcpy(gradient_out + in_shift, gradient_in + out_shift, in_w * sizeof(float));
    }
}

void cpuUpSample(float *out, float *in1,
                 uint32_t scale, 
                 uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w) 
{
    uint32_t out_h = in1_h * 2;
    uint32_t out_w = in1_w * 2;
    uint32_t step_nchw = in1_n * in1_c * in1_h * in1_w;
    uint32_t cntBlock = CPUBLOCK;
    uint32_t nBlockWidth = step_nchw / cntBlock;
    uint32_t step_out_hw = out_h * out_w;
    uint32_t step_in1_hw = in1_h * in1_w;
    #pragma omp parallel for
    for (uint32_t bid = 0; bid < cntBlock; ++bid) {
        for (uint32_t tid = 0; tid < nBlockWidth; ++tid) {
            uint32_t idx = bid * nBlockWidth + tid;
            uint32_t i, j, k;
            i = idx / step_in1_hw;
            j = (idx - i * step_in1_hw) / in1_w;
            k = (idx - i * step_in1_hw) - j * in1_w;
            for (uint32_t l = 0; l < scale; ++l) {
                for (uint32_t m = 0; m < scale; ++m) {
                    out[i * step_out_hw + (j * scale + l) * out_w + (k * scale + m)] = in1[idx];
                }
            }
        }
    }
    for (uint32_t idx = cntBlock * nBlockWidth; idx < step_nchw; ++idx) {
        uint32_t i, j, k;
        i = idx / step_in1_hw;
        j = (idx - i * step_in1_hw) / in1_w;
        k = (idx - i * step_in1_hw) - j * in1_w;
        for (uint32_t l = 0; l < scale; ++l) {
            for (uint32_t m = 0; m < scale; ++m) {
                out[i * step_out_hw + (j * scale + l) * out_w + (k * scale + m)] = in1[idx];
            }
        }
    }
}

void cpuDownSample(float *out, float *in1,
                   uint32_t scale,
                   uint32_t in1_n, uint32_t in1_c, uint32_t in1_h, uint32_t in1_w) 
{
    uint32_t out_h = in1_h / scale;
    uint32_t out_w = in1_w / scale;
    uint32_t step_nchw = in1_n * in1_c * out_h * out_w;
    uint32_t cntBlock = CPUBLOCK;
    uint32_t nBlockWidth = step_nchw / cntBlock;
    uint32_t step_out_hw = out_h * out_w;
    uint32_t step_in1_hw = in1_h * in1_w;
    #pragma omp parallel for
    for (uint32_t bid = 0; bid < cntBlock; ++bid) {
        for (uint32_t tid = 0; tid < nBlockWidth; ++tid) {
            uint32_t idx = bid * nBlockWidth + tid;
            uint32_t i, j, k;
            i = idx / step_out_hw;
            j = (idx - i * step_out_hw) / out_w;
            k = (idx - i * step_out_hw) - j * out_w;
            for (uint32_t l = 0; l < scale; ++l) {
                for (uint32_t m = 0; m < scale; ++m) {
                    out[idx] += in1[i * step_in1_hw + (j * scale + l) * in1_w + (k * scale + m)];
                }
            }
        }
    }
    for (uint32_t idx = cntBlock * nBlockWidth; idx < step_nchw; ++idx) {
        uint32_t i, j, k;
        i = idx / step_out_hw;
        j = (idx - i * step_out_hw) / out_w;
        k = (idx - i * step_out_hw) - j * out_w;
        for (uint32_t l = 0; l < scale; ++l) {
            for (uint32_t m = 0; m < scale; ++m) {
                out[idx] += in1[i * step_in1_hw + (j * scale + l) * in1_w + (k * scale + m)];
            }
        }
    }
} 

} // namespace gainet
