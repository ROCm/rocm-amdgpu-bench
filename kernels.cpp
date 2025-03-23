/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip/hip_runtime.h>
#include <string>
#include <stdlib.h>
#include <cstdio>
#include <vector>
#include <cassert>

#include "kernels.h"

__global__ void LDS_bw(int numIter, float *dummy)
{
     const uint32_t tid = threadIdx.x;
     __shared__ uint8_t shmem[64];


     if (tid == 0)
     {
        #pragma unroll
        for (int i=0;i<63;i++)
            shmem[i] = i+1;

        shmem[63] = 0;
     }

     __syncthreads();

     uint32_t index = tid;
     #pragma unroll 64
     for(uint32_t iter = 0; iter < numIter; iter++)
         index = shmem[index];

     dummy[tid] = (float )index;

}


using int32_16vec = __attribute__((__vector_size__(16 * sizeof(int)))) int;
using bf16_2vec = __attribute__((__vector_size__(1 * sizeof(__2i16))))  short;
using bf16_4vec = __attribute__((__vector_size__(2 * sizeof(__2i16))))  short;
using f32_16vec = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using f16_2vec = __attribute__((__vector_size__(2 * sizeof(__2f16))))  float;
using f64_4vec = __attribute__((__vector_size__(4 * sizeof(double)))) double;


__global__ void mfma_i8(int iter, float *dummy)
{
    // Output: 16 I32 registers
    int32_16vec result = {0};

// MI100/MI200
#if not defined(__gfx940__) and not defined(__gfx941__) and not defined(__gfx942__)
    // Input: 1 I32 register
    int a = threadIdx.x;

    // CDNA1/2: v_mfma_i32_32x32x8i8 ops: 32x32x8x2 = 16384
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_i32_32x32x8i8(a, a, result, 0, 0, 0);
    }
// MI300
#else
    // Input: 2 I32 registers
    // builting mfma expects I64 input
    long a =  threadIdx.x;

    // CDNA3: v_mfma_i32_32x32x16_i8 ops: 32x32x16x2 = 32768
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_i32_32x32x16_i8(a, a, result, 0, 0, 0);
    }
#endif

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}


__global__ void mfma_f8(int iter, float *dummy)
{
// MI300 series only - note gfx940/gfx941/gfx942 uses fnuz f8
#if defined(__gfx940__) or defined(__gfx941__) or defined(__gfx942__)
    // Input: 2 F32 registers
    // builtin mfma expects double input
    double a =  threadIdx.x;

    // Output: 16 F32 registers
    f32_16vec result = {0};

    // CDNA3: v_mfma_f32_32x32x16_fp8_fp8 ops: 32x32x16x2 = 32768
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
#endif
}


__global__ void mfma_bf16(int iter, float *dummy)
{
    // Output: 16 F32 registers
    f32_16vec result = {0};

// MI100/MI200
#if not defined(__gfx940__) and not defined(__gfx941__) and not defined(__gfx942__)
    // Input: 1 F32 register
    // builtin mfma expects 2 short registers
    bf16_2vec a;
    a[1] = a[0]= threadIdx.x;

    // CDNA1/2: v_mfma_f32_32x32x4bf16 ops: 32x32x4x2 = 8192
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x4bf16(a, a, result, 0, 0, 0);
    }
//MI300
#else
    // Input: 2 F32 registers
    // builting mfma expects 4 short registers
    bf16_4vec a;
    a[3] = a[2] = a[1] = a[0]= threadIdx.x;

    // CDNA3: v_mfma_f32_32x32x8_bf16 ops: 32x32x8x2 = 16384
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, a, result, 0, 0, 0);
    }
#endif

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
}


__global__ void mfma_f16(int iter, float *dummy)
{
    // Input: 2 F32 registers
    f16_2vec a;
    a[1] = a[0] = threadIdx.x;

    //Output: 16 F32 registers
    f32_16vec result = {0};

    // CDNA2: v_mfma_f32_32x32x8f16 ops: 32x32x8x2 = 16384
    // CDNA3: v_mfma_f32_32x32x8_f16
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x8f16(a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }

}


__global__ void mfma_f32(int iter, float *dummy)
{
    // Input: 1 F32 register
    float a =  threadIdx.x;

    // Output: 16 F32 registers
    f32_16vec result = {0};

    // CDNA2: v_mfma_f32_32x32x2f32 ops: 32x32x2x2 = 4096
    // CDNA3: v_mfma_f32_32x32x2_f32
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f32_32x32x2f32(a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }

}


__global__ void mfma_f64(int iter, float *dummy)
{
#if not defined(__gfx908__)
    // Input: 1 F64 register
    double a =  threadIdx.x;

    // Output: 4 F64 registers
    f64_4vec result = {0};

    // CDNA2: v_mfma_f64_16x16x4f64 ops: 16x16x4x2 = 2048
    // CDNA3: v_mfma_f64_16x16x4_f64
    for(int i = 0; i < iter; ++i)
    {
        result = __builtin_amdgcn_mfma_f64_16x16x4f64(a, a, result, 0, 0, 0);
    }

    if (result[0] != 2*result[0])
    {
        dummy[0] = result[0];
    }
#endif
}
