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

#ifndef __KERNELS_H__

// definitions of templated kernels

template <typename T, int cacheSize, int workgroup_size>
__global__ void Cache_bw(const T *memBlock, T *dummy, int numIter)
{
  const int thread_id = threadIdx.x;
  constexpr int cache_count = cacheSize / sizeof(T);

  T sink;

  sink = 0;
  for (int iter = 0; iter < numIter; ++iter)
  {
#pragma unroll 32
    for (int i = 0; i < cache_count; i += workgroup_size)
    {
      // if the size of the memory block is small (e.g., the size
      // of L1), then we need a slightly more complicated index
      // calculation. Otherwise, the compiler holds all the loads
      // in the inner loop in registers upon the first pass of the
      // outer loop, and it doesn't do the loads upon subsequent
      // passes of the outer loop.
      // OTOH, if the size of the memory block is larger (such as L2
      // size), experimentation showed that the overhead of the more
      // complicated index calculation has a noticeable effect on BW,
      // so we use a simpler index expression instead. This works since
      // for larger memory blocks, the compiler cannot hold the loads
      // of the inner loop in registers anymore, as it can with L1-sized
      // buffers.
      if constexpr (cache_count / workgroup_size <= 32)
      {
        sink += memBlock[(thread_id + i + iter) % cache_count];
      }
      else
      {
        sink += memBlock[thread_id + i];
      }
    }
  }

  dummy[thread_id] = sink;
}


template<typename T>
__global__ void HBM_bw(T *dst, const T *src)
{
    const uint32_t gid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    const uint32_t tid = hipThreadIdx_x;

    dst[gid] = src[gid];
}


template<typename T, int nFMA>
__global__ void flops_benchmark(T *buf, uint32_t nSize)
{
    const uint32_t gid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    const uint32_t nThreads  = gridDim.x * blockDim.x;
    const uint32_t nEntriesPerThread = (uint32_t) nSize / nThreads;
    const uint32_t maxOffset = nEntriesPerThread * nThreads;

    T *ptr;
    const T y = (T) 1.0;

    ptr = &buf[gid];
    T x = (T) 2.0;

    for(uint32_t offset=0; offset < maxOffset; offset += nThreads)
    {
        for(int j=0; j<nFMA; j++)
        {
            x = ptr[offset] * x + y;
        }
    }

    ptr[0] = -x;

}

enum MX_DATAFORMATS
{
    FP8,
    BF8,
    FP6,
    BF6,
    FP4
};

// declarations of non-templated kernels
__global__ void LDS_bw(int numIter, float *dummy);
__global__ void mfma_i8(int iter, float *dummy);
__global__ void mfma_f8f6f4(int iter, float *dummy, MX_DATAFORMATS datatype);
__global__ void mfma_f8(int iter, float *dummy);
__global__ void mfma_bf16(int iter, float *dummy);
__global__ void mfma_f16(int iter, float *dummy);
__global__ void mfma_f32(int iter, float *dummy);
__global__ void mfma_f64(int iter, float *dummy);

#define __KERNELS_H__
#endif