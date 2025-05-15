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

#ifndef __ROOFLINE_H__
#define __ROOFLINE_H__

#if AMDGPU_TARGET == GFX90A

#define L2_CACHE_SIZE           (8*1024*1024)
#define L1_CACHE_SIZE           (16*1024)
#define LDS_SIZE                (65536)
#else

// Default settings
#define L2_CACHE_SIZE           (8*1024*1024)
#define L1_CACHE_SIZE           (16*1024)
#define LDS_SIZE                (65536)

#endif

struct arch_size_specs {
    uint64_t L1_size;
    uint64_t L2_size;
    uint64_t MALL_size;
    uint64_t LDS_size;
    uint64_t CUs;
};

enum {
    MFMA_F4_OPS = 131072, // for f8f6f4 scaling mfma builting
    MFMA_F6_OPS = 131072, // for f8f6f4 scaling mfma builting
    MFMA_F8_OPS = 32768,
    MFMA_F16_OPS  = 16384,
    MFMA_F32_OPS  = 4096,
    MFMA_F64_OPS  = 2048,
#if AMDGPU_TARGET == GFX90A
    MFMA_I8_OPS   = 16384,
    MFMA_BF16_OPS = 8192,
#else
    MFMA_I8_OPS   = 32768,
    MFMA_BF16_OPS = 16384,
#endif
};

const int SIMDS_PER_CU = 4;

const int DEFAULT_WORKGROUP_SIZE = 256;
const int DEFAULT_WORKGROUPS     = 8192;
const int DEFAULT_THREADS        = (DEFAULT_WORKGROUP_SIZE * DEFAULT_WORKGROUPS);
const int DEFAULT_NUM_EXPERIMENTS = 100;
const int DEFAULT_NUM_ITERS       = 10;
const int DEFAULT_DATASET_SIZE    = (512 * 1024 * 1024);

#endif //__ROOFLINE_H__
