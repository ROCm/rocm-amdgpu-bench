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

#include "common.h"
const char ver[] = "v1.0";
void help(void)
{
    printf("Usage: roofline [-d DEVICE] [-o csvfile.csv]\n");
    printf("Version: %s\n", ver);
    printf("\n\n optional arguments:\n");
    printf("\n -h, --help           Show this help message and exit\n");
    printf("\n -d, --device         GPU device ID\n");
    printf("\n -v, --verbose        Vebose logging\n");
    printf("\n -q, --quiet          Quiet logging\n");
    printf("\n -w, --workgroup      Number of workgroups\n");
    printf("\n -s, --wsize          Workgroup size\n");
    printf("\n -i, --iter           Number of iterations\n");
    printf("\n -t, --dataset        Number of dataset entries\n");
    printf("\n -e, --experiments    Number of experiments\n");
    printf("\n -o, --output         Output file name\n");
    printf("\n\n");
}

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void showProgress(float percentage) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

void stats(float *samples, int entries, float *mean, float *stdev, float *confidence)
{
    float mean_val, stdev_val, conf_val;

    mean_val = 0;
    for(int i=0; i<entries; i++)
    {
        mean_val += samples[i];
    }

    mean_val = mean_val / entries;

    stdev_val = 0;
    for(int i=0; i<entries; i++)
    {
        stdev_val += (samples[i] - mean_val) * (samples[i] - mean_val);
    }

    stdev_val = sqrtf(stdev_val / entries);

    // return
    mean[0] = mean_val;
    stdev[0] = stdev_val;
    confidence[0] = 1.960 * stdev_val / sqrtf(entries);

}

std::string device_arch(int device_id){
  hipDeviceProp_t props;
  HIP_ASSERT(hipGetDeviceProperties(&props, device_id));
  std::string device_arch(props.gcnArchName);
  auto colon_pos = device_arch.find(':');
  if(colon_pos != std::string::npos){
    device_arch.erase(colon_pos);
  }
  return device_arch;
}

int device_compute_units(int device_id) {
    hipDeviceProp_t props;
    HIP_ASSERT(hipGetDeviceProperties(&props, device_id));
    return props.multiProcessorCount;
}
