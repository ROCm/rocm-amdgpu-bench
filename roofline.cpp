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
#include <hip/hip_ext.h>
#include <hip/hip_version.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <stdlib.h>
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cassert>
#include <getopt.h>
#include <math.h>
#include <map>

#include "roofline.h"
#include "kernels.h"
#include "common.h"

int main(int argc, char **argv)
{
    using arch_size_t = std::unordered_map<std::string, arch_size_specs>;
    arch_size_t arch_sizes;
    //                                     L1         L2               MALL              LDS        #CUs
    arch_sizes["gfx908"] = arch_size_specs{16 * 1024, 8 * 1024 * 1024, 0, /*          */ 64 * 1024, 120}; // MI100
    arch_sizes["gfx90a"] = arch_size_specs{16 * 1024, 8 * 1024 * 1024, 0, /*          */ 64 * 1024, 104}; // MI200 per die
    arch_sizes["gfx940"] = arch_size_specs{32 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024, 64 * 1024, 228}; // MI300A
    arch_sizes["gfx941"] = arch_size_specs{32 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024, 64 * 1024, 304}; // MI300X A0
    arch_sizes["gfx942"] = arch_size_specs{32 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024, 64 * 1024, 304}; // MI300X

    using cache_bw_kernel_t = decltype(Cache_bw<float, 1, 1>);
    using cache_bw_kernel_selector_t = std::unordered_map<std::string, cache_bw_kernel_t *>;

    cache_bw_kernel_selector_t L1_bw_kernel_selector;
    L1_bw_kernel_selector["gfx908"] = Cache_bw<float, 16 * 1024, 256>;
    L1_bw_kernel_selector["gfx90a"] = Cache_bw<float, 16 * 1024, 256>;
    L1_bw_kernel_selector["gfx940"] = Cache_bw<float, 32 * 1024, 256>;
    L1_bw_kernel_selector["gfx941"] = Cache_bw<float, 32 * 1024, 256>;
    L1_bw_kernel_selector["gfx942"] = Cache_bw<float, 32 * 1024, 256>;

    cache_bw_kernel_selector_t L2_bw_kernel_selector;
    L2_bw_kernel_selector["gfx908"] = Cache_bw<float, 8 * 1024 * 1024, 256>;
    L2_bw_kernel_selector["gfx90a"] = Cache_bw<float, 8 * 1024 * 1024, 256>;
    L2_bw_kernel_selector["gfx940"] = Cache_bw<float, 4 * 1024 * 1024, 256>;
    L2_bw_kernel_selector["gfx941"] = Cache_bw<float, 4 * 1024 * 1024, 256>;
    L2_bw_kernel_selector["gfx942"] = Cache_bw<float, 4 * 1024 * 1024, 256>;

    cache_bw_kernel_selector_t MALL_bw_kernel_selector;
    MALL_bw_kernel_selector["gfx940"] = Cache_bw<float, 64 * 1024 * 1024, 256>;
    MALL_bw_kernel_selector["gfx941"] = Cache_bw<float, 64 * 1024 * 1024, 256>;
    MALL_bw_kernel_selector["gfx942"] = Cache_bw<float, 64 * 1024 * 1024, 256>;

    hipDeviceProp_t props;
    float eventMs;
    hipEvent_t start, stop;

    static struct option long_options[] =
        {
            {"help", no_argument, 0, 'h'},
            {"verbose", no_argument, 0, 'v'},
            {"quiet", no_argument, 0, 'q'},
            {"device", required_argument, 0, 'd'},
            {"workgroups", required_argument, 0, 'w'},
            {"wsize", required_argument, 0, 's'},
            {"iter", required_argument, 0, 'i'},
            {"dataset", required_argument, 0, 't'},
            {"experiments", required_argument, 0, 'e'},
            {"output", required_argument, 0, 'o'},
            {0, 0, 0, 0}};
    int option_index = 0;
    int c;
    std::cout << "Empirical Roofline Calculation" << std::endl
              << "Copyright © 2022  Advanced Micro Devices, Inc. All rights reserved." << std::endl;

    /* default values */
    int devID = -1;
    int workgroupSize = DEFAULT_WORKGROUP_SIZE;
    int numWorkgroups = DEFAULT_WORKGROUPS;

    int datasetEntries = DEFAULT_DATASET_SIZE;
    int numIters = DEFAULT_NUM_ITERS;
    int numExperiments = DEFAULT_NUM_EXPERIMENTS;
    int numBenchmarks = 11;
    int verbose = 0;
    bool quiet = false;
    const char *csvFile = "roofline.csv";

    /* Performance number */
    float HBM_bandwidth = -1;
    float L2cache_bandwidth = -1;
    float L1cache_bandwidth = -1;
    float LDS_bandwidth = -1;

    // Minimum version check -> require 5.1 or newer

    int hipVersion;
    HIP_ASSERT(hipRuntimeGetVersion(&hipVersion));
    const int MAJOR_VER_MIN = 5;
    const int MINOR_VER_MIN = 1;

    if (hipVersion < (MAJOR_VER_MIN * 10000000 + MINOR_VER_MIN * 100000))
    {
        printf("\n");
        printf("Error: ROCm version %i.%i or newer is required.\n", MAJOR_VER_MIN, MINOR_VER_MIN);
        exit(1);
    }

    // CLI parsing
    while (1)
    {
        c = getopt_long(argc, argv, "d:w:i:s:o:hvq", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c)
        {

        case 'd':
            devID = atoi(optarg);
            break;
        case 's':
            workgroupSize = atoi(optarg);
            break;
        case 'w':
            numWorkgroups = atoi(optarg);
            break;
        case 't':
            datasetEntries = atoi(optarg);
            break;
        case 'i':
            numIters = atoi(optarg);
            break;
        case 'o':
            csvFile = optarg;
            break;
        case 'v':
            verbose = 1;
            break;
        case 'q':
            quiet = true;
            break;

        case 'h':
        default:
            help();
            exit(1);
        }
    }

    int totalThreads = numWorkgroups * workgroupSize;

    int numGpuDevices;
    HIP_ASSERT(hipGetDeviceCount(&numGpuDevices));

    using archs_t = std::unordered_set<std::string>;
    archs_t supported_archs{"gfx908", "gfx90a", "gfx940", "gfx941", "gfx942"};

    if ((devID >= 0) && (devID < numGpuDevices))
    {
        auto gcnArch = device_arch(devID);
        quiet = false;
        if (not supported_archs.contains(gcnArch))
        {
            printf("Unsupported device architecture \"%s\" will be skipped\n", gcnArch.c_str());
        }
    }

    printf("Total detected GPU devices: %d\n", numGpuDevices);

    uint64_t totalBytes;
    uint64_t totalFlops;
    float mean, stdev, confidence;
    float *samples;
    std::map<std::string, std::map<std::string, float>> statsMap[numGpuDevices];

    void *d_src;
    void *d_dst;
    void *memBlock;
    float *dummy;

    std::ofstream ofile;
    // Measurement data
    samples = (float *)calloc(numExperiments, sizeof(float));

    // CSV file, header
    ofile.open(csvFile);
    ofile << "device,HBMBw,HBMBwLow,hbmBwHigh,MALLBw,MALLBwLow,MALLBwHigh,";
    ofile << "L2Bw,L2BwLow,L2BwHigh,L1Bw,L1BwLow,L1BwHigh,LDSBw,LDSBwLow,LDSBwHigh,";
    ofile << "FP32Flops,FP32FlopsLow,FP32FlopsHigh,FP64Flops,FP64FlopsLow,FP64FlopsHigh,";
    ofile << "MFMABF16Flops,MFMABF16FlopsLow,MFMABF16FlopsHigh,";
    ofile << "MFMAF16Flops,MFMAF16FlopsLow,MFMAF16FlopsHigh,";
    ofile << "MFMAF32Flops,MFMAF32FlopsLow,MFMAF32FlopsHigh,";
    ofile << "MFMAF64Flops,MFMAF64FlopsLow,MFMAF64FlopsHigh,";
    ofile << "MFMAI8Ops,MFMAFI8OpsLow,MFMAI8OpsHigh\n";

    for (int dev = 0; dev < numGpuDevices; dev++)
    {
        float currBenchmark = 0.0;
        if (quiet)
        {
            showProgress((float)dev / numGpuDevices);
        }
        HIP_ASSERT(hipGetDeviceProperties(&props, dev));

        /* Skip incompatible devices */
        auto gcnArch = device_arch(dev);
        if ((not supported_archs.contains(gcnArch)) || ((devID >= 0) && (dev != devID)))
        {
            printf("GPU Device %d: Skipped\n", dev);
            continue;
        }
        if (!quiet)
        {
            hipDeviceProp_t props;
            HIP_ASSERT(hipGetDeviceProperties(&props, dev));

            printf("GPU Device %d (%s) with %d CUs: Profiling...\n", dev, gcnArch.c_str(),
                   props.multiProcessorCount);
        }

        HIP_ASSERT(hipSetDevice(dev));

        /* Perf metrics */
        std::vector<float> perf_metrics;

        /* **********************************************
         *
         * HBM BW benchmarking
         *
         * **********************************************/
        int workgroupsPerCU = 20 * 1024;
        numWorkgroups = arch_sizes[gcnArch].CUs * workgroupsPerCU;
        datasetEntries = numWorkgroups * workgroupSize;
        HIP_ASSERT(hipMalloc(&d_src, datasetEntries * sizeof(double)));
        HIP_ASSERT(hipMalloc(&d_dst, datasetEntries * sizeof(double)));
        hipLaunchKernelGGL((HBM_bw<double>), dim3(numWorkgroups), dim3(workgroupSize), 0, 0, (double *)d_dst, (const double *)d_src);
        HIP_ASSERT(hipDeviceSynchronize());

        totalBytes = (uint64_t)datasetEntries * sizeof(double) * 2;
        currBenchmark++;
        for (int n = 0; n < numExperiments; n++)
        {
            initHipEvents(start, stop);
            hipLaunchKernelGGL((HBM_bw<double>), dim3(numWorkgroups), dim3(workgroupSize),
                               0, 0, (double *)d_dst, (const double *)d_src);
            stopHipEvents(eventMs, start, stop);

            samples[n] = (float)totalBytes / eventMs / 1e6;
            if (!quiet)
            {
                showProgress((float)n / numExperiments);
            }
        }

        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["HBM BW"]["mean"] = mean;
            statsMap[dev]["HBM BW"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nHBM BW, GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, traffic:%lu bytes, duration:%.1f ms, mean:%.1f GB/sec, stdev=%.1f GB/sec\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalBytes, eventMs, mean, stdev);
        }

        HIP_ASSERT(hipFree(d_src));
        HIP_ASSERT(hipFree(d_dst));

        /* **********************************************
         *
         * MALL BW benchmarking
         *
         * **********************************************/

        numExperiments = DEFAULT_NUM_EXPERIMENTS;
        currBenchmark++;
        int cacheSize = arch_sizes[gcnArch].MALL_size;
        archs_t mall_unsupported{"gfx908", "gfx90a"};
        if (mall_unsupported.contains(gcnArch))
        {
            totalBytes = 0;
            samples[0] = 0;
            numExperiments = 1;
            eventMs = 0;
            if (!quiet)
            {
                showProgress(1);
            }
        }
        else
        {
            HIP_ASSERT(hipMalloc(&memBlock, cacheSize));
            HIP_ASSERT(hipMalloc(&dummy, workgroupSize * sizeof(float)));
            auto MALL_bw_kernel = MALL_bw_kernel_selector[gcnArch];

            // warm up first with one iteration, filling the cache
            numWorkgroups = 128 * arch_sizes[gcnArch].CUs;
            numIters = 1;
            hipLaunchKernelGGL((MALL_bw_kernel), dim3(numWorkgroups), dim3(workgroupSize), 0, 0,
                               (const float *)memBlock, dummy, numIters);
            HIP_ASSERT(hipDeviceSynchronize());

            numIters = 1;
            totalBytes = (unsigned long)numWorkgroups * numIters * cacheSize;
            for (int n = 0; n < numExperiments; n++)
            {

                initHipEvents(start, stop);
                hipLaunchKernelGGL((MALL_bw_kernel), dim3(numWorkgroups), dim3(workgroupSize), 0, 0,
                                   (const float *)memBlock, dummy, numIters);
                stopHipEvents(eventMs, start, stop);

                samples[n] = (float)totalBytes / eventMs / 1e6;
                if (!quiet)
                {
                    showProgress((float)n / numExperiments);
                }
            }
            HIP_ASSERT(hipFree(memBlock));
            HIP_ASSERT(hipFree(dummy));
        }
        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["MALL BW"]["mean"] = mean;
            statsMap[dev]["MALL BW"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nMALL BW, GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, traffic:%lu bytes, duration:%.1f ms, mean:%.1f GB/sec, stdev=%.1f GB/sec\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalBytes, eventMs, mean, stdev);
        }

        /* **********************************************
         *
         * L2 BW benchmarking
         *
         * **********************************************/

        numExperiments = DEFAULT_NUM_EXPERIMENTS;
        cacheSize = arch_sizes[gcnArch].L2_size;
        HIP_ASSERT(hipMalloc(&memBlock, cacheSize));
        HIP_ASSERT(hipMalloc(&dummy, workgroupSize * sizeof(float)));
        auto L2_bw_kernel = L2_bw_kernel_selector[gcnArch];

        // warm up first with one iteration, filling the cache
        numWorkgroups = 128 * arch_sizes[gcnArch].CUs;
        numIters = 1;
        hipLaunchKernelGGL((L2_bw_kernel), dim3(numWorkgroups), dim3(workgroupSize), 0, 0,
                           (const float *)memBlock, dummy, numIters);
        HIP_ASSERT(hipDeviceSynchronize());

        numIters = 10;
        totalBytes = (unsigned long)numWorkgroups * numIters * cacheSize;
        currBenchmark++;
        for (int n = 0; n < numExperiments; n++)
        {

            initHipEvents(start, stop);
            hipLaunchKernelGGL((L2_bw_kernel), dim3(numWorkgroups), dim3(workgroupSize), 0, 0,
                               (const float *)memBlock, dummy, numIters);
            stopHipEvents(eventMs, start, stop);

            samples[n] = (float)totalBytes / eventMs / 1e6;
            if (!quiet)
            {
                showProgress((float)n / numExperiments);
            }
        }

        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["L2 BW"]["mean"] = mean;
            statsMap[dev]["L2 BW"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nL2 BW, GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, traffic:%lu bytes, duration:%.1f ms, mean:%.1f GB/sec, stdev=%.1f GB/sec\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalBytes, eventMs, mean, stdev);
        }

        HIP_ASSERT(hipFree(memBlock));
        HIP_ASSERT(hipFree(dummy));

        /* **********************************************
         *
         * L1 BW benchmarking
         *
         * **********************************************/
        numExperiments = DEFAULT_NUM_EXPERIMENTS;
        cacheSize = arch_sizes[gcnArch].L1_size;
        HIP_ASSERT(hipMalloc(&memBlock, cacheSize));
        HIP_ASSERT(hipMalloc(&dummy, workgroupSize * sizeof(float)));
        auto L1_bw_kernel = L1_bw_kernel_selector[gcnArch];

        // warm up first with one iteration, filling the cache
        numWorkgroups = 128 * arch_sizes[gcnArch].CUs;
        numIters = 1;
        hipLaunchKernelGGL((L1_bw_kernel), dim3(numWorkgroups), dim3(workgroupSize), 0, 0,
                           (const float *)memBlock, dummy, numIters);
        HIP_ASSERT(hipDeviceSynchronize());

        numIters = 100;
        totalBytes = (unsigned long)numWorkgroups * numIters * cacheSize;
        currBenchmark++;
        for (int n = 0; n < numExperiments; n++)
        {

            initHipEvents(start, stop);
            hipLaunchKernelGGL((L1_bw_kernel), dim3(numWorkgroups), dim3(workgroupSize), 0, 0,
                               (const float *)memBlock, dummy, numIters);
            stopHipEvents(eventMs, start, stop);

            samples[n] = (float)totalBytes / eventMs / 1e6;
            if (!quiet)
            {
                showProgress((float)n / numExperiments);
            }
        }

        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["L1 BW"]["mean"] = mean;
            statsMap[dev]["L1 BW"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nL1 BW, GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, traffic:%lu bytes, duration:%.1f ms, mean:%.1f GB/sec, stdev=%.1f GB/sec\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalBytes, eventMs, mean, stdev);
        }

        HIP_ASSERT(hipFree(memBlock));
        HIP_ASSERT(hipFree(dummy));

        /* **********************************************
         *
         * LDS BW benchmarking
         *
         * **********************************************/
        numExperiments = DEFAULT_NUM_EXPERIMENTS;
        HIP_ASSERT(hipMalloc(&dummy, workgroupSize * sizeof(float)));

        // warm up first use default setting
        numWorkgroups = 128 * arch_sizes[gcnArch].CUs;
        numIters = 2000;
        hipLaunchKernelGGL(LDS_bw, dim3(numWorkgroups), dim3(workgroupSize), 0, 0, 10, dummy);
        HIP_ASSERT(hipDeviceSynchronize());

        totalBytes = (unsigned long)numWorkgroups * workgroupSize * numIters * 4;
        currBenchmark++;
        for (int n = 0; n < numExperiments; n++)
        {

            initHipEvents(start, stop);
            hipLaunchKernelGGL(LDS_bw, dim3(numWorkgroups), dim3(workgroupSize), 0, 0, numIters, dummy);
            stopHipEvents(eventMs, start, stop);

            samples[n] = (float)totalBytes / eventMs / 1e6;
            if (!quiet)
            {
                showProgress((float)n / numExperiments);
            }
        }

        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["LDS BW"]["mean"] = mean;
            statsMap[dev]["LDS BW"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nLDS BW, GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, traffic:%lu bytes, duration:%.1f ms, mean:%.1f GB/sec, stdev=%.1f GB/sec\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalBytes, eventMs, mean, stdev);
        }

        HIP_ASSERT(hipFree(dummy));

        /* **********************************************
         *
         * Peak FLOPs benchmarking
         *
         * **********************************************/
        numExperiments = DEFAULT_NUM_EXPERIMENTS;
        HIP_ASSERT(hipMalloc(&memBlock, DEFAULT_DATASET_SIZE));

        // warm up first use default setting
        numWorkgroups = 128 * arch_sizes[gcnArch].CUs;

        int numThreads = numWorkgroups * workgroupSize;

        /* FP32 benchmark */
        int nSize = DEFAULT_DATASET_SIZE / sizeof(float) / numThreads * numThreads;

        hipLaunchKernelGGL((flops_benchmark<float, 1024>), dim3(numWorkgroups), dim3(workgroupSize), 0, 0, (float *)memBlock, nSize);
        HIP_ASSERT(hipDeviceSynchronize());

        totalFlops = (uint64_t)nSize * 1024 * 2;
        currBenchmark++;
        for (int n = 0; n < numExperiments; n++)
        {

            initHipEvents(start, stop);
            hipLaunchKernelGGL((flops_benchmark<float, 1024>), dim3(numWorkgroups), dim3(workgroupSize), 0, 0, (float *)memBlock, nSize);
            stopHipEvents(eventMs, start, stop);

            samples[n] = (float)totalFlops / eventMs / 1e6;
            if (!quiet)
            {
                showProgress((float)n / numExperiments);
            }
        }

        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["Peak FLOPs (FP32)"]["mean"] = mean;
            statsMap[dev]["Peak FLOPs (FP32)"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nPeak FLOPs (FP32), GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, FLOP:%lu, duration:%.3f ms, mean:%.1f GFLOPS, stdev=%.1f GFLOPS\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalFlops, eventMs, mean, stdev);
        }

        /* FP64 benchmark */
        numExperiments = DEFAULT_NUM_EXPERIMENTS;
        nSize = DEFAULT_DATASET_SIZE / sizeof(double) / numThreads * numThreads;
        hipLaunchKernelGGL((flops_benchmark<double, 1024>), dim3(numWorkgroups), dim3(workgroupSize), 0, 0, (double *)memBlock, nSize);
        HIP_ASSERT(hipDeviceSynchronize());

        totalFlops = (uint64_t)nSize * 1024 * 2;
        currBenchmark++;
        for (int n = 0; n < numExperiments; n++)
        {

            initHipEvents(start, stop);
            hipLaunchKernelGGL((flops_benchmark<double, 1024>), dim3(numWorkgroups), dim3(workgroupSize), 0, 0, (double *)memBlock, nSize);
            stopHipEvents(eventMs, start, stop);

            samples[n] = (float)totalFlops / eventMs / 1e6;
            if (!quiet)
            {
                showProgress((float)n / numExperiments);
            }
        }

        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["Peak FLOPs (FP64)"]["mean"] = mean;
            statsMap[dev]["Peak FLOPs (FP64)"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nPeak FLOPs (FP64), GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, FLOP:%lu, duration:%.1f ms, mean:%.1f GFLOPS, stdev=%.1f GFLOPS\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalFlops, eventMs, mean, stdev);
        }

        HIP_ASSERT(hipFree(memBlock));

        /* **********************************************
         *
         * MFMA benchmarking
         *
         * **********************************************/
        HIP_ASSERT(hipMalloc(&dummy, 64 * sizeof(float)));

        // warm up first use default setting
        numWorkgroups = 128 * arch_sizes[gcnArch].CUs;
        numIters = 2000;

        /* MFMA-BF16 */
        numExperiments = DEFAULT_NUM_EXPERIMENTS;
        currBenchmark++;
        archs_t mfma_bf16_unsupported{"gfx940", "gfx941", "gfx942"};
        if (mfma_bf16_unsupported.contains(gcnArch))
        {
            totalFlops = 0;
            samples[0] = 0;
            numExperiments = 1;
            eventMs = 0;
            if (!quiet)
            {
                showProgress(1);
            }
        }
        else
        {
            totalFlops = (uint64_t)numWorkgroups * SIMDS_PER_CU * numIters * MFMA_BF16_OPS;
            for (int n = 0; n < numExperiments; n++)
            {

                initHipEvents(start, stop);
                hipLaunchKernelGGL(mfma_bf16, dim3(numWorkgroups), dim3(workgroupSize), 0, 0, numIters, dummy);
                stopHipEvents(eventMs, start, stop);

                samples[n] = totalFlops / eventMs / 1e6;
                if (!quiet)
                {
                    showProgress((float)n / numExperiments);
                }
            }
        }
        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["Peak MFMA FLOPs (BF16)"]["mean"] = mean;
            statsMap[dev]["Peak MFMA FLOPs (BF16)"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nPeak MFMA FLOPs (BF16), GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, FLOP:%lu, duration:%.1f ms, mean:%.1f GFLOPS, stdev=%.1f GFLOPS\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalFlops, eventMs, mean, stdev);
        }

        /* MFMA-F16 */
        numExperiments = DEFAULT_NUM_EXPERIMENTS;
        totalFlops = (uint64_t)numWorkgroups * SIMDS_PER_CU * numIters * MFMA_F16_OPS;
        currBenchmark++;
        for (int n = 0; n < numExperiments; n++)
        {

            initHipEvents(start, stop);
            hipLaunchKernelGGL(mfma_f16, dim3(numWorkgroups), dim3(workgroupSize), 0, 0, numIters, dummy);
            stopHipEvents(eventMs, start, stop);

            samples[n] = totalFlops / eventMs / 1e6;
            if (!quiet)
            {
                showProgress((float)n / numExperiments);
            }
        }

        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["Peak MFMA FLOPs (F16)"]["mean"] = mean;
            statsMap[dev]["Peak MFMA FLOPs (F16)"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nPeak MFMA FLOPs (F16), GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, FLOP:%lu, duration:%.1f ms, mean:%.1f GFLOPS, stdev=%.1f GFLOPS\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalFlops, eventMs, mean, stdev);
        }

        /* MFMA-F32 */
        numExperiments = DEFAULT_NUM_EXPERIMENTS;
        totalFlops = (uint64_t)numWorkgroups * SIMDS_PER_CU * numIters * MFMA_F32_OPS;
        currBenchmark++;
        for (int n = 0; n < numExperiments; n++)
        {

            initHipEvents(start, stop);
            hipLaunchKernelGGL(mfma_f32, dim3(numWorkgroups), dim3(workgroupSize), 0, 0, numIters, dummy);
            stopHipEvents(eventMs, start, stop);

            samples[n] = totalFlops / eventMs / 1e6;
            if (!quiet)
            {
                showProgress((float)n / numExperiments);
            }
        }

        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["Peak MFMA FLOPs (F32)"]["mean"] = mean;
            statsMap[dev]["Peak MFMA FLOPs (F32)"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nPeak MFMA FLOPs (F32), GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, FLOP:%lu, duration:%.1f ms, mean:%.1f GFLOPS, stdev=%.1f GFLOPS\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalFlops, eventMs, mean, stdev);
        }

        /* MFMA-F64 */
        numExperiments = DEFAULT_NUM_EXPERIMENTS;
        currBenchmark++;
        archs_t mfma_f64_unsupported{"gfx908"};
        if (mfma_f64_unsupported.contains(gcnArch))
        {
            totalFlops = 0;
            samples[0] = 0;
            numExperiments = 1;
            eventMs = 0;
            if (!quiet)
            {
                showProgress(1);
            }
        }
        else
        {
            totalFlops = (uint64_t)numWorkgroups * SIMDS_PER_CU * numIters * MFMA_F64_OPS;
            for (int n = 0; n < numExperiments; n++)
            {

                initHipEvents(start, stop);
                hipLaunchKernelGGL(mfma_f64, dim3(numWorkgroups), dim3(workgroupSize), 0, 0, numIters, dummy);
                stopHipEvents(eventMs, start, stop);

                samples[n] = totalFlops / eventMs / 1e6;
                if (!quiet)
                {
                    showProgress((float)n / numExperiments);
                }
            }
        }

        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["Peak MFMA FLOPs (F64)"]["mean"] = mean;
            statsMap[dev]["Peak MFMA FLOPs (F64)"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nPeak MFMA FLOPs (F64), GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, FLOP:%lu, duration:%.1f ms, mean:%.1f GFLOPS, stdev=%.1f GFLOPS\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalFlops, eventMs, mean, stdev);
        }

        /* MFMA-I8 */
        numExperiments = DEFAULT_NUM_EXPERIMENTS;
        currBenchmark++;
        archs_t mfma_i8_unsupported{"gfx940", "gfx941", "gfx942"};
        if (mfma_i8_unsupported.contains(gcnArch))
        {
            totalFlops = 0;
            samples[0] = 0;
            numExperiments = 1;
            eventMs = 0;
            if (!quiet)
            {
                showProgress(1);
            }
        }
        else
        {
            totalFlops = (uint64_t)numWorkgroups * SIMDS_PER_CU * numIters * MFMA_I8_OPS;
            for (int n = 0; n < numExperiments; n++)
            {

                initHipEvents(start, stop);
                hipLaunchKernelGGL(mfma_i8, dim3(numWorkgroups), dim3(workgroupSize), 0, 0, numIters, dummy);
                stopHipEvents(eventMs, start, stop);

                samples[n] = totalFlops / eventMs / 1e6;
                if (!quiet)
                {
                    showProgress((float)n / numExperiments);
                }
            }
        }

        stats(samples, numExperiments, &mean, &stdev, &confidence);

        perf_metrics.push_back(mean);
        perf_metrics.push_back(mean - confidence);
        perf_metrics.push_back(mean + confidence);

        if (quiet)
        {

            statsMap[dev]["Peak MFMA IOPs (I8)"]["mean"] = mean;
            statsMap[dev]["Peak MFMA IOPs (I8)"]["stdev"] = stdev;
            showProgress(((float)dev + currBenchmark / numBenchmarks) / numGpuDevices);
        }
        else
        {
            printf("\nPeak MFMA IOPs (I8), GPU ID: %d, workgroupSize:%d, workgroups:%d, experiments:%d, IOP:%lu, duration:%.1f ms, mean:%.1f GOPS, stdev=%.1f GOPS\n",
                   dev, workgroupSize, numWorkgroups, numExperiments, totalFlops, eventMs, mean, stdev);
        }

        /* Save to CSV */
        ofile << dev << std::setprecision(8);
        for (auto element : perf_metrics)
            ofile << "," << element;

        ofile << "\n";
    }

    if (quiet)
    {
        printf("\nGPU ID");
        printf("|HBM BW%-13s", " ");
        printf("|L2 BW%-13s", " ");
        printf("|L1 BW%-13s", " ");
        printf("|LDS BW%-13s", " ");
        printf("|FP32%-15s", " ");
        printf("|FP64%-15s", " ");
        printf("|BF16%-16s", " ");
        printf("|F16%-17s", " ");
        printf("|F32%-16s", " ");
        printf("|F64%-16s", " ");
        printf("|I8%-18s| \n", " ");

        for (int gpuID = 0; gpuID < numGpuDevices; gpuID++)
        {
            printf("\n%-6d| ", gpuID);
            for (auto statPair : statsMap[gpuID])
            {
                for (auto benchmarkPair : statPair.second)
                {
                    std::string stat = statPair.first;
                    std::string benchmarkType = benchmarkPair.first;
                    if (benchmarkType == "mean")
                    {
                        printf("%-7.2f ",
                               statsMap[gpuID][stat][benchmarkType]);
                    }
                    if (benchmarkType == "stdev")
                    {
                        printf("(%-7.2f%%)|",
                               statsMap[gpuID][stat][benchmarkType]);
                    }
                }
            }
        }
    }

    printf("\n");

    free(samples);
    ofile.close();

    return 0;
}
