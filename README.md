[![build](https://github.com/ROCm/rocm-amdgpu-bench/actions/workflows/build.yml/badge.svg)](https://github.com/ROCm/rocm-amdgpu-bench/actions/workflows/build.yml)

# rocm-amdgpu-bench: a roofline utility

This repository houses a proprietary rooflining utility for
benchmarking AMD Instinct GPUs starting with MI200. The `roofline`
utility directly measures performance for a variety of micro-benchmarks including:

* HBM Bandwidth
* MALL Bandwidth
* L2 Bandwidth
* L1 Bandwidth
* LDS Bandwidth
* Peak FLOPs (FP8)
* Peak FLOPs (FP32)
* Peak FLOPs (FP64)
* Peak MFMA FLOPs (F8)
* Peak MFMA FLOPs (BF16)
* Peak MFMA FLOPs (F16)
* Peak MFMA FLOPs (F32)
* Peak MFMA FLOPs (F64)
* Peak MFMA IOPs (I8)

This utility is an input dependency for AMD's
[ROCm Compute Profiler](https://github.com/ROCm/rocprofiler-compute) performance tool for profiling
machine learning/HPC workloads.


## Building locally

Compilation requires a working HIP environment that is typically
provided via a local [ROCm](https://rocm.docs.amd.com/en/latest/)
install (e.g. via installation of the `rocm-hip-sdk` package) along
with the `cmake` build tool. ROCm 6.2 or later is required to build.

### Configure

```
$ mkdir build
$ cd build
$ cmake ..
```

### Build

```
$ make
```

If successful, a `./roofline` binary will be present in the build directory.

### Packaging

To generate Debian and RPM packages for the resulting roofline binary, issue the following:

```
$ make package
```
