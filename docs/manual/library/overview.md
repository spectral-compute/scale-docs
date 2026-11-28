# Overview

The SCALE runtime library is an implementation of the CUDA Runtime,
Driver, and Math APIs.

It serves two key functions:

- Providing CUDA-equivalent implementations of these APIs when building
  CUDA programs for AMD targets (since the HIP runtime semantics do not
  match up).
- Providing HIP-equivalent implementations of the HIP APIs when building
  HIP programs for NVIDIA targets (by implementing the needed semantics
  atop the CUDA APIs, which are then provided by NVIDIA's toolkit).

## Selecting an API version

The SCALE runtime library can impersonate every version of the CUDA
API back to 8.0. You can select the API version you want in one of
two ways:

1. Set the `__SCALE_CUDA_VER_MAJOR__` and `__SCALE_CUDA_VER_MINOR__`
   macros (to eg `12` and `9`, respectively). This must be done on
   the command-line for `.cu` files, and before including any CUDA
   headers for other types of file.
2. Set the `SCALE_CUDA_VERSION` environment variable (to eg. `12.9`).

This is required only at compile-time. The runtime library itself
contains every version of every API function.
