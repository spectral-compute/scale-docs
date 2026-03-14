# API Coverage Report

These pages provide a diff between the SCALE Runtime API's headers and
the NVIDIA documentation.

This resource makes it easy to determine which CUDA APIs are supported
by SCALE when cross-compiling to AMD targets.

When compiling for NVIDIA GPUs, NVIDIA's own libraries are used. In that
case, you'll have access to the same set of APIs as are provided by that
version of the NVIDIA CUDA Toolkit.

- [Driver API](./api-driver.md)
- [Math API](./api-math.md)
- [Runtime API](./api-runtime.md)

The lists are based on the official Nvidia documentation and use the same layout.

## Presentation

The lists are presented using `diff` syntax. Missing APIs are prefixed with a `-`,
causing them to highlight red in the listing.
This makes it easy to see which entries are available and which [may be missing](#correctness).

For example:

```diff
const char * cudaGetErrorName(cudaError_t);
__device__ const char * cudaGetErrorName(cudaError_t);
const char * cudaGetErrorString(cudaError_t);
__device__ const char * cudaGetErrorString(cudaError_t);
cudaError_t cudaGetLastError();
-__device__ cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
-__device__ cudaError_t cudaPeekAtLastError();
```

In this example, functions `cudaGetErrorName` and `cudaGetErrorString` are available on both host and device.
Functions `cudaGetLastError` and `cudaPeekAtLastError` are available on host, and are not available on device.

## Correctness

The lists may say that something is unavailable when it's not the case.
This may happen for a few reasons.

NVIDIA documentation may differ from what CUDA provides in reality.
An example of that is differences in `const`-ness of some function arguments.
In such cases SCALE may be forced to maintain "bug compatibility" and the
functions stop matching what NVIDIA documentation promises.

SCALE retains support for some old APIs NVIDIA have since deleted.

Found a mistake?

[Let us know](../../../contact/report-a-bug.md), or [file a pull request](https://github.com/spectral-compute/scale-docs)
