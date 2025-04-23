# API Coverage Report

These pages provide a diff between SCALE's headers and the NVIDIA 
documentation, describing which APIs are supported by SCALE.

- [Driver API](./api-driver.md)
- [Math API](./api-math.md)
- [Runtime API](./api-runtime.md)

The lists are based on the official Nvidia documentation and use the same layout.

## Presentation

The lists are presented using `diff` syntax highlighting of code blocks.
This allows seeing which entries are available and which [may be missing](#correctness).
Missing entries are prefixed with `-` (a minus) which paints them red in the list.

By default, `__host__` qualifier is assumed for functions, it is removed if present.
Functions that are qualified as `__host__ __device__` are split into two separate entries.

Here is an example:

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

Many functions are called conditionally and may never get used in certain scenarios.
For some of those functions, SCALE may provide an empty implementation.
By doing this, SCALE allows more projects pass compilation and linking.
We don't want to list such empty functions as available, so we manually mark them as missing to avoid confusion.

The code that compares entries from SCALE against NVIDIA documentation may contain imperfections.
For this reason, some successful matches may simply get missed.

Note that these lists currently **don't check members of types** such as struct fields or enum variants.

[Reach out to us](../contact/report-a-bug.md) if you experience problems for any of these reasons.
Possible problems with the entries require our attention on a case-by-case basis.
Your feedback will help us find possible inconsistencies and prioritise our work to fix them.
