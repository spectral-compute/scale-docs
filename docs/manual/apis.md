# Introduction to implemented APIs

This section of the manual lists CUDA APIs that are implemented by SCALE.
It is split in three subsections:

- [Driver API](./api-driver.md)
- [Math API](./api-math.md)
- [Runtime API](./api-runtime.md)

The lists are based on the official Nvidia documentation and use the same layout.
Every heading links back to the page where its entries originate from.
We compare those entries against SCALE source code and identify the parts that are present or missing.

There are several types of entries:

- **Macros**, for which we only compare the names, as some of the values differ naturally between NVIDIA CUDA and SCALE
- **Types**, for which we compare:
    - their names
    - the exact way they are defined
- **Functions**, for which we compare:
    - their names
    - types and default values of their arguments (argument names are stripped among some other things)
    - their type arguments if present

The rules for comparing functions are also applied to function pointer types which are sometimes `typedef`'d or are themselves listed as function arguments.

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
In such cases SCALE may be forced to maintain "bug compatibility" and the functions stop matching what NVIDIA documentation promises.

Many functions are called conditionally and may never get used in certain scenarios.
For some of those functions, SCALE may provide an empty implementation.
By doing this, SCALE allows more projects pass compilation and linking.
We don't want to list such empty functions as available, so we manually mark them as missing to avoid confusion.

The code that compares entries from SCALE against NVIDIA documentation may contain imperfections.
For this reason, some successful matches may simply get missed.

Note that these lists currently **don't check members of types** such as struct fields or enum variants.

[Reach out to us](../contact/report-a-bug.md) if you experience problems for any of these reasons.
Possible problems with the entires would require our attention on a case-by-case basis.
Your feedback will help us find possible inconsistencies and prioritise our work to fix them.
