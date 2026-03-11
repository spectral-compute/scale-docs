# SCALE on NVIDIA GPUs

Just like with AMD GPUs, SCALE can be used with NVIDIA GPUs.
The most popular programming language used for NVIDIA GPUs is NVIDIA CUDA, and SCALE is made to be a drop-in replacement for it.
Compared to NVIDIA CUDA, SCALE can offer additional diagnostics, optimisations and optional features that allow the developers write better code.


## Enhancements

### PTX diagnostics

A lot of CUDA code includes blocks of [Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html).
Inline PTX is often used to access niche instructions, many of which have several modifiers and require multiple operands.
NVIDIA's `nvcc` does not parse inline PTX and provides [virtually no validation](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#incorrect-ptx) for it, which can make writing **correct** inline PTX a great challenge.

SCALE, on the other hand, parses those blocks of Inline PTX and provides proper warnings and errors for them.
The diagnostics range from fairly simple ones such as syntax validation, to advanced ones that involve variable type information, and that are even aware of the code outside of the `asm()` blocks!

For the following input, SCALE will produce an error about a missing semicolon, while NVIDIA's `nvcc` will report a fatal `ptxas` error caused by a "syntax error" without any indication of its cause or its source.

```cu
__device__ int ptxAdd(int x, int y) {
    int out;
    asm("add.u32 %0, %1, %2" : "=r"(out) : "r"(x), "r"(y));
    return out;
}
```

Here is a more serious example, in which NVIDIA's `nvcc` silently introduces undefined behaviour.
The example passes a generic memory pointer to an instruction that needs a shared memory pointer, a `cvta` instruction should have been added.

```cu
__global__ void foo() {
   __shared__ int example[32];
   int out;
   asm("ld.shared.b32 %0, [%1];": "=r"(out) : "l"(&example[threadIdx.x]));
}
```

In this case, SCALE produces a detailed warning and even suggests a solution:

```
[...]:7:27: warning: passing generic address-space pointer to non-generic PTX memory instruction is undefined behaviour [-Wptx-binding-as-addre
ss]
    7 |    asm("ld.shared.b32 %0, [%1];": "=r"(out) : "l"(&example[threadIdx.x]));
      |                           ^
[...]:7:27: note: use a PTX cvta instruction to convert the C++ pointer to the correct PTX address space
    7 |    asm("ld.shared.b32 %0, [%1];": "=r"(out) : "l"(&example[threadIdx.x]));
      |                           ^
1 warning generated when compiling for [...].
```

Given the same input, NVIDIA's `nvcc` silently chops off half of the pointer and introduces undefined behaviour.


### Flexible PTX `"C"` constraints

Inline PTX supports a special type of input [constraint](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints) `"C"` that receives an "array of const char" at compile-time.
It exists to make metaprogramming easier, and it can be used, for example, to customise PTX instruction modes.

In SCALE, `"C"` constraints are more flexible than in NVIDIA's `nvcc`.
For the following code, NVIDIA's `nvcc` would complain that *"The constant-expression for the 'C' constraint did not evaluate to the address of a variable"*, while SCALE will compile it successfully.

```cu
enum class PtxSaturation {
    NONE, SAT
};

constexpr const char* ptxSatToStr(PtxSaturation RM) {
    switch (RM) {
        case PtxSaturation::NONE:
            return "";
        case PtxSaturation::SAT:
            return ".sat";
    }

    return "ERROR";
}

template <PtxSaturation Sat>
__device__ float compute_add(float a, float b) {
    float result;
    asm("add.f32%1 %0, %2, %3;"
        : "=f"(result)
        : "C"(ptxSatToStr(Sat)),
          "f"(a), "f"(b));
    return result;
}

__global__ void kern(float *result, float a, float b) {
    *result++ = compute_add<PtxSaturation::NONE>(a, b); // generates add.f32
    *result   = compute_add<PtxSaturation::SAT>(a, b);  // generates add.f32.sat
}
```


### Structs in PTX

Some common NVIDIA CUDA types are represented as `struct`s, which makes it difficult to use them with inline PTX with NVIDIA's `nvcc`.
For types like `__half` and `__half2`, `reinterpret_cast`s have to be used to supply them to and receive them from the `asm()` blocks.
SCALE removes this restriction, allowing automatic conversions to happen.

The following example would not compile with NVIDIA's `nvcc`, as `asm()` expects to receive and return scalar types, while `__half` and `__half2` are structs.
With SCALE, it just works.

```cu
#include <cuda_fp16.h>

__global__ void add_half(__half* x, __half* y) {
    __half out;
    asm("add.f16 %0, %1, %2;" : "=h"(out) : "h"(*x), "h"(*y));
    *x = out;
}

__global__ void add_half2(__half2* x, __half2* y) {
    __half2 out;
    asm("add.f16x2 %0, %1, %2;" : "=r"(out) : "r"(*x), "r"(*y));
    *x = out;
}
```


### Better lambda support

NVIDIA CUDA has a concept of [Extended Lambdas](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda), rules of which are quite convoluted.
Notably, in NVIDIA CUDA, a lambda is an extended lambda only if it is annotated with `__device__` and the enclosing function is annotated with `__host__`.
There is a lot of complexity associated with extended lambdas, including [a set of type traits for metaprogramming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda-type-traits).

SCALE makes all lambda `__host__ __device__` unless that's impossible or explicitly specified otherwise.
When lambda sidedness is not specified, it looks at all the functions it calls, and all the variables it captures, and works out what its sidedness can be.
This behaviour is compatible with NVIDIA's `nvcc`, and it makes lambdas more intuitive and allows for better compiler diagnostics.
It also renders the metaprogramming using NVIDIA CUDA's Extended Lambda Type Traits redundant.

If SCALE finds that an unannotated lambda can neither be `__host__` nor `__device__`, an error is raised.
Otherwise, if SCALE finds any contradictions with the detected sidedness, errors will be raised at sites that attempt to call the lambda and provides a `note:` that explains what makes the lambda `__host__`-only or `__device__`-only.

TODO: example


### `__host__` and `__device__` attributes on class & struct declarations

SCALE allows annotating class and struct declarations with `__host__` and `__device__` attributes.
It works as a shortcut for declaring all of the member functions with the same attributes.

```cu
struct __host__ HostOnly {
    int f();
};

struct __device__ DeviceOnly {
    int f();
};

struct __host__ __device__ HostAndDevice {
    int f();
};

// The above is equivalent to:
//
// struct HostOnly {
//    int __host__ f();
// };
//
// struct DeviceOnly {
//    int __device__ f();
// };
//
// struct HostAndDevice {
//     int __host__ __device__ f();
// };


__host__ void host_function(HostOnly &host_only, DeviceOnly &device_only, HostAndDevice &host_and_device) {
    host_only.f();
    // device_only.f();  // Will produce an error
    host_and_device.f();
}

__host__ __device__ void host_device_function(HostOnly &host_only, DeviceOnly &device_only, HostAndDevice &host_and_device) {
    // host_only.f();    // Will produce a warning
    // device_only.f();  // Will produce an error
    host_and_device.f();
}

__device__ void device_function(HostOnly &host_only, DeviceOnly &device_only, HostAndDevice &host_and_device) {
    // host_only.f();    // Will produce an error
    device_only.f();
    host_and_device.f();
}
```
