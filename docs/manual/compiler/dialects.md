# CUDA Language Support

The CUDA programming language is not formally specified. The "standard" is
therefore approximately "whatever `nvcc` does". Although `clang` supports
compiling CUDA, it supports a somewhat different dialect compared to `nvcc`.

You can read more about (some of) the specific differences in
[the LLVM manual page about it](https://llvm.org/docs/CompileCudaWithLLVM.html#dialect-differences-between-clang-and-nvcc). The LLVM manual
unironically notes "This section is painful, hopefully you can skip it and
live your life blisfully unaware".

Many CUDA programs are written with `nvcc` in mind andcannot be successfully
compiled with `clang` because they depend on one of these quirks.

HIP experiences the same problem: the HIP compiler is based on LLVM, so HIP
is closer to "LLVM-dialect CUDA" than it is to "nvcc-dialect CUDA". This
causes some CUDA programs to fail in "interesting" ways when ported to HIP. It's
not really the case that you can remap all the API calls to the HIP ones and
expect it to work: nvcc-CUDA, and LLVM-CUDA/HIP have quite different C++
semantics.

SCALE resolves this issue by offering two compilers:

- `"nvcc"`: replicates the behaviour of NVIDIA's `nvcc`,
  allowing existing CUDA programs to compile directly.
- `clang`: providing clang's usual clang-dialect-CUDA support, with our
  opt-in language extensions.

## SCALE Language Enhancements

SCALE's `nvcc` aims to be compatible with all existing code, but does not replicate
bugs except when strictly necessary. In many cases, we can improve on the behaviour
of the NVIDIA compiler without breaking existing programs.

### Modern C++

SCALE accepts all C++ features supported by the version of Clang is is based on
(currently {{current_llvm_version}}). Clang maintains an exhaustive list of which
versions support which C++ features [here](https://clang.llvm.org/cxx_status.html).

As of SCALE 1.6.0, all of C++23 and much of C++26 is supported.

### Better lambda support

NVIDIA CUDA has a concept of [Extended Lambdas](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda), which have quite convoluted rules.

In NVIDIA CUDA, a lambda is an extended lambda only if it is annotated with `__device__` and the enclosing function is annotated with `__host__`.
There is a lot of complexity associated with extended lambdas, including [a set of type traits for metaprogramming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda-type-traits).

SCALE makes all lambda `__host__ __device__` unless that's impossible or explicitly specified otherwise.
When lambda sidedness is not specified, it looks at all the functions it calls, and all the variables it captures, and works out what its sidedness can be.
This behaviour is compatible with NVIDIA's `nvcc`, makes lambdas more intuitive, and allows for better compiler diagnostics.
It also renders the metaprogramming using NVIDIA CUDA's Extended Lambda Type Traits redundant.

If SCALE finds that an unannotated lambda can neither be `__host__` nor `__device__`, an error is raised.
Otherwise, if SCALE finds any contradictions with the detected sidedness, errors will be raised at sites that attempt to call the lambda and provides a `note:` that explains what makes the lambda `__host__`-only or `__device__`-only.

TODO: example


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
