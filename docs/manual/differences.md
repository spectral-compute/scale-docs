# Differences from NVIDIA CUDA

There are some areas where SCALE's implementation of a certain feature also
found in NVIDIA CUDA has different behaviour. This document does not
enumerate _missing_ CUDA APIs/features.

## Defects

### NVRTC differences

SCALE's current implementation of the nvrtc API works by calling the
compiler as a subprocess instead of a library. This differs from how
NVIDIA's implementation works, and means that the library must be able to
locate the compiler to invoke it.

If your program that uses the rtc APIs fails with errors relating to being
unable to locate the compiler, ensure that SCALE's `nvcc` is first in PATH.

### Stream synchronization

SCALE does not yet support
[per-thread default stream behaviour](http://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html).

Instead, the default stream is used in place of the per-thread default stream.
This will not break programs, but is likely to reduce performance.

A workaround which will also slightly improve the performance of your
program when run on NVIDIA GPUs is to use nonblocking CUDA streams
explicitly, rather than relying on the implicit CUDA stream.

### Host-side `__half` support

The CUDA API allows many `__half` math functions to be used on both host and 
device.

When compiling _non-CUDA_ translation units, you can include `<cuda_fp16.h>` 
and use the `__half` math APIs in host code.

NVIDIA's CUDA implementation converts the `__half` to 32-bit `float`, does the
calculation, and converts back.

SCALE only allows these functoins to be used when the host compiler supports 
compiling fp16 code directly (via the `_Float16` type). Current versions of 
both gcc and clang both support this.

This difference only applies to _non-CUDA_ translation units (ie. not `.cu` 
files) using compilers at least 2 years old.

This means:

- All `__half` APIs work in host code in `.cu` files.
- `__half` APIs that perform floating point math will not compile in host 
  code in non-CUDA translation units if an old host compiler is used.
- The outcome of `__half` calculations on host/device will always be the same.
- APIs for using `__half` as a storage type are always supported.

SCALE bundles a modern host compiler at `<SCALE_DIR>/targets/gfxXXX/bin/gcc` 
you can use as a workaround if this edgecase becomes a problem.

## Enhancements

### Contexts where CUDA APIs are forbidden

NVIDIA's implementation forbids CUDA APIs in various contexts, such as from
host-side functions enqueued onto streams.

This implementation allows CUDA API calls in such cases.

### Static initialization and deinitialization

This implementation permits the use of CUDA API functions during global static
initialization, `thread_local` static initialization, and
`thread_local` static deinitialization.

It is not permitted to use CUDA API functions during static deinitialization.

This is more permissive than what is allowed by NVIDIA's implementation.

### Device `printf`

SCALE's device `printf` accepts an unlimited number of arguments if you compile
with at least C++11.

If you target an older version of C++ then it is limited to 32, like NVIDIA's
implementation.

### Contexts

If `cuCtxDestroy()` is used to destroy the context that is current to a
different CPU thread, and that CPU thread then issues an API call that
depends on the context without first setting a different context to be
current, the behaviour is undefined.

In NVIDIA's implementation, this condition returns
`CUDA_ERROR_CONTEXT_IS_DESTROYED`.

Matching NVIDIA's behaviour would have incurred a small performance penalty
on many operations to handle an edgecase that is not permitted.
