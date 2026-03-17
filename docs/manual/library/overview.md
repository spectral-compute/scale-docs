# Overview

The SCALE runtime library is an implementation of the CUDA Math, Driver,
and Runtime APIs. This is necessary when compiling CUDA programs for
non-NVIDIA targets, where those libraries are not present.

The SCALE Runtime library is only needed when building for AMD GPUs.
When targeting NVIDIA, the NVIDIA libraries are used.

The purpose of the SCALE runtime library is to provide CUDA-compatible
implementations of these APIs, so existing programs work as expected.
Each function in these APIs should do _exactly_ what the corresponding
one from NVIDIA's implementation does.

See also:
- [Performance Tuning Guide](./performance-tuning.md)
- [API Extensions](./runtime-extensions.md)

## Comparison to HIP

The SCALE runtime library differs from the HIP runtime library in that it
aims to precisely replicate the behaviour of the corresponding CUDA APIs.

Many HIP APIs do subtly different things compared to their similarly-named
CUDA counterparts, creating a compatibility obstacle. The goals of the
two libraries are therefore slightly different: HIP appears to aim to provide
a "more sensible" API, whereas SCALE aims to provide a drop-in replacement
that makes CUDA programs work out of the box.
