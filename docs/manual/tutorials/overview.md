# Overview

SCALE consists of two main components, each of which has their own
section in this manual:

- The compiler.
- The runtime library.

The compiler serves the same function as NVIDIA's `nvcc`. It compiles
your `.cu` files for the GPU you want to use. The editor integrations
are provided by the compiler.

The runtime library contains implementations of the CUDA Runtime, driver,
and math APIs. This component is only used if you are compiling for a
non-NVIDIA platform. When compiling for an NVIDIA GPU, we use NVIDIA's
runtime libraries.
