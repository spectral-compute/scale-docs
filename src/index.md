# SCALE by Spectral Compute

## What is SCALE?

SCALE is a GPGPU programming toolkit that allows CUDA applications to be
natively compiled for AMD GPUs.

SCALE does not require the CUDA program or its build system to be modified.

Support for more GPU vendors and CUDA APIs is in development.

To get started:

- See the [tutorial](./manual/02-how-to-use.md).
- Review the [examples](./examples/index.md)
- [Contact us](#contact-us) for help.

## How does it work?

SCALE has several key innovations missing from other cross-platform GPGPU
solutions:

- The SCALE compiler accepts the same command-line options and CUDA dialect
  as `nvcc`, serving as a drop-in replacement.
- The compiler accepts inline PTX `asm`, and can compile it for AMD GPUs.
- Seamless integration with existing build tools by "looking like" an
  installation of the NVIDIA CUDA Toolkit.
  This eliminates the need to modify your project's build system: just point
  `cmake` (or equivalent) to SCALE instead of NVIDIA CUDA, and your compilation
  will produce AMD-compatible binaries instead.

## What projects have been tested?

We validate the correctness of SCALE by compiling open-source CUDA projects and
running their tests.
The following popular open-source projects are part of our nightly automated
tests and pass fully:

- [NVIDIA Thrust](https://github.com/NVIDIA/thrust)
- [Blender Cycles](https://github.com/blender/cycles)
- [AMGX](https://github.com/NVIDIA/AMGX)
- [llama-cpp](https://github.com/ggerganov/llama.cpp)
- [faiss](https://github.com/facebookresearch/faiss)
- [xgboost](https://github.com/dmlc/xgboost)
- [GOMC](https://github.com/GOMC-WSU/GOMC)
- [stdgpu](https://github.com/stotko/stdgpu)
- [hashcat](https://github.com/hashcat/hashcat)

## Which GPUs are supported?

The following GPU targets are supported, and covered by our nightly tests:

- AMD `gfx1030` (Navi 21, RDNA 2.0)
- AMD `gfx1100` (Navi 31, RDNA 3.0)

The following GPU targets have undergone ad-hoc manual testing and "seem to 
work":

- AMD `gfx1010`
- AMD `gfx1101`

We are working on supporting the following GPUs:

- AMD `gfx900` (Vega 10, GCN 5.0)

[Contact us](#contact-us) if you want us to expedite support for a particular AMD GPU
architecture.

## What are the components of SCALE?

SCALE consists of:

- An `nvcc`-compatible compiler capable of compiling nvcc-dialect CUDA for AMD
  GPUs, including PTX asm.
- Implementations of the CUDA runtime and driver APIs for AMD GPUs.
- Open-source wrapper libraries providing the "CUDA-X" APIs by delegating to the
  corresponding ROCm libraries.
  This is how libraries such as `cuBLAS` and `cuSOLVER` are handled.

## What are the differences between SCALE and other solutions?

SCALE aims to be fully compatible with NVIDIA CUDA.
We believe that users should not have to maintain multiple codebases or
compromise on performance to support multiple GPU vendors. A toolchain that 
can compile CUDA for multiple vendors is the best way to achieve that.

SCALE's language is a _superset_ of NVIDIA CUDA.

SCALE offers some [language extensions](./manual/language-extensions.md)
that can make writing GPU code easier and more efficient.

SCALE is a work in progress. If there is a missing API that is blocking your
attempt to use SCALE, please contact us so we can prioritise its development.

## Contact us

<!-- TODO: provide contact info -->
