# SCALE by Spectral Compute

## What is SCALE?

SCALE is a GPGPU programming toolkit that allows CUDA applications to be
natively compiled for AMD GPUs.

SCALE does not require the CUDA program or its build system to be modified.

Support for more GPU vendors and CUDA APIs is in development.

To get started:

- See the [tutorial](./manual/how-to-use.md).
- Review the [examples](./examples/README.md).
- Check out the [FAQ](./manual/faq.md)
- [Contact us](#contact-us) for help.

## How does it work?

SCALE has several key innovations compared to other cross-platform GPGPU
solutions:

- SCALE accepts CUDA programs as-is. No need to port them to another 
  language. This is true even if your program uses inline PTX `asm`.
- The SCALE compiler accepts the same command-line options and CUDA dialect
  as `nvcc`, serving as a drop-in replacement.
- "Impersonates" an installation of the NVIDIA CUDA Toolkit, so existing 
  build tools and scripts like `cmake` _just work_.

## What projects have been tested?

We validate SCALE by compiling open-source CUDA projects and running their
tests.

The following open-source projects are currently part of our nightly automated 
tests and pass fully:

| Project                                             | Version Tested |
|-----------------------------------------------------|----------------|
| [AMGX](https://github.com/NVIDIA/AMGX)              | `v2.4.0`       |
| [Blender Cycles](https://github.com/blender/cycles) | `v4.2.0`       |
| [faiss](https://github.com/facebookresearch/faiss)  | `v1.9.0`       |
| [FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2)  | `4e02604`      |
| [GOMC](https://github.com/GOMC-WSU/GOMC)            | `9fc85f`       |
| [GPUJPEG](https://github.com/CESNET/GPUJPEG)        | `3e045d1`      |
| [gpu_jpeg2k](https://github.com/ePirat/gpu_jpeg2k)  | `ee715e9`      |
| [hashcat](https://github.com/hashcat/hashcat)       | `6716447dfc`   |
| [llama-cpp](https://github.com/ggerganov/llama.cpp) | `b1500`        |
| [NVIDIA Thrust](https://github.com/NVIDIA/thrust)   | `756c5af`      |
| [stdgpu](https://github.com/stotko/stdgpu)          | `3b7d712`      |
| [xgboost](https://github.com/dmlc/xgboost)          | `v2.1.0`       |

The scripts we use to build and test these projects (and others that do not 
yet entirely work) are available
[on github](https://github.com/spectral-compute/scale-validation). You can use
these to reproduce our results (and find bugs!).

## Which GPUs are supported?

The following GPU targets are supported, and are covered by our nightly tests:

- AMD `gfx900` (Vega 10, GCN 5.0)
- AMD `gfx1030` (Navi 21, RDNA 2.0)
- AMD `gfx1100` (Navi 31, RDNA 3.0)

The following GPU targets have undergone ad-hoc manual testing and "seem to
work":

- AMD `gfx1010`
- AMD `gfx1101`
- AMD `gfx1102`

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

Instead of providing a [new way](https://xkcd.com/927/) to write GPGPU 
software, SCALE allows programs written using the widely-popular CUDA
language to be directly compiled for AMD GPUs.

SCALE aims to be fully compatible with NVIDIA CUDA. We believe that users 
should not have to maintain multiple codebases or compromise on performance
to support multiple GPU vendors.

SCALE's language is a _superset_ of NVIDIA CUDA, offering some opt-in
[language extensions](./manual/language-extensions.md)
that can make writing GPU code easier and more efficient for users who wish
to move away from `nvcc`.

SCALE is a work in progress. If there is a missing API that is blocking your
attempt to use SCALE, please contact us so we can prioritise its development.

## Contact us

There are multiple ways to get in touch with us:

 - Join our [Discord](https://discord.gg/KNpgGbTc38)
 - Send us an e-mail at [hello@spectralcompute.co.uk](mailto:hello@spectralcompute.co.uk)
