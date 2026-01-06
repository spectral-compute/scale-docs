# SCALE by Spectral Compute

## What is SCALE?

SCALE is a GPGPU programming toolkit that can natively compile CUDA
applications for AMD GPUs.

SCALE does not require the CUDA program or its build system to be modified.

Support for more GPU vendors and CUDA APIs is in development.

## How do I use SCALE?

1. [Install SCALE](./manual/how-to-install.md).
2. Activate SCALE, eg. `. /opt/SCALE/scaleenv gfx1100`
3. Compile your application, following the same steps you would use for
   NVIDIA CUDA.

## What projects have been tested?

We validate SCALE by compiling open-source CUDA projects and running their
tests. The list of currently-tested projects and their compatibility status
can be found [here](https://github.com/spectral-compute/scale-validation?tab=readme-ov-file#current-status)

## Which GPUs are supported?

The following GPU targets are supported:

- AMD `gfx900` (Vega 10, GCN 5.0)
- AMD `gfx906` (MI50/MI60)
- AMD `gfx908` (MI100)
- AMD `gfx90a` (MI210/MI250)
- AMD `gfx942` (MI300x)
- AMD `gfx950` (MI350x/MI355x)
- AMD `gfx1010`
- AMD `gfx1030` (Navi 21, RDNA 2.0)
- AMD `gfx1100` (Navi 31, RDNA 3.0)
- AMD `gfx1151` (Strix-halo etc.)
- AMD `gfx1101`
- AMD `gfx1102`
- AMD `gfx1200` (RX 9060 XT etc.)
- AMD `gfx1201` (RX 9070 XT etc.)

[Contact us](#contact-us) if you want us to expedite support for a particular AMD GPU
architecture.

## What are the components of SCALE?

SCALE consists of:

- An `nvcc`-compatible compiler capable of compiling nvcc-dialect CUDA for AMD
  GPUs, including PTX asm.
- An implementation of the CUDA runtime, driver and math APIs for AMD GPUs.
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
