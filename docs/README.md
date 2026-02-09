# SCALE by Spectral Compute

## What is SCALE?

SCALE is a GPGPU programming toolkit that can natively compile CUDA
applications for AMD GPUs.

SCALE does not require the CUDA program or its build system to be modified.

Support for more GPU vendors and CUDA APIs is in development.

## How do I use SCALE?

1. [Install SCALE](./manual/how-to-install.md).
2. Activate SCALE, eg. `. /opt/SCALE/scaleenv gfx1100`
3. Compile your [CUDA code with SCALE](https://docs.scale-lang.com/stable/manual/how-to-use/), following the same steps as for NVIDIA CUDA.

## What projects have been tested?

We validate SCALE by compiling open-source CUDA projects and running their
tests. The list of currently-tested projects and their compatibility status
can be found [here](https://github.com/spectral-compute/scale-validation?tab=readme-ov-file#current-status).

Join our [Discord](https://discord.com/invite/KNpgGbTc38) to let us know what projects are missing (_or_ support our mission by contributing yourself).

## What are examples of using SCALE?

Our [SCALE Examples](https://docs.scale-lang.com/stable/examples/) section show very basic examples of using SCALE and are a great place to start your hands-on experience.

We welcome contributions from our developer community. Join our [Discord](https://discord.com/invite/KNpgGbTc38) to share _your_ SCALE projects.

## Which GPUs are supported?

The following GPU targets are supported:

### AMD
Full technical specifications [available here](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html).

#### Enterprise
| Name                         | Architecture        | LLVM target |
|------------------------------|---------------------|-------------|
| MI350X / MI355X              | CDNA 4              | gfx950      |
| MI300A / MI300X / MI325X     | CDNA 3              | gfx942      |
| MI210 / MI250 / MI250X       | CDNA 2              | gfx90a      |
| MI100                        | CDNA 1              | gfx908      |
| MI50 / MI60                  | GCN 5.1             | gfx906      |
| MI25                         | GCN 5.0             | gfx900      |

#### Consumer
| Name                         | Architecture        | LLVM target |
|------------------------------|---------------------|-------------|
| Radeon AI PRO R9600D / R9700 | RDNA 4              | gfx1201     |
| Radeon RX 9070               | RDNA 4              | gfx1201     |
| Radeon RX 9060               | RDNA 4              | gfx1200     |
| Ryzen AI Max+ PRO 395        | RDNA 3.5            | gfx1151     |
| Radeon RX 7600               | RDNA 3              | gfx1102     |
| Radeon PRO v710 / W7700      | RDNA 3              | gfx1101     |
| Radeon RX 7700 / 7800        | RDNA 3              | gfx1101     |
| Radeon PRO W7800 / W7900     | RDNA 3              | gfx1100     |
| Radeon RX W7900 XT / XTX     | RDNA 3              | gfx1100     |
| Radeon PRO v620 / W6800      | RDNA 2              | gfx1030     |
| Radeon RX 6800 / 6900 / 6950 | RDNA 2              | gfx1030     |
| Radeon Pro W5700             | RDNA 1              | gfx1010     |


[Contact us](#contact-us) if you want us to expedite support for a particular AMD GPU
architecture.

## What are the components of SCALE?

SCALE consists of:

- An `nvcc`-compatible compiler capable of compiling nvcc-dialect CUDA for AMD
  GPUs, including PTX asm.
- An implementation of the CUDA runtime, driver and math APIs for AMD GPUs.
- Wrapper libraries providing the "CUDA-X" APIs by delegating to the
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
