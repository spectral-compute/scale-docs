# SCALE by Spectral Compute

## What is SCALE?

SCALE is a CUDA-compatible GPU programming toolkit targeting both
NVIDIA and AMD GPUs.

Instead of porting programs away from CUDA, SCALE offers a vendor-neutral platform
that allows users to keep their existing codebases without compromising on performance
or developer productivity.

SCALE is in active development. [Get in touch](./contact/join-our-discord.md) if the wheels fall off.

## What can SCALE do?

Examples of what SCALE can be used for:

- [Compile existing CUDA](./manual/tutorials/how-to-use.md) codebases for AMD GPUs,
  with zero code changes. Inline asm? No problem.
- [Provide valuable compiler diagnostics](./manual/compiler/diagnostic-reference.md),
  helping you spot problems early.
- [IDE integration](./manual/tutorials/editors/vscode.md) that properly understands CUDA.
- Improve the perforance of CUDA programs on NVIDIA GPUs (sometimes).

## How does it work?

SCALE provides:

- A drop-in replacement for NVIDIA `nvcc`, capable of compiling nvcc-dialect
  CUDA for AMD & NVIDIA GPUs.
- An implementation of the CUDA runtime, driver and math APIs for AMD GPUs.
- Wrapper libraries providing the "CUDA-X" APIs by delegating to the
  corresponding ROCm libraries for AMD GPUs.

For NVIDIA targets, SCALE replaces `nvcc`,
offering improved compiler diagnostics and - sometimes - performance.

For AMD targets, SCALE also provides the CUDA runtime,
driver, and math libraries. SCALE's implementations are often faster than HIP,
and match the behaviour of the NVIDIA CUDA APIs more precisely.

![Compilation Trajectories](./manual/imgs/OneFileThreeDestines.svg){ align=left }

## What projects have been tested?

We validate SCALE by compiling open-source CUDA projects and running their
tests. The list of currently-tested projects and their compatibility status
can be found [here](https://github.com/spectral-compute/scale-validation?tab=readme-ov-file#current-status).

We also offer some [toy programs](./examples/README.md) to illustrate the process in a very simple
environment.

Join our [Discord](https://discord.com/invite/KNpgGbTc38) to let us know what projects are missing, or support our mission by contributing yourself.

## Which GPUs are supported?

Below is a list of currently supported GPU targets.

=== "AMD"

    Full technical specifications [available here](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html).

    === "Enterprise"

        | Name                         | Architecture        | LLVM target |
        |------------------------------|---------------------|-------------|
        | MI350X / MI355X              | CDNA 4              | gfx950      |
        | MI300A / MI300X / MI325X     | CDNA 3              | gfx942      |
        | MI210 / MI250 / MI250X       | CDNA 2              | gfx90a      |
        | MI100                        | CDNA 1              | gfx908      |
        | MI50 / MI60                  | GCN 5.1             | gfx906      |
        | MI25                         | GCN 5.0             | gfx900      |

    === "Consumer"

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

=== "NVIDIA"

    You can find the list of devices for each of the compute capabilities in the NVIDIA developer documentation: [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus).

    For devices older than `sm_75`, you must install a version of the NVIDIA CUDA Toolkit older
    than 13.0. In NVIDIA mode, SCALE provides only the compiler: the libraries used are still
    NVIDIA's.

    | Compute Capability | LLVM Target | Example GPU |
    |--------------------|-------------|-------------|
    | 12.0               | `sm_120`, `sm_120a` | NVIDIA RTX PRO 6000 Blackwell, GeForce RTX 5090 |
    | 10.1               | `sm_101`, `sm_101a` | NVIDIA B100 |
    | 10.0               | `sm_100`, `sm_100a` | NVIDIA B200 |
    | 9.0                | `sm_90`, `sm_90a`   | NVIDIA H200 |
    | 8.9                | `sm_89`     | NVIDIA RTX 6000 Ada, GeForce RTX 4090 |
    | 8.7                | `sm_87`     | Jetson AGX Orin |
    | 8.6                | `sm_86`     | NVIDIA A40, NVIDIA RTX A6000, GeForce RTX 3090 |
    | 8.0                | `sm_80`     | NVIDIA A100 |
    | 7.5                | `sm_75`     | NVIDIA T4, QUADRO RTX 8000, NVIDIA T1200, GeForce RTX 2080 |
    | 7.2                | `sm_72`     | Jetson AGX Xavier |
    | 7.0                | `sm_70`     | NVIDIA V100, NVIDIA TITAN V |
    | 6.2                | `sm_62`     | Jetson TX2 |
    | 6.1                | `sm_61`     | Tesla P40, GeForce GTX 1080 |
    | 6.0                | `sm_60`     | Tesla P100 |
    | 5.3                | `sm_53`     | Jetson Nano |
    | 5.2                | `sm_52`     | Tesla M60, GeForce GTX 980 |
    | 5.0                | `sm_50`     | GeForce GTX 750 |
    | 3.7                | `sm_37`     | Tesla K80 |
    | 3.5                | `sm_35`     | Tesla K40, GeForce GTX 780 |
    | 3.2                | `sm_32`     | Tegra K1 |
    | 3.0                | `sm_30`     | Tesla K10, GeForce GTX 770 |
    | 2.1                | `sm_21`     | Quadro 2000 |
    | 2.0                | `sm_20`     | Quadro Plex 7000 |

## How does SCALE compare to other solutions?

Instead of providing a [new way](https://xkcd.com/927/) to write GPU
software, SCALE aims to augment CUDA (the de-facto standard).

SCALE aims to be fully compatible with NVIDIA CUDA. We believe that users
should not have to maintain multiple codebases or compromise on performance
to support multiple GPU vendors.

SCALE's language is a _superset_ of NVIDIA CUDA. Explore offering some
[enhancements](./manual/compiler/dialects.md) and opt-in
[extensions](./manual/compiler/language-extensions.md).

SCALE is a work in progress. If something gets in your way, please contact us.

## Contact us

There are multiple ways to get in touch with us:

 - Join our [Discord](https://discord.gg/KNpgGbTc38)
 - Send us an e-mail at [hello@spectralcompute.co.uk](mailto:hello@spectralcompute.co.uk)
