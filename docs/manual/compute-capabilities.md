# Compute Capability Mapping

"Compute capability" is a numbering system used by NVIDIA's CUDA tools to
represent different GPU targets. The value of the ``__CUDA_ARCH__`` macro is
derived from this, and it's how you communicate with `nvcc` to request the
target to build for.

GPUs from other vendors have their own numbering scheme, such as AMD's
`gfx1234` format.

CUDA projects sometimes do numeric comparisons on the compute capability
value to enable/disable features using the preprocessor. This is a problem,
since those comparisons are inherently meaningless when targeting non-NVIDIA
hardware.

There is no meaningful mapping between compute capability numbers and the
hardware of other vendors.

SCALE addresses this problem by providing a "CUDA installation directory"
for each supported GPU target. By default, the `nvcc` in each of these
directories maps *every* compute capability number to the corresponding AMD
GPU target.

That works fine for single-architecture builds and most simple cases,
but has two obvious flaws:
- Building for multiple architectures is unrepresentable.
- nvRTC code may use arbitrary logic to choose a different target.

SCALE provides two tools to address this issue:
- The compiler and RTC APIs accept AMD architectures eg. `-arch gfx1100`, so
  you can modify your build scripts or nvRTC code to explicitly ask for the
  correct architecture, entirely opting-out of the mapping tricks.
- ccmap configuration, which allows you to use config files or environment
  variables to create an arbitrary mapping between compute capability numbers
  and real GPU architectures.

## Configuration file format

The configuration file provides the answer to two questions:

- For a given compute capability, which GPU should we compile for?
- Given a device binary, which compute capability should we say it has?

Each line consists of a GPU architecture, a space, and a compute capbility
number. Entries are tried in order, and the first applicable one is used,
so it's possible to unambiguously map more than one ISA and compute
capability to each other.

A line consisting of just a GPU architecture name is a wildcard which
maps all remaining compute capilibite sto that GPU architecture.

Lines starting with `#` are comments.

### Example

```
# The library will report compute capability 6.1 for gfx900 devices. The compiler will use gfx900 for `sm_61` or
# `compute_61`.
gfx900 61

# The library will report compute capability 8.6 for gfx1030 devices. The compiler will use gfx1030 for any of `sm_80`,
# `compute_80`, `sm_86`, or `compute_86`.
gfx1030 86
gfx1030 80

# The compiler will use gfx1100 for any compute capability other than 6.1, 8.0, or 8.6.
gfx1100
```

## Search locations for the library

The library searches for a compute capability map in the following order:

- The file pointed the `SCALE_CCMAP` environment variable.
- `../share/scale/ccmap.conf` relative to the directory containing `libredscale.so`.
- `${HOME}/.scale/ccmap.conf`
- `/etc/scale/ccmap.conf`

## Search locations for the compiler

The compiler searches for a compute capability map in the following order:

- The file pointed to by the `--cuda-ccmap` flag.
- The file pointed the `SCALE_CCMAP` environment variable.
- `../share/scale/ccmap.conf` relative to the directory containing the
  compiler executable.
