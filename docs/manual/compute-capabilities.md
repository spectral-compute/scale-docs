# Compute Capability Mapping

"Compute capability" is a numbering system used by NVIDIA's CUDA tools to
represent different GPU targets. The value of the `__CUDA_ARCH__` macro is
derived from this, and it's how you communicate with `nvcc` to request the
target to build for.

AMD GPUs are identified using architecture IDs like `gfx1030`.

Buildsystems for existing CUDA projects are generally unable to accept AMD
format target identifiers. Existing CUDA projects also frequently do numeric
comparisons on the compute capability value to enable/disable features using
the preprocessor. If we simply used the "GFX number" as the compute
capability, these numeric comparisons would malfunction.

To solve this, SCALE:

- Provides one "CUDA installation directory" per AMD target, all of which
  map the NVIDIA compute capability "sm_86" to the corresponding AMD GPU target.
- Aims to support all CUDA APIs everywhere, including ones that NVIDIA
  deprecated/removed for newer compute capability targets.

86 was chosen because it's a "fairly new" NVIDIA target that should enable
most features in most projects. The default mapping number is likely to
increase over time.

Behind the scenes, this mapping is governed by a configuration file. Users
who wish to take direct control of the process (either by changing the
mapping, or modifying their project to accept both AMD and NVIDIA-format
target identifiers) may do so. The remainder of this document coves this topic.

The special target directory `<SCALE>/targets/gfxany` is a SCALE toolchain
with no Compute Capability Mapping configuration. This may be used in
combination with `clang++`'s usual flags for selecting a GPU target.

## Default numbering scheme

By default (when no ccmap is in use), AMD GPUs are assigned a compute
capability of at least 60000.0, or `sm_600000`.

For a given GPU, e.g: `gfx1030`, the last two digits are each represented by
two digits, and the remaining digits are copied verbatim. If one
of the last digits is a letter, then the numbering continues from 10. The
resulting number is used as the major compute capability. The minor version
number is currently unused and always zero. For example: `gfx1030` is
represented as compute capability 100300.0 (`sm_1003000`), and `gfx90c` is
represented as compute capability 90012.0 (`sm_900120`).



## Configuration file format

The configuration file is a newline separated list of entries. Each entry
consists of an ISA name, e.g: `gfx1030` and a
compute capability represented as an integer, e.g: `86`, separated by a space.
The entries are tried in order, so it's
possible to map more than one ISA and compute capability to each other
unambiguously. If the space and compute
capability are omitted, then the compiler associates all NVIDIA compute
capabilities with the specified GPU.

If no entry is found for the given GPU or if no configuration file is found,
then the library reports a compute
capability with a large major version defined by the default numbering scheme.

If no entry is found for the given GPU or if no configuration file is found, the
compiler does not translate a compute
capability to an AMD ISA.

Lines starting with `#` and empty lines are ignored.

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

- The file pointed the `REDSCALE_CCMAP` environment variable. It is an error if
  this environment variable is set but the
  file to which it points does not exist.
- `../share/redscale/ccmap.conf` relative to the directory
  containing `libredscale.so`. This search location is intended
  for users who build different installation trees for different GPUs. Packagers
  should not place a configuration here.
- `${HOME}/.redscale/ccmap.conf`
- `/etc/redscale/ccmap.conf`

## Search locations for the compiler

The compiler searches for a compute capability map in the following order:

- The file pointed to by the `--cuda-ccmap` flag. It is an error if this flag is
  given but the file to which it points
  does not exist.
- The file pointed the `REDSCALE_CCMAP` environment variable. It is an error if
  this environment variable is set but the
  file to which it points does not exist.
- `../share/redscale/ccmap.conf` relative to the directory containing the
  compiler binary (e.g: `nvcc`) if that
  directory is in a CUDA installation directory. This search location is
  intended for users who build different
  installation trees for different GPUs. Packagers should not place a
  configuration here.
