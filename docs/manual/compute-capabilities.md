# Compute Capability Mapping

"Compute capability" is a numbering system used by NVIDIA's CUDA tools to
represent different GPU targets. The value of the `__CUDA_ARCH__` macro is
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

This approach works, but has one obvious downside: it makes fat binaries 
unrepresentable.

To resolve that, your buildsystem must be at least somewhat SCALE-aware: 
compute capabilities are not a sufficiently powerful abstraction to model 
the needs of a cross-vendor fat binary.

The special `gfxany` target directory is a "CUDA installation directory" 
that does not perform this compute capability mapping at all. Instead, you 
may provide your own arbitrary mapping from GPU targets to CC-number - or 
use no such mapping at all (if your program doesn't use the CC-number for 
metaprogramming). We recommend CUDA progras be written using more portable 
and reliable means of detecting the existence of features: even within 
NVIDIA's universe, the CC number is a rather blunt instrument.

The remainder of this document explains how the compute capability mapping 
configuration works for users of the `gfxany` target.

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
