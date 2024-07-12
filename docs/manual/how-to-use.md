# Compile CUDA with SCALE

This guide covers the steps required to compile an existing CUDA project for an
AMD GPU using SCALE.

SCALE creates directories that aim to impersonate the NVIDIA CUDA Toolkit (from
the point of view of your build system). Compilation with SCALE is therefore
a matter of telling your build system that the CUDA installation path
is one provided by SCALE, rather than the one provided by NVIDIA.

## Install SCALE

[Install SCALE](./how-to-install.md), if you haven't already.

## Identifying GPU Target

If you don't already know which AMD GPU target you need to compile for, you can
use the `scaleinfo` command provided by SCALE to find out:

```bash
scaleinfo
```

Example output:

```
Found 1 CUDA devices
Device 0 (00:23:00.0): AMD Radeon Pro W6800 - gfx1030 (AMD) <amdgcn-amd-amdhsa--gfx1030>
    Total memory: 29.984375 GB [32195477504 B]
    Free memory: 29.570312 GB [31750881280 B]
    Warp size: 32
    Maximum threads per block: 1024
    Maximum threads per multiprocessor: 2048
    Multiprocessor count: 30
    Maximum block dimensions: 1024x1024x1024
    Maximum grid dimensions: 2147483647x4194303x4194303
    Global memory size: 29.984375 GB [32195477504 B]
    Shared memory size: 64.000000 kB [65536 B]
    Constant memory size: 2.000000 GB [2147483647 B]
    Clock rate: 2555000 kHz
    Memory clock rate: 1000000 kHz
```

In this example, the GPU target ID is `gfx1030`.

If your GPU is not listed in the output of this command, it is not currently
supported by SCALE.

If the `scaleinfo` command is not found, ensure
that `<SCALE install path>/bin` is in `PATH`.

## Point your build system at SCALE

To allow compilation without build system changes, SCALE provides a series of
directories that are recognised by build systems as being CUDA Toolkit
installations. One such directory is provided for each supported AMD GPU
target. These directories can be found at `<SCALE install
path>/targets/gfxXXXX`, where `gfxXXXX` is the name of an AMD GPU target,
such as `gfx1030`.

You must tell your build system to use the "CUDA Toolkit" corresponding to the
desired AMD GPU target.

For example: to build for `gfx1030` you would tell your build system that
CUDA is installed at `<SCALE install path>/targets/gfx1030`.

The remainder of this document assumes that `SCALE_PATH` is an environment
variable you have set to such a path (for example:
`/opt/scale/targets/gfx1030`).

### CMake

Add SCALE's `nvcc` first in `PATH`:

```bash
export PATH="${SCALE_PATH}/bin:$PATH"
```

Then add these two arguments to your `cmake` invocation:

```
# Replace with the path to your SCALE install, followed by the name of the
# AMD GPU target you want to compile for.
-DCMAKE_CUDA_COMPILER="${SCALE_PATH}/bin/nvcc"

# See "Why sm_86?" below
-DCMAKE_CUDA_ARCHITECTURES=86
```

This will work for any modern CMake project that is using CMake's native
CUDA support.

You can check CMake's output to verify it has properly detected SCALE instead of
picking up NVIDIA CUDA (if it is installed):

```
-- The CUDA compiler identification is NVIDIA 12.5.999
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /opt/scale/gfx1030/bin/nvcc
```

- The compiler ID should be "NVIDIA", followed by a version number ending in
  `999`.
- The "Check for working CUDA compiler" line should point to the SCALE nvcc
  compiler, not the NVIDIA one.
- Other paths (such as that of cublas, for example) should be pointing to
  the SCALE versions, not the NVIDIA ones.

### Others

Most build systems will use environment variables and information from invoking
the first `nvcc` found in `PATH` to determine where CUDA is. As a result, 
the following works for many other build systems:

```bash
# Update accordingly.
SCALE_INSTALL_DIR=/opt/scale/targets/gfx1030

export PATH="${SCALE_INSTALL_DIR}/bin:$PATH"
export CUDA_HOME="${SCALE_INSTALL_DIR}"
export CUDA_PATH="${SCALE_INSTALL_DIR}"

# If your system has a very old C++ compiler that chokes on SCALE compilers, 
# you could add the following to build all your C++ code with the modern 
# version of `clang++` bundled with SCALE (and send us a bug report!)
#export CC="${SCALE_INSTALL_DIR}/bin/clang"
#export CXX="${SCALE_INSTALL_DIR}/bin/clang++"
#export CUDAHOSTCXX="${SCALE_INSTALL_DIR}/bin/clang++"

<Your usual build here>
```

A build-system-specific way of specifying you wish to compile for sm_86 may
also be required.

You can verify that SCALE has been correctly added to `PATH` by executing
`nvcc --version`. You should see output like:

```
nvcc: NVIDIA (R) Cuda compiler driver
Actually, no. This is the SCALE compiler, and the first/last line of this output are lies to make CMake work.
clang version 17.0.0
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/scale/targets/gfx1030/bin
Cuda compilation tools, release 12.5, V12.5.999
```

## Next steps

- Understand why all compilation magically uses `sm_86` by reading about
  [Compute Capability Mapping](compute-capabilities.md)
- Learn about [CUDA dialects](dialects.md) and [SCALE language extensions](language-extensions.md)
- [Report a bug](../contact/report-a-bug.md)
