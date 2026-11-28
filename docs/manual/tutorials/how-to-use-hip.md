# Compile HIP with SCALE

Compiling HIP programs with SCALE can usually be done using the same
commands that you'd use with ROCM.

Just like CUDA programs built with SCALE, HIP programs use the SCALE
compiler, and run on the SCALE runtime library (AMD) or the NVIDIA
CUDA Runtime (NVIDIA).

To compile HIP, we use a tool called `shoulder`, to inject an appropriate
API mapping header into the build.

## The easy way: `scaleenv`

*TL;DR - the steps are the same as [building cuda with scale](./how-to-use.md):
activating `scaleenv` sets up your environment for building both cuda and hip.*

Building either cuda or hip code shouldn't require a conflicting environment. As
such, shoulder just uses the same `scaleenv` script as building cuda code does.

Usage is simple:

1. Activate the environment for the GPU target you want to build for.
2. Run the commands you normally use to build the project.

To activate it:

```
# AMD
source /opt/scale/bin/scaleenv /opt/scale gfx1201
```

You can exit by typing `scaleenv deactivate` or closing your terminal.

HIP programs built with shoulder should be able to link to ROC/HIP maths
libraries (rocBLAS, rocRAND, ...), however we don't currently ship all of them.
We expect to ship a wider range of maths libraries soon.

## Identifying GPU Target

If you don't know what target ID to use for your GPUs, here's how:

  ```bash
  # Just print the target IDs
  amdgpu-arch

  # Print more thorough info about the AMD GPUs present.
  scaleinfo
  ```

  If the `scaleinfo` command is not found, ensure
  that `<SCALE install path>/bin` is in `PATH`.
