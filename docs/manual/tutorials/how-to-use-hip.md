# Compile HIP with SCALE

Compiling HIP programs with SCALE can usually be done using the same
commands that you'd use with ROCM.

Just like CUDA programs built with SCALE, HIP programs use the SCALE
compiler, and run on the SCALE runtime library (AMD) or the NVIDIA
CUDA Runtime (NVIDIA).

To compile HIP, we use a tool called `shoulder`, to inject an appropriate
API mapping header into the build.

## The easy way: `shoulder/env`

Shoulder offers a "`venv`-flavoured" environment management script to allow
building HIP projects. This sets `PATH` and other environment variables
that affect buildsystems (eg. CMake), instructing them to use shoulder, SCALE's compiler,
and/or libraries, as appropriate.

Usage is simple:

1. Activate the environment for the GPU target you want to build for.
2. Run the commands you normally use to build the project.

To activate it:

```
# AMD
source /opt/scale/shoulder/env /opt/scale gfx1201
```

You can exit by typing `shoulder deactivate` or closing your terminal. 

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
