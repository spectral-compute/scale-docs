# Compile HIP with SCALE

Like with CUDA, Compiling HIP programs with SCALE can usually be done using the exact same
commands that you'd use with ROCM.

To compile HIP, we use a tool called `shoulder`, which translates HIP calls to CUDA.
This means you can compile your HIP program using our compiler and runtime, for AMD.

Support for NVIDIA's runtime and/or compiler is coming soon.

## The easy way: `shoulder/env`

Shoulder offers a "`venv`-flavoured" environment management script to allow
"magically" building HIP projects. This set PATH and other environment variables
that affect buildsystems (eg. CMake), instructing them to use shoulder, SCALE's compiler,
and/or libraries, as appropriate.

The concept is simple:

1. Activate the environment for the GPU target you want to build for.
2. Run the commands you normally use to build the project.

To activate it:

```
# AMD
source /opt/scale/shoulder/env /opt/scale gfx1201
```

You can exit by typing `shoulder deactivate` or closing your terminal. 

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

## Next steps

- Explore [diagnostics](../compiler/diagnostics.md) or [optimisation](../compiler/optimisation-flags.md) flags.
- [Report a bug](../../contact/report-a-bug.md)
