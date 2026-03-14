# Compile CUDA with SCALE

Compiling CUDA programs with SCALE can typically be done using the exact same
commands that are used to compile using the NVIDIA CUDA Toolkit.

SCALE's compiler can be called as `nvcc` and accepts the same commandline arguments
as NVIDIA's one. This makes it trivial switch compilers (SCALE vs. NVIDIA), and to
switch target GPU vendor (NVIDIA vs. AMD).

## Possible Compilations

There are three ways to compile CUDA files:

![Compilation Trajectories](../imgs/OneFileThreeDestines.svg)

1. Using NVIDIA nvcc to compile you GPU code, combined with NVIDIA's libraries.
   AKA: not using SCALE at all.
2. Using SCALE to compile your GPU kernels, combined with NVIDIA's runtime
  libraries, to run on NVIDIA GPUs.
3. Using SCALE's compiler and runtime libraries to build and run on AMD GPUs.

## The easy way: `scaleenv`

SCALE offers a "`venv`-flavoured" environment management script to allow
"magically" building CUDA projects. This set PATH and other environment variables
that affect buildsystems (eg. CMake), instructing them to use SCALE's compiler
and/or libraries, as appropriate.

The concept is simple:

1. Activate the `scaleenv` for the GPU target you want to build for.
2. Run the commands you normally use to build the project.

To activate a scaleenv:

```
# AMD
source /opt/scale/bin/scaleenv gfx1201

# NVIDIA
source /opt/scale/bin/scaleenv sm_89
```

You can exit a `scaleenv` by typing `deactivate` or closing your terminal. Scaleenv is
the build method we use for [scale-validation](https://github.com/spectral-compute/scale-validation), where we test various open-source CUDA projects.

!!! tip

    Check the output of `nvcc --version` to make sure the SCALE compiler is first in `PATH`.

    SCALE offers a compiler flag `--require-scale`, which does nothing. Adding this flag to the `NVCC_APPEND_FLAGS` environment variable will cause accidential use of NVIDIA nvcc to result in an error.


## Identifying GPU Target

If you don't know what target ID to use for your GPUs, here's how:

=== "AMD"

    Commands for getting info about AMD GPUs:

    ```bash
    # Just print the target IDs
    amdgpu-arch

    # Print more thorough info about the AMD GPUs present.
    scaleinfo
    ```

    If the `scaleinfo` command is not found, ensure
    that `<SCALE install path>/bin` is in `PATH`.

=== "NVIDIA"

    Commands for getting info about NVIDIA GPUs:

    ```bash
    # Just print the target IDs
    nvptx-arch

    # Print more thorough info about the NVIDIA GPUs present.
    nvidia-smi
    ```

## How it really works

=== "NVIDIA"

    When building CUDA programs, buildsystems typically ask nvcc where to
    find the CUDA libraries (cudart and friends) by parsing the output
    of `nvcc -v`.

    By placing the SCALE compiler first in `PATH`, it is picked up and used
    instead of NVIDIA nvcc. Then, it reports the library locations appropriately
    to cause the build system to link the NVIDIA CUDA runtime.

=== "AMD"

    To allow compilation without build system changes, SCALE provides a series of
    directories that are recognised by build systems as being CUDA Toolkit
    installations. One such directory is provided for each supported AMD GPU
    target. These directories can be found at `<SCALE install
    path>/targets/gfxXXXX`, where `gfxXXXX` is the name of an AMD GPU target,
    such as `gfx1030`.

    To achieve the desired effect, we need the build system to use the "CUDA
    toolkit" corresponding to the desired AMD GPU target.

    For example: to build for `gfx1030` you would tell your build system that
    CUDA is installed at `<SCALE install path>/targets/gfx1030`.

    All `scaleenv` is actually doing is setting various environment variables up
    to make this happen. It's just a shell script: open it to see the variables
    it is manipulating.

    ## Finding the libraries at runtime

    For maximum compatibility with projects that depend on NVIDIA's "compute
    capability" numbering scheme, SCALE provides one "cuda mimic directory" per
    supported GPU target that maps the new target to "sm_86" in NVIDIA's
    numbering scheme.

    This means that each of the `target` subdirectories contains
    identically-named libraries, so SCALE cannot meaningfully add them to the
    system's library search path when it is installed. The built executable/library
    therefore needs to be told how to find the libraries via another mechanism,
    such as:

    - [rpath](https://en.wikipedia.org/wiki/Rpath). With CMake, the simplest
      thing that "usually just works" is to add
      `-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON` to your cmake incantation.
    - Set `LD_LIBRARY_PATH` to include `${SCALE_DIR}/lib` at runtime. `scaleenv`
      does this, so if you keep that enabled when running your programs things
      will just work.

    Support for multiple GPU architectures in a single binary ("Fat binaries")
    is in development.

## Next steps

- Learn about [CUDA dialects](../compiler/dialects.md)
- Explore [diagnostics](../compiler/diagnostics.md) or [optimisation](../compiler/optimisation-flags.md) flags.
- [Report a bug](../../contact/report-a-bug.md)
