# Compile CUDA with SCALE

This guide covers the steps required to compile an existing CUDA project for an
AMD GPU using SCALE.

SCALE makes this as easy as possible by convincingly impersonating the 
NVIDIA CUDA Toolkit (from the point of view of your build system).

To use SCALE, we must simply cause your build system to use the "CUDA 
installation" offered by SCALE.

[Install SCALE](./how-to-install.md), if you haven't already.

## Identifying GPU Target

If you don't already know which AMD GPU target you need to compile for, you can
use the `scaleinfo` command provided by SCALE to find out:

```bash
scaleinfo | grep gfx
```

Example output:

```
Device 0 (00:23:00.0): AMD Radeon Pro W6800 - gfx1030 (AMD) <amdgcn-amd-amdhsa--gfx1030>
```

In this example, the GPU target ID is `gfx1030`.

If your GPU is not listed in the output of this command, it is not currently
supported by SCALE.

If the `scaleinfo` command is not found, ensure
that `<SCALE install path>/bin` is in `PATH`.

## The easy way: `scaleenv`

SCALE offers a "`venv`-flavoured" environment management script to allow 
"magically" building CUDA projects.

The concept is simple:

1. Activate the `scaleenv` for the AMD GPU target you want to build for.
2. Run the commands you normally use to build the project for an NVIDIA GPU.
3. AMD binaries are sneakily produced instead of NVIDIA ones.

To activate a scaleenv:

```
source /opt/scale/bin/scaleenv gfx1030
```

You can exit a `scaleenv` by typing `deactivate` or closing your terminal.

While the environment is active: simply run the usual `cmake`/`make`/etc.
commands needed to build the project, and it will build for whatever AMD 
target you handed to `scaleenv`.

## How it really works

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

- Learn about [CUDA dialects](dialects.md) and [SCALE language extensions](language-extensions.md)
- [Report a bug](../contact/report-a-bug.md)
