# Troubleshooting

This page provides tips for solving common problems encountered when trying 
to compile or run CUDA programs with SCALE.

## Crashes

Please [report a bug](../contact/report-a-bug.md)

## "No such function: cuBlas/cuFFt/cuSolverSomethingSomething()"

If you project needs a missing "CUDA-X" API (cuBLAS, cuFFT, cuSOLVER and
friends), this is most likely something you can fix yourself by submitting a
patch to the [open-source library wrapper project](https://github.com/spectral-compute/scale-library-wrappers).
So long as an equivalent function is available in a ROCm library, the wrapper
code is trivial.

## CUDA API errors

The [`SCALE_EXCEPTIONS` feature](runtime-extensions.md#scale_exceptions1) can
be helpful for getting more information about many failures.

## wave64 issues

All current NVIDIA GPU have a warp size of 32, so many CUDA programs are 
written in a way that assumes this is always the case.

Some AMD GPUs have a warp size of 64, which can cause problems for CUDA code 
written in this way.

SCALE offers tools to address this problem:

- APIs that operate on warp masks accept and return a new type: 
  `cudaWarpSize_t`. This is an integer with as many bits as there are 
  threads in a warp on the target GPU.
- Some APIs (such as `__ffs()`) have extra overloads for `cudaWarpSize_t`, so
  common patterns (such as `__ffs(__ballot(...))`) just work.
- The SCALE compiler will emit compiler warnings when values that represent 
  warp masks are implicitly truncated to 32 bits.

To write code that works correctly on both platforms:

- Use `auto` instead of `uint32_t` when declaring a variable that is 
  intended to contain a warp mask. With NVIDIA `nvcc` this will map to
  `uint32_t`, and with SCALE this will map to `cudaWarpSize_t`, producing 
  correct behaviour on both platforms.
- Avoid hardcoding the constant "32" to represent warp size, instead using 
  the global `warpSize` available on all platforms.

## Initialization errors or no devices found

The SCALE runtime can fail to initialise if:

- The AMD kernel module is out of date.
- `/dev/kfd` is not writable by the user running the program.
- There are no supported GPUs attached.

This situation produces error messages such as:

```
$ SCALE_EXCEPTIONS=1 ./myProgram
terminate called after throwing an instance of 'redscale::SimpleException'
  what():  cudaDeviceSynchronize: No usable CUDA devices found., CUDA error: "no device"
Aborted (core dumped)
```

```
$ /opt/scale/bin/scaleinfo
Error getting device count: initialization error
```

```
$ /opt/scale/bin/hsakmtsysinfo
terminate called after throwing an instance of 'std::runtime_error'
  what():  HSAKMT Error 20: Could not open KFD
Aborted (core dumped)
```

### Verify you have a supported gpu

Run `/opt/scale/bin/hsasysinfo | grep 'Name: gfx` to determine the 
architecture of your GPU, and determine if it is one of the supported 
architectures listed [here](../README.md#which-gpus-are-supported)

### Ensure `/dev/kfd` is writable

Ensure your user is in the group that grants access to `/dev/kfd`. On Ubuntu,
this is via membership of the `render` group:
`sudo usermod -a -G render USERNAME`

You could temporarily make `/dev/kfd` world-writable via: `sudo chmod 666 
/dev/kfd`.

### Ensure kernel module is up to date

If the GPU is still not detected by SCALE, the kernel driver may be out of 
date. On Ubuntu, follow the instructions in
[the installation guide](how-to-install.md#debian-like-linux-debian-ubuntu-mint)
for installing `amdgpu-dkms`.

## Cannot find shared object

The correct library search path for a SCALE binary can be target dependent due
to [compute capability mapping](./compute-capabilities.md). This can lead
to runtime errors where the SCALE libraries cannot be found, such as:

```
error while loading shared libraries: libredscale.so: cannot open shared object file: No such file or directory
```

Two ways to solve this problem are:
 - Set `LD_LIBRARY_PATH` to the SCALE target library directory, such as:
   `LD_LIBRARY_PATH=/opt/scale/targets/gfx1030/lib:$LD_LIBRARY_PATH` for `gfx1030`.
 - Compile your program is compiled with that directory in RPATH:
   [rpath](https://en.wikipedia.org/wiki/Rpath).

## CMake: Error running link command: no such file or directory

CMake tries to detect the linker to use based on the compiler. For SCALE's
`nvcc`, it uses `clang++` as the linker. If this does not exist in your `PATH`,
the result is an error like the one in the example below.

A good solution is to make sure SCALE's `nvcc` is at the start of your `PATH`.
This will place our `clang++` on your path too, avoiding the problem.

```bash
# Adjust for the target you want to use.
export PATH=/opt/scale/targets/gfx1030/bin:$PATH
```

#### Example error

```
-- The CUDA compiler identification is NVIDIA 12.5.999
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - failed
-- Check for working CUDA compiler: /opt/scale/targets/gfx1030/bin/nvcc
-- Check for working CUDA compiler: /opt/scale/targets/gfx1030/bin/nvcc - broken
CMake Error at /usr/local/share/cmake-3.29/Modules/CMakeTestCUDACompiler.cmake:59 (message):
  The CUDA compiler

    "/opt/scale/targets/gfx1030/bin/nvcc"

  is not able to compile a simple test program.

  It fails with the following output:

    Change Dir: '/home/user/test/cmake/build/CMakeFiles/CMakeScratch/TryCompile-vLZLYV'

    Run Build Command(s): /usr/local/bin/cmake -E env VERBOSE=1 /usr/bin/gmake -f Makefile cmTC_185e7/fast
    /usr/bin/gmake  -f CMakeFiles/cmTC_185e7.dir/build.make CMakeFiles/cmTC_185e7.dir/build
    gmake[1]: Entering directory '/home/user/test/cmake/build/CMakeFiles/CMakeScratch/TryCompile-vLZLYV'
    Building CUDA object CMakeFiles/cmTC_185e7.dir/main.cu.o
    /opt/scale/targets/gfx1030/bin/nvcc -forward-unknown-to-host-compiler   "--generate-code=arch=compute_86,code=[compute_86,sm_86]" -MD -MT CMakeFiles/cmTC_185e7.dir/main.cu.o -MF CMakeFiles/cmTC_185e7.dir/main.cu.o.d -x cu -c /home/user/test/cmake/build/CMakeFiles/CMakeScratch/TryCompile-vLZLYV/main.cu -o CMakeFiles/cmTC_185e7.dir/main.cu.o
    Linking CUDA executable cmTC_185e7
    /usr/local/bin/cmake -E cmake_link_script CMakeFiles/cmTC_185e7.dir/link.txt --verbose=1
    clang++ @CMakeFiles/cmTC_185e7.dir/objects1.rsp -o cmTC_185e7 @CMakeFiles/cmTC_185e7.dir/linkLibs.rsp -L"/opt/scale/targets/gfx1030/lib"
    Error running link command: no such file or directorygmake[1]: *** [CMakeFiles/cmTC_185e7.dir/build.make:102: cmTC_185e7] Error 2
    gmake[1]: Leaving directory '/home/user/test/cmake/build/CMakeFiles/CMakeScratch/TryCompile-vLZLYV'
    gmake: *** [Makefile:127: cmTC_185e7/fast] Error 2





  CMake will not be able to correctly generate this project.
Call Stack (most recent call first):
  CMakeLists.txt:2 (project)


-- Configuring incomplete, errors occurred!
```

## Half precision intrinsics not defined in C++

If you're using `__half` in host code in a non-cuda translation unit, you 
might get an error claiming the function you want does not exist:

```
error: ‘__half2float’ was not declared in this scope
```

This problem can be resolved by using newer C++ compiler.

This issue is discussed in more detail in the [Differences from NVIDIA CUDA](differences.md#host-side-__half-support) 
section.
