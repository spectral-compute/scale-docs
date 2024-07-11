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

## wave64 issues

Some AMD GPUs have a warp size of 64, not 32. All current NVIDIA GPU have a 
warp size of 32. This discrepancy can cause problems in CUDA code that has 
been written in a way that assumes a warp size of 32.

Although NVIDIA's documentation encourages writing code in a way that does 
not depend on warp size, much real-world code ends up being hardcoded for a warp
size of 32.

SCALE offers tools to address this problem:

- APIs that operate on warp masks accept and return a new type: 
  `cudaWarpSize_t`. This is an integer with as many bits as there are 
  threads in a warp on the target GPU.
- Some APIs (such as `__ffs()`) have extra overloads for `cudaWarpSize_t`, so
  common patterns (such as `__ffs(__ballot(...))`) just work.
- The SCALE compiler will emit compiler warnings when values that represent 
  warp masks are implicitly truncated to 32 bits.

To write code that works correctly on both platforms:

- Use `auto` not `uint32_t` when declaring a variable that is intended to 
  contain a warp mask. With NVIDIA `nvcc` this will map to `uint32_t`, and 
  with SCALE this will map to `cudaWarpSize_t`, producing correct behaviour 
  on both platforms.
- Avoid hardcoding the constant "32" to represent warp size, instead using 
  the global `warpSize` available on all platforms.

## Initialization errors or no devices found

The SCALE runtime can fail to initialize and list devices if the AMD kernel module is out of date, or if `/dev/kfd` is
not writable to the user running the program. This may also happen if there are no supported GPUs.

 - Make sure your GPU is one of the supported GPUs listed [here](../README.md) by running
  `/opt/scale/bin/hsasysinfo | grep 'Name: gfx`. If this does not show anything or returns an error, continue to the
   next step.
 - Make `/dev/kfd` is writable.
    - This can be tested temporarily by making it world-writable: `sudo chmod 666 /dev/kfd`.
    - Make sure the user is in `/dev/kfd`'s group. For example, on Ubuntu: `sudo usermod -a -G render USERNAME`.
 - If `/dev/kfd` is writable, the kernel driver may be out of date. On Ubuntu, follow the instructions in
   [the installation guide](../how-to-install.md) for installing `amdgpu-dkms`.

#### Example error messages

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
