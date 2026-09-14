# Warp Size Options

The CUDA language offers eveything you need to model devices with varying warp sizes,
such as:

- `cudaDevAttrWarpSize` to query the warp size of a particular device from the host.
- The warpSize constant in device code.

However, many CUDA codebases have hardcoded the assumption that the GPU warp size is 32, since
that's true of all NVIDIA devices.

## Warp Size Modes

SCALE offers two different modes for handling warp sizes

### Warp32 Emulation

Warp32 emulation causes the warp size reported to the program to always be 32, and
the compiler/library inserts appropriate extra math to make that work. This *usually*
has no measurable effect on performance, but certain code patterns will become
very slow on warp64 devices (eg. if control flow sends alternating warps in
different directions).

### Native Warp Size

In native warp size mode, the warp size reported to the program matches that of the
hardware, and code is required to be properly warp size agnostic. Such changes can be
done in a way that is compatible with NVIDIA devices, has no performance downside,
and can likely be merged into the original CUDA codebase.

SCALE offers compiler diagnostics (including for inline PTX) to help you detect
places where the warp size has been hardcoded. This cannot be 100% effective, since
the warp32 assumption may be baked into arbitrarily complicated arithmetic.

## Mode Selection

- SCALE's `nvcc` defaults to warp32 emulation, since that is most likely to provide an
  out-of-the-box-working experience for existing codebases.
- SCALE's `clang` defaults to native warp size.

The mode can be selected using a preprocessor macro set on the command line:

- `-DSCALE_WARP_SIZE=32`: Enables warp32 emulation.
- `-DSCALE_WARP_SIZE=SCALE_WARP_SIZE_NATIVE`: Enables native-warp-size mode.

Explicitly setting SCALE_WARP_SIZE to a value larger than the native warp size of the
device is not allowed (eg. `SCALE_WARP_SIZE=64` is allowed, but only for a natively
wave64 device.). The emulation code can only do smaller warps on larger hardware, not
the other way around.

## Mixing Warp Sizes

Code built in either mode is fully interoperable with code built in the
other mode. You can mix emulated and native mode, as well as mixing devices
with different warp sizes in a single fatbinary. You can link relocatable
device objects together, even if they're for the same device and built in
conflicting warp size modes. It is possible for user code to break this
interopability if the warp size is part of the ABI (such as if you use
the warp size as a template parameter, expand only one version of the
template, and then another translation unit tries to link against the
other one).
