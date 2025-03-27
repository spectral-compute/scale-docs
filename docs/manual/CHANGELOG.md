# What's new?

{% if false %}

## NEXT

- Packages for Rocky9 are now available.
- SCALE_EXCEPTIONS now supports a non-fatal, print-only mode for projects 
  that create exceptions intentionally.
- Huge improvement to performance of device-level atomics.
- `--device-c` no longer inappropriately triggers the linker.
- Denorm-flushing optimisations are now applied in all cases they're 
  supposed to be.
- Added a compiler error for certain patterns of undefined-behaviour atomic 
  operations banned by the CUDA language (in cases the compiler manages to 
  prove it!)
- Added `nvcc -compress-mode`

{% endif %}

## Unstable-2025.03.24

### Platform

- Upgraded from llvm18.1.8 to llvm19.1.7. Much zoominess ensues.
- Added faux "stubs" libraries to placate non-cmake buildsystems
- Added some fake symbols to satisfy ancient/insane buildsystems.
- Added `SCALE_CUDA_VERSION` environment variable to tell SCALE to impersonate a
  specific version of CUDA.

### Library Enhancements

- Fix a crash when initialising SCALE with many GPUs with huge amounts of memory.
- Faster startup times, especially on many-GPU machines.
- Added CUDA IPC APIs. Among other things, this enables CUDA-MPI 
  applications to work, including AMGX's distributed mode.
- Fixed lots of multi-GPU brokenness.
- Added lots more of cuSolver and cuSPARSE
- Filled in some missing NVTX APIs
- Implemented the `CU_CTX_SYNC_MEMOPS` context flag.
- Fixed accuracy issues in some of the CUDA Math APIs.
- fp16 headers no longer produce warnings for projects that include them without
  `-isystem`.
- Improved performance and correctness of cudaMemcpy/memset by finishing the 
  move away from hsa's buggy implementation.
- Fix some wave64 issues with `cooperative_groups.h`.
- Support for `grid_sync()`.
- Fix subtle issues with pointer attribute APIs.
- Implicit context initialisation rules now more closely match NVIDIA's, fixing
  some projects that depended on checking primary context state.
- Improvements to C89 compatibility of headers.

### Compiler Enhancements

- `__launch_bounds__` now works correctly, significantly improving performance.
- Added the nvcc `-prec-sqrt` and `-prec-div` flags.
- Corrected the behaviour of the nvcc `-odir` flag in during dependency generation.
- `use_fast_math` now matches nivida's behaviour, instead of mapping to clang's
  `-ffast-math`, which does too much.
- Support broken template code in more situations in nvcc mode.
- Allow invalid const-correctness in unexpanded template code in nvcc mode.
- Allow trailing commas in template argument lists in nvcc mode.
- Fix a parser crash when explicitly calling `operator<<<int>()` in CUDA mode.

### Thirdparty Project demos

Things that now appear to work include:

- CUDA-aware MPI
- MAGMA

## Unstable-2025.02.19

### Platform

- Support for simulating a warp size of 32 even on wave64 platforms, fixing 
  many projects on such platforms.
- Support for `bfloat16`.
- Upgraded from llvm17 to llvm18.1.8.
- Support for rocm 6.3.1
- Availabiliy of an Ubuntu package repo to simplify installation/upgrades.

### Library Enhancements

- Added software emulated WMMA APIs, and `wmma`/`mma` PTX instructions. 
  Hardware-accelerated versions are in development.
- Added more Cooperative Groups APIs.
- Rewritten device allocator to work around HSA bugs and performance issues.
- Significant performance improvements for most warp-level cooperative 
  operations.
- Various random APIs added.

### PTX

- Compiler diagnostics for unused PTX variables and attempts to return 
  the carry-bit.
- PTX variable references and `{}` now work correctly between `asm` blocks
  within the same function.
- Added PTX `C` constraints (dynamic asm strings).
- Added the new mixed-precision `add/sub/fma` FP instructions.
- Added `membar` instruction.
- Partial support for `fence` instruction.
- half-float PTX instructions now work correctly even if `cuda_fp16.h` has not 
  been included.
- Fixed various parsing issues (undocumented syntax quirks etc.).
- Fixed an issue where template-dependent asm strings were mishandled.
- Fixed various miscompilations.

### Thirdparty Project demos

The `scale-validation` repo now has working demos for the following:

- whisper.cpp
- TCLB

## Release 1.2.0 (2024-11-27)

### Library Enhancements

- Support for `gfx900` architecture.
- Support for `gfx1102` architecture.

### PTX

- [Improved handling of wave64](./inline-ptx.md#wave64-considerations) in inline PTX.
- Various inline PTX compilation fixes.

### Other

- Support for Ubuntu 24.04.
- Upgraded to ROCm 6.2.2.

## Release 1.1.0 (2024-10-31)

### Library Enhancements

- Added much of the CUDA graph API.
- Improvements to multi-GPU handling.
- Fixed rare shutdown-time segfaults.
- Added many random API functions. As usual, see [The diff](./apis.md).

### PTX

- `f16x2`, `u16x2` and `s16x2` types.
- `fns` instruction
- Fixed miscompile of `sad` instruction.

### Thirdparty Project demos

The `scale-validation` repo now has working demos for the following 
additional projects:

- FLAMEGPU2
- GPUJPEG
- gpu_jpeg2k

## Release 1.0.2.0 (2024-09-05)

Documented a record of the CUDA APIs already available in SCALE, and those still to come: [Implemented APIs](./apis.md).

### Library Enhancements

- Kernel arguments larger than 4kb no longer crash the library.
- Programs that ignore CUDA error codes can no longer get stuck in a state
  where the library always returns the error code you ignored.
- Fixed synchronisation bugs when using synchronous `cuMemset*` APIs.
- Fixed implicit synchronisation behaviour of `cuMemcpy2D/cuMemcpy2DAsync()`.
- Fixed precision issues in fp16 `exp2()`, `rsqrt()`, and `h2log()`.
- `cudaEventRecord` for an empty event no longer returns a time in the past.
- Fixed occupancy API behaviour in edgecases that are not multiples of warp
  size.
- Fixed rare crashes during static de-initialisation when library wrappers
  were in use.
- All flags supported by SCALE's nvcc are now also accepted by our nvrtc
  implementation.
- Various small header fixes.

### Compiler Enhancements

- `decltype()` now works correctly for `__host__ __device__` functions.
- `-Winvalid-constexpr` no longer defaults to `-Werror`, for consistency
  with nvcc.
- PTX variable names including `%` are no longer rejected.
- Support for nvcc's nonstandard permissiveness surrounding missing
  `typename` keywords in dependent types.
- Support for nvcc's wacky "split declaration" syntax for `__host__ __device`
  functions (with a warning):
  ```
  int foo();
  __device__ int foo();
  __host__ int foo() {
      return 5;
  }
  // foo() is a __host__ __device__ function. :D
  ```
- Newly-supported compiler flags (all of which are aliases for
  standard flags, or combinations thereof):
    * `-device-c`
    * `-device-w`
    * `-pre-include`
    * `-library`
    * `-output-file`
    * `-define-macro`
    * `-undefine-macro`

### New CUDA APIs

#### Math APIs

- `exp10(__half)`
- `exp2(__half)`
- `rcp(__half)`
- `rint(__half)`
- `h2exp10(__half2)`
- `h2exp2(__half2)`
- `h2rcp(__half2)`
- `h2rint(__half2)`

## Release 1.0.1.0 (2024-07-24)

This release primarily fixes issues that prevent people from successfully
compiling their projects with SCALE. Many thanks to those users who
submitted bug reports.

### CUDA APIs

- The `extra` argument to `cuLaunchKernel` is now supported.
- Added support for some more undocumented NVIDIA headers.
- Fix various overload resolution issues with atomic APIs.
- Fix overload resolution issues with min/max.
- Added various undocumented macros to support projects that are explicitly
  checking cuda include guard macros.
- `lrint()` and `llrint()` no longer crash the compiler. :D
- Newly supported CUDA APIs:
    * `nvrtcGetNumSupportedArchs`
    * `nvrtcGetSupportedArchs`
    * `cudaLaunchKernelEx`, `cuLaunchKernelEx`, `cudaLaunchKernelExC`: some
     of the performance-hint
    launch options are no-ops.
    * `__vavgs2`, `__vavgs4`
    * All the `atomic*_block()` and `atomic*_system()` variants.

### Compiler

- Improved parsing of nvcc arguments:
     * Allow undocumented option variants (`-foo bar`, `--foo bar`,
       `--foo=bar`, and `-foo=bar` are always allowed, it seems).
     * Implement "interesting" quoting/escaping rules in nvcc arguments, such as
       embedded quotes and `\,`. We now correctly handle cursed arguments like:
       `'-Xcompiler=-Wl\,-O1' '-Xcompiler=-Wl\,-rpath\,/usr/lib,-Wl\,-rpath-link\,/usr/lib'`
- Support for more nvcc arguments:
    * NVCC-style diagnostic flags: `-Werror`, `-disable-warnings`, etc.
    * `--run`, `--run-args`
    * `-Xlinker`, `-linker-options`
    * `-no-exceptions`, `-noeh`
    * `-minimal`: no-op. Exact semantics are undocumented, and build times
      are reasonably fast anyway.
    * `-gen-opt-lto`, `-dlink-time-opt`, `-dlto`. No-ops: device LTO not yet
      supported.
    * `-t`, `--threads`, `-split-compile`: No-ops: they're flags for making
      compilation faster and are specific to how nvcc is implemented.
    * `-device-int128`: no-op: we always enable int128.
    * `-extra-device-vectorization`: no-op: vectorisation optimisations are
      controlled by the usual `-O*` flags.
    * `-entries`, `-source-in-ptx`, `-src-in-ptx`: no-ops: there is no PTX.
    * `-use-local-env`, `-idp`, `-ddp`, `-dp`, etc.: ignored since they are
      meaningless except on Windows.

- Allow variadic device functions in non-evaluated functions.
- Don't warn about implicit conversion from `cudaLaneMask_t` to `bool`.
- `__builtin_provable` no longer causes compiler crashes in `-O0`/`-O1` builds.
- Fixed a bug causing PTX `asm` blocks inside non-template, non-dependent
  member functions of template classes to sometimes not be compiled,
  causing PTX to end up in the AMD binary unmodified.
- CUDA launch tokens with spaces (ie.: `myKernel<< <1, 1>> >()`) are now
  supported.
- Building non-cuda C translation units with SCALE-nvcc now works.

### Other

- The `meson` build system no longer regards SCALE-nvcc as a "broken" compiler.
- `hsakmtsysinfo` no longer explodes if it doesn't like your GPU.
- New documentation pages.
- Published more details about thirdparty testing, including the build scripts.

## Release 1.0.0.0 (2024-07-15)

Initial release
