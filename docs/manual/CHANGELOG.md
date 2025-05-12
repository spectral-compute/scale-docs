# What's new?

## Release 1.3.0 (2025-04-23)

### Platform

- Upgraded from llvm17 to llvm19.1.7.
- Support for rocm 6.3.1.
- Support for `gfx902` architecture.
- Enterprise edition: Support for new architectures:
    - `gfx908`
    - `gfx90a`
    - `gfx940`
    - `gfx942`

### Packaging

- Packages for Rocky9 are now available.
- Package repos for Ubuntu and Rocky9 to simplify
  installation/upgrades.

### New Features

- Added `scaleenv`, a new and much easier way to [use SCALE](./how-to-use.md).
- Support for simulating a warp size of 32 even on wave64 platforms, fixing
  many projects on such platforms.
- Support for `bfloat16`.
- Compatibility improvements with non-cmake buildsystems.
- Added `SCALE_CUDA_VERSION` environment variable to tell SCALE to impersonate a
  specific version of CUDA.
- `SCALE_EXCEPTIONS` now supports a non-fatal mode.

### Library wrappers

- Added most of cuFFT.
- Added lots more cuSolver and cuSPARSE.
- Filled in some missing NVTX APIs.
- Added homeopathic quantities of nvtx3.

### Library

- Lazy-initialisation of primary contexts now works properly, fixing some 
  subtle lifecycle issues.
- Added some missing undocumented headers like `texture_types.h`.
- Added the IPC memory/event APIs
- Added many multi-GPU APIs
- Added `cuMemcpyPeer`/`cuMemcpyPeerAsync`.
- Rewritten device allocator to work around HSA bugs and performance issues.
- Fix a crash when initialising SCALE with many GPUs with huge amounts of memory.
- Added CUDA IPC APIs. Among other things, this enables CUDA-MPI
  applications to work, including AMGX's distributed mode.
- Fixed lots of multi-GPU brokenness.
- Implemented the `CU_CTX_SYNC_MEMOPS` context flag.
- Fixed accuracy issues in some of the CUDA Math APIs.
- fp16 headers no longer produce warnings for projects that include them without
  `-isystem`.
- Improved performance and correctness of cudaMemcpy/memset.
- Fix subtle issues with pointer attribute APIs.
- Improvements to C89 compatibility of headers.
- Added more Cooperative Groups APIs.
- Support for `grid_sync()`.
- Fix some wave64 issues with `cooperative_groups.h`.

### Compiler

- `__launch_bounds__` now works correctly, significantly improving performance.
- Device atomics are now much more efficient.
- Denorm-flushing optimisations are no longer skipped when they aren't
  supposed to be.
- Ability to use DPP to optimise warp shuffles in some cases. Currently, 
  this only works if the individual shfl is provably equivalent to a DPP op, 
  not when loop analysis would be required. `__shfl_xor` is your friend.

### NVCC Interface

- Corrected the behaviour of the nvcc `-odir` flag in during dependency generation.
- Added the nvcc `-prec-sqrt` and `-prec-div` flags.
- `use_fast_math` now matches nivida's behaviour, instead of mapping to clang's
  `-ffast-math`, which does too much.
- `--device-c` no longer inappropriately triggers the linker.
- Newly-supported `nvcc` flags:
    * `-arch=native`
    * `-jump-table-density` (ignored)
    * `-compress-mode` (ignored)
    * `-split-compile-extended` (ignored)

### NVCC Semantics

- Support broken template code in more situations in nvcc mode.
- Allow invalid const-correctness in unexpanded template code in nvcc mode.
- Allow trailing commas in template argument lists in nvcc mode.
- Fix a parser crash when explicitly calling `operator<<<int>()` in CUDA mode.
- Fix a crash when using `--compiler-options` to pass huge numbers of
  options through to `-Wl`.

### Diagnostics

- Warning for unused PTX variables
- Error for attempts to return the carry bit (undefined behaviour on NVIDIA).
- Compiler diagnostic to catch some undefined behaviour patterns with CUDA
  atomics.

### PTX

- New instructions supported
    - `sm_100` variants of `redux`.
    - Mixed-precision `add/sub/fma` FP instructions.
    - `membar`
    - `bar.warp.sync`
    - `fence` (partial)
    - `mma` (software emulated)
    - `wmma` (software emulated)
- Fixed parsing of hex-float constants.
- Support for PTX `C` constraints (dynamic asm strings).
- f16/bf16 PTX instructions no longer depend on the corresponding C++ header.
- asm blocks can now refer to variables declared in other asm blocks, including
  [absolutely cursed](https://gist.github.com/ChrisKitching/73f66a422af926a6dbdcd045442c4440) patterns.
- Fixed an issue where template-dependent asm strings were mishandled.
- Fixed various parsing issues (undocumented syntax quirks etc.).
- Fixed a crash when trying to XOR floating point numbers together.

### Thirdparty Project demos

Things that now appear to work include:

- CUDA-aware MPI
- MAGMA
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
