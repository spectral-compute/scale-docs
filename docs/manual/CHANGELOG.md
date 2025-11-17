# What's new?

## Release 1.5.0 (2025-11-15)

### Platform

- Compiler is now based on llvm 20.1.8
- Using rocm 7.1 versios of rocBLAS etc.

### Supported architectures

- All architectures are now enabled in the free version of SCALE. See new EULA for details.
- Newly-supported AMD GPU architectures:
    * `gfx950` (MI350x/MI355x)
    * `gfx1151` (Strix-halo etc.)
    * `gfx1200` (RX 9060 XT etc.)
    * `gfx1201` (RX 9070 XT etc.)
    * `gfx908` (MI100)
    * `gfx906` (MI50/MI60)
- NVIDIA support.


## Compiler: Inline PTX support

- Improve PTX diagnostics for unused instruction flags
- Support for `q` constraints (128-bit asm inputs)
- Diagnostics for implicit truncation via asm constraints.
- New PTX instructions:
    * `movmatrix`
    * `bar`/`barrier`: only cases that can be represented as `__synthreads_*`.

## Compiler: NVCC emulation

- New NVCC flags:
    * `-keep`
    * `-keep-dir`
    * `-link`
    * `-preprocess` (alias of `-E`)
    * `-libdevice-directory`/`-ldir`: Meaningless
    * `-target-directory`/`target-dir`: Meaningless
    * `-cudadevrt`: no-op because our devrt implementation uses no smem, so no point.
    * `-opt-info`: alias of `-Rpass`, so it can do more than just `-inline`
- Fix resolution of relative paths for nvcc's option-files flag.
- Fix compiler crash when `fence.cluster` appears in inline PTX in dead code.
- Accept even more cases of invalid identifiers in unused code in nvcc mode.
- Improvements and fixes to the `__shfl*` optimiser. More DPP, less UB.
- Add a compiler warning to complain about mixing `__constant__` and `constexpr`.

## Compiler (misc.)

- Introduce builtins for conversion between arbitrary types (fp8 here we come)
- Further improvements to deferred diagnostics, especialy surrounding typo'd identifiers
- Add `[[clang::getter]]`
- Fix kernel argument buffer alignment sometimes being wrong
- Microsoft extensions are no longer enabled in CUDA mode.
- Arithmetic involving `threadIdx` and friends now compiles.
- Various small optimiser enhancements.

## Runtime

- Fix a rare race condition resolving cudaEvents
- Slightly improve performance of *every* API by optimising error handling routines.
- Introduce nvtx3 support as a header-only library, like it should be.
- New API: `cudaEventRecordWithFlags`
- Respect `CUDA_GRAPHS_USE_NODE_PRIORITY` environment variable.
- Implement UUID-query APIs, and make them consistent between the driver API and nvml
- Support the new `__nv_*` atomic functions
- Support (and ignore) the ancient `CU_CTX_MAP_HOST/cudaDeviceMapHost` flags. This feature is always enabled.
- Startup-time improvements.
- Fix crash when empty CUDA graphs are executed.
- Fix occasionally picking the wrong cache coherence domain for SDMA engines and breaking everything.
- fp16x2/bf16x2 are now trivially-copyable in C++11+: a very tiny extension.
- Fix intermittent crash in cuMemUnmap
- Add the new CUDA13 aligned vector types
- Workaround SDMA bug that gave incorrect results for cuMemsetD16 on some devices.
- Many random header compatibility/typedef fixes.
- Accuracy improvements to `tanh()` and `sin()`.
- Fix crash when millions of cudaStreams are destroyed all at once.
- Crash _correctly_ when the GPU executes a trap instruction.
- Fix the GPU trap handler on GFX94x devices.
- Fix a few C89-compatibility issues in less-commonly-used headers.
- Make headers warning-free on GCC, since not everyone uses `-isystem` properly.
- Slightly improve performance of nvRTC.
- Raise an error if the user attempts to execute PTX as AMDGPU machine code, instead of actually trying it.
- Fix a few cases where the runtime library and RTC compiler would disagree about what architecture to build for.

## Release 1.4.2 (2025-09-19)

### Platform/Packaging

- Simplified packaging/installation: removed dependency on rocm package repos. Removal of more rocm components from the package is in development.
- Using rocm 6.4.1 versions of rocBLAS etc.

### Compiler Diagnostics

- Warn about PTX instruction flags that don't actually do anything.
- Warn about PTX variable declarations leaking into the enclosing function, since this may cause ptxas failures
  when building for NVIDIA.
- Warn about passing a generic address space C++ pointer into an asm operand for a non-generic PTX memory instruction if the
  corresponding addrspacecast is not a no-op.
- Detect when the user seems to have gotten `cvta.to` and `cvta` mixed up.
- Cleanly reject `.param` address space in PTX: using that in inline asm is undefined behaviour.
- Diagnose accidental address operands (eg. use of `[]` for `cvta` is a common screwup).
- Proper errors for integer width mismatch in PTX address space conversions. Implicitly truncating pointers is *bad*.
- Disallow overloads that differ only in return type and sideness in clang-dialect mode.
- Implement a fancier-than-nvcc edition of `-Wdefault-stream-launch`, warning about implicit use of the legacy default stream via any launch mechanism, with opt-in support for warning about any use of the default stream (ptds or not).
- PTX diagnositcs now correctly point at the offending operand, where applicable.
- Correctly report source locations for diagnostics in PTX blocks that use `C` constraints.

### Compiler Optimisation

- Use DPP to optimise applicable `__shfl*/__reduce*` calls, and loops that perform common reduction/cumsum idioms.
- Improved instruction selection for shared-memory reads of sub-word types.
- Vectorisation support for `__shfl*` (eg. turns multiple i8 shuffles into an i32 or i64 shuffle). Particularly useful for exploiting architectures that have support for i64 shuffles.
- Ability to move extending conversions across shuffles, to avoid shuffling useless bits.
- Don't generate redundant warp reductions prior to uniform atomic ops.
- Constant propagation for `__shfl*`.
- Improved program startup time in the presence of CUDA translation units with no device code.
- Improvements to `i1` vectorisation.
- Don't discard `__builtin_provable` too early when doing LTO or building bitcode libraries.

### NVCC Compatibility

- Support for `__shared__` and `__constant__` keywords an anonymous unions/structs.
- Allow `__device__` keyword to redundantly accompany `__shared__` or `__constant__`.
- Remove spurious warning about passing C++ pointers to memory ops.
- Commas in asm statement argument lists are, it turns out, optional.
- Support `forward-unknown-opts`.
- Correctly handle missing `template` keywords in more dependent name lookup scenarios.
- Tolerate wrong-sided access during constant evaluation.
- Cleaner diagnostics when handling redundant commas in the presence of multiple templates closing at once.
- Allow out-of-line redeclaration of namespaced template functions with conflicting signatures (this becomes an overload, when it should be a compile error).

### Inline PTX Support

- Support for undocumented syntax quirks, like spaces in the middle of directives.
- Support for a wider variety of constant expressions.
- Fix miscompilation of `min.xorsign.abs`.
- Fix miscompilation of `testp.finite` in `-ffast-math` mode.
- Correct behaviour of `dp4a/dp2a` in the presence of overflow.
- Correctly parse identifiers including `%` on asm blocks with no inputs/outputs.
- Fix miscompile of shifts/bmsk with extremely large shift amounts.
- Implement insane implicit asm-input conversion rules.
- Fix miscompiles in video instructions using min/max as the secondary op, since real behaviour
  turned out to differ from the manual.
- Avoid compiler crash when trying to constant-evaluate a PTX `n` input that has a type error.
- `tf32` support, via upcasting to `fp32`.
- Fix some corner cases of mixing vector splats with `_` operands.
- Fix crash parsing video byte selectors.
- Correctly handle implicit vector-of-vector types, because it turns out those are at least semantically a thing.
- Newly-supported instructions:
    * `cp.async.*`
    * `min/max`, 3-input edition.
    * `add/sub/mul/fma/cvt` with non-default rounding modes.
    * `mma/wmma` for sub-byte integer types.
    * `mma/wmma` for all remaining shapes, in all datatypes except fp8/6/4.
    * `red/atom` for vector types.
    * `cvt.pack`, 4-input edition.
    * `nanosleep`
- Newly-supported special registers:
    * `%globaltimer`
    * `%globaltimer_hi`
    * `%globaltimer_lo`
    * `%warpid`
    * `%nwarpid`
    * `%smid`
    * `%nsmid`

### Compiler Misc.

- Fixed compiler crash when doing `decltype(lambda)` in certain conditions.
- `__syncthreads_and()` and friends no longer use any shared memory.
- Improvements to compile speed, espeically in the presence of inline PTX. Extremely so in the presence of `mma/wmma`.

### Library: New APIs

See the [API diffs](https://docs.scale-lang.com/stable/manual/apis/) for precise information.

- Added most commonly-used NVML APIs.
- More `cuSOLVER/cuSPARSE`.
- Non-default-rounding-mode APIs (conversions and arithmetic).
- Added 1D CUarray copy APIs.
- Added device-side versions of device/context property query APIs (`cudaDeviceGetAttribute` etc).
- `bmma_sync`.
- `atomicAdd` for `float2` and `float4`.
- Added API for programmatically controlling SCALE's exception behaviour.

### Library: Fixes

- Allocator improvements prevent premature OOM due to address space fragmentation in long-running applications with a high level of memory churn.
- Fix const-correctness in coop-groups.
- Removed some macro leaks.
- `__sad()` now behaves correctly in the presence of integer overload.
- Fix bugs in FFT plan creation.
- Fix some edgecases relating to delting the active CUcontext before popping it.
- Fixed use of `cuCtxSetCurrent` when stack is empty
- Don't crash when unloading a module at the same time as launching one of its kernels.
- `scaleenv` now works with zsh, and does not pollute the shell environment.
- Endless tiny fixes, random macros, header compatibility tweaks, etc.
- Device PCI IDs now match the format of NVIDIA CUDA _exactly_.
- Fix some edgecases where denorms weren't being flushed, but should be.
- Stream creation is faster.
- Don't crash when the printf buffer is larger than 4GB.
- Fix rare hang when using the IPC APIs.

## Release 1.3.1 (2025-05-12)

### Compiler

- Fixed a bug in the handling of weak device-side symbols which broke the
  device binaries for certain projects.
- Fixed various PTX miscompilations.
- Added support for approximate-math PTX instructions (`lg2.approx` and
  friends).

### Library

- Fixed many small bugs in the device-side APIs.
- Per-thread-default-stream actually works now, rather than silently using
  the legacy stream.
- Fixed a race condition in the fft library.

### Thirdparty Project demos

- GROMACS now works. SCALE appears to support a wider selection of AMD
  architectures than the HIP port, and seems to perform somewhat better (on
  MI210, at least!).

## Release 1.3.0 (2025-04-23)

### Platform

- Upgraded from llvm17 to llvm19.1.7.
- Support for rocm 6.3.1.
- Support for `gfx902` architecture.
- Enterprise edition: Support for new architectures:
    - `gfx908`
    - `gfx90a`
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
