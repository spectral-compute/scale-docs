# What's New

## Release 1.7.5 (2026-11-03)

Our largest release yet.

### Highlights

- LLVM 23
- Surface objects + more texture features.
- `CUlibrary`/`CUkernel` APIs.
- Mostly-completed cuFFT, cuSOLVER, cuSPARSE, and NPP.
- Substantial performance improvements, including:
    * Most gfx11 kernels ~2x faster
    * Kernel launch times
    * New compiler optimisations for NVIDIA devices.
- Vast numbers of bugs fixed.

Much of this work was made possible by validating SCALE against third-party projects, including PyTorch,
QUDA, hypre, Ginkgo, AMGX, HOOMD-blue, DEM-Engine, Viskores, cuPHY, ERF, llama.cpp, stdgpu,
jitify and nixl.

### Platform

- The compiler is now based on LLVM 23.1.
- Tarballs are smaller.
- A `version.json` is installed, as in NVIDIA's CUDA toolkits.
- Added `scaleenv.fish` for fish shell users.
- `CUDA_VISIBLE_DEVICES` is now supported.

### Compiler

- Based on LLVM 23.1.
- Fixed various problems that lead to linker errors.
- `__CUDA_ARCH__` /`__CUDA_ARCH_LIST__` now work correctly in multi-arch builds.
- Fixed `-fdevice-entity-has-hidden-visibility` and `-fstatic-global-template-stub` being enabled
  by mistake.
- Device-side `<<<>>>` launches are now rejected with a clear diagnostic, while direct calls to
  `cudaLaunchDevice()` (as in some Thrust versions) still compile.
- Variadic kernels are now rejected.
- Fixed a crash on reference captures of structured bindings.
- Captures in nested lambdas now resolve correctly through enclosing contexts.
- Fixed many crashes in semantic analysis.
- Inline managed variables and managed variable templates are now allocated.
- Device-side device-attribute queries report the right warp size when emulating 32-wide warps.
- `__noinline__` no longer clashes with libstdc++ 13's headers.
- The CUDA headers can now be included inside `extern "C" {}`.
- `__maxnreg__` is now defined.
- Device code may now be up to 1.5 GiB per device (but perhaps shouldn't be).
- Kernels launched only through uninlined stubs now benefit from launch-based optimisations.
- Added `__builtin_provably_uniform(X)`, which tells whether a value is provably the same across all
  threads executing together.

#### nvcc Compatibility

- Added host-compiler emulation: we now determinee which host compiler NVCC would use, then
  pretend to be that when handling the host side of CUDA files. Support (GCC, clang, MSVC, and PGI).
- Newly-supported nvcc flags.
  - `--resource-usage`
  - `--brief-diagnostics`
  - `--lib`
  - `--cudart`
  - `--fatbin` (treated like `--cubin`)
  - `--compress-mode=balance`
- `-G`, `-e`, `-time`, `-O`, `--optimize` and `--profile` now have nvcc's meanings.
- `-odir` now applies to default output names.
- nvcc options that have no effect under SCALE are now accepted instead of being rejected or
  misparsed as clang options:
  - `-ewp`
  - `-reloc-ptx`
  - `-nodlink`
  - `-Ofc`
  - `-device-stack-protector`
  - `-fdevice-time-trace`
  - `-objtemp`
  - `-arbin`
  - `-Xarchive`
  - `-nohdmoveforward`
  - `--forward-unknown-to-host-linker`
  - `--force-cl-env-setup`
  - `-diag-error`
  - `-diag-warn`
- `--compress-mode` is now passed to NVIDIA's `fatbinary` only from CUDA 12.8 onwards.
- Fixed more cursed flag parsing rules around  `-L`, `-l`, `-D`, `-U`, `-o`, `-include`, `-MF` and `-MT`.
- `-Xlinker` lists are now split on commas.
- Explicit `--fmad`, `--ftz` and `--prec-*` flags now override `--use_fast_math`, as with nvcc.
- `nv_diag_suppress` (and `-diag-suppress`) now actually suppress diagnostics.
- ANSI colour codes now work in static_assert messages.

#### Code Generation

- Kernels on RDNA3 GPUs are substantially faster (~2x).
- Approximate division is now faster.
- NaN handling is now much faster and matches NVIDIA mode closely.
- Many improvements to the shuffle optimiser:
  - Improved optimisation of common shuffle patterns/reductions.
  - Masked and sub-warp reductions are now optimised.
  - Fixed several crashes and miscompilations.
  - Concurrent, disjoint reductions are now supported.
- Fixed miscompilations:
  - Loops with divergent control flow no longer deadlock.
  - Register spill reloads could read stale values on AMD GPUs.
  - `v_dot2` is no longer formed from FMA chains without permission to approximate.
  - Integer to bf16 conversions were incorrectly rounded on all targets, including the host.
  - A `StructurizeCFG` crash (seen in SHOULDER).
  - An assertion copying aperture registers.
  - An infinite loop recovering from an AMDGPU assembly operand error at the end of a file.
  - An assertion on calls with a mismatched function type.
- Improved NVIDIA code generation via some new peephole optimisations.

#### PTX Support

- Newly-supported instructions:
  - `suld.b` and `sust.b`.
  - `tex` with `.v4.f16` and `.v2.f16x2` results.
  - `sqrt` with non-default rounding modes.
  - `setp` on `f16x2` and `bf16x2`.
  - `min`/`max` `.xorsign.abs` on `f16x2` and `bf16x2`.
  - Vector `atom`/`red` `min`/`max` on `f16`, `bf16`, `f16x2` and `bf16x2`.
  - Directed `cvt` from `f64` to `f16`/`bf16`, and between `f16` and `bf16`.
  - `barrier.sync` as a spelling of `bar.sync`.
- `div.full` and `sqrt.approx` are now much faster.
- Fixed miscompilations:
  - `0f` floating point hex literals are no longer interpreted as zero.
  - `shr.u64` (and `shl`) by a constant now actually shifts.
  - Signed `add.cc`, `sub.cc` and `mad.cc` now set the carry-bit correctly.
  - `cvt.sat` between integer types now clamps correctly.
  - Vector loads didn't extend elements into wider registers.
  - `vote.uni` is no longer interpreted as `vote.all`.
  - Instructions that write their own guard predicate now work correctly.
  - Byte and half selectors on video instructions are no longer ignored.
  - Video instruction secondary options now behave correctly.
  - `mma .xor.popc` now actually computes XOR, not OR.
  - `.relu` conversions no longer turn a negative NaN into 0.
  - `%warpid` is now the warp ID, not the lane index.
  - `%nwarpid` now gives the correct value in the last warp of a block with an
    incomplete final warp.
  - PTX cache operators are now respected (`ld.{cg/cs/cv`, `st.{cg,cs,wt}`)
  - `tex.base` no longer applies the texture's LOD bias.
  - `bar.sync` no longer hangs if some lanes have exited.
  - Several `shfl.sync` edgecases no longer exhibit undefined behaviour.
- Integer `asm` inputs narrower than their constraint are now sign- or
  zero-extended according to their type (`"l"(-5)` previously gave
  4294967291), with a warning unless they are constants. Other size mismatches
  are rejected, as by nvcc.
- Operand size rules for `ld`, `st` and `cvt` now match `ptxas`.
- Labels are scoped to their `{}` block.
- Unreachable instructions are now a warning, not an error.
- Declarations like `.reg .b32 r<1000000000>;` no longer hang the compiler.
- Very large `asm` blocks no longer take enormous amounts of time to process.
- PTX debuginfo directives no longer hang the compiler.
- Whitespace is now tolerated in more places in inline PTX.
- Fixed codegen of `%clock_hi`, `%current_graph_exec`, `%total_smem_size`,
  `%aggr_smem_size` and `%dynamic_smem_size`
- Fixed codegen of special registers used as members of a PTX vector,
  literal.
- `WARP_SZ` can now be used as a 64-bit operand.
- Improved tolerance to invalid PTX, avoiding `clangd` crashes.
- Improvements to PTX diagnostics, including stricter type and flag checking.

### Runtime Library

#### Error Handling

- API argument validation and error codes now match NVIDIA's across most of the API surface.
- Errors no longer incur so much CPU overhead.
- Errors from failed asserts, `__trap()` and trapping surface accesses are now reported at every
  later synchronisation point, in every thread.
- On RDNA3, a failed assert or `__trap()` now ends the wave and reports the error instead of
  aborting the process.
- The CUDA error log is thread-safe, and its iterators work after the log wraps.
- Invalid device `printf` placeholders no longer crash.
- Failed asserts print to stderr.

#### Devices and Contexts

- Added support for `CUDA_VISIBLE_DEVICES` and  `ROCR_VISIBLE_DEVICES`.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` sorts by the full PCI address.
- Device UUIDs agree with those reported by NVML.
- Initialisation errors are reported from the first `cudaGetDeviceCount()`, and from whichever
  call needed initialisation.
- Primary contexts:
  - Implemented `cuDevicePrimaryCtxSetFlags()`.
  - `cudaInitDevice()` flags apply to an existing primary context.
  - `cudaFree(nullptr)` initialises the primary context.
  - Resetting the primary context clears its flags.
  - `cuCtxGetCurrent()` reports the runtime's primary context.
- `cudaGetDeviceFlags()`/`cuCtxGetFlags()` report the current context's flags.
- `cuCtxGetId(nullptr)` returns the current context's ID.
- `cudaSetValidDevices()` is permitted while a context is current, and no longer disables every
  device when given an empty or rejected list.
- Peer access is now tracked per context. `cuCtxEnablePeerAccess()`/`cuCtxDisablePeerAccess()`
  are implemented, and `cudaDeviceCanAccessPeer()` reports the correct direction.
- Kernels launch correctly when peer links only go one way.
- The maximum grid size in y and z is now 65535, as on NVIDIA.
- Added `cudaDeviceProp::memPitch` and `cudaDeviceGetTexture1DLinearMaxWidth()` and friends.
- `clock()` and `clock64()` are monotonic and 64 bits wide on RDNA3.
- The profiler start/stop APIs succeed.

#### Streams, Events and Synchronisation

- Legacy default stream semantics are now more correct (and crazy).
- `cudaFreeHost()` now waits for the device, as `cudaFree()` does (seen in cuPHY).
- `cudaEventElapsedTime()` not being ready yet is no longer treated as an error.
- `cudaStreamGetFlags()` reports blocking streams correctly.
- Stream and launch attributes have NVIDIA's default values.
- Launch attribute queries read stream attributes instead of writing them.
- Each capture session has a distinct, non-zero ID.
- Added the `CU_STREAM_ATTRIBUTE_*` aliases, `cuStreamAttachMemAsync()` and `CUmemAttach_flags`.

#### Graphs

- Fixed many graph bugs:
  - Event wait nodes wait for the event's latest record.
  - `cudaGraphExecUpdate()` keeps launch order and flags.
  - Kernel arguments stay alive until launches using them finish.
  - Setting a child graph's kernel node parameters through the parent's exec works.
  - Changing a copy node to a 1D copy resets its height and depth.
  - Kernel node attributes are copied in the right direction.
  - Graph execs no longer get invalidated when a node's enabled state doesn't actually change.
  - Graph scheduling no longer stalls when dependencies are on other devices or behind blocked
    roots.
- Graph memory allocations are aligned and reserved as on NVIDIA, and each allocation is backed
  by its own size.
- Added `cuGraphInstantiateWithFlags()`.
- Fixed the C++ overloads of the graph symbol copy APIs, and the output arrays of the graph list
  queries.

#### Memory

- Device-to-device copies are significantly faster.
- Mid-sized allocations are much faster (up to ~500x for cold 1 MiB `cudaMalloc()`).
- Destroying a memory pool no longer unmaps blocks that its parent still owns.
- Default memory pool properties are zero-initialised. Two threads can no longer both create a
  device's default memory pool.
- Added the typed `cudaMallocFromPoolAsync()` overloads.
- Implemented `CU_POINTER_ATTRIBUTE_BUFFER_ID` and `CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE`.
- `cuPointerGetAttribute()` reports managed memory as device memory, and rejects all attributes
  of unknown memory.
- Multicast APIs report being unsupported, as NVIDIA does on unsupported hardware.
- Exporting VMM allocations as handle types they weren't created with is rejected.
- `cudaMemcpyToArray()` no longer corrupts memory on 2D arrays.
- Fixed the legacy v1 `cuMemcpy3D` symbols, and added `cuTexRefSetAddress()`.
- The legacy v1 `cuDeviceTotalMem()` saturates instead of wrapping.
- Legacy v1 driver stubs return `CUDA_ERROR_INVALID_CONTEXT`.
- Unregistering memory from `cudaMallocHost()` no longer asserts.

#### Modules and Libraries

- Implemented `CUlibrary`, including fatbin modules and `cudaLibraryGetGlobal()`. Modules are now
  owned by their contexts, and repeated lookups of a function return the same handle.
- Added `cudaGetFuncBySymbol()`.
- All driver functions added since CUDA 11.4 are now available through `cuGetProcAddress()`.
- Fixed stale instruction caches after module reload. Previously a newly-loaded kernel could run
  the code of an unloaded one (seen in jitify).
- Loading, launching and unloading modules from multiple threads is now safe.
- NVRTC no longer initialises the CUDA context, so it works without visible devices.
- The C++ API wrappers honour the per-thread default stream macros.

#### Textures and Surfaces

- Added surface objects and the `surf1D`, `surf2D`, `surf2DLayered` and `surf3D` APIs.
- 3D mipmaps with more than one level are now sampled correctly.
- Mip levels are laid out with their rounded-up sizes.
- Linear textures whose size isn't a whole number of elements are allowed.
- Texture descriptor queries return the correct host information.
- Destroying the null texture object is a no-op.
- Three-channel driver array and texture formats are rejected, as on NVIDIA.
- Out-of-bounds TMA elements are filled with NaN when the tensor map asks for it.

#### Warp-Level Operations and Cooperative Groups

- Warp intrinsics are now more correct, and optimise better.
  - `__shfl_*_sync()` honours its lane mask, including operands past the width.
  - `__reduce_*_sync()` with a partial mask no longer reads stale values.
  - `__all()` now works.
- `__fns()` and `fns` handle out-of-range bases.
- Named barriers no longer hang when lane 0 of a wavefront has exited.
- `__nanosleep(0)` returns immediately.
- Grid synchronisation:
  - Fixed hangs when the barrier's generation wraps.
  - Fixed arrival counting for 2D/3D blocks.
- Cooperative groups:
  - Added `grid_group::dim_threads()`/`thread_index()`,
    `cg::this_thread()`, `cg::thread_rank()` and `cg::group_size()`.
  - `thread_block_tile::ballot()` and `match_all()` compile, and their masks are tile-relative.
  - `coalesced_group`'s `shfl()`, `shfl_up()`, `shfl_down()`, ballot and match masks are correct.
  - `cg::reduce()`:
    - Reducing a `coalesced_group` returns the whole group's result.
    - Multi-warp tiles no longer require `block_tile_memory`.
    - Nested multi-warp tiles no longer interfere with one another.
  - Each multi-warp `thread_block_tile` has its own barrier.
  - `meta_group_size()` counts a partial last tile.
  - `binary_partition()` gives the correct meta group ranks.
  - Converting a group to `thread_group` and back no longer corrupts memory.
  - `grid_group`'s cluster queries report correct values for non-cluster launches.
  - `cg::memcpy_async()` now accepts `void*`.
  - Fixed compatibility with C++03 and C++11.
- Shipped `cuda/pipeline`.

#### Atomics and Memory Operations

- Added atomic min and max for f16, bf16, f16x2 and bf16x2.
- `atomicInc()`/`atomicDec()` work on fine-grained memory, and with a limit of `UINT_MAX`.
- The `__nv_atomic_*()` functions:
  - Work on mapped host and peer memory.
  - Accept integer memory orders.
  - Use NVIDIA's `__NV_THREAD_SCOPE_*` values.
- Shared-memory atomics are cheaper.
- `__ldcg()`, `__ldcv()` (and corresponding PTX instructions) honour their cache operators.
- Vector type alignment is capped at 16 bytes.
- Added `make_double4_16a()` and the other aligned vec4 make functions.
- `dim3` converts to `uint3` on the host.
- Device-side `cudaMalloc()` and `cudaFree()` are declared `extern "C"`.

### Maths Library

- New functions:
  - `cyl_bessel_i0f()`, `cyl_bessel_i1f()`
  - `jnf()`, `ynf()` (integer-order Bessel functions)
  - `normcdfinvf()`
  - Single-precision `norm*f()` functions.
  - `__isinff()`, `__finitef()`, `__signbitf()` and their double versions
  - `__tanhf()`
  - Host-side `rcbrt()` and `rcbrtf()`
- `-prec-div`, `-prec-sqrt` and fast maths now behave exactly as with nvcc.
- Accuracy improvements:
  - Double `pow()` is now within CUDA's error bound.
  - Double `lgamma()` and `tgamma()` are now less wrong.
  - Double `norm*()` is fixed.
  - Directed-rounding float reciprocals, divisions and square roots, `__frsqrt_rn()` and
    `__drcp_rn()` are correctly rounded.
  - `__fmaf_ieee_rn()` keeps subnormals.
  - `normcdff()`, `powf()`  no longer lose accuracy to contraction.
  - Complex division and `abs` no longer overflow/underflow early.
  - bf16 `sin`/`cos` use the high-precision implementations
  - bf16 `hrsqrt()` no longer double rounds.
- Faster `pow()` for common constant exponents and bases.
- NaN, infinity and saturation handling:
  - The exponential functions, `sinpif()`/`cospif()`/`sincospif()`, `atan2()`, `fmod()`,
    `remainder()`, `remquo()`, `logb()`, `nextafter()`, `hypot()` and `lgamma()` return the correct
    NaN.
  - `fmod()`, `remainder()` and `remquo()` return quiet NaNs.
  - `nan()` builds the requested NaN.
  - Float-to-integer conversions now clamp correctly.
  - `lround()`, `lrint()` and their `long long` forms saturate as nvcc does.
  - `scalbln()` handles exponents outside the range of `int`.
  - `__fdividef()` divides by subnormals as nvcc does.
- Many small maths and bit-manipulation functions are faster.
- Half, bf16 and minifloats:
  - Saturating conversions handle NaN/saturation correctly.
  - bf16 `__hmax`/`__hmin` handle NaNs and signed zeros correctly.
  - Fixed bf16 `__high2float()`/`__low2float()`.
  - Added the pairwise FP4 conversions, the missing E8M0 constructors from 64-bit integers, and
    the missing minifloat-to-bf16/char operators.
  - Fixed E8M0 overflow when rounding toward zero.
  - Minifloats now convert to bf16 correctly.
- SIMD and integer intrinsics:
  - Fixed `__uhadd()` and friends.
  - Fixed rounding in `__vavgs2()`/`__vavgs4()`.
  - Fixed `__vaddus2()`/`__vaddus4()` saturation.
  - Fixed `__vadd`/`__vsub` with zero low lanes.
  - Fixed the signedness of `__viaddmin_u16x2()`.
  - Added the missing device-side `half2char_rz()` and friends.
- The bitwise OR and XOR reduction operators return `T`.

### CUDA-X Libraries

- **cuFFT** has been rewritten:
  - Fixed many correctness bugs:
    - Non-square 2D/3D plans no longer transpose.
    - In-place real transforms use the padded layout.
    - Batched real transforms use the right distance.
    - User work areas are honoured.
    - `cufftXtExecDescriptor*()` no longer overwrites its input.
  - `cufftHandle` is now an `int`
  - Now works with `SCALE_EXCEPTIONS`.
  - Added:
    - `cufftSetPlanPropertyInt64()` and friends
    - `cufftXtSetWorkArea()`, `cufftXtSetWorkAreaPolicy()`, `cufftXtQueryPlan()`
    - The newer result codes and `cufftCompatibility`
  - Plans with arbitrary strides are supported.
  - Callbacks report `CUFFT_NOT_IMPLEMENTED`, as NVIDIA's shared library does.
- **cuFFTW**: added.
- **cuSPARSE** now covers essentially all of NVIDIA's APIs. Not yet supported:
  - The half-precision pruning APIs
  - `SpMMOp`/`SpMVOp`
  - CUDA 13.1's `SpGEAM`
  - Diagonal-only `SpSV`/`SpSM` updates
  - Sliced-ELL `SpMV`
  - 1- or 2-byte value transposes in `csr2cscEx2`
- **cuSOLVER** is now essentially complete:
  - All APIs now have C linkage.
  - `_bufferSize` APIs report real workspace sizes, and the user's workspace is used.
  - Invalid arguments and `info` values are now handled correctly.
  - `syevj`/`gesvdj` report non-convergence correctly. `gesvdj` returns V rather than V^H.
  - `gesvdaStridedBatched` reports residual norms.
  - `cusolverDnGetStream()` returns the stream that was set.
- **NPP**: added, covering around 500 of the total ~6000 APIs. More on the way.
- **cuBLAS**:
  - Newly-added APIs:
    - `cublas<t>syrkx()`
    - `cublasCgemm3m()`, `cublasZgemm3m()`, `cublasCgemm3mStridedBatched()`
    - `cublasRotEx()`
    - `cublasCsyrkEx()`, `cublasCherkEx()` and their 3M variants
    - Mixed-precision `gemvBatched` and `gemvStridedBatched`
    - `cublas{S,D}gemmGroupedBatched()`, `cublasGemmGroupedBatchedEx()`
    - `cublasSetLoggerCallback()`, `cublasGetLoggerCallback()`
    - `cublasLtMatmulAlgoGetHeuristicForStream()`, `cublasLtMatmulAlgoCheckForStream()`
  - Added the newer operation, maths mode, compute type and algorithm enumerators.
  - cuBLASLt attribute, tile, stage and numerical-flag values now correct.
  - `cublasSideMode_t` is numbered as in cuBLAS.
  - Complex GEMMs through the `*gemm*Ex` paths now work (seen in QUDA).
  - Handle structs have NVIDIA's names.
- **cuRAND**:
  - Fixed the signature of `curand_log_normal()`.
  - The sobol64 `curand_init()` now takes a 64-bit offset.
- **NVML**:
  - Added `nvmlDeviceGetFieldValues()` and the field IDs.
  - NVML no longer crashes when shut down during static destruction.
  - Devices are counted without initialising CUDA.
  - UUIDs match those reported by the CUDA runtime APIs.
  - Fixed `nvmlSystemGetProcessName()` with empty command lines.
- **NVTX** defines the unsuffixed A/W name macros.
- The maths libraries:
  - Respect `CUDA_VISIBLE_DEVICES`.
  - Work in whichever order they are first used.
  - No longer crash at exit or when handles are destroyed from static destructors.
  - No longer deadlock on some GFX9 GPUs (seen in llama.cpp).

## Release 1.7.4 (2026-10-08)

This release substantially improves the accuracy of the
single-precision maths library, adds mipmapped textures and 3D memcpy, brings
`-ffast-math` division and floating-point contraction in line with nvcc, and
fixes a wide range of bugs in graphs, cooperative launches, and occupancy
calculations.

### Platform

- Using ROCm 7.14.1 versions of rocBLAS etc.
- Fixed ROCm libraries in tarball packages being mangled by an outdated `patchelf`.
- Added the `bin2c` tool.

### Compiler

- Floating-point division now honours `-prec-div=false`/`__CUDA_PREC_DIV=0`
  (and therefore `-ffast-math`/`-use_fast_math`) by using approximate division,
  as nvcc does. This can be a very large speedup for division-heavy kernels.
- Floating-point contraction now matches nvcc: CUDA defaults to
  `-ffp-contract=fast-honor-pragmas`, so code that disables contraction via
  pragma is respected (including in linked bitcode). `-fmad=false` is now also
  passed on to `ptxas`.
- Reduced size of PTX output on NVIDIA targets.
- Added support for variadic functions in device code.
- `warpSize` no longer reports incorrect values in device code on some wave64 devices.
- Kernels that contain (but do not execute) `printf` no longer incur a performance penalty.
- Texture objects no longer impact register pressure so much.
- Added support for `nvcc -V`.
- Fixed `--dependency-output` crashing the compiler. This broke JIT builds of
  PyTorch extensions, FlashInfer, and SGLang kernels.
- Fixed `-keep`/`-keep-dir` when a configuration file is in use.
- Fixed `-fno-cuda-device-side-default` doing the opposite of what it says.
- Fixed an infinite loop when `-gencode` named a virtual architecture with
  nothing after it.
- Fixed a compiler crash when targeting `sm_100f`, `sm_101f`, or `sm_120f`.
- Fixed `sm_110a`/`sm_110f` rejecting builtins that need a minimum SM version
  (such as `__nanosleep()`).
- Fixed kernels being registered twice on NVIDIA when using recent CUDA versions.
- Fixed compiler crashes when instantiating lambdas in certain contexts (seen in
  NCCL), and when lambda parameter types contain errors.
- Fixed spurious wrong-side diagnostics arising from default arguments.
- Improved the error produced when trying to use `threadIdx` and friends in host code.
- Fixed miscompilation of atomic min/max on vectors of floats.
- Fixed host-side `max`/`min` of `unsigned long` truncating to 32 bits.
- Fixed a linker bug that led to some device symbols being dropped.
- Fixed the shuffle optimiser overlooking cheap 16- and 4-lane lowerings.
- Fixed stray `$` characters in merged host/device dependency files.
- Fixed stray debug output to stderr when compiling with some precision flags.
- Fixed the wrong SCALE installation being selected when several are present
  in the default locations.

#### PTX Support

- Newly-supported PTX opcodes:
  - `tex` (1D, 2D, 3D, 1D/2D layered, cubemap and layered cubemap), including
    LOD and offset variants.
- Inline PTX `add`, `sub` and `mul` without an explicit rounding mode may now
  contract into FMAs, as `ptxas` would allow.
- Unused cache flags on memory instructions are now ignored rather than producing a warning.
- Fixed conversions from `float` to fp8 types.
- Fixed the two-output form of `setp`.
- Fixed a crash when binding a float value to an `"n"` operand.
- Fixed insufficient operand checking for `atom` and the `vadd` family.
- Fixed user declarations being able to shadow the CUDA APIs used by PTX lowering. Hopefully nobody was actually doing this...

### Runtime Library

- Fixed a collision with glibc's C23 maths headers.
- Textures:
  - Added mipmapped arrays and mipmapped textures (runtime and driver APIs), including LOD bias.
  - Added software emulation for some missing texture features on gfx90a.
  - Fixed wide linear textures.
  - Added f16 textures.
  - Read mode is now ignored for float textures.
  - Added support for texture-related cudaDeviceAttr queries.
- Added support for `cudaDevAttrMaxPitch`, and various NUMA/PCI-related device attribute queries.
- Improved kernel launch performance.
- Improved performance of 3D memcpy.
- 3D memcpy can now be captured into CUDA graphs.
- Fixed a race condition that could lead to copies involving pageable
  host memory not being properly waited for.
- Newly-added APIs:
  - `cuGraphicsResourceGetMappedPointer()`
  - `cuGreenCtxGetDevResource()`
  - `cuEventElapsedTime_v2()`
  - `cudaDeviceFlushGPUDirectRDMAWrites()` and friends
  - `cudaLaunchAttributeValue::preferredClusterDim`
  - `cudaGraphDependencyType`, `cudaDeviceSyncMemops`,
    `cudaMemPoolCreateUsageHwDecompress`, `CUarray_cubemap_face`,
    `CUstreamUpdateCaptureDependencies_flags`, `CUshared_carveout`
- `__trap()` now produces `cudaErrorLaunchFailure` in `cudaGetLastError()`.
- Explicit-scope atomics no longer use the wrong scope.
- Fixed deadlocks in `grid.sync()`.
- Fixed several defects in the occupancy calculator, which could badly
  underestimate achievable occupancy. Occupancy APIs also validate their
  inputs more strictly.
- Improved TLB behaviour for some allocation sizes.
- Graph fixes:
  - Fixed numerous bugs in graph memory nodes.
  - Fixed several graph-api-related deadlocks.
  - Fixed a hang when destroying the graph scheduler.
  - Fixed swapped arguments in `cudaGraphKernelNodeCopyAttributes()`.
- Fixed indexing defect in the prefetch APIs.
- `cuCtxDetach()` no longer refuses to work except on contexts where it cannot possibly work.
- `cuCtxGetApiVersion()` now returns a real CUDA API version.
- Fixed `cuDeviceGetName()` with negative lengths.
- Fixed backwards access flags in memory import.
- Probing unknown pointers via the VMM APIs now returns `cudaErrorInvalidValue` instead of asserting.
- NVML no longer pollutes `cudaGetLastError()`.
- Fixed typos in some minifloat API declarations.
- Fixed allocation handles and host pointers sometimes being 32 bits wide,
  which broke nixl.
- `cooperative_groups.h` is now accepted by non-cuda translation units.
- `cudaTypedefs.h` is no longer missing most of its per-thread-default-stream variants.

### Maths Library

- Added `remquof()`, `erfcxf()`, `j0f()`, `j1f()`, `y0f()` and `y1f()`.
- Improved accuracy of `erff()`, `erfinvf()`, `erfcinvf()` (which was
  catastrophically wrong in its tail), `tgammaf()`, `lgammaf()`, `cbrtf()`,
  `logf()`, `log1pf()`, `log10f()`, `exp10f()`, and `tanhf()`, all of which now
  meet NVIDIA's documented error bounds.
- `coshf()` no longer overflows early.
- Faster `rsqrtf()`, using the hardware instruction.
- Fixed double-precision `sinpi()`, `cospi()`, `lgamma()` and `tgamma()`
  near their zeroes.
- `__fadd_rn()`, `__fmul_rn()` and friends never fuse into FMAs, as CUDA
  promises.
- `__fsqrt_rn()` is now correctly rounded even under `-ffast-math`.
- Fixed `fdividef()`, which divided approximately by default and was not found
  by user code. `__fdividef()` now uses the hardware reciprocal.
- The fast approx functions (`__tanf()`, `__powf()`, `__tanhf()`, `__sinf()`, etc.)
  now follow NVIDIA's documented implementations, and fast-math now maps
  `tanh`, `exp10` and `log10` to the fast versions.
- `cuCfmaf()` now rounds correctly.
- Many signed-zero, infinity, and flushed-denormal fixes, including in `erf()`,
  `erfc()`, `cbrt()`, `fmodf()`, `remainderf()`, `remquof()`, `nextafter()`,
  `nextafterf()`, `__saturatef()`, `frexpf()`, `powf()`, `tgammaf()`,
  `erfinvf()`, `scalblnf()`, and `modf()`.
- Adding missing conversions between `half2`/`bfloat162` and their "raw" counterparts.
- Fixed `half` increment/decrement operators.

### CUDA-X Libraries

- Declared `cublasGetStatusString()` in `cublasLt.h`.
- Added the rest of `cublasGemmAlgo_t`.
- cuBLASLt now accepts the `FAST` and `PEDANTIC` compute types.
- Internal cuBLASLt headers are no longer installed.

## Release 1.7.3 (2026-09-14)

This release adds Rocky 8 support (and support for older glibc in general), hugely
expands texture API support, introduces initial support for TMA APIs, and has the
usual selection of many bugfixes and performance improvements.

### Platform

- Rocky/RHEL 8 package is now available.
- More compiler components now ship statically linked, reducing opportunities to have
  exciting conflicts with other LLVM-based compilers (such as the Cray compiler).
- Using rocm 7.2.4 versions of rocBLAS etc.

### Compiler

- Fixed various compiler crashes in the nvcc semantics mode, in both `clang`
  and `clangd`.
- Fixed an NVIDIA-specific crash when a shuffle loop optimised to
  a `redux` instruction on a target without `redux` hardware.
- Fixed some NVIDIA-specific math API overload issues.
- Fixed spurious complaints about initialisation on `__shared__` variables.
- Fixed incorrect `__CUDA_ARCH__` values when the ccmap feature was in use.
- Various warnings that could not be turned off now have control flags.
- Fixed bug causing stack variables to sometimes be less aligned than
  they should be when building for NVIDIA targets.
- Improved behaviour of warnings in nvcc mode, avoiding cases where
  warnings would sometimes "vanish".
- Added (no-op) support for `#pragma nv_diag_suppress`. Maybe someday we'll
  find the error code reference manual...
- Various nvcc semantics alignment improvements (ignoring `template` in more places,
  etc.).
- Reject kernels using too much `param` memory at compile-time. This avoids having
  `ptxas` do it on NVIDIA platforms, and makes it happen at all on AMD.
- Improved unroll behaviour in long kernels.
- Deduce max-threads-per-block from kernel callsites.

#### PTX Support

- Fixed a crash that could occur when inline PTX contained type errors.
- Fixed miscompilation of certain PTX memory instructions in nvcc mode.
- Newly-supported PTX opcodes:
  - `mbarrier`
  - `cp.async.bulk.*`
  - `cp.async.bulk.tensor.*`
  - `cp.async.bulk.prefetch.*`
  - `fence.proxy.async`

### Runtime Library

- Improvements to "No device found" error messages.
- Added `cuMemGetMemPool()`.
- Added missing templated `cudaMallocAsync()`.
- Improved kernel launch performance.
- Eliminated some unnecessary synchronisation when launching graphs involving
  host-to-host copies.
- Fixed a discrepancy between the device-side runtime API's warpSize property
  and that reported by the warpSize global on wave64 devices.
- Added the [CUDA Execution Context Management API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EXECUTION__CONTEXT.html).
- Added support for Green Contexts.
- A slew of small header compatibility improvements.
- Minifloat conversions are now less wrong.
- Vector elements no longer get permuted during vector-minifloat
  conversions.
- Fixed accuracy issues in several device-side math APIs (mostly hyperbolic trig
  and logarithms).
- Zero-sized memset of nullptr is now allowed.
- Zero-sized `cudaMallocPitch()` now works correctly.
- Fixed 3D textures on GFX12 devices.
- Added layered texture support.
- Added cubemap support.
- Added Lod/Grad texture samplers.
- Fixed unresolved symbol issue for `sincosf()`.
- `cuArray3DGetDescriptor()` now behaves properly for 3D textures.
- Added software-emulated TMA for devices that lack hardware support.
- Cooperative groups: added block and block-tile reductions.
- Cooperative groups: Introduced the `__v1` namespace used by newer CUDA versions.

### CUDA-X Libraries

- Fixed C89 compatibility issues in cuFFT, cuSOLVER, and cuBLAS headers.
- `cufftGetVersion()` now returns the cuFFT version.
- Added some missing `*_v2` cuBLAS APIs.
- Fixed various hangs/crashes from calling into cuSolver/cuSparse.
- curand-device is now in the bitcode library, so much less sensitive to header
  strangeness.
- Avoid shipping any rocm headers. They mostly weren't used, but caused problems
  for people trying to use HIP while SCALE was installed.
- Fixed subtle typing issues of the curand direction vector APIs.
- Added a few more npp types.
- `curandState` now correctly defaults to XORWOW.

## Release 1.7.2 (2026-07-19)

This release adds new CUDA APIs, and fixes bugs from previous releases of SCALE.

In the compiler, we have continued our progress towards full NVCC compatibility,
adding support for a number of undocumented features of the CUDA language. The
runtime library has gained a number of new CUDA APIs and had some of its most
long-standing hangs/crashes fixed. We have fixed bugs in both the compiler and
runtime library.

### Compiler

- Fixed miscompiles of PTX `elect` instruction.
- Fixed NVIDIA-specific build failures relating to `__assert_fail()`.
- Fixed crash when using a redundant `.sat` modifier in inline PTX.
- Fix a crash when empty asm blocks were used with PTX constraint letters
  that were not valid asm constraints for the AMDGPU backend.
- Fixed a crash when compiling shuffles whose source lanes evaluate to undefined
  behaviour.
- Fixed a crash when lowering some warp reductions that fully constant-propagate.
- Fixed miscompilation of `bar.sync 0`.

### Compiler Optimiser

- Huge performance improvement for atomic arithmetic operations on non-managed
  device memory for GFX10/GFX11 devices.
- Improvements to the shuffle optimiser, allowing it to handle even more
  complex scenarios.
- Fixed many compiler crashes and spurious compiler errors relating to the NVCC
  dialect.

### NVCC Compatibility

- Support the NVCC dialect's more permissive attribute placement rules.
- Implement NVCC's strange handling of `+` constraints.
- Added `#pragma nv_exec_check_disable`.
- Improved handling of constructors for `__shared__` variables, more closely
  matching `nvcc`'s behaviour.
- Initializer-lists now work in the arguments to `<<<>>>`.
- Fixed some defects in function-emission rules that led to the need to patch
  CCCL's `EmptyKernel`.
- Newly-supported `nvcc` flags:
    * `-device-entity-has-hidden-visibility`
    * `-static-global-template-stub`
    * `-generate-dependencies-with-compile`
    * `-generate-nonsystem-dependencies-with-compile`

### Library

- Added a new leak-detection feature (see documentation for runtime extensions).
- Added the CUDA 11 variants of the entrypoint access APIs.
- Added `curand_uniform2_double`.
- Graph API:
    * Added Graph Memory Nodes and related APIs.
    * Added `cudaGraphNodeGetToolsId`/`cudaGraphNodeGetLocalId`.
    * Fixed various header/typedef compatibility issues.
    * Added some of the missing driver APIs.
    * Increased how much of the API works when `long long` is unavailable.
    * Fixed bugs that yielded hangs when external event nodes are in use.
    * `cuGraphDebugDotPrint` now always prints dependency edges correctly.
    * graphExec flags no longer get forgotten.
    * Fixed a crash when a recording session involves secondary streams that
      contribute no nodes.
    * Fixed a crash when deleting a cudaStream in the middle of a capture
      session.
    * Fixed a hang when deleting still-active `cudaGraphExec_t`s.
- Fixed a startup-time crash on some older GPUs.
- The deprecated stack-based launch APIs (`cudaSetupArgument()` etc.) are now
  associated with the thread, not the CUDAContext, as is required.
- Added `cudaMemcpy2DFromArrayAsync`.
- Added `cuMemcpy3D/cuMemcpy3DAsync`.
- Improvements to the RTC header environment, making it more compatible with
  the many and varied hacks people use in this domain.
- Fix further math function declaration compatibility issues.
- `nodiscard` on `cudaError_t` is now opt-in.
- Added `__device_builtin__`.
- Fixed conversion ambiguity between bf16/fp16.
- Added stream and kernel node attribute APIs.
- `warpSize` is no longer `constexpr` in nvcc-dialect mode, after finding real
  programs that manage to break when it is. It will, of course, constant-propagate.
- Fixed some missing enums/API defects and crashes in `cublasLt`.
- `cublas` and `cublasLt` handles are now layout-compatible, as is required.
- Fixed a hang when combining `cuCtxRecordEvent` with deletion of still-busy
  CUDA streams.
- Emit the proper cudaError code when trying to allocate more memory than
  exists in the universe.
- Various random little APIS added.

As always, see [the apidiff](../library/apidiff/apis.md) for the precise library completeness status.

## Release 1.7.1 (2026-06-01)

Primarily a library update, fixing many bugs in the graph API
and introducing many previously-missing APIs.

### Platform

- Updated bundled rocm math libraries to vesion 7.2.3
- Updated bundled cccl version to 2.8.2

### Library

- Added CUDA Logging API.
- Added stream-ordered memory allocator (mempool) APIs.
- Added CUDA User Object APIs.
- `CUDA_LOG_FILE` and `CUDA_LAUNCH_BLOCKING` environment variables now supported.
- Graph API: added node enable/disable
- Graph API: fixed bugs causing incorrect graph topologies during multi-
  stream recording.
- Graph API: added support for external event nodes.
- Graph API: dot-debug printer now prints much more information.
- Many other graph API bugs fixed.
- Expanded cusolver/cusparse API support
- Fixed bug in bf16 `rint()`.
- Fixed an issue where extra overloads of `isinf()`/`isinan()` were provided,
  causing some programs to pick the wrong one during overload resolution.
- Device-side assert failures now print to stderr, not stdout.
- Printf buffer is now flushed in more cases, matching nvidia's behaviour.
- Added `h2tanh(__bfloat162)`.
- Added `cuCtxWait/RecordEvent()`.
- Added some `mbarrier` APIs.

See [apidiffs](../library/apidiff/apis.md) for updated API compatibility status.

### Compiler

- New PTX instruction: `mbarrier`.
- Support for `-Xptxas -gpu-name`.
- `compiler-rt` is now bundled.

## Release 1.7.0 (2026-05-12)

### Compiler

- Significant improvements to nvcc semantics emulation.
- `-save-temps` no longer breaks compilation.
- Fixed many compiler crashes in the presence of invalid code.
- Support for `__managed__` keyword.
- Compiler is now built with `RPATH` not `RUNPATH`, preventing strange crashes on systems with unusual library search path configurations.
- Improvements to optimisation of host-side of CUDA translation units
- Improvements to performance of device-side memory operations, particularly atomics.
- Fix occasional miscompilation of kernels that have `__launch_bounds__`.
- Fix issues with subtle symbol emissions rules which led to missing device symbols in some projects.
- Enable the `shfl` optimiser for NVIDIA mode, too (pattern-matching shufle sequences onto `redux`, etc.)
- `-fdeclspec` is no longer implicitly enabled in NVIDIA mode.
- Newly-supported NVCC flag: `--keep-device-functions`

### PTX Support

- Fixed compiler crashes when compiling `atom` and `red` instructions.
- Fix a compiler crash when using certain constraint letters with empty inline PTX asm blocks in AMD mode. Apparently people do that?
- Added support for `bar.sync X, Y` on all architectures.

### Library

- Fix bug in `cooperative_groups` reduce function.
- Managed memory is now *much* faster on XNACK-compatible devices (when `SCALE_AMD_XNACK` is enabled).
- Small performance improvements to almost every API.
- Improved error messages for certain common screwups, such as building for the wrong architecture.
- `__bf16` is no longer corrupted when implicitly converted to `__nv_bfloat16_raw`.
- `cudaMemPrefetchAsync` now works correctly when targeting the host on multi-CPU systems.
- Add undocumented `__isnan()` and `__assertfail()` device functions.
- OpenGL Interop support
- More Graph API support (child graphs, etc.).
- Fix miscalculation of SGPR occupancy leading to reduced performance.
- GPU-side assert fails now produce the proper CUDA error code at the next sync point.
- Fixed a bug where CUDA graphs with converging dependency edges would sometimes fail to launch.
- Device-side printf now accepts host pointers as operands to `%s`.
- Memory allocated with `cudaMallocHost` is now visible to all devices by default.
- Kernel launches no longer fail when an uninitialised GPU is present (but not being used).
- Add missing `htanh(__bfloat16)` function.
- Added cublasLt.
- Lots more cusparse, cusolver APIs
- Lots more cuda apis (list being calculated)

## Release 1.6.1 (2026-03-13)

### Supported architectures

- Added support for most NVIDIA GPUs.

### Compiler

- `clangd` works now, providing meaningful diagnostics for *both sides* of CUDA translation units, in either clang or nvcc dialect mode.
- Fixed miscompiles of constexpr-evaluated expressions involving errors.
- Fix an edgecase of SFINAE behaving differently from NVIDIA nvcc.
- Fix various compiler crashes relating to diagnostic deferral.
- Fix a compiler crash caused by attempting to convert an i1 to bf16.
- Various performance improvements.
- Don't reject `__launch_bounds__` expressions containing commas.
- Added an optimisation to detect cumsums and lower them using DPP/PERMLANE.
- CUDA `printf` is now subjected to type checking of the format arguments against the format string speifiers.
- PTX diagnostics now prettily underline the offending element.
- Slightly improved compile times for all CUDA translation units.
- `elect` instruction no longer crashes the compiler.
- PTX parser no longer rejects labels immediatley followed by `}`.
- Newly-supported NVCC flags:
    * `--gpu-architecture`
    * `--gpu-code`

### Library (AMD)

- Managed memory is now approximatley 50x faster.
- `cudaDeviceSynchronize()` is now much faster.
- `XNACK` may be used to accelerate certain workloads on supportd hardware now.
- C++03 is now supported.
- `scaleenv` no longer explodes when users set non-default bash flags.
- Various header compatibility fixes/tweaks.
- Added `atomicAdd_block/system` for `float2`/`float4`.


## Release 1.5.1 (2026-02-03)

### Compiler: NVCC emulation

- `-Wregister` no longer defaults to an error in nvcc dialect mode.
- `-Xfatbin` and `-Xptxas` are now parsed to extract meaningful flags from them.
- `nvcc -v`'s path outputs now reflect the actual CUDA install search process.
- Fixed an edgecase of the `target_overload` attribute that led to incompatibility with some versions of the standard library.
- Don't crash when a variable is illegally declared `__host__ __device__`.
- `nv_weak` is now an alias of `weak`.
- Added support for `-ptx` flag.
- Fixed some edgecases of argument translation for list-form arguments.
- Dependency information is no longer duplicated for CUDA mode (for `-M` and friends).
- *Even more* missing-typename relaxations to support the nvcc dialect.
- Unqualified lookups of dependent members in base classes is now alloewd in nvcc mode.
- Fix some builtin macro pedantry so programs that check with `#if <foo> == 1` work, as well as simply `#ifdef <foo>`, work.
- Implement some simple IR-level optimisations for `__byte_perm()`.
- `--extended-lambda` and `--relaxed-constexpr` now drive the same macro as they do in NVIDIA's implementation, even though they still do not change the behaviour of the semantic analyzer.

### Compiler: Inline PTX support

- Add minifloat types (fp8/6/4) to the PTX type system, pending implementation of the corresponding tensor core ops.
- Fix subtle miscompiles surrounding edgecases for `.relu` PTX operations. Seems the semantics of the `.relu` operator differ between different opcodes...
- Fix miscompile when both `.satf` and `.relu` are provided on the same instruction.
- Implement `__nv_is_extended_device_lambda_with_preserved_return_type`. Although it is largely meaningless given our proper lambda suppport, user code relies on it.
- Treat `.mmio` as an alias of `.volatile`, at least until we support some devices with actual IO devices stuck to the GPU.
- Fix a compiler crash when passing a pointer to a struct directly into a PTX asm block.
- Fix a compiler crash when compiling certain SIMD PTX instructions, such as `fma.sat.f32x2`.
- Fix a type inference failure for instructions include vector expressions with constants in them, such as `asm("mov.b32 %0, {0, %1};" : "=f"(f) : "h"(h));`.
- Fix a compiler crash when a mandatory round-mode operand was omitted from a float-to-int `cvt` instruction.
- Fix failure to compile certain cases of `cvt`, such as `f64<->f16`.
- Fix failure to compile vector-splat cases of `cvt` for small float types.

### Compiler: Other

- Fix a miscompilation in which rounding-mode state would sometimes be leaked, leaving the rounding mode register set to a non-default value after executing certain combinations special-rounding-mode operations.
- `clangd` now crashes less often.

### Library

- Newly-added CUDA APIs:
    * `cudaMalloc3D`
    * `cuCtxSynchronize_v2`
    * `cuCtxGetDevice_v2`
    * `cuCtxWaitEvent`
    * `cuFuncGetName`
    * `cuFuncGetParamInfo`
    * `cuStreamWriteValue32`
- Add the NVML APIs for query process information.
- Small improvements to exception handling performance.
- Fix an implicit synchronisation bug for memsets. Memsets targeting pinned host memory are supposed to implicitly synchronise the affected stream.
- Fix a crash when using the `extra` argument of various launch APIs to pass a user-provided kernel argument buffer containing objects with "diamond of doom" inheritance.
- `half2char` now properly converts to *signed* char, not unsigned char.
- Fixed handling of `-0` and `NaN` in various Math API routines.
- Fixed a few cases where denorms were flushed on the "wrong side" of the operation. NVIDIA don't document which operations flush denorms before doig the operation (eg. min()), and which ones flush afterwards.
- Fixed an endianness issue affecting bf16x2 assembly APIs such as `__low2bfloat16`.
- Entirely disabled SDMA for gfx9xx devices except gfx90a, due to apparent hardware bugs. This should resolve issues where cudaMemcpys would mysteriously not synchronise with the host even when asked to. Seems SDMA barriers sometimes just don't work...?
- Fixed various header defects (missing `typedef enum`, implicit STL includes, etc.)

### Other

- scaleenv no longer leaks PS1 to the env.
- The scripts in scale-validation are now much simpler, hopefully making it more obvious that they are just copies of the upstream build scripts!


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

### Compiler: Inline PTX support

- Improve PTX diagnostics for unused instruction flags
- Support for `q` constraints (128-bit asm inputs)
- Diagnostics for implicit truncation via asm constraints.
- New PTX instructions:
    * `movmatrix`
    * `bar`/`barrier`: only cases that can be represented as `__synthreads_*`.

### Compiler: NVCC emulation

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

### Compiler (misc.)

- Introduce builtins for conversion between arbitrary types (fp8 here we come)
- Further improvements to deferred diagnostics, especialy surrounding typo'd identifiers
- Add `[[clang::getter]]`
- Fix kernel argument buffer alignment sometimes being wrong
- Microsoft extensions are no longer enabled in CUDA mode.
- Arithmetic involving `threadIdx` and friends now compiles.
- Various small optimiser enhancements.

### Runtime

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

#### NVCC Compatibility

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

- Added `scaleenv`, a new and much easier way to [use SCALE](../tutorials/how-to-use.md).
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

- [Improved handling of wave64](../compiler/inline-ptx.md#wave64-considerations) in inline PTX.
- Various inline PTX compilation fixes.

### Other

- Support for Ubuntu 24.04.
- Upgraded to ROCm 6.2.2.


## Release 1.1.0 (2024-10-31)

### Library Enhancements

- Added much of the CUDA graph API.
- Improvements to multi-GPU handling.
- Fixed rare shutdown-time segfaults.
- Added many random API functions. As usual, see [The diff](../library/apidiff/apis.md).

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


## Release 1.0.2 (2024-09-05)

Documented a record of the CUDA APIs already available in SCALE, and those still to come: [Implemented APIs](../library/apidiff/apis.md).

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


## Release 1.0.1 (2024-07-24)

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
