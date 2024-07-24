# What's new?

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
  member functions of template classes would sometimes not be compiled, 
  causing PTX to end up in the AMD binary unmodified.
- CUDA launch tokens with spaces (ie.: `myKernel<< <1, 1>> >()`) are now 
  supported.
- Building non-cuda C translation units with SCALE-nvcc now works.

### Other

- The `meson` build system no longer regards SCALE-nvcc as a "broken" compiler.
- `hsakmtsysinfo` no longer explodes if it doesn't like your GPU.
- Fleshed out the documentation pages slightly more.
 
## Release 1.0.0.0 (2024-07-15)

Initial release
