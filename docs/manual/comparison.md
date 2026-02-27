# Comparison to other solutions

## HIP

[HIP](https://github.com/ROCm/HIP) is AMD's answer to CUDA. It is superficially
similar to CUDA, providing a similar programming language and similar APIs.
An automatic `hipify` tool exists to partially automate the process of
rewriting your code from CUDA to HIP.

We believe HIP does not solve the "CUDA compatibility problem" because:

- HIP does not address the  CUDA [dialect problem](./dialects.md).
  HIP's language is almost identical to LLVM-dialect CUDA, which is
  quite different from the dialect
  of CUDA accepted by NVIDIA `nvcc`. Consequently, many CUDA programs fail in
  strange ways after porting, if they compile at all.
- HIP has no support for inline PTX `asm` blocks in CUDA code. These must be
  manually removed or guarded by macros. SCALE simply accepts them and
  compiles them for AMD.
- HIP's support for NVIDIA is via wrapper APIs rather than simply using
  NVIDIA's tools directly as a SCALE-based solution does.
- `hipify` is unable to handle many CUDA code constructs, such as complex
  macros.
- HIP runtime APIs have subtly different semantics than the corresponding
  CUDA APIs. `hipify` operates under the incorrect assumption that all
  `cudaFoo()` can be mapped to `hipFoo()` to achieve the same effect, which
  is not the case. SCALE aims to carefully reproduce the behaviour of
  NVIDIA's CUDA APIs.

Most projects that use HIP mitigate these issues by maintaining separate
HIP and  CUDA codebases, or one codebase that converts to HIP or CUDA via
complex preprocessor macros. This significantly increases maintenance costs.

We relatively often encounter projects with a significant performance or
correctness discrepancy between their CUDA and HIP editions, because one
or the other gets more attention, or gets changes merged more promptly.
Such source fragmentation is bad for everyone.

## ZLUDA

[ZLUDA](https://github.com/vosen/ZLUDA) is a PTX JIT for AMD GPUs. On program
startup, ZLUDA grabs the PTX from the CUDA binary and compiles it for your AMD
GPU.

ZLUDA is a useful tool for end-users to run CUDA programs on
otherwise-unsupported GPUs, without the involvement of the authors of the
program (or even access to the source code!).

There are some downsides:

- JIT on startup can lead to startup-time delays.
- Reliance on dll-injection is a bit "hacky", and tends to make antivirus
  software angry.
- ZLUDA's approach to providing AMD support inherently depends on tools
  provided by NVIDIA. NVIDIA controls the design of the PTX language and the
  compilers that produce it, and manipulate both to optimise outcomes for
  their hardware.
- Compiling source code directly to AMDGPU machine code should
  offer greater opportunities for optimisation than working backwards from
  PTX that has already been optimised for a specific NVIDIA target. ZLUDA
  has to deal with many of the same problems that a decompiler does, which
  are intractible in general.
- ZLUDA does not support AMD GPUs with a wave size of 64, which includes
  many exciting datacenter devices such as the MI300.

We believe that ZLUDA fills a useful niche, but that software distributors
should have the power to compile their CUDA source code directly to the
machine code of multiple GPU vendors, without reliance on tools maintained
by NVIDIA.
