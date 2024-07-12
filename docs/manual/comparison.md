# Comparison to other solutions

## HIP

[HIP](https://github.com/ROCm/HIP) is AMD's answer to CUDA. It is superficially
similar to CUDA, providing a similar programming language and similar APIs. 
An automatic `hipify` tool exists to partially automate the process of 
rewriting your code from CUDA to HIP.

We believe HIP does not solve the "CUDA compatibility problem" because:

- The CUDA [dialect problem](./dialects.md). HIP's language is almost 
  identical to LLVM-dialect CUDA, which is quite different from the dialect 
  of CUDA accepted by `nvcc`. Consequently, some CUDA programs fail in 
  strange ways after porting.
- HIP has no support for inline PTX `asm` blocks in CUDA code. These must be
  manually removed or guarded by macros. SCALE simply accepts them and
  compiles them for AMD.
- HIP's support for NVIDIA is via wrapper APIs rather than simply using 
  NVIDIA's tools directly as a SCALE-based solution does.
- `hipify` is unable to handle many CUDA code constructs, such as complex 
  macros.

To avoid these issues, many projects end up maintaining separate HIP and 
CUDA codebases (or one codebase that converts to HIP or CUDA via complex
preprocessor macros).

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
  their hardware specifically.
- Compiling source code directly to AMDGPU machine code should
  offer greater opportunities for optimisation than working backwards from
  PTX that has already been optimised for a specific NVIDIA target.

We believe that ZLUDA fills a useful niche, but that software distributors 
should have the power to compile their CUDA source code directly to the 
machine code of multiple GPU vendors, without reliance on tools maintained 
by NVIDIA.
