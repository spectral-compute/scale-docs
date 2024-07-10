# CUDA Dialects

The CUDA programming language is not formally specified. The "standard" is 
therefore approximately "whatever `nvcc` does". Although `clang` supports 
compiling CUDA, it supports a somewhat different dialect compared to `nvcc`.

You can read more about (some of) the specific differences in
[the LLVM manual page about it](https://llvm.org/docs/CompileCudaWithLLVM.html#dialect-differences-between-clang-and-nvcc)

This leads to a problem: most CUDA code is written with `nvcc` in mind, but 
the only open-source compiler available with a CUDA frontend is `clang`. Many 
real-world CUDA programs cannot be successfully compiled with `clang` 
because they depend on `nvcc`'s behaviour.

HIP experiences the same problem: the HIP compiler is based on LLVM, so HIP 
is closer to "LLVM-dialect CUDA" than it is to "nvcc-dialect CUDA". This 
causes some CUDA programs to fail in "interesting" ways when ported to HIP. It's
not really the case that you can remap all the API calls to the HIP ones and
expect it to work: nvcc-CUDA, and LLVM-CUDA/HIP have quite different C++ 
semantics.

SCALE resolves this issue by offering two compilers:

- `"nvcc"`: a clang frontend that replicates the behaviour of `nvcc`, allowing 
  existing CUDA programs to compile directly. This is similar to how LLVM 
  already provides `clang-cl`: a frontend to clang that replicates the 
  quirks of the 
  Microsoft compiler for c++.
- `clang`: providing clang's usual clang-dialect-CUDA support, with our 
  opt-in language extensions.

Existing projects can be compiled without modification using the 
`nvcc`-equivalent compiler. Users of clang-dialect CUDA may use the provided 
clang compiler to compile for either platform, optionally using the language 
extensions.
