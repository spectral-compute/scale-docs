# Compiler warnings

There are some differences in how NVIDIA's `nvcc` and the SCALE
compiler in "nvcc mode" interpret compiler options relatig to warnings.

## `clang++` flags

The SCALE compiler accepts all of `clang++`'s usual flags in addition to those
provided by nvcc, except where doing so would create an ambiguity.

## Compiler warnings

The SCALE compiler has the same default warning behaviour as `clang`, which
is somewhat more strict than `nvcc`. Warnings may be disabled with the usual
`-Wno-` flags documented in the [clang diagnostics reference](https://clang.llvm.org/docs/DiagnosticsReference.html).

There may be value in *enabling* even more warnings to find further issues
and improve your code.

Note that the end of every compiler warning message tells you the name of
the warning flag it is associated with, such as:

```
warning: implicit conversion from 'int' to 'float' changes value from
2147483647 to 2147483648 [-Wimplicit-const-int-float-conversion]
```

By changing `-W` to `-Wno-`, you obtain the flag required to disable that
warning.

The SCALE implementation of the CUDA runtime/driver APIs uses `[[nodiscard]]`
for the error return codes, meaning you'll get a warning from code that
ignores potential errors from CUDA APIs. This warning can be disabled via
`-Wno-unused-value`.

## `-Werror`

`nvcc`'s `-Werror` takes an argument specifying the types of warnings that
should be errors, such as:

```
-Werror reorder,default-stream-launch
```

This differs from clang's syntax, which consists of either a lone `-Werror`
to make all warnings into errors, or a set of `-Werror=name` flags to make
specific things into errors.

In `nvcc` mode, the SCALE compiler accepts only the `nvcc` syntax, but
allows the same set of diagnostic names accepted by `clang` (as well
as the special names supported by NVIDIA's `nvcc`). For example:

```
nvcc -Werror=documentation,implicit-int-conversion foo.cu
clang++ -Werror=documentation -Werror=implicit-int-conversion foo.cu
```

Since SCALE enables more warnings than nvcc does by default, many projects
using `-Werror` with nvcc will not compile without one of:

- Disabling `-Werror`.
- Disabling the corresponding warnings.
- Using diagnostic pragmas to disable the corresponding warnings in a region
  of code.

## Diagnostic control pragmas

The SCALE compiler does not currently support `#pragma nv_diag_suppress` or
`#pragma diag_suppress` because the set of integers accepted by these pragmas
does not appear to be documented, so we do not know which diagnostics should
be controlled by which pragmas. Using these pragmas in your program will
produce an "unrecognised pragma ignored" warning, which can itself be disabled
with `-Wno-unknown-pragmas`.

SCALE supports clang-style diagnostic pragmas, as documented [here](https://clang.llvm.org/docs/UsersManual.html#controlling-diagnostics-via-pragmas).
This can be combined with preprocessor macros to achieve the desired effect:

```c++
#if defined(__clang__) // All clang-like compilers, including SCALE.
#pragma clang diagnostic ignored "-Wunused-result"
#elif defined(__NVCC__) // NVCC, but not clang. AKA: nvidia's one.
#pragma nv_diag_suppress ...
#endif
```
