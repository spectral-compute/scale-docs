# Compiler warning flags

There are some differences in how NVIDIA's `nvcc` and the SCALE 
compiler in "nvcc mode" interpret compiler flags.

## `clang++` flags

The SCALE compiler accepts all of `clang++`'s usual flags in addition to those provided by nvcc,
except where doing so would create an ambiguity.

## Compiler warnings

The SCALE compiler has the same default warning behaviour as `clang`, which 
is somewhat more strict than `nvcc`. Warnings may be disabled with the usual 
`-Wno-` flags documented in the [clang diagnostics reference](https://clang.llvm.org/docs/DiagnosticsReference.html).

There may be value in *enabling* even more warnings to find further issues 
and improve your code.

The SCALE implementation of the CUDA runtime/driver APIs uses `[[nodiscard]]`
for the error return codes, meaning you'll get a warning from code that 
ignores potential errors from CUDA APIs. This warning can be disabled via 
`-Wno-`.

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
using `-Werror` with nvcc will not compile without either disabling the flag 
or fixing the underlying code issues.
