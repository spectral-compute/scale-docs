# Compiler Optimisation Flags

When using the `nvcc` frontend to SCALE, it matches the behaviour of 
NVIDIA's compiler as closely as possible.

This means disabling some optimisations that are enabled by default by clang,
since those break certain programs which rely on the behaviour of NVIDIA's 
compiler. In some cases, such reliance represents undefined behaviour in the
affected program.

This page documents some of these differences, and how to opt-in to these 
optimisations on a case-by-case basis. In general, all `clang++` flags are 
also accepted by SCALE in `nvcc` mode.

You may be able to simply switch some of these features on and immediately 
gain performance.

tl;dr: try `-ffast-math -fstrict-aliasing` and see if your program explodes.

## Floating Point Optimisation

NVIDIA's compiler provides a `-use_fast_math` flag that relaxes some 
floating point rules to improve optimisation. It's documented to do exactly
these things:

- Use some less precise math functions (eg. `__sinf()` instead of `sinf()`).
- Enables less precise sqrt/division
- Flushes denorms to zero.
- Fuse multiply-adds into FMA operations.

SCALE mirrors this behaviour when you use this flag in our `nvcc` emulation 
mode, aiming to produce the same results as NVIDIA's compiler.

SCALE also provides all of `clang`'s flags,
allowing access to its more aggressive floating point optimisations, 
such as:

- Assume infinities and NaNs never happen
- Allow algebraic rearrangement of floating point calculations

Full details about these flags are available in the [clang user manal](https://releases.llvm.org/19.1.0/tools/clang/docs/UsersManual.html).

These optimisations can be controlled per-line using the [fp control pragmas](https://releases.llvm.org/19.1.0/tools/clang/docs/LanguageExtensions.html#extensions-to-specify-floating-point-flags).

This allows you to either:
- Specify the compiler flag (to enable an optimisation by default) and then 
  switch it off for special code regions (ie. opt-out mode).
- Opt-in to the optimisation in regions of code where you know it to be safe.

These flags will affect the performance of functions 
in the SCALE implementation of the CUDA Math API.

These flags do not affect the accuracy of the results of the Math API, but 
do apply assumptions about the range of possible inputs. For example: if 
you enable "assume no infinities", all infinity-handling logic will be removed
from the Math API functions, making them slightly more efficient. Flags like
"reassociate" and "use reciprocal math" do not affect the behaviour of the
math functions.

Each call to a math function will be optimised separately, using the set of 
fp optimisation flags in effect at that point in that file. You can use 
pragmas to mix different optimisation flags at different points within the 
same file. It is OK to compile different source files with different fp 
optimisation flags and then link them together.

## Strict aliasing

By default, in C++, the compiler assumes that pointers to unrelated types 
(eg `float` and `int`) never point to the same place. This can significantly
improve optimisation by improving instruction reordering and ILP.

Unfortunately, NVIDIA's `nvcc` does not do strict-aliasing optimisations, 
and enabling it breaks some CUDA programs. SCALE-nvcc therefore disables 
this by default.

You can explicitly enable this class of optimisations in SCALE by adding 
`-fstrict-aliasing`. This may break your program if it contains TBAA violations.
We recommend you find and fix such violations, since they are undefined
behaviour. This would mean your code is only working correctly because NVIDIA's
compiler doesn't currently exploit this type of optimisation: something which
may change in the future!
