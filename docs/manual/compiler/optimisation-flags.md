# Optimisation Flags

When using the `nvcc` frontend to SCALE, it matches the behaviour of
NVIDIA's compiler as closely as possible.

This means disabling some optimisations that NVIDIA nvcc does not do, and
which break programs which rely on this fact. This usually represents
undefined behaviour in the affected program.

This page documents some of these differences, and how to opt-in to those
optimisations. You may be able to simply switch some of these features on
and immediately gain performance.

tl;dr: try `-ffast-math -fstrict-aliasing` and see if your program explodes.

## Floating Point Optimisation

NVIDIA's compiler provides a `-use_fast_math` flag that relaxes some
floating point rules to improve optimisation. It's documented to do exactly
these things:

- Use some less precise math functions (eg. `__sinf()` instead of `sinf()`).
- Use less precise sqrt/division
- Flush denorms to zero.
- Fuse multiply-adds into FMA operations.

SCALE does likewise when you use this flag in our `nvcc` emulation
mode, aiming to produce the same results as NVIDIA's compiler.

SCALE also provides all of `clang`'s flags,
allowing access to its more aggressive floating point optimisations,
such as:

- Assume infinities and NaNs never happen
- Allow algebraic rearrangement of floating point calculations

Full details about these flags are available in the [clang user manal](https://releases.llvm.org/{{current_llvm_version}}/tools/clang/docs/UsersManual.html).

These optimisations can be controlled per-line using the [fp control pragmas](https://releases.llvm.org/{{current_llvm_version}}/tools/clang/docs/LanguageExtensions.html#extensions-to-specify-floating-point-flags).

This allows you to either:

- Enable an option with the compiler flag, then use a pragma to
  switch it off for special code regions.
- Opt-in to the optimisation in regions of code where you know it to be safe.

These flags will affect the behaviour of functions
in the SCALE implementation of the CUDA Math API, based on the set of flags
that was enabled at each callsite.

These flags do not affect the accuracy of the Math API, but
do apply assumptions about the range of possible inputs. For example: if
you enable "assume no infinities", all infinity-handling logic will be removed.
Only the inf, nan, and ignore-signed-zero flags will affect
the output of the math functions. Flags like reassoc will not degrade the
accuracy of these routines.

It is OK to compile different source files with different fp
optimisation flags and then link them together.

## Strict aliasing

C++ rules require the compiler to assume that pointers to unrelated types
(eg `float` and `int`) never point to the same place. This can dramatically
improve performance.

Unfortunately, NVIDIA's `nvcc` does not do this,
and some CUDA programs have come to rely on this.

You can explicitly enable this optimisation in SCALE by adding
`-fstrict-aliasing`. This may break your program if it contains TBAA violations.
We recommend you find and fix such violations, since they are undefined
behaviour. This would mean your code is only working correctly because NVIDIA's
compiler doesn't currently exploit this type of optimisation: something which
may change in the future!
