# Compiler Diagnostics Reference

This page documents the meaning of various compiler diagnostics provided by 
SCALE, which are not found in other compilers. Diagnostics not listd here 
are provided by clang, and may be found in the [Clang Compiler Diagnostics
Reference](https://clang.llvm.org/docs/DiagnosticsReference.html)

As with most other diagnostics, these are enabled by default and may be switched
off with `-Wno-<name of diagnostic>`, such as `-Wno-ptx-binding-as-address`, or
using clang diagnostic pragmas.

Since many of them represent undefined behaviour even on NVIDIA platforms, 
fixing the underlying problem is recommended.

## `-Wptx-binding-as-address`

Any pointer argument to a PTX `asm()` statement is passed as a generic address.
It is therefore invlaid to directly use an `asm()` input as an address operand
to any PTX instruction that doesn't use generic addressing. Such code will
work correctly any time the generic address space has identical layout to the
target address space of the instruction (as is relatively often the case for
`global`, for example), but will fail randomly on some targets.

To achieve correct behaviour across all GPUs, use the [`cvta`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta)
PTX instruction to convert the incoming pointer to the desired address space
before passing it to the PTX memory instruction. In cases where this conversion
is a no-op, the optimiser will remove this extra step. In other cases, the
usual address-space-inference process performed by the optimiser will attempt
to rewrite the pointer to be defined in the target address space for its entire
lifetime, so all address space casts can be eliminated.

## `-Wptx-unused-local-variable`

Identifies unused local variables (`.reg` declarations) in `asm()` blocks in 
device functions.

## `-Wptx-local-variable-leak`

Identifies PTX variable declarations that may lead to `ptxas` failures when 
compiling for NVIDIA.

When a device function contains a PTX variable declaration, repeated 
inlining of calls to it may lead to duplicate variable declarations in the 
generated PTX.

```c++
__device__ void suffering() {
    asm(".reg .b32 pain;");
}

__global__ void explode() {
    // The function body will inline twice, causing this kernel's final PTX to
    // contain two declarations of the same PTX variable. This produces a 
    // confusing ptxas error. 
    suffering();
    suffering();
}
```

To resolve this, all device functions that make PTX `.reg` declarations should
enclose them in PTX `{}`. This limits the scope of the inlined variable 
declarartion to the inlined functoin body, allowing multiple copies to 
coexist.

This issue can never cause a problem when building for AMD targets.

## `-Wptx-wave64`

Detects hardcoded lanemask operands that have all zeros in the top 32 bits when
compiling for native wave64 mode. Such code is likely a mistake, such as 
hardcoding `0xFFFFFFFF` for a ballot's mask argument instead of the more 
portable `-1`. On a wave64 target, `0xFFFFFFFF` is really 
`0x00000000FFFFFFFF`, turning off half the warp, when the intent was likely 
to turn every thread on.

Note that SCALE's default compilation mode is to emulate a warp size of 32 
on all targets, so you can usually ignore this class of problems initially. 
Most programs don't suffer a measurable performance degredation from this 
emulation process, but certain patterns (such as sending alternating warps 
down different control flow paths) would be pathological. It is desirable to 
migrate your code to be truly warp-size portable.

## Errors relating to the PTX carry bit

PTX offers [extended-precision integr math](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-integer-arithmetic-instructions)
instructions, with implicit carry-out and carry-in. However, the PTX manual
notes:

> The condition code register is not preserved across calls and is mainly
> intended for use in straight-line code sequences for computing
> extended-precision integer addition, subtraction, and multiplication.

It is therefore undefined behaviour to end one device function with a PTX
`asm()` statement that writes the carry-bit, and then try to read that stored
carry-bit using an `asm()` statement at the start of another device function. If
you need to write that kind of code, you can either:

- Use compiler intrinsics to access add-with-carry operations more directly.
- Use int128 types (where possible) to avoid having to have this kind of asm
  entirely.
- Use macros instead of functions for affected regions of code.
- Refactor so both the reader and writer of the carry bit are in the same 
  device function.

When compiling for NVIDIA, such code will _usually_ work if the functions
inline and the asm blocks end up adjacent (so there is no actual function 
call to discard the carry bit). This is optimiser-dependent behaviour, and will
fail if the compiler decides to reorder code or not perform the inlining.

When compiling for AMD with SCALE, we cannot create that behaviour, so it is
simply an error to attempt to return the carry-bit.

