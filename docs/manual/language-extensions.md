# Language Extensions

SCALE has various opt-in language extensions that aim to improve the 
experience of writing GPU code. More language extensions are in development.

Language extensions are currently only available when compiling in `clang` 
mode. Using language extensions in `nvcc` mode would break compatibility 
with NVIDIA's `nvcc` compiler.

Since NVIDIA's compiler does not support SCALE language extensions, if you 
want to retain the ability to compile for NVIDIA GPUs you must do one of two 
things:

- Guard use of language extensions behind the `__REDSCALE__` macro, hiding 
  it from NVIDIA's `nvcc`.
- Use SCALE's `clang` compiler to compile for both NVIDIA and AMD targets. 
  This will require changes to your build system.

## `[[clang::loop_unroll]]`

GPU code frequently contains loops that need to be partially unrolled, and which
have the property that the degree of unrolling is a tradeoff between ILP and
register usage.

Finding the optimal amount to unroll is not usually possible in the compiler
because the number of threads to be used is a runtime value. Programmers
therefore usually want to set unroll depth by hand.

The existing `#pramga unroll N` allows this to be set at the preprocessor level.
The new `[[clang::loop_unroll N]]` allows doing this in a template-dependent
way:

```c++
template<int UnrollAmount>
__device__ void example(int n) {
    [[clang::loop_unroll UnrollAmount]]
    for (int i = 0; i < n; i++) {
        // ...
    }
}
```

## `__builtin_provable(bool X)`

`__builtin_provable(X)` accepts a boolean, `X`, and:

- If the compiler is able to prove, during optimisation, that `X` is a 
  compile-time constant true, the entire expression evaluates to a 
  compile-time constant true.
- Otherwise (if `X` is unknown, or provably false), the entire expression 
  evaluates to a compile-time constant false.

This allows you to write code that opportunistically optimises for a special 
case, without the risk of runtime branching overhead or the inconvenience of 
propagating this information through your entire program using templates. 
For example:

```c++
__device__ int myCleverFunction(int input) {
    if (__builtin_provable(input % 2 == 0)) {
        // Special fast code for the case where `input` is divisible
        // by 2 goes here.
    } else {
        // Slow, general case goes here.
    }
}
```

During optimisation, as calls to `myCleverFunction` get inlined, the 
compiler may be able to prove that `input % 2 == 0` for specific calls to 
this function. Those cases will be compiled with the "fast path", while all 
others will be compiled to the "slow path". The `if` statement will never 
compile to an actual conditional.

Since there are no guarantees that the optimiser is able to prove the 
condition, the program must produce identical outputs from either path, or 
the behaviour is undefined.

This feature differs from the standard c++17 `if constexpr` in that it is 
not required that the input boolean be `constexpr`. `__builtin_provable()` 
communicates with the optimiser, not the template system. Consequently:

- You don't need to use templates to propagate "optimisation knowledge" 
  throughout the program.
- Compilation may be faster, as a result of not having to template everything.
- Some cases may be missed where optimisation fails. Such cases are probably 
  independently worth investigating (_Why_ did optimisation fail? That's a 
  source of additional slowness).

## Improved support for non-32 warpSize

Not all AMD GPUs have a warp size of 32. To mitigate this, we offer a variety
of compiler and API features:

- `cudaLaneMask_t`: A type that is an integer with the number of bits as a CUDA
  warp. This should be used when using functions such as `__ballot()` to avoid
  discarding half the bits.
- Use of `cudaLaneMask_t` in appropriate places in the CUDA APIs (such as 
  the return value of `__ballot()`)
- Diagnostics to catch implicit casts from `cudaLaneMask_t` to narrower types.

In practice, this means the compiler detects the majority of cases where code
is written in a way that will break on a device with a warp size of 64.

Programmers should modify their CUDA code to be agnostic to warp size. 
NVIDIA's documentation recommends this practice, but lots of real-world CUDA 
code does it incorrectly because no current NVIDIA hardware has a warp size 
other than 32.

Since NVIDIA's `nvcc` does not have `cudaLaneMask_t`, programmers should use
`auto` to declare the return types of functions such as `__ballot()` that return
it. This will compile correctly on all platforms.
