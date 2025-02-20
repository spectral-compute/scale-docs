# Inline PTX support

SCALE accepts inline PTX `asm` blocks in CUDA programs and will attempt to 
compile it for AMD along with the rest of your program.

## Wave64 considerations

A small number of PTX instructions depend on the warp size of the GPU being 
used. Since all NVIDIA GPUs and many AMD ones have a warp size of 32, much 
code implicitly relies on this. As a result, issues can appear when 
targeting wave64 devices.

SCALE provides several tools and compiler warnings to help you write 
portable PTX code. In most cases only small tweaks are required to get things
working. Since so little PTX actually depends on the warp size, most 
projects are unaffected by the issues documented in this section. 
Nevertheless, it is useful to adjust your code to be warp-size-agnostic, 
since doing so can be done with no downsides.

### Querying warp size

PTX defines the `WARP_SZ` global constant which can be used to access the
warp size directly. It's a compile-time constant in nvidia's implementation 
as well as in SCALE, so there is no cost to using this and doing arithmetic 
with it (like with `warpSize` in CUDA code).

### Lanemask inputs

The length of lanemask operands on instructions will always have a number of 
bits equal to the warp size on the target GPU. For 
example, when compiling for a wave64 GPU, the lanemask argument to `shfl.sync`
is a `b64`, not `b32`.

The following rules are applied to help detect problems with such operands:

- If a non-constant lanemask operand is used, and its bit-length is <= the 
  warp size, an error is raised.
- If a constant lanemask operand is used with no 1-bits in the high 32 bits, 
  while compiling for a wave64 architecture, a warning is raised (which can 
  be disabled). This catches the common case of hardcoded lanemasks like 
  `0xFFFFFFFF` which will typecheck as `b64`, but will probably not do what 
  you want.

In the common case where you want an all-ones lanemask, the most convenient 
thing to do is write `-1` instead of `0xFFFFFFFF`: this will give you the 
correct number of 1-bits in all cases, including on nvidia platforms.

### The `c` argument to `shfl` instructions

The `shfl` PTX instruction has a funky operand, `c`, used for clamping etc.
See [the documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync).

The `c` operand is really two operands packed together: `cval` in
bits 0-4, and `segmask` in bits 8-12. For wave64, an extra bit is needed. Since
there is space for an extra bit in each of these values, we simply add it in
the obvious place.

A portable way of reasoning about this is to assume that `cval` is in bits 0-7
and `segmask` in bits 8-15.

Here's a concrete example of a reverse cumsum that works on either warp size:

```c++
__global__ void shuffleRevCumsumKernel(float *dst)
{
    float out;
    const int C = warpSize - 1;
    asm(
    ".reg .f32 Rx;"
    ".reg .f32 Ry;"
    ".reg .pred p;"
    "mov.b32 Rx, %1;"
    "shfl.sync.down.b32  Ry|p, Rx, 0x1,  %2, -1;"
    "@p  add.f32        Rx, Ry, Rx;"
    "shfl.sync.down.b32  Ry|p, Rx, 0x2,  %2, -1;"
    "@p  add.f32        Rx, Ry, Rx;"
    "shfl.sync.down.b32  Ry|p, Rx, 0x4,  %2, -1;"
    "@p  add.f32        Rx, Ry, Rx;"
    "shfl.sync.down.b32  Ry|p, Rx, 0x8,  %2, -1;"
    "@p  add.f32        Rx, Ry, Rx;"
    "shfl.sync.down.b32  Ry|p, Rx, 0x10, %2, -1;"
    "@p  add.f32        Rx, Ry, Rx;"
    
    // One extra shuffle is needed for the larger warp size.
    #if __SCALE_WARP_SIZE__ > 32
    "shfl.sync.down.b32  Ry|p, Rx, 0x20, %2, -1;"
    "@p  add.f32        Rx, Ry, Rx;"
    #endif // __SCALE_WARP_SIZE__
    "mov.b32 %0, Rx;"
    : "=f"(out) : "f"(1.0f), "n"(C)
    );

    dst[threadIdx.x] = out;
}
```

And here's how to do a portable butterfly shuffle reduction:

```c++
__global__ void shuffleBflyKernel(float *dst)
{
    const int C = warpSize - 1;

    float out;
    asm(
    ".reg .f32 Rx;"
    ".reg .f32 Ry;"
    ".reg .pred p;"
    "mov.b32 Rx, %1;"
    #if __SCALE_WARP_SIZE__ > 32
    "shfl.sync.bfly.b32  Ry, Rx, 0x20, %2, -1;"
    "add.f32        Rx, Ry, Rx;"
    #endif // __SCALE_WARP_SIZE__
    "shfl.sync.bfly.b32  Ry, Rx, 0x10, %2, -1;"
    "add.f32        Rx, Ry, Rx;"
    "shfl.sync.bfly.b32  Ry, Rx, 0x8,  %2, -1;"
    "add.f32        Rx, Ry, Rx;"
    "shfl.sync.bfly.b32  Ry, Rx, 0x4,  %2, -1;"
    "add.f32        Rx, Ry, Rx;"
    "shfl.sync.bfly.b32  Ry, Rx, 0x2,  %2, -1;"
    "add.f32        Rx, Ry, Rx;"
    "shfl.sync.bfly.b32  Ry, Rx, 0x1,  %2, -1;"
    "add.f32        Rx, Ry, Rx;"
    "mov.b32 %0, Rx;"
    : "=f"(out) : "f"((float) threadIdx.x), "n"(C)
    );

    dst[threadIdx.x] = out;
}
```

## Dialect differences

The SCALE compiler accepts a more permissive dialect of PTX than NVIDIA's 
implementation does. 

### Integer lengths

Most PTX instructions are defined to work only for a specific, arbitrary set 
of integer types. We didn't bother to implement such restrictions except in 
cases where they are needed for correctness, so many PTX instructions accept 
a wider selection of types than nvcc accepts.

One amusing consequence of this is that most of the simple instructions work 
for *any* bit-length: `add.s17` is allowed (but will of course lead to 
extra sext/trunc instructions, so isn't necessarily a good idea).

### Divergent `exit`

AMD hardware does not seem to have a mechanism by which individual threads 
can terminate early (only entire warps). As a result, the `exit` 
instruction may be used only in converged contexts. We transform it into
approximately:

```c++
if (__activemask() == -1) {
    exit_entire_warp();
} else {
    // This situation is unrepresentable
    __trap();
}
```

Code that uses `exit` as a performance optimisation for nvidia hardware may 
benefit from being adjusted for AMD.

## Empty `asm volatile` blocks

To cater to "interesting" user code, the SCALE compiler will not touch
`asm volatile` blocks containing no instructions. We've seen
real-world CUDA code that uses these as a kind of ad-hoc optimisation
barrier to prevent the compiler breaking programs that contain undefined
behaviour. This pragmatic choice should reduce how often such broken programs
fail to function, but such code is broken by definition.

Note that the `volatile` on non-empty `volatile asm` blocks has no effect on the
behaviour of the SCALE compiler. `volatile` asm is a conservative feature that
allows the compiler to model "unknowable" implicit dependencies of the actions
takenby the inline asm. Since we're compiling the asm to IR, the *actual*
dependencies and properties of everything it does are known and modelled. This
can improve optimisation, but may break programs that have undefined behaviour
that was being hidden by the optimisation barrier effect of the volatile asm 
block.

## `asm` input/output types

`nvcc` doesn't appear to consistently follow its own tying rules for PTX asm 
inputs/outputs. It allows the following invalid things to occur in some cases 
(and real programs depend on this):

- Writes to read-only asm bindings are permitted (such as writing to an "r") 
  constraint. The result of the write is not visible to the caller: it's 
  effectively a temporary inside the scope of the asm block.
- `=r` (write-only) constraints can be used in a read-write fashion (as if 
  they were `+r`).
- Values passed to the asm block are sometimes, but not always, type checked,
  implicitly widened, or implicitly truncated.

To avoid having to characterise and define the perimeter of this buggy 
behaviour, SCALE's implementation defines the following consistent rules 
which are intended to maximise compatibility (and minimise "weirdness"):

- All read-only inputs may be written to. The results of these writes are 
  visible only within the scope of the asm block (as if they were local 
  variables being passed by value into a function).
- All write-only outputs are implicitly read-write. ie.: there is no 
  difference between `+r` and `=r`.
- The type of an input or output binding is governed by the type of the 
  expression, not the constraint letter. Once "in PTX", the usual PTX rules 
  about implicit truncation/widening/etc. apply. This nuance won't change 
  the behaviour of programs unless they rely on using a "too short" PTX 
  constraint type to truncate a value, and then implicitly widen it within 
  PTX (hence zeroing out some of the bits). Since such truncations are 
  inconsistently applied even with nvidia nvcc mode, they are probably best 
  achieved with an explicit cast.

## Performance considerations

In most cases, there isn't a performance penalty from using PTX asm in CUDA 
code: it will usually convert to the same IR as the C++ you could have 
written instead, and may actually be faster due to not needing to be as
conservative about optimisation compared to the usual rules of asm blocks.

Since the compiler _effectively_ converts it to the CUDA code you could have 
written to achieve the same effect without the use of the PTX asm, it 
doesn't come with the optimisation-hindering downsides asm blocks 
normally imply. The compiler will respect the ordering/synchronisation/etc. 
requirements of each operation individually, rather than having to regard an 
entire `asm volatile` block as an opaque, immutable unit.

Programs that have already added support for HIP might have multiple 
codepaths: one for CUDA that uses inline PTX, and one for AMD which doesn't. 
In such cases, it is worth testing both to see which is fastest.

## Supported constraints

The following PTX constraint letters are supported. See above commentary on 
nuances regarding how they are interpreted.

`h`: u16
`r`: u32
`l`: u64
`f`: f32
`d`: f64
`n`: constants
`C`: dynamic asm strings


## Supported instructions

The following instructions are currently supported.

Caveat: since the `bf16`, `fp8` and `tf32` floating point formats are not 
currently supported in SCALE, they are also not supported here.


| Instruction                      | Notes                     |
|----------------------------------|---------------------------|
| abs                              |                           |
| activemask                       |                           |
| add                              |                           |
| addc                             |                           |
| and                              |                           |
| atom                             |                           |
| bfe                              |                           |
| bfi                              |                           |
| bfind                            |                           |
| bfind.shiftamt                   |                           |
| bmsk                             |                           |
| bra                              |                           |
| brev                             |                           |
| brkpt                            | Currently a no-op         |
| clz                              |                           |
| cnot                             |                           |
| copysign                         |                           |
| cvt                              |                           |
| cvt.pack                         |                           |
| discard                          | Currently a no-op         |
| div                              |                           |
| dp2a                             |                           |
| dp4a                             |                           |
| elect                            |                           |
| exit                             | Only from convergent code |
| fence                            | Memory ranges unsupported |
| fma                              |                           |
| fns                              |                           |
| griddepcontrol.launch_dependents | Currently a no-op         |
| griddepcontrol.wait              | Currently a no-op         |
| isspacep                         |                           |
| ld                               |                           |
| ld.nc                            |                           |
| ldmatrix                         |                           |
| ldu                              |                           |
| lop3                             |                           |
| mad                              |                           |
| mad24                            |                           |
| madc                             |                           |
| match.all                        |                           |
| match.any                        |                           |
| max                              |                           |
| max.xorsign.abs                  |                           |
| membar                           |                           |
| min                              |                           |
| min.xorsign.abs                  |                           |
| mma                              | `wmma.mma` likely faster  |
| mov                              |                           |
| mul                              |                           |
| mul24                            |                           |
| nanosleep                        |                           |
| neg                              |                           |
| not                              |                           |
| or                               |                           |
| pmevent                          | Currently a no-op         |
| popc                             |                           |
| prefetch                         |                           |
| prefetchu                        |                           |
| prmt                             |                           |
| prmt.b4e                         |                           |
| prmt.ecl                         |                           |
| prmt.ecr                         |                           |
| prmt.f4e                         |                           |
| prmt.rc16                        |                           |
| prmt.rc8                         |                           |
| rcp                              |                           |
| red                              |                           |
| redux                            |                           |
| rem                              |                           |
| sad                              |                           |
| selp                             |                           |
| set                              |                           |
| setp                             |                           |
| shf.l                            |                           |
| shfl.bfly                        |                           |
| shfl.down                        |                           |
| shfl.idx                         |                           |
| shfl.up                          |                           |
| shf.r                            |                           |
| shl                              |                           |
| shr                              |                           |
| slct                             |                           |
| st                               |                           |
| stmatrix                         |                           |
| sub                              |                           |
| subc                             |                           |
| szext                            |                           |
| testp.finite                     |                           |
| testp.infinite                   |                           |
| testp.normal                     |                           |
| testp.notanumber                 |                           |
| testp.number                     |                           |
| testp.subnormal                  |                           |
| trap                             |                           |
| vabsdiff                         |                           |
| vadd                             |                           |
| vmax                             |                           |
| vmin                             |                           |
| vote.all                         |                           |
| vote.any                         |                           |
| vote.ballot                      |                           |
| vote.uni                         |                           |
| vshl                             |                           |
| vshr                             |                           |
| vsub                             |                           |
| wmma.load                        |                           |
| wmma.store                       |                           |
| wmma.mma                         |                           |
| xor                              |                           |
