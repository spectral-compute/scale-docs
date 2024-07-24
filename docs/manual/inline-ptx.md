# Inline PTX support

SCALE accepts inline PTX `asm` blocks in CUDA programs and will attempt to 
compile it for AMD along with the rest of your program.

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
instruction may be used only in converged contexts.

## Empty `asm volatile` blocks

To cater to "interesting" user code, the SCALE compiler will not touch
`asm volatile` blocks containing no instructions. We've seen
real-world CUDA code that uses these as a kind of ad-hoc optimisation
barrier to prevent the compiler breaking programs that contain undefined
behaviour. This pragmatic choice should reduce how often such broken programs
fail to function, but such code is broken by definition.

## `asm` input/output types

`nvcc` doesn't appear to consistently follow its own tying rules for PTX asm 
inputs/outputs. It the following invalid things to occur in some cases (and real
programs depend on this):

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
  inconsistently applied even in nvidia mode, they are probably best achieved
  with an explicit cast. In any case, we have thus far never seen such code 
  in the wild!

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
In such cases, it is worth experimentally determining which is actually 
superior: the result can vary on a case-by-case basis.

## Supported constraints

The following PTX constraint letters are supported. See above commentary on 
nuances regarding how they are interpreted.

`h`: u16
`r`: u32
`l`: u64
`f`: f32
`d`: f64
`n`: constants

The `"C"` constraint type is in development, and seems likely to prove 
useful for authors wishing to generalise their PTX code for wave sizes other 
than 32.

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
| fma                              |                           |
| griddepcontrol.launch_dependents | Currently a no-op         |
| griddepcontrol.wait              | Currently a no-op         |
| isspacep                         |                           |
| ld                               |                           |
| ld.nc                            |                           |
| ldu                              |                           |
| lop3                             |                           |
| mad                              |                           |
| mad24                            |                           |
| madc                             |                           |
| match.all                        |                           |
| match.any                        |                           |
| max                              |                           |
| max.xorsign.abs                  |                           |
| min                              |                           |
| min.xorsign.abs                  |                           |
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
| xor                              |                           |
