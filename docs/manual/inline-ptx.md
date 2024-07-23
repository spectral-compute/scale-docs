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
can terminate early, so the `exit` instruction may be used only in converged 
contexts.

### Lane masks

Instructions that return (or accept) lanemasks, such as `activemask.b32`, will 
sneakily return/accept a 64-bit value instead when compiled for a wave64 AMD 
target. If this leads to type errors, it means your code needs to be changed to
work correctly on targets where warpSize is not 32!

## Empty `asm volatile` blocks

To cater to "interesting" user code, the SCALE compiler will not touch
`asm volatile` blocks containing no instructions. We've seen
real-world CUDA code that uses these as a kind of ad-hoc optimisation
barrier to prevent the compiler breaking programs that contain undefined
behaviour. This pragmatic choice should reduce how often broken programs
fail to function, but such code is broken by definition.

## Performance considerations

In most cases, there isn't a performance penalty from using PTX asm in CUDA 
code: it will usually convert to the same IR as the C++ you could have 
written instead, and may actually be faster due to having to be less 
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
| brkpt                            |                           |
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
| pmevent                          |                           |
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
