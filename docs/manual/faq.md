# Frequently asked questions

## How do I report a problem?

[Contact us](../contact/report-a-bug.md)

Bug reports - no matter how small - accelerate the SCALE project.

Let's work together to democratise the GPGPU market!

## When will `<some GPU>` be supported?

Expanding the set of supported GPUs is an ongoing process. We already
support some GPUs ROCM has dropped support for (eg. gfx900), and are
working to expand the set further. If there's a device you want to bring
to our attention, please get in touch.

## When will `<some CUDA API>` be supported?

We prioritise CUDA APIs based on the number and popularity of third-party
projects requiring the missing API.

If you'd like to bring a missing API to our attention,
[Contact us](../contact/report-a-bug.md)

## Can't NVIDIA just change CUDA and break SCALE?

Of course, we have no control over what NVIDIA does with the CUDA toolkit.

Although it is possible for NVIDIA to change/remove APIs in CUDA or PTX,
doing so would break every CUDA program that uses these functions. Those
programs would then be broken on both SCALE and NVIDIA's platform. It
seems unlikely that NVIDIA would do something that breaks existing programs.

NVIDIA can *add new things* to CUDA which we don't support. Projects are free to
choose whether or not to use any new features that are added in the future,
and may choose to use feature detection macros to conditionalise dependence
on non-essential new features. Projects face a similar choice when deciding
whether or not to use SCALE's steadily growing set of features that go beyond
NVIDIA's CUDA.

## Does SCALE depend on NVIDIA's compiler/assembler/etc.?

No.

Although much of this manual talks about "nvcc", it is important to
understand the distinction between the two things this can refer to:

- The SCALE compiler, which is named "nvcc" for compatibility. This is the
  name build scripts expect, so if we named it anything else then nothing
  would work!
- NVIDIA's proprietary CUDA compiler, `nvcc`.

SCALE provides a _thing called nvcc_, which is in fact absolutely nothing to
do with NVIDIA's `nvcc`. Our "nvcc" is built on top of the open-source
clang/llvm compiler, and has no dependency on NVIDIA's compiler.



SCALE does not make use of "nvvm", either.

## Isn't CUDA inherently optimised for NVIDIA hardware?

Not really.

CUDA is already a cross-platform language/toolkit. NVIDIA have been making
CUDA-capable GPUs for more than 20 years, with many drastic changes in
the hardware during that period. You can nevertheless compile CUDA programs
and run them on NVIDIA cards spanning many years and multiple hardware
architecture redesigns, because CUDA *is a cross-platform toolkit*.

The only problem is that, of course, NVIDIA didn't extend this
cross-platform compatibility to other vendors.

Claiming "CUDA is optimised for NVIDIA GPUs" is a bit like saying
"C++ is optimised for Intel CPUs". If compilers can be taught to
build the same C++ program for Intel/ARM/AMD CPUs with good performance,
why can't we do that for CUDA? It's a different and challenging
compiler research problem, but it is possible.

## Does'

## Does it work on Windows?

Not yet. :(

Windows support is a challenge that we
