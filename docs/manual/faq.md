# Frequently asked questions

## How do I report a problem?

Strange compiler errors? Performance not as great as expected? Something else
not working as expected?

[Contact us](https://docs.scale-lang.com/contact/report-a-bug/)

Bug reports - no matter how small - accelerate the SCALE project.

Let's work together to democratise the GPGPU market!

## When will <some GPU> be supported?

Expanding the set of supported GPUs is an ongoing process. At present we're 
being very conservative with the set of GPUs enabled with SCALE to avoid 
use on platforms we currently have zero ability to test on.

If your GPU is supported by ROCm, it'll probably become available on SCALE a 
little sooner than if it is not, since it won't break our "CUDA-X" library 
delegation mechanism.

## When will <some CUDA API> be supported?

We prioritise CUDA APIs based on the number and popularity of third-party 
projects requiring the missing API.

If you'd like to bring a missing API to our attention,
[Contact us](https://docs.scale-lang.com/contact/report-a-bug/)

## Can't NVIDIA just change CUDA and break SCALE?

Of course, we have no control over what NVIDIA does with the CUDA toolkit.

Although it is possible for NVIDIA to change/remove APIs in CUDA or PTX, 
doing so would break every CUDA program that uses these functions. Those
programs would then be broken on both SCALE and NVIDIA's platform.

NVIDIA can *add new things* to CUDA which we don't support. Projects are free to 
choose whether or not to use any new features that are added in the future, 
and may choose to use feature detection macros to conditionalise dependence 
on non-essential new features. Projects face a similar choice when deciding 
whether or not to use SCALE's steadily growing set of features that go beyond
NVIDIA's CUDA.

### Does SCALE depend on NVIDIA's compiler/assembler/etc.?

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
