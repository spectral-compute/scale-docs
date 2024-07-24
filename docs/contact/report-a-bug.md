# Report a Bug

SCALE is still in active development, so you may encounter bugs. If you run 
into problems, contact us by:

- Joining our [Discord](https://discord.gg/KNpgGbTc38)
- Creating [a ticket](https://github.com/spectral-compute/scale-validation/issues)
- Sending us an e-mail at [hello@spectralcompute.co.uk](mailto:hello@spectralcompute.co.uk)

The remainder of this page provides information about how to make your 
report as helpful as possible.

## "No such function: cudaSomethingSomething()"

If your project fails to compile due to a missing CUDA Runtime or Driver API
function, [get in touch][get-in-touch]: this helps us prioritise work by fixing
the holes that have the most demand first.

## "No such function: cuBlas/cuFFt/cuSolverSomethingSomething()"

If your project needs a missing "CUDA-X" API (cuBLAS, cuFFT, cuSOLVER and
friends), this is most likely something you can fix yourself by submitting a
patch to the [open-source library wrapper project](https://github.com/spectral-compute/scale-library-wrappers).
So long as an equivalent function is available in a ROCm library, the wrapper
code is trivial.

## Compiler crash

When the compiler crashes, it creates temporary files containing a reproducer
for the compiler crash, like this:

```
********************

PLEASE ATTACH THE FOLLOWING FILES TO THE BUG REPORT:
Preprocessed source(s) and associated run script(s) are located at:
clang++: note: diagnostic msg: /tmp/a-02f191.cpp
clang++: note: diagnostic msg: /tmp/a-02f191.sh
clang++: note: diagnostic msg:

********************
```

These files will contain the preprocessed version of the source file that broke
the compiler, among other things.
If you are able to share this with us, it will significantly increase the
usefulness of the bug report.

If the source file contains sensitive/proprietary information, this could be
destroyed by reducing the testcase using [cvise][cvise]. Alternatively, a bug
report consisting of just the compiler output is still helpful - especially if
it relates to PTX.

[cvise]: https://github.com/marxin/cvise/

## GPU Crash

If your GPU code crashes with SCALE but not with NVIDIA's compiler, more
useful information can be harvested by enabling some environment variables
that dump extra information. If you are able, sharing the output obtained
from reproducing the crash with one or both of these enabled can be helpful:

- `REDSCALE_CRASH_REPORT_DETAILED=1` will dump extra information from the
  GPU trap handler. This includes register state and some symbol names, so
  it is unlikely to contain any sensitive/proprietary information from your code.
- `REDSCALE_CRASH_DUMP=somefilename` will write the crashing machine code to
  a file. This makes it easier to investigate the problem, but it means that you're
  sharing the compiled version of the crasing GPU kernel with us.

## Something else

It will be helpful if you provide the output of the following commands along
with your report:

```
lspci | grep VGA
redscaleinfo
hsainfo
hsakmtinfo
```

Running your program with the environment variable `SCALE_EXCEPTIONS=1` set might give a more detailed error that would
be helpful to us too.

[get-in-touch]: ../README.md#contact-us
