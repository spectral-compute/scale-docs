# Overview

The SCALE runtime library is an implementation of the CUDA Math, Driver,
and Runtime APIs. This is necessary when compiling CUDA programs for
non-NVIDIA targets, where those libraries are not present.

Currently, the SCALE runtime only supports AMD.

## Comparison to HIP

The SCALE runtime library differs from the HIP runtime library in that it
aims to precisely replicate the behaviour of the corresponding CUDA APIs.

Many HIP APIs do subtly different things compared to their similarly-named
CUDA counterparts, creating a compatibility obstacle. The goals of the
two libraries are therefore slightly different: HIP appears to aim to provide
a "more sensible" API, whereas SCALE aims to provide a drop-in replacement
that makes CUDA programs work out of the box.
