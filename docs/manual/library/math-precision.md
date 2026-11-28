# Maths Function Precision

This page documents the accuracy characteristics of the math functions in SCALE's device runtime
library.

The tables below express the error in terms of [ULP](https://en.wikipedia.org/wiki/Unit_in_the_last_place). Roughly, this means "how many floating point numbers away from the
most-correct representable value can it be?"

Bold values indicate that the function was tested for *all* inputs (every possible bit pattern).
Other values were calculated by testing billions of arguments per function, chosen to
include the values maths functions often get wrong and the ranges where results are hardest to
get right (which varies somewhat based on the properties of each function).

Where a function's result is infinite or zero, its sign matches NVIDIA's implementation.
Where it is a NaN, its sign and payload are unspecified: you'll get *a* NaN, but not
necessarily the same one NVIDIA produce. This is because there would be a meaningful
performance cost to matching that, and the correct way to compare NaNs (`isnan()`)
doesn't care.

The accuracy bounds cover all supported GPU architectures.

Functions not listed here have an accuracy of 0ULP for all inputs. Empty
cells in the table indicate that the CUDA API does not provide that function
for that data type.

NVIDIA documents its own bounds in the
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).

<!-- math-precision-tables -->
