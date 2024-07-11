# BLAS example

This example demonstrates SCALE's compatibility with cuBLAS APIs by using 
cuBLAS to perform a double-precision dot-product on an AMD GPU.

cuBLAS APIs are forwarded to use the relevant ROCm APIs.
Note that the example links to `cublas` in its [`CMakeLists.txt`](#cmakeliststxt-used).

## Example source code

```cpp
---8<--- "examples/src/blas/blas.cu"
```

## `CMakeLists.txt` used

```cmake
---8<--- "examples/src/blas/CMakeLists.txt"
```
