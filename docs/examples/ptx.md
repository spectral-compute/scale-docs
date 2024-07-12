# PTX example

This example demonstrates SCALE's support for inline PTX. A lot of real-world 
CUDA code uses inline PTX asm blocks, which are inherently NVIDIA-only. There is no 
need to rewrite those when using SCALE: the compiler just digests them and 
outputs AMD machine code.

This example uses C++ templates to access the functionality of the PTX 
`lop3` instruction, used in various ways throughout the kernel.

Build and run the example by following the [general instructions](./README.md).

## Extra info

- [Using inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
- [PTX ISA reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

PTX instructions used:

- [`add`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-add)
- [`lop3`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3)

## Example source code

```cpp
---8<--- "examples/src/ptx/ptx.cu"
```

## `CMakeLists.txt` used

```cmake
---8<--- "examples/src/ptx/CMakeLists.txt"
```
