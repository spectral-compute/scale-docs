# Basic example

This is simple vector-sum kernel using CUDA. 

The example:

- Generates test data on the host
- Sends data to the device
- Launches a kernel on the device
- Receives data back from the device
- Checks that the data is correct.

Build and run the example by following the [general instructions](./README.md).

## Example source code

```cpp
---8<--- "examples/src/basic/basic.cu"
```

## `CMakeLists.txt` used

```cmake
---8<--- "examples/src/basic/CMakeLists.txt"
```
