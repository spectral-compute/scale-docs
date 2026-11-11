# API Extensions

This page documents APIs found in SCALE, but not in NVIDIA CUDA.

Some of SCALE's API extensions require the `scale.h` header to be included.

## Programmatic Exception Enablement

SCALE's exception mode may also be controlled programmatically:

```c++
scale::Exception::setMode(scale::ExceptionMode::THROW); // Throw exceptions
scale::Exception::setMode(scale::ExceptionMode::PRINT); // Print errors
scale::Exception::setMode(scale::ExceptionMode::OFF); // Match CUDA behaviour
```

Even when exceptions are disabled, you can access a `scale::Exception` object
containing the descriptive error message from the most recent failure using
`scale::Exception::last()`:

```c++
cudaError_t e = cudaSomething();
if (e != cudaSuccess) {
    const scale::Exception &ex = scale::Exception::last();
    std::cerr << "CUDA error: " << ex.what() << '\n';
}
```

The error accessed by this API is the same one you'd get from using the CUDA
API `cudaGetLastError()`, just more descriptive.

## Graphics interop

OpenGL interop requires `AMD_DEBUG=noexporteddcc` to be set in the GL process environment, so Mesa decompresses DCC on DMABUF export.
Export it yourself if not using `scaleenv`.
