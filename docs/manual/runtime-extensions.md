# API Extensions

SCALE has some runtime/library features not found in NVIDIA's CUDA Toolkit.

## Environment variables

Some extra features can be enabled by environment variables.

### `SCALE_EXCEPTIONS`

Errors from the CUDA API can be hard to debug, since they simply return an
error code that the host program has to do something with.

SCALE provides an environment variable to make any error from the CUDA API
produce a observable result.

Setting `SCALE_EXCEPTIONS=1` will cause all CUDA APIs to throw descriptive 
exceptions instead of returning C-style error codes.

Setting `SCALE_EXCEPTIONS=2` will print the error messages to stderr, but not
throw them. This is helpful for programs that deliberately create CUDA errors
as part of their processing.

In cases where CUDA APIs are expected to return a value other than 
`cudaSuccess` during normal operation (such as `cudaStreamQuery()`, an 
exception will not be thrown except if an exceptional case arises.

## API Extensions

Some of SCALE's API extensions require the `scale.h` header to be included. 

### Programmatic Exception Enablement

SCALE exceptions (see documentation of `SCALE_EXCEPTIONS` environment 
variable above) may also be enabled/disabled programmatically using:

```c++
scale::Exception::enable(); // To enable.
scale::Exception::enable(false); // To disable.
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
