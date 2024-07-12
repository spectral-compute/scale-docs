# API Extensions

SCALE has some few runtime/library features not found in NVIDIA's CUDA Toolkit.

## Environment variables

Some extra features can be enabled by environment variables.

### `SCALE_EXCEPTIONS=1`

Setting `SCALE_EXCEPTIONS=1` will cause all CUDA APIs to throw 
exceptions instead of returning C-style error codes. The exceptions contain 
descriptive error messages that make it clear what the problem is. For 
example: instead of simply returning `cudaErrorInvalidArgument`, you might 
get an exception thrown containing the message `"The second argument is not 
allowed to be nullptr"`.

In cases where CUDA APIs are expected to return a value other than 
`cudaSuccess` during normal operatoin (such as `cudaStreamQuery()`, an 
exception will not be thrown except if an exceptional case arises.

## API Extensions

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

#ifdef __REDSCALE__
if (e != cudaSuccess) {
    const scale::Exception &ex = scale::Exception::last();
    std::cerr << "CUDA error: " << ex.what() << '\n';
}
#endif __REDSCALE__
```

The error accessed by this API is the same one you'd get from using the CUDA
API `cudaGetLastError()`, just more descriptive.
