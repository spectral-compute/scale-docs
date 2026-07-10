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

### `SCALE_DETECT_LEAKS`

The SCALE runtime library has a limited ability to detect resource leaks.

To enable this feature, run your program with the `SCALE_DETECT_LEAKS=1`
environment variable set. Upon exit, the program will print output similar
to:

```
Free nodes: 7 / 7 [capacity: 4096]. 0 leaked.
Free events: 21 / 21 [capacity: 4096]. 0 leaked.
```

If any resources are found to have leaked, the program will abort and print
a count of leaked resources.

Leaked resources indicate one of three things:

- Your program forgot to delete some CUDA resource. For example: a
  `cudaGraphExec_t` was not passed to `cudaGraphExecDestroy()` and leaked.
- Some of your program's CUDA resources are destroyed during static
  deinitialisation and - due to to chance - the SCALE library's static
  destructors ran before yours. This scenario may lead to undefined behaviour
  when running with either SCALE or NVIDIA CUDA, since it may end up deleting
  a resource after the CUDA library has been de-initialised.
- A bug in SCALE causing it to leak resources internally.

This leak detector is an emergent property of an implementation detail of the
SCALE runtime library for AMD GPUs. Certain resources are very expensive to
create/destroy, so we use object pooling internally to improve performance.
The leak detector simply checks that every object created for the pool has
made its way back to the pool by the time the SCALE library is being
unloaded.

The resources being counted are the nodes/events used internally to represent
work in the GPU work queues. Most APIs that "do GPU stuff" result in the
creation/use of one or more of these resources. The leak checker does not provide
any way to map a particular leaked object back to the API call that created it.

The leak detector does not detect memory leaks, file descriptor leaks, or any other
kind of leak. It finds only leaks of resouces that the SCALE runtime handles using
object pools.

## API Extensions

Some of SCALE's API extensions require the `scale.h` header to be included.

### Programmatic Exception Enablement

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
