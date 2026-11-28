# Environment Variables

This page documents environment variables that affect the behaviour
of the SCALE runtime library.

## `SCALE_EXCEPTIONS`

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

## `SCALE_CUDA_VERSION`

Allows selecting a specific version of the CUDA API.

Set to a CUDA version number like "12.9" and SCALE will expose that
version of the CUDA API to your application. This is required only
at compile-time.

This effect can also be achieved by setting the
`__SCALE_CUDA_VER_MAJOR__` and `__SCALE_CUDA_VER_MINOR__`
macros.

## `SCALE_DETECT_LEAKS`

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

## `CUDA_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`

`CUDA_VISIBLE_DEVICES` works as it does on NVIDIA: it selects which devices
the program sees and the order it numbers them in. Its indices count the
devices in `CUDA_DEVICE_ORDER` order.

`ROCR_VISIBLE_DEVICES` is also supported, since some job schedulers set only
that on AMD systems. It both filters and orders: the program sees only the
listed GPUs, and device 0 is the first usable one listed. This order overrides
`CUDA_DEVICE_ORDER`, as `CUDA_VISIBLE_DEVICES` order does on NVIDIA.

- It applies only if `CUDA_VISIBLE_DEVICES` is not set.
- Its indices are ROCm's, counting every GPU in the system, so a list names
  the same GPUs whichever target you use. Listed GPUs SCALE can't use are
  skipped.
- As in ROCm, the list stops at the first entry that isn't a valid index or
  `GPU-` UUID, or that repeats one.
- A list that selects no usable GPU makes the first CUDA call fail with
  `cudaErrorNoDevice`.

## `CUDA_MODULE_LOADING`

Controls how device code is loaded. Possible values:

- `LAZY`: Device code is lazily-loaded and lazily-decompressed on first-use.
          This can dramatically improve performance of large applications.
- `EAGER`: All kernels are linked and loaded onto all enabled devices on
           program startup, or when the module is loaded (as applicable).
- `DEFAULT`: Equivalent to `LAZY`.

## `CUDA_LAUNCH_BLOCKING`

Possible values: `0`, `1`.

Causes all kernel launches to run synchronously. This is effectively the
same as sticking `cudaDeviceSynchronize()` before and after every kernel
launch.

Can be useful for debugging.

## `CUDA_LOG_FILE`

Causes descriptive error messages to be logged to the specified location.

Possible values:

- `stdout`: Logs to stdout.
- `stderr`: Logs to stderr.
- A file path (appends logs to that file).

The messages are the same ones you'd get with `SCALE_EXCEPTIONS`, plus
a log-line prefix to make the format match that produced by NVIDIA's
similar feature.

An example log-line could be:

```
[22:21:32.099][25642][CUDA][E][cuLogsDumpToMemory] buffer cannot be NULL
```
