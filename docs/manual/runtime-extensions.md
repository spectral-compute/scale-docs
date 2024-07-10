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
