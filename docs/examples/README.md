# SCALE Example Programs

These example programs are simple CUDA programs demonstrating the 
capabilities of SCALE.

SCALE is capable of much more, but these small demonstrations serve as a 
proof of concept of CUDA compatibility, as well as a starting point for 
users wishing to get into GPGPU programming.

## List of examples

Here is the list of examples that are currently available:

| Example             | What it is about           |
| ------------------- | -------------------------- |
| [Basic](./basic.md) | Usage in its simplest form |
| [PTX](./ptx.md)     | Using PTX Assembly         |
| [BLAS](./blas.md)   | Using BLAS maths wrapper   |

## Accessing the examples

The examples are hosted in [the public github repository](https://github.com/spectral-compute/scale-docs)
with the rest of this manual.

```sh
git clone https://github.com/spectral-compute/scale-docs.git
cd scale-docs/examples
```

## Using the examples

To build an example:
- [Install SCALE](../manual/how-to-install.md)
- [Decide on a GPU target](../manual/how-to-use.md#identifying-gpu-target)
- [Build the example using cmake](../manual/how-to-use.md#cmake)

The example repository includes a helper script, `example.sh` that can fully 
automate the process. Pass your SCALE target directory as the first argument,
and the example you want to build/run as the second:

```bash
# You should be in the `examples` directory of the `scale-docs` repository
./example.sh /opt/scale gfx1030 basic
```

For the specified example, this will:

1. Remove its build directory if it already exists.
2. Configure CMake for that example in a freshly-created build directory.
3. Build the example in that directory using Make.
4. Set the [`SCALE_EXCEPTIONS=1` environment variable][exceptions] for better 
   error reporting.
4. Run the example.

[exceptions]: ../manual/runtime-extensions.md#scale_exceptions
