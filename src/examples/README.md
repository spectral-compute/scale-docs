# About the examples

There are a few simple CUDA programs included in SCALE documentation to demonstrate key features of SCALE.

The examples don't aim to cover all the things available with SCALE.
Instead, they highlight individual features in isolation from each other.
This way, they can be used as a reference in your development process.

Additionally, you are welcome to use these examples as a starting point for your project.

## List of examples

Here is the list of examples that are currently available.
Read more about them in their corresponding pages.

| Example             | What it is about           |
| ------------------- | -------------------------- |
| [Basic](./basic.md) | Usage in its simplest form |
| [PTX](./ptx.md)     | Using PTX Assembly         |
| [BLAS](./blas.md)   | Using BLAS maths wrapper   |

## Accessing the examples

The examples are hosted in a public repository, together with the rest of SCALE documentation.
You can clone it using git:

```sh
git clone https://github.com/spectral-compute/scale-docs.git
cd scale-docs
```

You can also download it as a ZIP file:

```sh
wget -O scale-docs.zip https://github.com/spectral-compute/scale-docs/archive/refs/heads/main.zip
unzip scale-docs.zip
cd scale-docs-main
```

## Using the examples

To build and run the examples, you should have SCALE [installed on your machine](../manual/how-to-install.md).
You should also determine which [path to SCALE](../manual/how-to-use.md#identifying-gpu-target) to use, as it depends on your target GPU.

The example repository includes a helper script, `example.sh`, that configures, builds and runs the example of your choice.

Here is how you can use it for the [Basic](./basic.md) example:

```sh
# You should be in the root directory of the repository when running this
./example.sh {SCALE_DIR} basic
```

For the specified example, this will:

1. Remove its build directory if it already exists
2. Configure CMake for that example in a freshly-created build directory
3. Build the example in that directory using Make
4. Set the `SCALE_EXCEPTIONS=1` environment variable for better error reporting (read more [in the manual][exceptions])
4. Run the example

[exceptions]: ../manual/runtime-extensions.md#scale_exceptions1

---

For accessibilty, SCALE documentation portal includes the source code of the examples in its pages.
This is the source code of `example.sh` referenced above:

```sh
---8<--- "public/examples/example.sh"
```
