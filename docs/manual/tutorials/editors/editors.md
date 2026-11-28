# Overview

SCALE ships a build of [`clangd`](https://clangd.llvm.org/), the language server used by most
modern editors for C and C++, that understands CUDA as SCALE does. Pointing your editor at it
gives you, for both host and device code:

- Diagnostics from the SCALE compiler as you type, including SCALE's checks of
  [inline PTX](../../compiler/inline-ptx.md) and its other
  [CUDA-specific diagnostics](../../compiler/diagnostics.md).
- Correct handling of the nvcc [dialect](../../compiler/dialects.md), where the clangd from your
  distribution would reject code that nvcc accepts.
- Code completion, go-to-definition, find-references, hover, rename, inlay hints and semantic
  highlighting that see through the CUDA headers, Thrust, CUB and the rest of CCCL.

You don't need to build your project with SCALE to use this: if you normally build with NVIDIA's
`nvcc`, you only need to configure your project with SCALE once to produce a compilation database.

Setting it up takes three steps, the first of which is editor-specific:

1. [Tell your editor to use SCALE's `clangd`](#1-use-scales-clangd).
2. [Give `clangd` a compilation database](#2-generate-a-compilation-database) for your project.
3. [Add a `.clangd` file](#3-add-a-clangd-file) to your project.

## 1. Use SCALE's `clangd`

SCALE's `clangd` is `<SCALE_DIR>/llvm/bin/clangd`: `/opt/scale/llvm/bin/clangd` for a package
install. Instructions for specific editors:

- [Visual Studio Code](./vscode.md) (and VSCodium, Cursor, etc.)
- [Zed](./zed.md)
- [Neovim](./neovim.md)
- [Emacs](./emacs.md)
- [Helix](./helix.md)
- [Sublime Text](./sublime-text.md)
- [Kate](./kate.md)

Any other editor with LSP support works too: set its `clangd` command to SCALE's, and make sure it
starts `clangd` for `.cu` and `.cuh` files.

!!! warning

    Use the `clangd` from the same SCALE installation as your compiler. Another `clangd` (from your
    distribution, or one your editor downloaded) parses CUDA as upstream clang does, and reports
    spurious errors on code that SCALE and nvcc accept.

## 2. Generate a compilation database

`clangd` needs to know how each file is compiled: include paths, macros, the target GPU, and
whether it's CUDA at all. It reads this from a `compile_commands.json` file, which it looks for in
each parent directory of the source file, and in a `build` subdirectory of each of them.

The compile commands decide which CUDA dialect `clangd` uses. Commands that call a compiler named
`nvcc` (SCALE's or NVIDIA's) get the nvcc dialect. Commands that call `clang++` get clang's
dialect.

=== "CMake"

    Configure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, using SCALE as the CUDA compiler. Either
    activate a [`scaleenv`](../how-to-use.md#the-easy-way-scaleenv) first:

    ```bash
    source /opt/scale/bin/scaleenv gfx1100
    cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    ```

    or name the compiler explicitly:

    ```bash
    cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_CUDA_COMPILER=/opt/scale/targets/gfx1100/bin/nvcc
    ```

    If your build directory is called `build`, `clangd` finds the database by itself. To
    configure from inside your editor (for example with VS Code's CMake Tools), or to keep the
    setting permanently, put it in a
    [`CMakePresets.json`](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html):

    ```json
    {
      "version": 6,
      "configurePresets": [
        {
          "name": "scale",
          "binaryDir": "${sourceDir}/build",
          "cacheVariables": {
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "CMAKE_CUDA_COMPILER": "/opt/scale/targets/gfx1100/bin/nvcc"
          }
        }
      ]
    }
    ```

=== "Make and others"

    Use [Bear](https://github.com/rizsotto/Bear) to record the compiler invocations of a full
    build:

    ```bash
    source /opt/scale/bin/scaleenv gfx1100
    make clean
    bear -- make
    ```

=== "Bazel"

    Use [Hedron's compile commands extractor](https://github.com/hedronvision/bazel-compile-commands-extractor)
    to write `compile_commands.json` to the root of your workspace.

If the database is somewhere `clangd` won't look, either symlink it into the project root, or
name its directory in your [`.clangd` file](#3-add-a-clangd-file):

```yaml
CompileFlags:
  CompilationDatabase: out/release
```

## 3. Add a `.clangd` file

Put a file called `.clangd` in the root of your project, containing:

```yaml
CompileFlags:
  Add: [-fno-gpu-defer-diag]
```

`clangd` analyses the host side of each CUDA file. Normally, errors in device code are deferred
until the device compilation, so `clangd` never reports them: mistakes in `__device__` functions
and inline PTX go unnoticed until you build. `-fno-gpu-defer-diag` reports them immediately. It's
harmless for non-CUDA files.

Optionally, you can also define `__CUDA_ARCH__` for CUDA files:

```yaml
CompileFlags:
  Add: [-fno-gpu-defer-diag]
---
If:
  PathMatch: .*\.cuh?
CompileFlags:
  Add: [-D__CUDA_ARCH__=890]
```

This makes `clangd` treat `#ifdef __CUDA_ARCH__` regions as active, so you get completion and
navigation in device-only code paths. The cost is that `#ifndef __CUDA_ARCH__` (host-only) regions
become inactive instead. Use the value that corresponds to the architecture you build for (`890`
for `sm_89`).

If `clangd` picks the wrong CUDA dialect (for example, because your compilation database invokes
`clang++` but your code is written for nvcc), add `-fcuda-nvcc-emulation` to force the nvcc
dialect.

See the [`clangd` configuration reference](https://clangd.llvm.org/config) for everything else
`.clangd` can do. Its settings can also be made global for all projects, in
`~/.config/clangd/config.yaml`.

## Check it's working

Create a `.cu` file in your project containing:

```cuda
__global__ void kernel() {
    asm("add.f32 foo, 1, 1;");
}
```

Your editor should report `unresolved identifier: foo` on the `asm` statement. If it doesn't:

- No diagnostic at all: the `-fno-gpu-defer-diag` flag isn't being applied. Check that `.clangd`
  is in a parent directory of the file.
- Errors about CUDA headers, or `__global__` not being recognised: `clangd` hasn't found a
  compile command for this file. New files have no entry in the database until you re-run
  CMake (or Bear).
- Still wrong: run `clangd --check` on the file from a terminal. It prints the compile command it
  used and every diagnostic, which usually makes the problem clear:

    ```bash
    /opt/scale/llvm/bin/clangd --check=path/to/file.cu
    ```

Most editors also have a way to show `clangd`'s log, which contains the same information. If a
change to `.clangd` or the compilation database doesn't seem to take effect, restart `clangd`.
