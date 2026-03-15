# VSCode Tutorial

SCALE's compiler can be connected to VSCode, giving you access to our
high-quality CUDA diagnostics directly in your editor.

Since the SCALE compiler understands the same command line arguments and
language dialect as NVIDIA `nvcc`, it can plug directly into your existing
nvcc project.

See the [general clangd information](./editors.md) for more context.

The [clangd documentation](https://clangd.llvm.org/installation#project-setup) may
also prove useful.

## One-time setup

- Install the `clangd` extension for vscode
- Open settings
- Search for "@ext:llvm-vs-code-extensions.vscode-clangd path"
- Replace "clangd" with the path to the SCALE clangd, eg. `/opt/scale/llvm/bin/clangd`.

=== "CMake"

    - Ensure you have the "CMake Tools" plugin installed.
    - Open the command pallette (CTRL+SHIFT+P)
    - Select "edit user-local cmake kits"
    - Add the SCALE compilers as kits, adjusting the installation directory appropriately:

    ```json
      {
        "name": "SCALE-nvc x86_64-pc-linux-gnu",
        "compilers": {
          "C": "/opt/scale/llvm/bin/clang",
          "CXX": "/opt/scale/llvm/bin/clang++",
          "CUDA": "/opt/scale/llvm/bin/clang-nvcc"
        },
        "isTrusted": true
      },
      {
        "name": "SCALE-clang x86_64-pc-linux-gnu",
        "compilers": {
          "C": "/opt/scale/llvm/bin/clang",
          "CXX": "/opt/scale/llvm/bin/clang++",
          "CUDA": "/opt/scale/llvm/bin/clang++"
        },
        "isTrusted": true
      },

    ```

=== "Other"

    Perform any steps required to add the SCALE compiler paths to the buildsystem you
    use with vscode. The editor integration doesn't need you to actually compile with
    these compilers, but it needs to be able to configure and emit a compilation
    database.


## Project-specific setup

These steps need to be taken for each project you work with, though of course
the repetition may be eliminated using a template.

- Add a `.clangd` file to the root of your project with content similar to this:

```
---
CompileFlags:
  Add: [-fno-gpu-defer-diag, -D__CUDA_ARCH__=1200]
---
```

Optionally, you may use the `-fno-gpu-defer-diag` flag during compilation in nvcc mode to see compiler errors that nvcc would normally ignore.

=== "CMake"

    - Find the `copycompilecommands` setting and set it to `${workspaceFolder}/compile_commands.json`. It could also be manually symlinked, but we find this option maximally convenient.
    - Find the "cmake configure args" settings and add these flags to it:
        `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
        `-DCMAKE_CUDA_ARCHITECTURES=90`  (Adjust as appropriate)

    !!! warning

        Not to be confused with the "CMake Build Options" option!

    - Right click your `CMakeLists.txt` and hit "clean reconfigure all projects". When prompted, select one of the cmake kits created earlier.
      * If you normally compile with nvcc, pick the `SCALE-nvcc` one. Otherwise, pick the `SCALE-clang` one.

    - Restart vscode

=== "Bazel"

    Use the [extractor extention](https://github.com/hedronvision/bazel-compile-commands-extractor) to have Bazel generate a configuration database for use by clangd, and ensure it's linked into the root of your workspace directory.

    - Restart vscode when you're done.

=== "Other"

    Refer to [clangd doucumentation](https://clangd.llvm.org/installation#project-setup)
    for now to obtain a compilation database for `clangd`.


    - Restart vscode when you're done.


### Check it's working

As a sanity check, create a new `.cu` file with the following contents:

```cu
__global__ void kernel() {
  asm("add.f32 foo, 1, 1;");
}
```

If everything is working correctly, you should get an red underline on "foo" telling you about the access of a nonexistent variable.
