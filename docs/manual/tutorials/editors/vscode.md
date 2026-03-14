# VSCode Integration

SCALE's compiler can be connected to VSCode, giving you access to our
high-quality CUDA diagnostics directly in your editor.

Since the SCALE compiler understands the same command line arguments and
language dialect as NVIDIA `nvcc`, it can plug directly into your existing
nvcc project.

In a nutshell:
- Follow instructions to set up clangd for your editor: https://clangd.llvm.org/installation
- Reconfigure it to use the clangd shipped with SCALE instead of the usual one.

## clangd extension setup

- Install the `clangd` extension for vscode
- Open settings
- Search for "@ext:llvm-vs-code-extensions.vscode-clangd path"
- Replace "clangd" with the path to the SCALE clangd, eg. `/opt/scale/llvm/bin/clangd`.
- Open the command pallette (CTRL+SHIFT+P)
- Select "edit user-local cmake kits"
- Add the SCALE compilers as kits, adjusting the installation directory appropriately:

```
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

## Project setup

- Set `copycompilecommands` to `${workspaceFolder}/compile_commands.json`
- Hit "clean reconfigure all projects", and select one of the cmake kits created earlier.
   * If you normally compile with nvcc, pick the `SCALE-nvcc` one. Otherwise, pick the `SCALE-clang` one.
- Add `-DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` to cmake settings (adjust arch as necessary).
- Add a `.clangd` file to the root of your project with content similar to this:

```
---
CompileFlags:
  Add: [-fno-gpu-defer-diag, -D__CUDA_ARCH__=1200]
---
```

Optionally, you may use the `-fno-gpu-defer-diag` flag during compilation in nvcc mode to see compiler errors that nvcc would normally ignore.

- Restart vscode.

### Check it's working

As a sanity check, create a new `.cu` file with the following contents:

```
__global__ void kernel() {
  asm("add.f32 foo, 1, 1;");
}
```

If everything is working correctly, you should get an red underline on "foo" telling you about the access of a nonexistent variable.
