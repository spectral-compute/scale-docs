# Overview

SCALE provides a version of [`clangd`](https://clangd.llvm.org/) that
understands nvcc-dialect CUDA, inline PTX, and provides other CUDA-specific
features.

`clangd` is a language server. Typically, it is run behind-the-scenes by your
editor to provide autocompletion, diagnostics, etc.

In general, using SCALE's `clangd` in your editor is the
same as [the process](https://clangd.llvm.org/installation) for upstream `clangd`,
except:

- You set the path to the `clangd` executable to the one that ships with
  SCALE (`<SCALE_DIR>/llvm/bin/clangd`)
- You add a file called `.clangd` to the root of your project with the following
  contents. This will add a few flags to the command line `clangd` uses to explore
  your code:

```
---
CompileFlags:
  Add: [-fno-gpu-defer-diag, -D__CUDA_ARCH__=1200]
---
```

These two options have the following effect:

- `-fno-gpu-defer-diag`: instructs the compiler to provide diagnostics for all code,
  including code that seems to be unused. CUDA compilation typically ignores code for
  the "other side".
- Defining `__CUDA_ARCH__` tends to cause more device code to be shown to clangd.
  For best results, you should give `clangd` a combination of macros that shows it
  the code from both sides, all the time.

If clangd fails to autodetect which CUDA dialect you are using from the compilation
database, you may need to add `-fcuda-nvcc-emulation` to explicitly enable the NVCC
dialect.

Editor-specific tutorials for setting up SCALE-clangd are available in this section
of the mnaual.

- [`vscode`](./vscode.md)
