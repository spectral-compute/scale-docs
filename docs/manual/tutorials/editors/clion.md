# CLion

CLion bundles its own modified `clangd`, and doesn't support replacing it, so it can't use SCALE's.

You can still build with SCALE from CLion by setting the CUDA compiler in your CMake profile
(`-DCMAKE_CUDA_COMPILER=/opt/scale/targets/gfx1100/bin/nvcc`), but CLion's code insight won't
reflect SCALE's dialect or diagnostics.
