# Sublime Text

This page assumes you've read the [general instructions](./editors.md)

Install, from Package Control:

- [LSP](https://packagecontrol.io/packages/LSP) and
  [LSP-clangd](https://packagecontrol.io/packages/LSP-clangd).
- [CUDA C++](https://packagecontrol.io/packages/CUDA%20C%2B%2B), since Sublime Text has no
  built-in syntax for CUDA. LSP-clangd attaches to files that use it.

Then open `Preferences > Package Settings > LSP > Servers > LSP-clangd`, and set:

```json
{
  "system_binary": "/opt/scale/llvm/bin/clangd"
}
```

Other `clangd` options are set as `initialization_options` in the same file: it lists them all.
