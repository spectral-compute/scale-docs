# Helix

This page assumes you've read the [general instructions](./editors.md)

Helix uses `clangd` for C++, and treats `.cu` and `.cuh` files as C++. In
`~/.config/helix/languages.toml` (or a project's `.helix/languages.toml`):

```toml
[language-server.clangd]
command = "/opt/scale/llvm/bin/clangd"
args = ["--background-index"]
```

Run `hx --health cpp` to check which `clangd` Helix finds. Its log is at `~/.cache/helix/helix.log`.
