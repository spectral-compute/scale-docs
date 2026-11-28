# Kate

This page assumes you've read the [general instructions](./editors.md)

Kate's LSP client uses `clangd` for C and C++, and highlights `.cu` and `.cuh` files as C++. Open
`Settings > Configure Kate > LSP Client > User Server Settings`, and add:

```json
{
  "servers": {
    "c": {
      "servers": {
        "clangd": {
          "command": ["/opt/scale/llvm/bin/clangd", "-log=error", "--background-index",
                      "--limit-results=500", "--completion-style=bundled"]
        }
      }
    }
  }
}
```

Kate's C++ server settings inherit from the C ones, so this covers both. The `command` list
replaces the default one entirely, which is why it repeats Kate's default options.
