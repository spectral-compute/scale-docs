# Zed

This page assumes you've read the [general instructions](./editors.md).

Zed has built-in `clangd` support, and treats `.cu` and `.cuh` files as C++, so there's nothing to
install. You only need to point it at SCALE's `clangd`.

## Use SCALE's `clangd`

Add this to your settings (`zed: open settings` in the command palette, or edit
`~/.config/zed/settings.json`):

```json
{
  "lsp": {
    "clangd": {
      "binary": {
        "path": "/opt/scale/llvm/bin/clangd"
      }
    }
  }
}
```

The path must be absolute. Without this setting, Zed uses whichever `clangd` is first in `PATH`,
and downloads an upstream `clangd` if there isn't one: neither understands CUDA as SCALE does.

To pass extra command-line options to `clangd`, add an `arguments` list next to `path`:

```json
"binary": {
  "path": "/opt/scale/llvm/bin/clangd",
  "arguments": ["--background-index", "--header-insertion=never"]
}
```

The same settings can go in a project's `.zed/settings.json`, if you only want SCALE's `clangd` for
some projects.

## Compilation database and `.clangd`

Follow the [general instructions](./editors.md#2-generate-a-compilation-database). Zed doesn't
configure CMake projects for you, so run CMake (or Bear) from a terminal, either Zed's own or
another. Zed's tasks (`.zed/tasks.json`) are a convenient way to keep the configure command
with the project:

```json
[
  {
    "label": "Configure with SCALE",
    "command": "cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CUDA_COMPILER=/opt/scale/targets/gfx1100/bin/nvcc"
  }
]
```

After regenerating the database, run `editor: restart language server` if Zed doesn't pick up
the change.

## Remote development

When you open a project on a remote machine over SSH, Zed runs language servers on that machine.
Set `path` to the location of SCALE on the remote machine.

## Check it's working

Follow the [general instructions](./editors.md#check-its-working). To see `clangd`'s log, run
`dev: open language server logs` and select `clangd`.
