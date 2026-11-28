# Neovim

This page assumes you've read the [general instructions](./editors.md), and that you're using
Neovim 0.11 or later with [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig).

Neovim recognises `.cu` and `.cuh` files as the `cuda` filetype, and nvim-lspconfig's `clangd`
configuration already attaches to it. You only need to point it at SCALE's `clangd`. In your
`init.lua`:

```lua
vim.lsp.config('clangd', {
  cmd = { '/opt/scale/llvm/bin/clangd', '--background-index' },
})
vim.lsp.enable('clangd')
```

`vim.lsp.config()` is merged over nvim-lspconfig's defaults, so this only replaces the command.
Alternatively, put the same table in `~/.config/nvim/after/lsp/clangd.lua`, as
`return { cmd = { ... } }`.

=== "Without nvim-lspconfig"

    Neovim itself doesn't ship a `clangd` configuration, so you need to give the rest of it too:

    ```lua
    vim.lsp.config('clangd', {
      cmd = { '/opt/scale/llvm/bin/clangd', '--background-index' },
      filetypes = { 'c', 'cpp', 'objc', 'objcpp', 'cuda' },
      root_markers = { '.clangd', 'compile_commands.json', '.git' },
    })
    vim.lsp.enable('clangd')
    ```

=== "Neovim 0.10 and earlier"

    Use nvim-lspconfig's older interface:

    ```lua
    require('lspconfig').clangd.setup({
      cmd = { '/opt/scale/llvm/bin/clangd', '--background-index' },
    })
    ```

If you install language servers with Mason, make sure it isn't also installing `clangd`: Mason's
is an upstream build.

## Compilation database and `.clangd`

Follow the [general instructions](./editors.md#2-generate-a-compilation-database). After
regenerating the database, run `:LspRestart` (`:lsp restart` on Neovim 0.12 and later) if `clangd`
doesn't pick up the change.

## Check it's working

Follow the [general instructions](./editors.md#check-its-working). `:checkhealth vim.lsp` shows
which `clangd` is attached to the current buffer, and where its log is.
