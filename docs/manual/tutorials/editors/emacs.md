# Emacs

This page assumes you've read the [general instructions](./editors.md).

## Associate CUDA files with a mode

Emacs has no built-in mode for CUDA, so `.cu` and `.cuh` files need one before a language server
can attach. Either use C++ mode for them:

```elisp
(add-to-list 'auto-mode-alist '("\\.cuh?\\'" . c++-mode))   ; or c++-ts-mode
```

or install [`cuda-mode`](https://github.com/chachi/cuda-mode) from MELPA, which adds some CUDA
keywords to C++ mode, and associates the extensions itself.

## Use SCALE's `clangd`

=== "Eglot"

    [Eglot](https://www.gnu.org/software/emacs/manual/html_node/eglot/) is built into Emacs 29
    and later. Add an entry for SCALE's `clangd` to the front of `eglot-server-programs`:

    ```elisp
    (with-eval-after-load 'eglot
      (add-to-list 'eglot-server-programs
                   '((c-mode c-ts-mode c++-mode c++-ts-mode)
                     . ("/opt/scale/llvm/bin/clangd" "--background-index"))))
    ```

    `cuda-mode` derives from `c++-mode`, so this entry covers it too. Start Eglot in a CUDA buffer
    with `M-x eglot`, or add `eglot-ensure` to the mode's hook.

=== "lsp-mode"

    Set these before `lsp-mode` is loaded:

    ```elisp
    (setq lsp-clients-clangd-executable "/opt/scale/llvm/bin/clangd"
          lsp-clients-clangd-args '("--header-insertion-decorators=0" "--background-index"))
    ```

    `lsp-clients-clangd-args` replaces the default arguments rather than adding to them, so keep
    `--header-insertion-decorators=0`. lsp-mode's `clangd` client has a low priority: if you have
    another C++ language server installed, such as `ccls`, add it to `lsp-disabled-clients`.

## Compilation database and `.clangd`

Follow the [general instructions](./editors.md#2-generate-a-compilation-database). After
regenerating the database, restart the server (`M-x eglot-reconnect` or
`M-x lsp-workspace-restart`) if `clangd` doesn't pick up the change.

## Remote development

Both Eglot and lsp-mode work over TRAMP, running `clangd` on the remote machine. Use the path to
SCALE on that machine.

## Check it's working

Follow the [general instructions](./editors.md#check-its-working). Eglot's log is in
`M-x eglot-events-buffer`; lsp-mode's is in the `*clangd::stderr*` buffer.
