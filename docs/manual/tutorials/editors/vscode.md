# Visual Studio Code

This page assumes you've read the [general instructions](./editors.md). It applies equally to
VS Code's derivatives, such as VSCodium and Cursor.

## Install the `clangd` extension

Install the [clangd extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
(`llvm-vs-code-extensions.vscode-clangd`). It's also on
[Open VSX](https://open-vsx.org/extension/llvm-vs-code-extensions/vscode-clangd), for editors that
use that.

Then point it at SCALE's `clangd`, in your user `settings.json` (`Preferences: Open User Settings
(JSON)` in the command palette):

```json
{
  "clangd.path": "/opt/scale/llvm/bin/clangd"
}
```

Leave `clangd.checkUpdates` off, and decline if the extension offers to download a `clangd` for
you: that would replace SCALE's with an upstream one.

If you have Microsoft's C/C++ extension installed, its IntelliSense competes with `clangd`, and
doesn't understand CUDA as SCALE does. The clangd extension offers to disable it when it
starts; accept, or set:

```json
{
  "C_Cpp.intelliSenseEngine": "disabled"
}
```

The C/C++ extension's debugger still works with IntelliSense disabled.

## Compilation database

Any of the methods in the [general instructions](./editors.md#2-generate-a-compilation-database)
work. If you use the [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)
extension, you can make it configure with SCALE in either of two ways:

=== "CMake presets"

    Add a `CMakePresets.json` to your project, as shown in the
    [general instructions](./editors.md#2-generate-a-compilation-database). CMake Tools uses
    presets automatically when the file exists: pick the `scale` preset when prompted, or with
    `CMake: Select Configure Preset`.

    This is the better option when you share the project with others, since the configuration
    lives in the repository.

=== "CMake kits"

    Run `CMake: Edit User-Local CMake Kits` from the command palette, and add a kit for each SCALE
    GPU target you use:

    ```json
    [
      {
        "name": "SCALE gfx1100",
        "compilers": {
          "C": "/opt/scale/llvm/bin/clang",
          "CXX": "/opt/scale/llvm/bin/clang++",
          "CUDA": "/opt/scale/targets/gfx1100/bin/nvcc"
        },
        "isTrusted": true
      }
    ]
    ```

    Then run `CMake: Select a Kit` and choose it. Add `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` to the
    `cmake.configureArgs` setting.

    !!! warning

        `cmake.configureArgs` is not the same as `cmake.buildArgs`.

CMake Tools builds in `${workspaceFolder}/build` by default, where `clangd` finds
`compile_commands.json` without further help. If you've changed `cmake.buildDirectory`, either set
`CompilationDatabase` in your `.clangd` file, or set `cmake.copyCompileCommands` to
`${workspaceFolder}/compile_commands.json`.

Run `CMake: Configure`, then `clangd: Restart language server`.

## Remote development

With the Remote - SSH, Dev Containers or WSL extensions, `clangd` runs on the remote side.
Install the clangd extension there (VS Code offers to), and use the path to SCALE on the remote
machine or container. Since it's the remote machine that needs SCALE, this is a convenient way to
work on a GPU machine from a laptop.

## Check it's working

Follow the [general instructions](./editors.md#check-its-working). To see `clangd`'s log, open the
Output panel (`View: Toggle Output`) and select `clangd` from the drop-down.
