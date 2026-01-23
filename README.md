# SCALE documentation

This repository contains the source code for the [SCALE](https://scale-lang.com) 
documentation and examples.

The documentation can be viewed at https://docs.scale-lang.com/.

The root of the documentation files is at [`docs/README.md`](docs/README.md).

Pull requests are welcomed!

The `master` branch holds the current version of the stable documentation.
`unstable` contains the current unstable documentation.

The [`mike`](https://github.com/jimporter/mike) tool is used for managing
documentation versions.

## Compiling the manual

Use uv for installing dependencies:

```sh
uv sync
```

To launch the devserver and see changes live:

```sh
uv run mkdocs serve
```

## Publishing

### To publish a new version of the manual for a stable release:

*Make sure not to prepend 'v' to the version number*

- Ensure your changes are all in master, and master is checked out.
- `mike deploy --push --update-aliases <SCALE VERSION NUMBER> stable`

To deploy an unstable release of the manual:

- Check out the `unstable` branch.
- `mike deploy --push unstable`

### To publish a new version of the manual for an unstable release:

*Make sure not to prepend 'v' to the version number*

- Ensure your changes are all in `unstable`, and `unstable` is checked out.
- `mike deploy --push unstable`
