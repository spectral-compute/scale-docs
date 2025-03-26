# SCALE documentation

This repository contains the source code for the [SCALE](https://scale-lang.com) 
documentation and examples.

The documentation can be viewed at https://docs.scale-lang.com/.

The root of the documentation files is at [`docs/README.md`](docs/README.md).

Pull requests are welcomed!

## Compiling the manual

Needed Python modules:

- `pymdown-extensions`
- `mkdocs-material`
- `mike`

Development:

```sh
mkdocs serve
```

Generating:

```sh
mkdocs build -f mkdocs-stable.yml
```
