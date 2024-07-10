# SCALE documentation

This is documentation for SCALE, hosted at https://docs.scale-lang.com/.

The root of the documentation files is here: [`docs/README.md`](docs/README.md).
It can be browsed directly, but by doing so, you would miss out on the enhancements applied to the hosted version.
One of such changes is inclusion of example code files into the documentation.

You can find the examples in the `examples` directory of this repository.
Read how to use them in the documentation.

## Generating public documentation

Dependencides:

```sh
yay -S mkdocs pymdown-extensions
pip install mkdocs-material
```

Development:

```sh
mkdocs serve
```

Generating:

```sh
mkdocs build
```

Good example for features:
- https://example-mkdocs-basic.readthedocs.io/en/latest/
