# Reference

Reference pages are generated from source where possible. Python signatures
come from autodoc, and the complete Click command tree comes from
sphinx-click.

```{toctree}
:maxdepth: 1

../api
../cli
support-matrix
../STABILITY
```

The API inventory and stability annotations are checked independently in a
real project environment so the lightweight Read the Docs mocks cannot hide
first-party import errors.
