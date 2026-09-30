# `src/`

Holds the `edgecnn` package — the shared library every notebook imports. Nothing else goes here.

The package is installed in editable mode, so `import edgecnn` works from any notebook, from the
tests, and from any directory, and your edits take effect immediately:

```powershell
pip install -e ".[dev]"
```

Notebooks *call* this code; they do not contain it. How code moves from a scratch notebook into
here is in [CONTRIBUTING.md → the promote rule](../CONTRIBUTING.md#the-promote-rule). What is
inside: [`edgecnn/README.md`](edgecnn/README.md).
