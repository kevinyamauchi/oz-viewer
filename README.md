# oz-viewer

[![License](https://img.shields.io/pypi/l/oz-viewer.svg?color=green)](https://github.com/kevinyamauchi/oz-viewer/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/oz-viewer.svg?color=green)](https://pypi.org/project/oz-viewer)
[![Python Version](https://img.shields.io/pypi/pyversions/oz-viewer.svg?color=green)](https://python.org)
[![CI](https://github.com/kevinyamauchi/oz-viewer/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinyamauchi/oz-viewer/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kevinyamauchi/oz-viewer/branch/main/graph/badge.svg)](https://codecov.io/gh/kevinyamauchi/oz-viewer)

A viewer for ome-zarr images.

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork kevinyamauchi/oz-viewer --clone
# or just
# gh repo clone kevinyamauchi/oz-viewer
cd oz-viewer
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
