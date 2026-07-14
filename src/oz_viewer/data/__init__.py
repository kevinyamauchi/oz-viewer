"""Example data for oz-viewer."""

from oz_viewer.data._blobs import make_example_zarr
from oz_viewer.data._cells3d import make_cells3d_zarr

__all__ = [
    "make_cells3d_zarr",
    "make_example_zarr",
]
