"""Shared pytest fixtures for oz-viewer tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def write_demo_ome(tmp_path: Path) -> Callable[[Literal["image", "plate"]], Path]:
    """Return a factory that writes demo OME-Zarr stores to tmp_path.

    Skips the test if zarr or ome-zarr are not installed.
    """
    try:
        from yaozarrs._demo_data import write_ome_image, write_ome_plate
    except ImportError:
        pytest.skip("zarr and ome-zarr are required for demo data fixtures")

    def _factory(store_type: Literal["image", "plate"] = "image") -> Path:
        if store_type == "image":
            path = tmp_path / "demo_image.zarr"
            write_ome_image(path)
        elif store_type == "plate":
            path = tmp_path / "demo_plate.zarr"
            write_ome_plate(path)
        else:
            raise ValueError(f"Unknown store_type: {store_type!r}")
        return path

    return _factory
