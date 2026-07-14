"""OME-Zarr metadata extraction shared by the convenience-based viewers.

This module holds the pure, Qt-free logic that turns an OME-Zarr store into the
handful of numbers the cellier ``convenience`` API needs: which axis is the
channel axis, the spatial extents, the contrast-limit range, sensible slider
decimals, and the voxel-to-world transform.  It has no cellier ``convenience``
dependency itself so it can be reused by both the single-panel viewer and the
orthoviewer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from oz_viewer.viewer._utils import _dtype_clim_max, _dtype_decimals, _perf_mark

if TYPE_CHECKING:
    from cellier.data.image import OMEZarrImageDataStore
    from cellier.transform import AffineTransform

    from oz_viewer._perf import StartupPerfTracer


class _ViewerGeometry(NamedTuple):
    """Scene geometry derived from OME-Zarr metadata; no Qt objects."""

    axis_names: tuple[str, ...]
    spatial_indices: list[int]
    spatial_ndim: int
    channel_axis: int | None
    n_channels: int
    voxel_to_world: AffineTransform
    #: World-space ``(shape - 1) * scale`` for every axis, used for centering.
    world_max_full: np.ndarray
    #: World-space ``(shape - 1) * scale`` for the spatial axes only.
    world_max_spatial: np.ndarray
    initial_clim_max: float
    clim_range: tuple[float, float]
    slider_decimals: int
    #: ``(near, far)`` clip distances derived from the world extents.
    depth_range: tuple[float, float]

    @property
    def spatial_axis_labels(self) -> tuple[str, ...]:
        """Axis labels of the spatial axes, in order."""
        return tuple(self.axis_names[i] for i in self.spatial_indices)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        """Global spatial axis indices as a tuple (``(z, y, x)`` when 3-D).

        The orthoviewer passes this to
        :class:`cellier.convenience.OrthoViewer` and to the overlay builders,
        which expect a length-3 ``(z, y, x)`` tuple.
        """
        return tuple(self.spatial_indices)

    @property
    def axis_ranges(self) -> dict[int, tuple[float, float]]:
        """World-space ``(0, world_max)`` per axis for the dims-slider ranges.

        Computed directly from the metadata rather than via
        ``cellier.convenience.axis_ranges_from_viewer`` so it works for
        multichannel visuals too (whose transform is spatial-only and cannot be
        mapped against the full-ndim store shape).
        """
        return {
            i: (0.0, round(float(self.world_max_full[i])))
            for i in range(len(self.axis_names))
        }

    def center_slice_indices(self) -> dict[int, float]:
        """World-coordinate midpoints for each spatial axis.

        Extra (non-spatial) axes such as channel are intentionally omitted so
        they keep their default position of ``0``.
        """
        return {
            axis: round(float(self.world_max_full[axis]) / 2.0)
            for axis in self.spatial_indices
        }


def extract_viewer_geometry(
    zarr_uri: str,
    *,
    channel_axis: int | None = None,
    perf: StartupPerfTracer | None = None,
    print_summary: bool = True,
) -> tuple[OMEZarrImageDataStore, _ViewerGeometry]:
    """Open an OME-Zarr store and extract the geometry the viewers need.

    Parameters
    ----------
    zarr_uri : str
        Path or URI to the OME-Zarr store.
    channel_axis : int or None, optional
        Axis index to treat as the channel dimension.  When ``None`` (default),
        the channel axis is auto-detected from the OME-Zarr axis metadata.
    perf : StartupPerfTracer or None, optional
        Optional startup performance tracer.
    print_summary : bool, optional
        Print a Rich table summarizing the store's metadata.  Default ``True``.

    Returns
    -------
    tuple[OMEZarrImageDataStore, _ViewerGeometry]
    """
    import yaozarrs
    from cellier.data.image import OMEZarrImageDataStore
    from cellier.transform import AffineTransform

    _perf_mark(perf, "geometry.start", zarr_uri=zarr_uri)
    data_store = OMEZarrImageDataStore.from_path(zarr_uri)
    _perf_mark(perf, "geometry.data_store_ready", n_levels=data_store.n_levels)

    group = yaozarrs.open_group(data_store.zarr_path)
    ome_image = group.ome_metadata()
    ms = ome_image.multiscales[data_store.multiscale_index]

    n_dims = len(data_store.level_shapes[0])

    # Detect channel axis from OME-Zarr axis metadata when not explicitly set.
    if channel_axis is None:
        for idx, ax in enumerate(ms.axes):
            if getattr(ax, "type", None) == "channel":
                channel_axis = idx
                break

    spatial_indices: list[int] = [i for i in range(n_dims) if i != channel_axis]
    spatial_ndim = len(spatial_indices)

    level_0_scale_full = np.array(
        ms.datasets[0].scale_transform.scale, dtype=np.float64
    )
    level_0_scale_spatial = level_0_scale_full[spatial_indices]

    vox_shape_full = np.array(data_store.level_shapes[0], dtype=np.float64)
    vox_shape_spatial = vox_shape_full[spatial_indices]
    world_extents_spatial = vox_shape_spatial * level_0_scale_spatial
    max_extent = float(world_extents_spatial.max())
    depth_range = (max(1.0, max_extent * 0.0001), max_extent * 10.0)

    world_max_full = (vox_shape_full - 1) * level_0_scale_full
    world_max_spatial = (vox_shape_spatial - 1) * level_0_scale_spatial

    if print_summary:
        _print_metadata_table(
            zarr_uri,
            data_store,
            channel_axis,
            spatial_ndim,
            level_0_scale_spatial,
            world_extents_spatial,
            depth_range,
        )
        _perf_mark(perf, "geometry.metadata_printed")

    voxel_to_world = AffineTransform.from_scale_and_translation(
        scale=tuple(level_0_scale_full)
    )

    initial_clim_max = _dtype_clim_max(data_store.dtype)
    slider_decimals = _dtype_decimals(data_store.dtype)
    n_channels = (
        int(data_store.level_shapes[0][channel_axis]) if channel_axis is not None else 0
    )

    geometry = _ViewerGeometry(
        axis_names=tuple(data_store.axis_names),
        spatial_indices=spatial_indices,
        spatial_ndim=spatial_ndim,
        channel_axis=channel_axis,
        n_channels=n_channels,
        voxel_to_world=voxel_to_world,
        world_max_full=world_max_full,
        world_max_spatial=world_max_spatial,
        initial_clim_max=initial_clim_max,
        clim_range=(0.0, initial_clim_max),
        slider_decimals=slider_decimals,
        depth_range=depth_range,
    )
    _perf_mark(perf, "geometry.ready")
    return data_store, geometry


def _print_metadata_table(
    zarr_uri: str,
    data_store: OMEZarrImageDataStore,
    channel_axis: int | None,
    spatial_ndim: int,
    level_0_scale_spatial: np.ndarray,
    world_extents_spatial: np.ndarray,
    depth_range: tuple[float, float],
) -> None:
    """Print the startup metadata summary as a Rich table."""
    from rich.console import Console
    from rich.table import Table

    spatial_label = "spatial" if channel_axis is not None else "ZYX"[-spatial_ndim:]
    table = Table(
        title=f"OME-Zarr  [dim]{zarr_uri}[/dim]",
        show_header=False,
        box=None,
        padding=(0, 2),
    )
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("dtype", str(data_store.dtype))
    table.add_row("axes", "  ".join(data_store.axis_names))
    table.add_row("units", "  ".join(str(u) for u in data_store.axis_units))
    table.add_row("levels", str(data_store.n_levels))
    for i, shape in enumerate(data_store.level_shapes):
        table.add_row(f"  level {i}", str(list(shape)))
    table.add_row(
        f"scale ({spatial_label})",
        "  ".join(f"{v:.4g}" for v in level_0_scale_spatial),
    )
    table.add_row(
        f"world extents ({spatial_label})",
        "  ".join(f"{v:.4g}" for v in world_extents_spatial),
    )
    table.add_row("depth range", f"near={depth_range[0]:.2f}  far={depth_range[1]:.0f}")
    Console().print(table)
