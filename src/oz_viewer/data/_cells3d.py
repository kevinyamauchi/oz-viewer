"""Synthetic multichannel OME-Zarr example from ``skimage.data.cells3d``.

Writes a two-channel fluorescence volume (cell membranes + nuclei) as a
3-level OME-Zarr v0.5 store with a channel axis, suitable for the multichannel
viewer.  The sample data is downloaded on first use via ``pooch`` (a
``scikit-image`` dependency), so the first call needs network access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Physical spacing (micrometers): z=0.29, y/x=0.26 at level 0, y/x halved per
# level.  c is a channel axis (unitless, scale 1.0).
_SCALE_Z = 0.29
_SCALE_YX_L0 = 0.26
_N_LEVELS = 3
_CHUNK_CZYX = (1, 1, 64, 64)

_DEFAULT_PATH = Path("cells3d.ome.zarr")


def make_cells3d_zarr(output_path: Path | str = _DEFAULT_PATH) -> Path:
    """Create a two-channel OME-Zarr from ``skimage.data.cells3d``.

    The stored array is ordered ``(c, z, y, x)`` -- channel-first, as required
    by the OME-Zarr v0.5 axis-ordering rule -- with two channels (channel 0:
    membranes, channel 1: nuclei) and three resolution levels downsampled in
    ``y``/``x`` only.

    Parameters
    ----------
    output_path : Path or str
        Directory to write the OME-Zarr store.  Defaults to
        ``cells3d.ome.zarr`` in the current working directory.  If the path
        already exists it is left untouched and returned.

    Returns
    -------
    Path
        Resolved path to the written (or pre-existing) store.
    """
    import zarr
    from skimage.data import cells3d
    from skimage.measure import block_reduce

    output_path = Path(output_path)
    if output_path.exists():
        print(f"Example dataset already exists at {output_path}")
        return output_path.resolve()

    print(f"Creating multichannel cells3d dataset at {output_path} ...")
    print("Loading skimage.data.cells3d() (downloads on first use) ...")
    # skimage returns (z, c, y, x); transpose to (c, z, y, x) so the channel
    # axis precedes all spatial axes.
    data_l0 = np.transpose(cells3d(), (1, 0, 2, 3)).astype(np.uint16)

    levels = [data_l0]
    for _ in range(_N_LEVELS - 1):
        down = block_reduce(levels[-1], block_size=(1, 1, 2, 2), func=np.mean).astype(
            np.uint16
        )
        levels.append(down)

    root = zarr.open_group(str(output_path), mode="w")
    datasets_meta = []
    for level, arr in enumerate(levels):
        zarr_arr = root.create_array(
            str(level),
            shape=arr.shape,
            chunks=_CHUNK_CZYX,
            dtype=np.uint16,
        )
        zarr_arr[:] = arr
        yx = _SCALE_YX_L0 * (2.0**level)
        datasets_meta.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, _SCALE_Z, yx, yx]},
                ],
            }
        )
        print(f"  Level {level}: shape={arr.shape}  scale=(z={_SCALE_Z}, yx={yx:.4f})")

    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": datasets_meta,
                "name": "cells3d",
            }
        ],
    }

    print("Done. Two channels (0: membranes, 1: nuclei), 3 resolution levels.")
    return output_path.resolve()
