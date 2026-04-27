"""Synthetic anisotropic OME-Zarr example dataset (ExpA-like scale)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ExpA: Z=5.0 µm, Y=X=6.55 µm; only Y/X are downsampled per level (Z fixed).
_SCALE_Z = 5.0
_SCALE_YX = 6.550032422660492
_SHAPE_ZYX = (200, 300, 300)
_N_LEVELS = 4
_N_BLOBS = 12
_BLOB_RADIUS_UM = 120.0
_CHUNK_ZYX = (32, 32, 32)

_DEFAULT_PATH = Path(__file__).parent / "example_anisotropic_blobs.ome.zarr"


def _make_blob_volume(
    shape_zyx: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    n_blobs: int,
    radius_um: float,
    seed: int = 42,
) -> np.ndarray:
    """Return a uint8 volume with blobs that are spherical in physical space."""
    nz, ny, nx = shape_zyx
    sz, sy, sx = spacing_zyx
    rng = np.random.default_rng(seed)
    volume = np.zeros(shape_zyx, dtype=np.uint8)

    rz = int(np.ceil(radius_um / sz))
    ry = int(np.ceil(radius_um / sy))
    rx = int(np.ceil(radius_um / sx))

    for _ in range(n_blobs):
        cz = rng.integers(rz, nz - rz)
        cy = rng.integers(ry, ny - ry)
        cx = rng.integers(rx, nx - rx)

        lz = np.arange(-rz, rz + 1)
        ly = np.arange(-ry, ry + 1)
        lx = np.arange(-rx, rx + 1)
        ZZ, YY, XX = np.meshgrid(lz, ly, lx, indexing="ij")
        mask = (ZZ * sz) ** 2 + (YY * sy) ** 2 + (XX * sx) ** 2 <= radius_um**2

        z0, z1 = max(0, cz - rz), min(nz, cz + rz + 1)
        y0, y1 = max(0, cy - ry), min(ny, cy + ry + 1)
        x0, x1 = max(0, cx - rx), min(nx, cx + rx + 1)

        mz0 = max(0, -(cz - rz))
        my0 = max(0, -(cy - ry))
        mx0 = max(0, -(cx - rx))
        mz1 = mz0 + (z1 - z0)
        my1 = my0 + (y1 - y0)
        mx1 = mx0 + (x1 - x0)

        volume[z0:z1, y0:y1, x0:x1] |= mask[mz0:mz1, my0:my1, mx0:mx1].astype(np.uint8)

    return volume


def make_example_zarr(output_path: Path = _DEFAULT_PATH) -> Path:
    """Create a synthetic anisotropic OME-Zarr with spherical blobs.

    Uses the same Z/YX scale ratio as ExpA (5.0 : 6.55 µm). Only Y and X are
    downsampled per level; Z stays fixed. Blobs appear as perfect spheres in
    all slice planes when the viewer applies correct coordinate transforms.

    Parameters
    ----------
    output_path : Path
        Directory to write the OME-Zarr store. Created if absent.

    Returns
    -------
    Path
        Resolved path to the written store.
    """
    import zarr

    output_path = Path(output_path)
    if output_path.exists():
        print(f"Example dataset already exists at {output_path}")
        return output_path.resolve()

    print(f"Creating example dataset at {output_path} ...")
    sz, syx = _SCALE_Z, _SCALE_YX
    nz, ny, nx = _SHAPE_ZYX

    data_l0 = _make_blob_volume(_SHAPE_ZYX, (sz, syx, syx), _N_BLOBS, _BLOB_RADIUS_UM)

    root = zarr.open_group(str(output_path), mode="w")
    datasets_meta = []

    for level in range(_N_LEVELS):
        factor = 2**level
        data = data_l0 if level == 0 else data_l0[:, ::factor, ::factor]

        arr = root.create_array(
            f"s{level}",
            shape=data.shape,
            chunks=_CHUNK_ZYX,
            dtype=np.uint8,
        )
        arr[:] = data

        datasets_meta.append(
            {
                "path": f"s{level}",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [sz, syx * factor, syx * factor]},
                ],
            }
        )
        print(
            f"  Level {level}: shape={data.shape}  "
            f"scale=(z={sz}, yx={syx * factor:.4f})"
        )

    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": datasets_meta,
                "name": "blobs",
            }
        ],
    }

    print(
        f"Done. Physical size: "
        f"z={nz * sz:.0f} µm, y={ny * syx:.0f} µm, x={nx * syx:.0f} µm"
    )
    print(f"Blob radius: {_BLOB_RADIUS_UM} µm  (spherical in world space)")
    return output_path.resolve()
