"""Headless build tests for the convenience-based orthoviewer.

These exercise the full Qt build path (``_build_and_show_ortho_qt``) offscreen:
the four synced panels, the build-time single/multichannel choice, the
re-attached 3D overlays, and the Qt control docks.  A live GPU render is not
exercised (there is no display in CI), only construction.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


def _dock_titles(window) -> list[str]:
    from PySide6.QtWidgets import QDockWidget

    return [d.windowTitle() for d in window.findChildren(QDockWidget)]


def test_single_channel_ortho_build(qapp, tmp_path):
    """Blobs (z,y,x) -> single-channel ortho: 4 panels, overlays, Rendering dock."""
    from oz_viewer.data._blobs import make_example_zarr
    from oz_viewer.viewer._orthoviewer import _build_and_show_ortho_qt

    zarr_path = make_example_zarr(output_path=tmp_path / "blobs.ome.zarr")
    handle = _build_and_show_ortho_qt(f"file://{zarr_path}")
    try:
        build = handle.build
        assert build.is_multichannel is False
        assert set(build.visuals) == {"xy", "xz", "yz", "vol"}
        assert set(build.viewer.scenes) == {"xy", "xz", "yz", "vol"}

        # Overlays attached: plane + three orientation-axis meshes, ISO profile.
        overlays = build.overlays
        assert overlays is not None
        assert len(overlays.axis_visual_ids) == 3
        assert overlays.transparency_manager.current_mode == "iso"

        # Qt-only appearance panel on the left dock; no channel dock.
        assert "Rendering" in _dock_titles(handle.window)
    finally:
        handle.close()


def test_multichannel_ortho_build(qapp, write_demo_ome):
    """Demo image (c,z,y,x) -> multichannel ortho: channel stacked, both docks."""
    from oz_viewer.viewer._orthoviewer import _build_and_show_ortho_qt

    zarr_path = write_demo_ome("image")
    handle = _build_and_show_ortho_qt(f"file://{zarr_path}")
    try:
        build = handle.build
        assert build.is_multichannel is True
        assert set(build.visuals) == {"xy", "xz", "yz", "vol"}

        # Channel axis (0) is stacked, not sliced, so no channel slider remains.
        for scene in build.viewer.scenes.values():
            assert 0 not in scene.dims.selection.slice_indices

        # Multichannel volume is locked to MIP; overlays present.
        assert build.overlays.transparency_manager.current_mode == "mip"

        # ChannelControls dock ("Left") plus the Qt-only volume group ("Volume").
        titles = _dock_titles(handle.window)
        assert "Left" in titles
        assert "Volume" in titles
    finally:
        handle.close()
