"""Tests for the ``sidecar=`` wiring on ``display_viewer`` / ``display_orthoviewer``.

These stub out the anywidget canvas build (``cellier.convenience.gui``, which
needs ``rendercanvas.anywidget``) and ``cellier.convenience.display`` itself,
so only the oz-viewer-owned wiring is exercised: turning the public
``sidecar: bool`` flag into a titled ``SidecarOptions`` and forwarding it to
``display()``.
"""

from __future__ import annotations

import pytest


def test_sidecar_options_false_returns_none():
    from oz_viewer.viewer._utils import _sidecar_options

    assert _sidecar_options(False, "OME-Zarr Viewer") is None


def test_sidecar_options_true_builds_titled_options():
    from cellier.convenience import SidecarOptions

    from oz_viewer.viewer._utils import _sidecar_options

    options = _sidecar_options(True, "OME-Zarr Viewer")
    assert isinstance(options, SidecarOptions)
    assert options.title == "OME-Zarr Viewer"


@pytest.fixture
def blobs_uri(tmp_path):
    from oz_viewer.data._blobs import make_example_zarr

    zarr_path = make_example_zarr(output_path=tmp_path / "blobs.ome.zarr")
    return f"file://{zarr_path}"


@pytest.fixture
def _stub_display(monkeypatch):
    """Stub the anywidget canvas build + ``cellier.convenience.display``.

    Returns a dict that ``display()`` calls populate with their ``sidecar``
    kwarg, so tests can assert on it without a real rendercanvas/anywidget
    stack.
    """
    import cellier.convenience as convenience
    import cellier.convenience.gui as gui_mod

    captured: dict = {}

    def _fake_display(viewer, layout, *, fit, sidecar=None, **kwargs):
        captured["sidecar"] = sidecar
        return "handle"

    monkeypatch.setattr(gui_mod, "build_canvas_widget", lambda *a, **k: object())
    monkeypatch.setattr(gui_mod, "build_ortho_grid_widget", lambda *a, **k: object())
    monkeypatch.setattr(convenience, "display", _fake_display)
    return captured


def test_display_viewer_forwards_sidecar_true(blobs_uri, _stub_display):
    from oz_viewer.viewer._viewer import display_viewer

    result = display_viewer(blobs_uri, sidecar=True)

    assert result == "handle"
    assert _stub_display["sidecar"].title == "OME-Zarr Viewer"


def test_display_viewer_defaults_sidecar_to_none(blobs_uri, _stub_display):
    from oz_viewer.viewer._viewer import display_viewer

    display_viewer(blobs_uri)

    assert _stub_display["sidecar"] is None


def test_display_orthoviewer_forwards_sidecar_true(blobs_uri, _stub_display):
    from oz_viewer.viewer._orthoviewer import display_orthoviewer

    result = display_orthoviewer(blobs_uri, sidecar=True)

    assert result == "handle"
    assert _stub_display["sidecar"].title == "OME-Zarr Orthoviewer"


def test_display_orthoviewer_defaults_sidecar_to_none(blobs_uri, _stub_display):
    from oz_viewer.viewer._orthoviewer import display_orthoviewer

    display_orthoviewer(blobs_uri)

    assert _stub_display["sidecar"] is None
