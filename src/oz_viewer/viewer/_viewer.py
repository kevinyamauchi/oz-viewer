"""OmeZarrViewer: single-panel 2D / 3D toggle viewer for OME-Zarr images."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from oz_viewer._perf import StartupPerfTracer

_DEFAULT_COLORMAPS: list[str] = [
    "viridis",
    "plasma",
    "grays",
    "white",
    "green",
    "blue",
    "red",
    "magenta",
    "cyan",
    "bop_blue",
    "bop_orange",
    "bop_purple",
    "i_blue",
    "i_bordeaux",
    "i_cyan",
    "i_forest",
    "i_green",
    "i_magenta",
    "i_orange",
    "i_purple",
    "i_red",
    "i_yellow",
]


# ---------------------------------------------------------------------------
# Geometry descriptor
# ---------------------------------------------------------------------------


class _ViewerGeometry(NamedTuple):
    """Scene geometry derived from OME-Zarr metadata; no Qt objects."""

    spatial_indices: list[int]
    spatial_ndim: int
    displayed_axes_2d: tuple[int, ...]
    displayed_axes_3d: tuple[int, ...]  # equals displayed_axes_2d when spatial_ndim < 3
    world_max_spatial: np.ndarray
    voxel_to_world: object
    axis_ranges: dict[int, tuple[float, float]]
    initial_slice_indices_2d: dict[int, float]
    initial_slice_indices_3d: dict[int, float]
    initial_clim_max: float
    clim_range: tuple[float, float]
    slider_decimals: int


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _dtype_clim_max(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def _dtype_decimals(dtype: np.dtype) -> int:
    return 0 if np.issubdtype(dtype, np.integer) else 2


def _perf_mark(perf: StartupPerfTracer | None, step: str, /, **fields: object) -> None:
    if perf is not None:
        perf.mark(step, **fields)


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------


class OmeZarrViewer:
    """Single-panel 2D / 3D toggle viewer for OME-Zarr images.

    For interactive use (IPython / Jupyter) call :func:`viewer`.
    For scripts and the CLI call :func:`launch_viewer`.
    """

    def __init__(
        self,
        controller,
        scene,
        canvas_widget,
        visual_model,
        geometry: _ViewerGeometry,
    ) -> None:
        from cellier.v2.gui.visuals._colormap import QtColormapComboBox
        from cellier.v2.gui.visuals._contrast_limits import QtClimRangeSlider
        from cellier.v2.gui.visuals._image import QtVolumeRenderControls
        from PySide6 import QtCore, QtWidgets
        from PySide6.QtWidgets import QStackedWidget

        self._controller = controller
        self._scene = scene
        self._canvas_widget = canvas_widget
        self._visual_model = visual_model
        self._geo = geometry

        si = geometry.spatial_indices
        # The axis that moves between displayed and sliced on 2D ↔ 3D toggle.
        self._sz0: int | None = si[-3] if geometry.spatial_ndim >= 3 else None
        self._displayed_axes_2d = geometry.displayed_axes_2d
        self._displayed_axes_3d = geometry.displayed_axes_3d
        self._active_mode = "2d"

        # Independent LOD bias values per mode.
        self._lod_bias_2d: float = visual_model.appearance.lod_bias
        self._lod_bias_3d: float = visual_model.appearance.lod_bias

        # Z world-coord to restore when re-entering 2D mode.
        self._saved_z: float = (
            float(geometry.initial_slice_indices_2d.get(self._sz0, 0))
            if self._sz0 is not None
            else 0.0
        )

        # ── Shared controls (single visual → auto-synced across modes) ───
        clim_range = geometry.clim_range
        slider_decimals = geometry.slider_decimals

        self._clim_slider = QtClimRangeSlider(
            visual_model.id,
            clim_range=clim_range,
            initial_clim=visual_model.appearance.clim,
            decimals=slider_decimals,
        )
        controller.connect_widget(
            self._clim_slider,
            subscription_specs=self._clim_slider.subscription_specs(),
        )

        self._colormap_combo = QtColormapComboBox(
            visual_model.id,
            initial_colormap=visual_model.appearance.color_map,
        )
        self._colormap_combo.add_colormaps(_DEFAULT_COLORMAPS)
        controller.connect_widget(
            self._colormap_combo,
            subscription_specs=self._colormap_combo.subscription_specs(),
        )

        # ── 3D-only render controls ───────────────────────────────────────
        self._render_controls = QtVolumeRenderControls(
            visual_model.id,
            dtype_max=clim_range[1],
            initial_render_mode=visual_model.appearance.render_mode,
            initial_threshold=visual_model.appearance.iso_threshold,
            initial_attenuation=visual_model.appearance.attenuation,
            decimals=slider_decimals,
        )
        controller.connect_widget(
            self._render_controls,
            subscription_specs=self._render_controls.subscription_specs(),
        )

        # ── Qt window ─────────────────────────────────────────────────────
        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("OME-Zarr Viewer")
        self._window.resize(1100, 750)

        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        # Side panel on the left.
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(280)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        root_layout.addWidget(panel)
        root_layout.addWidget(canvas_widget.widget, stretch=1)

        # ── Toggle button ─────────────────────────────────────────────────
        self._toggle_btn = QtWidgets.QPushButton("Switch to 3D")
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        if geometry.spatial_ndim < 3:
            self._toggle_btn.setEnabled(False)
            self._toggle_btn.setToolTip(
                "3D view requires at least 3 spatial dimensions"
            )
        panel_layout.addWidget(self._toggle_btn)

        self._mode_label = QtWidgets.QLabel("Mode: 2D")
        panel_layout.addWidget(self._mode_label)

        # ── Shared: contrast limits ───────────────────────────────────────
        clim_group = QtWidgets.QGroupBox("Contrast limits")
        QtWidgets.QVBoxLayout(clim_group).addWidget(self._clim_slider.widget)
        panel_layout.addWidget(clim_group)

        # ── Shared: colormap ──────────────────────────────────────────────
        cmap_group = QtWidgets.QGroupBox("Colormap")
        QtWidgets.QVBoxLayout(cmap_group).addWidget(self._colormap_combo.widget)
        panel_layout.addWidget(cmap_group)

        # ── Stacked: mode-specific controls ───────────────────────────────
        self._stack = QStackedWidget()
        panel_layout.addWidget(self._stack)

        # Page 0 — 2D controls
        page_2d = QtWidgets.QWidget()
        layout_2d = QtWidgets.QVBoxLayout(page_2d)
        layout_2d.setContentsMargins(0, 0, 0, 0)
        layout_2d.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        lod_2d_group = QtWidgets.QGroupBox("Fine-coarse tile bias")
        lod_2d_group.setToolTip("Bigger values use coarser tiles")
        self._lod_bias_2d_slider = self._make_lod_slider("2d")
        QtWidgets.QVBoxLayout(lod_2d_group).addWidget(self._lod_bias_2d_slider)
        layout_2d.addWidget(lod_2d_group)
        self._stack.addWidget(page_2d)

        # Page 1 — 3D controls
        page_3d = QtWidgets.QWidget()
        layout_3d = QtWidgets.QVBoxLayout(page_3d)
        layout_3d.setContentsMargins(0, 0, 0, 0)
        layout_3d.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        render_group = QtWidgets.QGroupBox("Render mode")
        QtWidgets.QVBoxLayout(render_group).addWidget(self._render_controls.widget)
        layout_3d.addWidget(render_group)
        lod_3d_group = QtWidgets.QGroupBox("Fine-coarse tile bias")
        lod_3d_group.setToolTip("Bigger values use coarser tiles")
        self._lod_bias_3d_slider = self._make_lod_slider("3d")
        QtWidgets.QVBoxLayout(lod_3d_group).addWidget(self._lod_bias_3d_slider)
        layout_3d.addWidget(lod_3d_group)
        self._stack.addWidget(page_3d)

        self._stack.setCurrentIndex(0)
        panel_layout.addStretch()

    # ------------------------------------------------------------------
    # LOD bias helpers
    # ------------------------------------------------------------------

    def _make_lod_slider(self, mode: str):
        from qtpy.QtCore import Qt
        from superqt import QLabeledDoubleSlider

        initial = self._lod_bias_2d if mode == "2d" else self._lod_bias_3d
        slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        slider.setRange(1e-6, 5.0)
        slider.setDecimals(2)
        slider.setValue(initial)

        def _on_released() -> None:
            if self._active_mode != mode:
                return
            value = slider.value()
            if mode == "2d":
                self._lod_bias_2d = value
            else:
                self._lod_bias_3d = value
            self._controller.update_appearance_field(
                self._visual_model.id, "lod_bias", value
            )

        slider.sliderReleased.connect(_on_released)
        return slider

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _on_toggle_clicked(self) -> None:
        from cellier.v2.events import DimsUpdateEvent

        current_slice = dict(self._scene.dims.selection.slice_indices)
        self._controller.cancel_pending_slices(self._scene.id)

        if self._active_mode == "2d":
            # Save the Z world-coord before it leaves slice_indices.
            if self._sz0 is not None:
                self._saved_z = float(
                    current_slice.get(
                        self._sz0,
                        self._geo.initial_slice_indices_2d.get(self._sz0, 0),
                    )
                )
            # sz0 becomes displayed in 3D — drop it from slice_indices.
            new_slice = {k: v for k, v in current_slice.items() if k != self._sz0}

            # Switch to stored 3D LOD bias.
            self._lod_bias_2d = self._lod_bias_2d_slider.value()
            self._controller.update_appearance_field(
                self._visual_model.id, "lod_bias", self._lod_bias_3d
            )
            self._lod_bias_3d_slider.blockSignals(True)
            self._lod_bias_3d_slider.setValue(self._lod_bias_3d)
            self._lod_bias_3d_slider.blockSignals(False)

            self._controller.incoming_events.emit(
                DimsUpdateEvent(
                    source_id=self._controller._id,
                    scene_id=self._scene.id,
                    slice_indices=new_slice,
                    displayed_axes=self._displayed_axes_3d,
                )
            )

            self._active_mode = "3d"
            self._mode_label.setText("Mode: 3D")
            self._toggle_btn.setText("Switch to 2D")
            self._stack.setCurrentIndex(1)

        else:
            # Restore sz0 into slice_indices.
            new_slice = dict(current_slice)
            if self._sz0 is not None:
                new_slice[self._sz0] = self._saved_z

            # Switch to stored 2D LOD bias.
            self._lod_bias_3d = self._lod_bias_3d_slider.value()
            self._controller.update_appearance_field(
                self._visual_model.id, "lod_bias", self._lod_bias_2d
            )
            self._lod_bias_2d_slider.blockSignals(True)
            self._lod_bias_2d_slider.setValue(self._lod_bias_2d)
            self._lod_bias_2d_slider.blockSignals(False)

            self._controller.incoming_events.emit(
                DimsUpdateEvent(
                    source_id=self._controller._id,
                    scene_id=self._scene.id,
                    slice_indices=new_slice,
                    displayed_axes=self._displayed_axes_2d,
                )
            )
            self._active_mode = "2d"
            self._mode_label.setText("Mode: 2D")
            self._toggle_btn.setText("Switch to 3D")
            self._stack.setCurrentIndex(0)

    # ------------------------------------------------------------------

    @property
    def window(self):
        return self._window

    def close_widgets(self) -> None:
        self._canvas_widget.close()
        self._clim_slider.close()
        self._colormap_combo.close()
        self._render_controls.close()


# ---------------------------------------------------------------------------
# Layer 1: ViewerModel builder (no Qt)
# ---------------------------------------------------------------------------


def build_viewer_model(
    zarr_uri: str,
    *,
    perf: StartupPerfTracer | None = None,
) -> tuple:
    """Build a ViewerModel for the viewer without constructing any Qt objects.

    Parameters
    ----------
    zarr_uri : str
        Path or URI to the OME-Zarr store.
    perf : StartupPerfTracer | None, optional
        Optional startup performance tracer.

    Returns
    -------
    tuple[cellier.v2.viewer_model.ViewerModel, _ViewerGeometry]
    """
    import yaozarrs
    from cellier.v2.data.image import OMEZarrImageDataStore
    from cellier.v2.scene.cameras import (
        OrbitCameraController,
        OrthographicCamera,
        PanZoomCameraController,
        PerspectiveCamera,
    )
    from cellier.v2.scene.canvas import Canvas
    from cellier.v2.scene.dims import (
        AxisAlignedSelection,
        CoordinateSystem,
        DimsManager,
    )
    from cellier.v2.scene.scene import Scene
    from cellier.v2.transform import AffineTransform
    from cellier.v2.viewer_model import DataManager, ViewerModel
    from cellier.v2.visuals._image import (
        MultiscaleImageAppearance,
        MultiscaleImageRenderConfig,
        MultiscaleImageVisual,
    )
    from rich.console import Console
    from rich.table import Table

    _perf_mark(perf, "viewer.model.start", zarr_uri=zarr_uri)
    data_store = OMEZarrImageDataStore.from_path(zarr_uri)
    _perf_mark(perf, "viewer.model.data_store_ready", n_levels=data_store.n_levels)

    group = yaozarrs.open_group(data_store.zarr_path)
    ome_image = group.ome_metadata()
    ms = ome_image.multiscales[data_store.multiscale_index]

    n_dims = len(data_store.level_shapes[0])

    # Detect channel axis from OME-Zarr axis metadata.
    channel_axis: int | None = None
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

    vox_shape_spatial = np.array(
        [data_store.level_shapes[0][i] for i in spatial_indices], dtype=np.float64
    )
    world_extents_spatial = vox_shape_spatial * level_0_scale_spatial
    max_extent = float(world_extents_spatial.max())
    depth_range = (max(1.0, max_extent * 0.0001), max_extent * 10.0)

    vox_shape_full = np.array(data_store.level_shapes[0], dtype=np.float64)
    world_max_full = (vox_shape_full - 1) * level_0_scale_full
    world_max_spatial = (vox_shape_spatial - 1) * level_0_scale_spatial

    # Print startup summary.
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
    _perf_mark(perf, "viewer.model.metadata_printed")

    cs = CoordinateSystem(name="world", axis_labels=tuple(data_store.axis_names))
    voxel_to_world = AffineTransform.from_scale_and_translation(
        scale=tuple(level_0_scale_full)
    )

    initial_clim_max = _dtype_clim_max(data_store.dtype)
    slider_decimals = _dtype_decimals(data_store.dtype)

    # axis_ranges covers all axes; dims sliders appear for non-displayed axes.
    axis_ranges = {i: (0.0, round(float(world_max_full[i]))) for i in range(n_dims)}

    def _mid(ax: int) -> float:
        return round(float(world_max_full[ax]) / 2.0)

    # 2D: display last 2 spatial axes; 3D: display last 3 spatial axes.
    if spatial_ndim >= 2:
        displayed_axes_2d: tuple[int, ...] = tuple(spatial_indices[-2:])
    else:
        displayed_axes_2d = tuple(spatial_indices)

    if spatial_ndim >= 3:
        displayed_axes_3d: tuple[int, ...] = tuple(spatial_indices[-3:])
    else:
        displayed_axes_3d = displayed_axes_2d

    _set_2d = set(displayed_axes_2d)
    _set_3d = set(displayed_axes_3d)
    initial_slice_indices_2d = {i: _mid(i) for i in range(n_dims) if i not in _set_2d}
    initial_slice_indices_3d = {i: _mid(i) for i in range(n_dims) if i not in _set_3d}

    # Single visual for both 2D and 3D rendering.
    visual = MultiscaleImageVisual(
        name="volume",
        data_store_id=str(data_store.id),
        level_transforms=data_store.level_transforms,
        appearance=MultiscaleImageAppearance(
            color_map="grays",
            clim=(0.0, initial_clim_max),
            lod_bias=1.0,
            force_level=None,
            frustum_cull=True,
            iso_threshold=initial_clim_max / 2.0,
            render_mode="mip",
            attenuation=1.0,
        ),
        render_config=MultiscaleImageRenderConfig(
            block_size=32,
            gpu_budget_bytes=2048 * 1024**2,
            gpu_budget_bytes_2d=64 * 1024**2,
        ),
        transform=voxel_to_world,
    )

    # Single canvas: orthographic camera for 2D, perspective for 3D.
    canvas = Canvas(
        cameras={
            "2d": OrthographicCamera(
                near_clipping_plane=depth_range[0],
                far_clipping_plane=depth_range[1],
                controller=PanZoomCameraController(enabled=True),
            ),
            "3d": PerspectiveCamera(
                fov=70.0,
                near_clipping_plane=depth_range[0],
                far_clipping_plane=depth_range[1],
                controller=OrbitCameraController(enabled=True),
            ),
        }
    )

    # Scene supports both render modes; starts in 2D.
    scene = Scene(
        name="main",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=displayed_axes_2d,
                slice_indices=initial_slice_indices_2d,
            ),
        ),
        render_modes={"2d", "3d"},
        lighting="none",
        visuals=[visual],
        canvases={canvas.id: canvas},
    )

    viewer_model = ViewerModel(
        data=DataManager(stores={data_store.id: data_store}),
        scenes={scene.id: scene},
    )
    _perf_mark(perf, "viewer.model.ready")

    geometry = _ViewerGeometry(
        spatial_indices=spatial_indices,
        spatial_ndim=spatial_ndim,
        displayed_axes_2d=displayed_axes_2d,
        displayed_axes_3d=displayed_axes_3d,
        world_max_spatial=world_max_spatial,
        voxel_to_world=voxel_to_world,
        axis_ranges=axis_ranges,
        initial_slice_indices_2d=initial_slice_indices_2d,
        initial_slice_indices_3d=initial_slice_indices_3d,
        initial_clim_max=initial_clim_max,
        clim_range=(0.0, initial_clim_max),
        slider_decimals=slider_decimals,
    )
    return viewer_model, geometry


# ---------------------------------------------------------------------------
# Layer 2: Qt bootstrap helper
# ---------------------------------------------------------------------------


def _ensure_qt_app():
    from PySide6.QtWidgets import QApplication

    if app := QApplication.instance():
        return app
    try:
        import IPython

        ip = IPython.get_ipython()
        if ip is not None:
            ip.enable_gui("qt6")
            return QApplication.instance()
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# Layer 3: Non-blocking show (interactive / Jupyter)
# ---------------------------------------------------------------------------


def viewer(
    zarr_uri: str,
    theme: str = "dark",
) -> OmeZarrViewer:
    """Open a viewer window without blocking.

    Intended for interactive use (Jupyter Lab, IPython). The Qt event loop
    must already be running or be startable via IPython's ``enable_gui``; this
    function sets that up automatically. For scripts use :func:`launch_viewer`.

    Parameters
    ----------
    zarr_uri : str
        Path or URI to the OME-Zarr store.
    theme : str
        Registered theme name. Defaults to ``"dark"``.
        Use ``oz_viewer.theme.list_themes()`` to see available themes.

    Returns
    -------
    OmeZarrViewer
        The viewer window object. Keep a reference to prevent garbage collection.
    """
    app = _ensure_qt_app()
    if app is None:
        raise RuntimeError(
            "No Qt event loop is running. "
            "Use launch_viewer() for scripts, or run inside IPython/Jupyter."
        )
    return _build_and_show_viewer(zarr_uri, theme=theme)


# ---------------------------------------------------------------------------
# Layer 4: Private async core
# ---------------------------------------------------------------------------


def _asyncio_exception_handler(context: dict) -> None:
    import traceback

    exc = context.get("exception")
    if isinstance(exc, asyncio.CancelledError):
        return

    msg = context.get("message", "unhandled exception in asyncio")
    task = context.get("task")
    handle = context.get("handle")
    source = (
        f"task {task._name}" if task else (repr(handle) if handle else "unknown source")
    )
    print(f"[asyncio] {msg} from {source}")
    if exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__)


async def _run_viewer_async(
    zarr_uri: str,
    theme: str = "dark",
    *,
    perf: StartupPerfTracer | None = None,
) -> None:
    import asyncio as _asyncio

    from PySide6.QtWidgets import QApplication

    _asyncio.get_event_loop().set_exception_handler(_asyncio_exception_handler)
    _perf_mark(perf, "viewer.async.start", theme=theme)

    v = _build_and_show_viewer(zarr_uri, theme=theme, perf=perf)
    _perf_mark(perf, "viewer.async.build_complete")

    app = QApplication.instance()
    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    app.aboutToQuit.connect(v.close_widgets)
    await close_event.wait()


# ---------------------------------------------------------------------------
# Layer 5: Blocking launcher (scripts / CLI)
# ---------------------------------------------------------------------------


def launch_viewer(
    zarr_uri: str,
    theme: str = "dark",
    *,
    perf: StartupPerfTracer | None = None,
) -> None:
    """Open a viewer window and block until it is closed.

    Creates a ``QApplication`` if one does not already exist, then runs the
    Qt + asyncio event loop via ``QtAsyncio``. Intended for scripts and the
    CLI. For interactive/Jupyter use, call :func:`viewer` instead.

    Parameters
    ----------
    zarr_uri : str
        Path or URI to the OME-Zarr store.
    theme : str
        Registered theme name. Defaults to ``"dark"``.
        Use ``oz_viewer.theme.list_themes()`` to see available themes.
    perf : StartupPerfTracer | None, optional
        Optional startup performance tracer.
    """
    import sys

    import fsspec.asyn as _fsspec_asyn
    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    _fsspec_asyn.get_loop()

    _perf_mark(perf, "viewer.launch.start", theme=theme)
    app = QApplication.instance() or QApplication([sys.argv[0]])  # noqa: F841
    _perf_mark(perf, "viewer.launch.qapp_ready")
    QtAsyncio.run(
        _run_viewer_async(zarr_uri, theme=theme, perf=perf),
        handle_sigint=True,
    )


# ---------------------------------------------------------------------------
# Shared builder (used by both viewer() and _run_viewer_async())
# ---------------------------------------------------------------------------


def _build_and_show_viewer(
    zarr_uri: str,
    theme: str = "dark",
    *,
    perf: StartupPerfTracer | None = None,
) -> OmeZarrViewer:
    """Build the full viewer from a zarr URI and show the window."""
    from PySide6.QtWidgets import QApplication

    from oz_viewer.theme import apply_theme

    _perf_mark(perf, "viewer.build.start", theme=theme)
    apply_theme(QApplication.instance(), theme)
    _perf_mark(perf, "viewer.build.theme_applied")

    from cellier.v2.controller import CellierController
    from cellier.v2.gui._scene import QtCanvasWidget
    from cellier.v2.render._config import (
        RenderManagerConfig,
        SlicingConfig,
        TemporalAccumulationConfig,
    )

    viewer_model, geometry = build_viewer_model(zarr_uri, perf=perf)
    _perf_mark(perf, "viewer.build.model_ready")

    controller = CellierController.from_model(
        viewer_model,
        render_config=RenderManagerConfig(
            slicing=SlicingConfig(batch_size=32, render_every=4),
            temporal=TemporalAccumulationConfig(enabled=False),
        ),
        widget_parent=None,
    )
    _perf_mark(perf, "viewer.build.controller_ready")

    scene = controller.get_scene_by_name("main")
    visual_model = next(iter(scene.visuals))

    canvas_id = controller.get_canvas_ids(scene.id)[0]
    canvas_view = controller.get_canvas_view(canvas_id)
    canvas_widget = QtCanvasWidget.from_scene_and_canvas(
        scene, canvas_view, axis_ranges=geometry.axis_ranges
    )
    controller.connect_widget(
        canvas_widget.dims_sliders,
        subscription_specs=canvas_widget.dims_sliders.subscription_specs(),
    )
    _perf_mark(perf, "viewer.build.canvas_widget_ready")

    v = OmeZarrViewer(
        controller=controller,
        scene=scene,
        canvas_widget=canvas_widget,
        visual_model=visual_model,
        geometry=geometry,
    )

    # Perf: track time to first paint.
    if perf is not None and perf.enabled:
        from PySide6.QtCore import QEvent, QObject, QTimer

        settled_timer = QTimer()
        settled_timer.setSingleShot(True)
        settled_timer.setInterval(300)

        def _on_settled() -> None:
            _perf_mark(perf, "viewer.canvas.startup_settled", quiet_ms=300)
            perf.report_rich_table()

        settled_timer.timeout.connect(_on_settled)

        class _PaintTracker(QObject):
            def eventFilter(self, watched, event):
                if event.type() == QEvent.Type.Paint:
                    _perf_mark(perf, "viewer.canvas.first_paint")
                    settled_timer.start()
                    canvas_widget.widget.removeEventFilter(self)
                return False

        paint_tracker = _PaintTracker()
        canvas_widget.widget.installEventFilter(paint_tracker)
        v._startup_perf_objects = (paint_tracker, settled_timer)

    v.window.show()
    _perf_mark(perf, "viewer.window.show_called")

    controller.fit_camera(scene.id)
    controller.reslice_scene(scene.id)

    return v
