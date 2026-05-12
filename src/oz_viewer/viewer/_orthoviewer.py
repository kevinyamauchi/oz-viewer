"""OME-Zarr orthoviewer: 4-panel viewer (XY, XZ, YZ, 3D)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    from PySide6 import QtWidgets

    from oz_viewer._perf import StartupPerfTracer

from oz_viewer.viewer._utils import (
    _asyncio_exception_handler,
    _dtype_clim_max,
    _dtype_decimals,
    _ensure_qt_app,
    _perf_mark,
)
from oz_viewer.viewer._widgets import (
    _DEFAULT_COLORMAPS,
    _MultiVisualClimSlider,
    _MultiVisualColormapCombo,
    _MultiVisualLodBiasSlider,
    build_channel_list_widget,
)

# ---------------------------------------------------------------------------
# Slider color styles for each 2D panel
# ---------------------------------------------------------------------------


def _make_slider_style(color_a: str, color_b: str) -> str:
    """Return a widget-level QSS string for an axis-identification dims slider.

    All sub-controls are styled so that the Fusion style engine does not
    override any individual element.  Structural colors (groove, handle,
    add-page) use ``palette()`` references so they adapt to whatever theme is
    active.  The ``sub-page`` gradient encodes the axis identity and is
    intentionally fixed regardless of theme.

    Parameters
    ----------
    color_a : str
        CSS color for the left/start stop of the sub-page gradient.
    color_b : str
        CSS color for the right/end stop of the sub-page gradient.

    Returns
    -------
    str
        QSS string suitable for ``widget.setStyleSheet()``.
    """
    return f"""
QSlider::groove:horizontal {{
    height: 6px;
    background: palette(mid);
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    height: 6px;
    background: qlineargradient(x1:0, y1:0.2, x2:1, y2:1,
        stop:0 {color_a}, stop:1 {color_b});
    border-radius: 3px;
}}
QSlider::add-page:horizontal {{
    height: 6px;
    background: palette(mid);
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 20px;
    height: 20px;
    margin: -7px 0;
    border-radius: 10px;
    background: #2a82da;
    border: 1px solid palette(shadow);
}}
QSlider::handle:horizontal:hover {{
    background: palette(light);
}}
QSlider:horizontal {{ min-height: 20px; }}
QLabel {{ font-size: 12px; }}
"""


_SLIDER_STYLE_XY = _make_slider_style("#bbf", "#55f")
_SLIDER_STYLE_XZ = _make_slider_style("#bfb", "#3a3")
_SLIDER_STYLE_YZ = _make_slider_style("#fdb", "#c60")

# Colors matching the slider gradients.
_PLANE_COLOR_XY = (0.33, 0.33, 1.00)  # blue
_PLANE_COLOR_XZ = (0.23, 0.67, 0.23)  # green
_PLANE_COLOR_YZ = (0.80, 0.40, 0.00)  # orange

_AXIS_FRACTION: float = 0.15
_AXIS_3D_LENGTH_FRACTION: float = 0.12
_AXIS_3D_CUBE_SIDE_FRACTION: float = 0.024
_AXIS_3D_PRISM_CROSS_SECTION_FRACTION: float = 0.020
_AXIS_3D_CUBE_COLOUR: tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0)
_N_FACES_PER_BOX: int = 12

_TRANSPARENCY_MODES: list[str] = ["weighted_blend", "weighted_solid", "blend", "add"]


class _WorldGeometry(NamedTuple):
    """Geometry derived from OME-Zarr metadata; returned by build_ortho_viewer_model."""

    spatial_indices: list[int]
    spatial_ndim: int
    world_max_spatial: np.ndarray
    spatial_mid_world: dict[int, float]
    voxel_to_world: object
    axis_ranges: dict[int, tuple[float, float]]
    excluded_axes: set[int] | None
    initial_clim_max: float
    clim_range: tuple[float, float]
    slider_decimals: int
    n_channels: int


@dataclass
class _VolTransparencyProfile:
    transparency_mode: str
    opacity: float


_ISO_DEFAULT_TRANSPARENCY = _VolTransparencyProfile("weighted_blend", 0.3)
_MIP_DEFAULT_TRANSPARENCY = _VolTransparencyProfile("weighted_blend", 1.0)


@dataclass
class _VisualRenderProfile:
    render_order: int
    depth_test: bool
    depth_write: bool
    transparency_mode: str
    opacity: float


_ISO_PLANE_PROFILE = _VisualRenderProfile(
    render_order=0,
    depth_test=True,
    depth_write=True,
    transparency_mode="blend",
    opacity=1.0,
)
_MIP_PLANE_PROFILE = _VisualRenderProfile(
    render_order=1,
    depth_test=False,
    depth_write=True,
    transparency_mode="weighted_blend",
    opacity=0.99,
)
_ISO_AXES_PROFILE = _VisualRenderProfile(
    render_order=1,
    depth_test=True,
    depth_write=True,
    transparency_mode="blend",
    opacity=1.0,
)
_MIP_AXES_PROFILE = _VisualRenderProfile(
    render_order=2,
    depth_test=False,
    depth_write=False,
    transparency_mode="blend",
    opacity=1.0,
)

# Multichannel MIP uses the same plane/axes settings as ISO so that the plane
# composites on top of the resolved OIT volume rather than being mixed into it.
# ---------------------------------------------------------------------------
# Volume transparency manager
# ---------------------------------------------------------------------------


class _VolTransparencyManager:
    """Manages transparency profiles for each render mode via the model layer."""

    def __init__(
        self,
        controller,
        vol_visual_id,
        *,
        vol_is_multichannel: bool = False,
        plane_visual_id=None,
        axis_visual_ids: list | None = None,
        initial_mode: str = "iso",
    ) -> None:
        self._controller = controller
        self._vol_visual_id = vol_visual_id
        self._vol_is_multichannel = vol_is_multichannel
        self._plane_visual_id = plane_visual_id
        self._axis_visual_ids: list = axis_visual_ids or []
        self._vol_profiles: dict[str, _VolTransparencyProfile] = {
            "iso": _VolTransparencyProfile(
                transparency_mode=_ISO_DEFAULT_TRANSPARENCY.transparency_mode,
                opacity=_ISO_DEFAULT_TRANSPARENCY.opacity,
            ),
            "mip": _VolTransparencyProfile(
                transparency_mode=_MIP_DEFAULT_TRANSPARENCY.transparency_mode,
                opacity=_MIP_DEFAULT_TRANSPARENCY.opacity,
            ),
        }
        self._plane_profiles: dict[str, _VisualRenderProfile] = {
            "iso": _ISO_PLANE_PROFILE,
            "mip": _MIP_PLANE_PROFILE,
        }
        self._axes_profiles: dict[str, _VisualRenderProfile] = {
            "iso": _ISO_AXES_PROFILE,
            "mip": _MIP_AXES_PROFILE,
        }
        self._current_mode = (
            initial_mode if initial_mode in self._vol_profiles else "iso"
        )

    @property
    def current_mode(self) -> str:
        return self._current_mode

    @property
    def current_profile(self) -> _VolTransparencyProfile:
        return self._vol_profiles[self._current_mode]

    def _apply_profile_to_mesh(self, visual_id, profile: _VisualRenderProfile) -> None:
        if visual_id is None:
            return
        c = self._controller
        c.update_appearance_field(visual_id, "render_order", profile.render_order)
        c.update_appearance_field(visual_id, "depth_test", profile.depth_test)
        c.update_appearance_field(visual_id, "depth_write", profile.depth_write)
        c.update_appearance_field(
            visual_id, "transparency_mode", profile.transparency_mode
        )
        c.update_appearance_field(visual_id, "opacity", profile.opacity)

    def _apply_vol_profile(self) -> None:
        """Apply the current volume profile via the model layer.

        For multichannel visuals each ChannelAppearance is mutated directly
        since they share no single .appearance object; the psygnal bridge then
        routes the changes to the render layer.  For single-channel visuals
        update_appearance_field is used so source-id threading is preserved.
        """
        vid = self._vol_visual_id
        if vid is None:
            return
        vol = self.current_profile
        depth_write = vol.transparency_mode == "weighted_solid"
        if self._vol_is_multichannel:
            visual = self._controller.get_visual_model(vid)
            for ch in visual.channels.values():
                ch.transparency_mode = vol.transparency_mode
                ch.opacity = vol.opacity
        else:
            c = self._controller
            c.update_appearance_field(vid, "transparency_mode", vol.transparency_mode)
            c.update_appearance_field(vid, "opacity", vol.opacity)
            c.update_appearance_field(vid, "depth_write", depth_write)

    def apply(self) -> None:
        self._apply_vol_profile()
        plane_profile = self._plane_profiles[self._current_mode]
        self._apply_profile_to_mesh(self._plane_visual_id, plane_profile)
        axes_profile = self._axes_profiles[self._current_mode]
        for vid in self._axis_visual_ids:
            self._apply_profile_to_mesh(vid, axes_profile)

    def on_render_mode_changed(self, new_mode: str) -> None:
        if new_mode in self._vol_profiles:
            self._current_mode = new_mode
        elif new_mode.startswith("mip"):
            self._current_mode = "mip"
        else:
            self._current_mode = "iso"
        self.apply()

    def update_transparency_mode(self, transparency_mode: str) -> None:
        self.current_profile.transparency_mode = transparency_mode
        self.apply()

    def update_opacity(self, opacity: float) -> None:
        self.current_profile.opacity = opacity
        self.apply()


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------


class OmeZarrOrthoViewer:
    """4-panel orthoviewer window: XY, XZ, YZ slices and a 3D volume."""

    def __init__(
        self,
        controller,
        scenes: dict,
        visuals: dict,
        canvas_widgets: dict,
        clim_range: tuple[float, float],
        slider_decimals: int = 2,
        axes_2d_overlay_ids: list | None = None,
        orient_3d_visual_ids: list | None = None,
        plane_visual=None,
        plane_store=None,
        initial_plane_opacity: float = 0.4,
        transparency_manager: _VolTransparencyManager | None = None,
        mc_transparency_manager: _VolTransparencyManager | None = None,
        channel_axis: int | None = None,
        n_channels: int = 0,
        spatial_ndim: int = 3,
        data_store=None,
        voxel_to_world=None,
        initial_mode: str = "single",
        initial_multichannel_visual_ids: list | None = None,
        initial_channel_appearances: dict | None = None,
        world_max_spatial=None,
        channel_syncer: _ChannelAxisSyncer | None = None,
    ):
        from PySide6 import QtCore, QtWidgets
        from PySide6.QtWidgets import QStackedWidget

        self._controller = controller
        self._scenes = scenes
        self._visuals = visuals
        self._canvas_widgets = canvas_widgets
        self._channel_axis = channel_axis
        self._n_channels = n_channels
        self._spatial_ndim = spatial_ndim
        self._data_store = data_store
        self._voxel_to_world = voxel_to_world
        self._clim_range = clim_range
        self._slider_decimals = slider_decimals
        self._mode = initial_mode
        # Single-channel visual controls (None until built in SC mode).
        self._2d_clim: _MultiVisualClimSlider | None = None
        self._2d_colormap: _MultiVisualColormapCombo | None = None
        self._2d_lod_bias: _MultiVisualLodBiasSlider | None = None
        self._3d_clim = None
        self._3d_colormap = None
        self._3d_render = None
        self._group_3d = None
        # Multichannel state.
        self._multichannel_visual_ids: list | None = initial_multichannel_visual_ids
        self._channel_appearances: dict | None = initial_channel_appearances
        # Tracks whether SC visuals + controls have been built.
        # True at init when starting in single mode; False when starting in
        # multichannel.
        self._single_channel_built = initial_mode == "single"
        # Store overlay/plane info for lazy SC page construction.
        self._axes_2d_overlay_ids: list = axes_2d_overlay_ids or []
        self._orient_3d_visual_ids: list = orient_3d_visual_ids or []
        self._plane_visual = plane_visual
        self._plane_store = plane_store
        self._initial_plane_opacity = initial_plane_opacity
        self._transparency_manager = transparency_manager
        self._mc_transparency_manager = mc_transparency_manager
        self._world_max_spatial = world_max_spatial
        self._channel_syncer = channel_syncer

        if initial_mode == "single" and visuals.get("xy") is not None:
            self._init_sc_controls()

        # ── Qt window ─────────────────────────────────────────────────────────
        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("OME-Zarr Orthoviewer")
        self._window.resize(1400, 900)

        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        grid_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_widget)
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)

        panels = [
            (0, 0, "xy", "XY  (slice Z)"),
            (0, 1, "xz", "XZ  (slice Y)"),
            (1, 0, "yz", "YZ  (slice X)"),
            (1, 1, "vol", "3D Volume"),
        ]
        for row, col, key, label in panels:
            cell = QtWidgets.QWidget()
            cell_layout = QtWidgets.QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(0)

            lbl = QtWidgets.QLabel(label)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 11px; padding: 2px;")
            cell_layout.addWidget(lbl)
            cell_layout.addWidget(canvas_widgets[key].widget, stretch=1)
            grid.addWidget(cell, row, col)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        root_layout.addWidget(panel)
        root_layout.addWidget(grid_widget, stretch=1)

        # ── Mode toggle button (only in multichannel-capable mode) ────────────
        if channel_axis is not None:
            _btn_text = (
                "Switch to Single Channel"
                if initial_mode == "multichannel"
                else "Switch to Multichannel"
            )
            toggle_btn = QtWidgets.QPushButton(_btn_text)
            toggle_btn.clicked.connect(self._on_mode_toggle)
            panel_layout.addWidget(toggle_btn)
            self._toggle_btn = toggle_btn
        else:
            self._toggle_btn = None

        # ── Stacked widget: page 0 = single-channel, page 1 = multichannel ───
        self._stack = QStackedWidget()
        panel_layout.addWidget(self._stack)

        # Page 0 — single-channel controls.
        page0 = QtWidgets.QWidget()
        self._page0_layout = QtWidgets.QVBoxLayout(page0)
        self._page0_layout.setContentsMargins(0, 0, 0, 0)
        self._page0_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        if initial_mode == "single":
            self._populate_sc_page(self._page0_layout)
        self._stack.addWidget(page0)

        # Page 1 — multichannel controls.
        self._multichannel_page = QtWidgets.QWidget()
        self._mc_page_layout = QtWidgets.QVBoxLayout(self._multichannel_page)
        self._mc_page_layout.setContentsMargins(0, 0, 0, 0)
        self._mc_page_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        if initial_mode == "multichannel" and self._channel_appearances is not None:
            self._populate_mc_page(self._mc_page_layout)
        self._stack.addWidget(self._multichannel_page)

        self._stack.setCurrentIndex(1 if initial_mode == "multichannel" else 0)

    # ------------------------------------------------------------------
    # SC control initialisation helpers
    # ------------------------------------------------------------------

    def _init_sc_controls(self) -> None:
        """Create and connect single-channel slider/combo widgets."""
        from cellier.v2.gui.visuals._colormap import QtColormapComboBox
        from cellier.v2.gui.visuals._contrast_limits import QtClimRangeSlider
        from cellier.v2.gui.visuals._image import QtVolumeRenderControls

        visuals = self._visuals
        controller = self._controller
        clim_range = self._clim_range
        slider_decimals = self._slider_decimals

        _2d_visual_ids = [
            visuals[k].id for k in ("xy", "xz", "yz") if visuals.get(k) is not None
        ]
        self._2d_clim = _MultiVisualClimSlider(
            _2d_visual_ids,
            clim_range=clim_range,
            initial_clim=visuals["xy"].appearance.clim,
            decimals=slider_decimals,
        )
        controller.connect_widget(
            self._2d_clim, subscription_specs=self._2d_clim.subscription_specs()
        )
        self._2d_colormap = _MultiVisualColormapCombo(
            _2d_visual_ids,
            initial_colormap=visuals["xy"].appearance.color_map,
        )
        controller.connect_widget(
            self._2d_colormap,
            subscription_specs=self._2d_colormap.subscription_specs(),
        )
        self._2d_lod_bias = _MultiVisualLodBiasSlider(
            _2d_visual_ids,
            initial_lod_bias=visuals["xy"].appearance.lod_bias,
        )
        controller.connect_widget(
            self._2d_lod_bias,
            subscription_specs=self._2d_lod_bias.subscription_specs(),
        )

        if visuals.get("vol") is not None:
            vol_id = visuals["vol"].id
            self._3d_clim = QtClimRangeSlider(
                vol_id,
                clim_range=clim_range,
                initial_clim=visuals["vol"].appearance.clim,
                decimals=slider_decimals,
            )
            controller.connect_widget(
                self._3d_clim, subscription_specs=self._3d_clim.subscription_specs()
            )
            self._3d_colormap = QtColormapComboBox(
                vol_id,
                initial_colormap=visuals["vol"].appearance.color_map,
            )
            self._3d_colormap.add_colormaps(_DEFAULT_COLORMAPS)
            controller.connect_widget(
                self._3d_colormap,
                subscription_specs=self._3d_colormap.subscription_specs(),
            )
            self._3d_render = QtVolumeRenderControls(
                vol_id,
                dtype_max=clim_range[1],
                initial_render_mode=visuals["vol"].appearance.render_mode,
                initial_threshold=visuals["vol"].appearance.iso_threshold,
                decimals=slider_decimals,
            )
            controller.connect_widget(
                self._3d_render,
                subscription_specs=self._3d_render.subscription_specs(),
            )

    def _populate_sc_page(self, page0_layout) -> None:
        """Fill the page-0 (single-channel) panel with controls."""
        from PySide6 import QtWidgets

        controller = self._controller
        axes_2d_overlay_ids = self._axes_2d_overlay_ids
        orient_3d_visual_ids = self._orient_3d_visual_ids
        plane_visual = self._plane_visual
        plane_store = self._plane_store
        initial_plane_opacity = self._initial_plane_opacity
        transparency_manager = self._transparency_manager

        group_2d = QtWidgets.QGroupBox("2D Rendering")
        layout_2d = QtWidgets.QVBoxLayout(group_2d)

        if self._2d_clim is not None:
            clim_2d_box = QtWidgets.QGroupBox("Contrast limits")
            QtWidgets.QVBoxLayout(clim_2d_box).addWidget(self._2d_clim.widget)
            layout_2d.addWidget(clim_2d_box)

        if self._2d_colormap is not None:
            cmap_2d_box = QtWidgets.QGroupBox("Colormap")
            QtWidgets.QVBoxLayout(cmap_2d_box).addWidget(self._2d_colormap.widget)
            layout_2d.addWidget(cmap_2d_box)

        if self._2d_lod_bias is not None:
            lod_2d_box = QtWidgets.QGroupBox("Fine-coarse tile bias")
            lod_2d_box.setToolTip("Bigger numbers render more coarse")
            QtWidgets.QVBoxLayout(lod_2d_box).addWidget(self._2d_lod_bias.widget)
            layout_2d.addWidget(lod_2d_box)

        if axes_2d_overlay_ids:
            from PySide6.QtWidgets import QCheckBox

            axes_2d_cb = QCheckBox("Show orientation axes")
            axes_2d_cb.setChecked(True)

            def _on_axes_2d_toggled(checked: bool) -> None:
                for oid in axes_2d_overlay_ids:
                    controller.set_overlay_visible(oid, checked)

            axes_2d_cb.toggled.connect(_on_axes_2d_toggled)
            layout_2d.addWidget(axes_2d_cb)

        page0_layout.addWidget(group_2d)

        if self._3d_clim is not None:
            from PySide6.QtWidgets import QCheckBox

            group_3d = QtWidgets.QGroupBox("3D Rendering")
            layout_3d = QtWidgets.QVBoxLayout(group_3d)
            self._group_3d = group_3d

            clim_3d_box = QtWidgets.QGroupBox("Contrast limits")
            QtWidgets.QVBoxLayout(clim_3d_box).addWidget(self._3d_clim.widget)
            layout_3d.addWidget(clim_3d_box)

            cmap_3d_box = QtWidgets.QGroupBox("Colormap")
            QtWidgets.QVBoxLayout(cmap_3d_box).addWidget(self._3d_colormap.widget)
            layout_3d.addWidget(cmap_3d_box)

            render_3d_box = QtWidgets.QGroupBox("Render mode")
            QtWidgets.QVBoxLayout(render_3d_box).addWidget(self._3d_render.widget)
            layout_3d.addWidget(render_3d_box)

            if transparency_manager is not None:

                def _on_render_mode_event(event) -> None:
                    if hasattr(event, "field") and event.field == "render_mode":
                        transparency_manager.on_render_mode_changed(event.value)

                self._3d_render.changed.connect(_on_render_mode_event)

            if orient_3d_visual_ids:
                orient_3d_cb = QCheckBox("Show orientation axes")
                orient_3d_cb.setChecked(True)

                def _on_orient_3d_toggled(checked: bool) -> None:
                    for vid in orient_3d_visual_ids:
                        controller.set_visual_visible(vid, checked)

                orient_3d_cb.toggled.connect(_on_orient_3d_toggled)
                layout_3d.addWidget(orient_3d_cb)

            page0_layout.addWidget(group_3d)

        if plane_visual is not None:
            from PySide6.QtCore import Qt
            from superqt import QLabeledDoubleSlider

            group_planes = QtWidgets.QGroupBox("Slice Overlay")
            layout_planes = QtWidgets.QVBoxLayout(group_planes)

            plane_opacity_label = QtWidgets.QLabel("Slice opacity")
            plane_opacity_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
            plane_opacity_slider.setRange(0.0, 1.0)
            plane_opacity_slider.setValue(initial_plane_opacity)
            plane_opacity_slider.setDecimals(2)

            def _on_plane_opacity_changed(value: float) -> None:
                if plane_visual is not None:
                    controller.update_appearance_field(
                        plane_visual.id, "opacity", value
                    )
                if plane_store is not None:
                    plane_store.colors = _make_plane_colors(value)
                    controller.reslice_visual(plane_visual.id)
                if orient_3d_visual_ids:
                    for vid in orient_3d_visual_ids:
                        controller.update_appearance_field(vid, "opacity", value)

            plane_opacity_slider.valueChanged.connect(_on_plane_opacity_changed)
            layout_planes.addWidget(plane_opacity_label)
            layout_planes.addWidget(plane_opacity_slider)

            _on_plane_opacity_changed(initial_plane_opacity)

            page0_layout.addWidget(group_planes)

        page0_layout.addStretch()

    def _populate_mc_page(self, mc_page_layout) -> None:
        """Fill the page-1 (multichannel) panel with per-channel controls."""
        from PySide6.QtCore import Qt as _Qt

        mc_page_layout.setAlignment(_Qt.AlignmentFlag.AlignTop)
        mc_page_layout.addWidget(
            build_channel_list_widget(
                self._channel_appearances, self._clim_range, self._slider_decimals
            )
        )
        if self._spatial_ndim == 3:
            mc_page_layout.addWidget(self._build_mc_3d_group())
        mc_page_layout.addStretch()

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _on_mode_toggle(self) -> None:
        to_multichannel = self._mode == "single"

        if to_multichannel:
            # Single → Multichannel: build MC visuals lazily if not yet done.
            if self._multichannel_visual_ids is None:
                self._build_multichannel()
        else:
            # Multichannel → Single: build SC visuals lazily if not yet done.
            if not self._single_channel_built:
                self._build_single_channel()

        # Toggle single-channel visual visibility.
        sc_keys = ["xy"] + (["xz", "yz", "vol"] if self._spatial_ndim == 3 else [])
        for key in sc_keys:
            v = self._visuals.get(key)
            if v is not None:
                self._controller.update_appearance_field(
                    v.id, "visible", not to_multichannel
                )

        # Toggle multichannel visual visibility.
        if self._multichannel_visual_ids:
            for vid in self._multichannel_visual_ids:
                self._controller.set_visual_visible(vid, to_multichannel)

        # Update scene dims to add/remove the channel axis from slice_indices.
        if self._channel_axis is not None:
            dim_scene_keys = ["xy"] + (["xz", "yz"] if self._spatial_ndim == 3 else [])
            for key in dim_scene_keys:
                scene = self._scenes[key]
                current = dict(scene.dims.selection.slice_indices)
                if to_multichannel:
                    current.pop(self._channel_axis, None)
                else:
                    current[self._channel_axis] = 0
                self._controller.update_slice_indices(scene.id, current)

        # Show/hide the channel slider in each 2D canvas and enable/disable sync.
        if self._channel_axis is not None:
            nds: set[int] = {self._channel_axis} if to_multichannel else set()
            for key in ("xy", "xz", "yz"):
                cw = self._canvas_widgets.get(key)
                if cw is not None:
                    cw.dims_sliders.non_displayed_sliders = nds
            if self._channel_syncer is not None:
                self._channel_syncer.enabled = not to_multichannel

        # Show/hide the 3D controls group (only relevant for 3-D spatial data).
        if self._group_3d is not None:
            self._group_3d.setVisible(not to_multichannel)

        # Apply the correct transparency manager for the target mode.
        if to_multichannel and self._mc_transparency_manager is not None:
            self._mc_transparency_manager.apply()
        elif not to_multichannel and self._transparency_manager is not None:
            self._transparency_manager.apply()

        self._stack.setCurrentIndex(1 if to_multichannel else 0)
        self._mode = "multichannel" if to_multichannel else "single"
        if self._toggle_btn is not None:
            self._toggle_btn.setText(
                "Switch to Single Channel"
                if to_multichannel
                else "Switch to Multichannel"
            )

    # ------------------------------------------------------------------
    # Lazy multichannel build (single→multichannel, called at most once)
    # ------------------------------------------------------------------

    def _build_multichannel(self) -> None:
        from cellier.v2.visuals._channel_appearance import ChannelAppearance
        from cellier.v2.visuals._image import MultiscaleImageRenderConfig

        initial_clim_max = self._clim_range[1]
        colormaps = _DEFAULT_COLORMAPS

        self._channel_appearances = {
            i: ChannelAppearance(
                color_map=colormaps[i % len(colormaps)],
                clim=(0.0, initial_clim_max),
            )
            for i in range(self._n_channels)
        }

        render_config = MultiscaleImageRenderConfig(
            block_size=32,
            gpu_budget_bytes=512 * 1024**2,
            gpu_budget_bytes_2d=64 * 1024**2,
        )

        scene_keys = ["xy"] + (["xz", "yz", "vol"] if self._spatial_ndim == 3 else [])
        self._multichannel_visual_ids = []
        _lazy_mc_vol_visual_id = None
        for key in scene_keys:
            scene = self._scenes[key]
            mc_visual = self._controller.add_multichannel_image_multiscale(
                data=self._data_store,
                scene_id=scene.id,
                channel_axis=self._channel_axis,
                channels=self._channel_appearances,
                name=f"multichannel_{key}",
                render_config=render_config,
                transform=self._voxel_to_world,
            )
            # Hide immediately; the toggle will show it.
            for ch in mc_visual.channels.values():
                ch.visible = False
            self._multichannel_visual_ids.append(mc_visual.id)
            if key == "vol":
                _lazy_mc_vol_visual_id = mc_visual.id

        # Build the MC transparency manager if we now have a 3D MC visual and
        # the plane/axis model IDs are already in place.
        if _lazy_mc_vol_visual_id is not None and self._mc_transparency_manager is None:
            plane_id = self._plane_visual.id if self._plane_visual is not None else None
            self._mc_transparency_manager = _VolTransparencyManager(
                self._controller,
                _lazy_mc_vol_visual_id,
                vol_is_multichannel=True,
                plane_visual_id=plane_id,
                axis_visual_ids=list(self._orient_3d_visual_ids),
                initial_mode="mip",
            )

        self._populate_mc_page(self._mc_page_layout)

    # ------------------------------------------------------------------
    # Lazy single-channel build (multichannel→single, called at most once)
    # ------------------------------------------------------------------

    def _build_single_channel(self) -> None:
        """Hot-add SC visuals and build the page-0 controls panel."""
        from cellier.v2.visuals._image import (
            MultiscaleImageAppearance,
            MultiscaleImageRenderConfig,
        )

        initial_clim_max = self._clim_range[1]
        coarsest_level = self._data_store.n_levels - 1

        common_appearance = MultiscaleImageAppearance(
            color_map="viridis",
            clim=(0.0, initial_clim_max),
            lod_bias=1.0,
            force_level=None,
            frustum_cull=True,
            iso_threshold=0.2,
            render_mode="mip",
        )
        common_render_config = MultiscaleImageRenderConfig(
            block_size=32,
            gpu_budget_bytes=512 * 1024**2,
            gpu_budget_bytes_2d=64 * 1024**2,
        )

        sc_2d_keys = ["xy"] + (["xz", "yz"] if self._spatial_ndim == 3 else [])
        for key in sc_2d_keys:
            scene = self._scenes[key]
            visual = self._controller.add_image_multiscale(
                data=self._data_store,
                scene_id=scene.id,
                appearance=common_appearance,
                name=f"{key}_volume",
                render_config=common_render_config,
                transform=self._voxel_to_world,
            )
            self._controller.update_appearance_field(visual.id, "visible", False)
            self._visuals[key] = visual

        if self._spatial_ndim == 3:
            vol_appearance = MultiscaleImageAppearance(
                color_map="white",
                clim=(0.0, initial_clim_max),
                lod_bias=1.0,
                force_level=coarsest_level,
                frustum_cull=False,
                iso_threshold=initial_clim_max / 2.0,
                render_mode="iso",
            )
            vol_render_config = MultiscaleImageRenderConfig(
                block_size=32,
                gpu_budget_bytes=2048 * 1024**2,
                gpu_budget_bytes_2d=64 * 1024**2,
            )
            vol_visual = self._controller.add_image_multiscale(
                data=self._data_store,
                scene_id=self._scenes["vol"].id,
                appearance=vol_appearance,
                name="vol_volume",
                render_config=vol_render_config,
                transform=self._voxel_to_world,
            )
            self._controller.update_appearance_field(vol_visual.id, "visible", False)
            self._visuals["vol"] = vol_visual

            plane_id = self._plane_visual.id if self._plane_visual is not None else None
            self._transparency_manager = _VolTransparencyManager(
                self._controller,
                vol_visual.id,
                plane_visual_id=plane_id,
                axis_visual_ids=list(self._orient_3d_visual_ids),
            )

        self._init_sc_controls()
        self._populate_sc_page(self._page0_layout)
        self._single_channel_built = True

    def _build_mc_3d_group(self) -> QtWidgets.QGroupBox:
        """Build a '3D Rendering' panel for multichannel mode, mirroring SC mode."""
        from PySide6 import QtWidgets
        from PySide6.QtCore import Qt
        from superqt import QLabeledDoubleSlider

        group = QtWidgets.QGroupBox("3D Rendering")
        layout = QtWidgets.QVBoxLayout(group)

        tm = self._mc_transparency_manager

        # Transparency mode combo
        mode_box = QtWidgets.QGroupBox("Transparency mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_box)
        mode_combo = QtWidgets.QComboBox()
        for m in _TRANSPARENCY_MODES:
            mode_combo.addItem(m)
        if tm is not None:
            mode_combo.setCurrentText(tm.current_profile.transparency_mode)

        def _on_mode_combo(text: str) -> None:
            if tm is not None:
                tm.update_transparency_mode(text)

        mode_combo.currentTextChanged.connect(_on_mode_combo)
        mode_layout.addWidget(mode_combo)
        layout.addWidget(mode_box)

        # Volume opacity slider
        opacity_box = QtWidgets.QGroupBox("Volume opacity")
        opacity_layout = QtWidgets.QVBoxLayout(opacity_box)
        opacity_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        opacity_slider.setRange(0.0, 1.0)
        opacity_slider.setSingleStep(0.05)
        opacity_slider.setDecimals(2)
        if tm is not None:
            opacity_slider.setValue(tm.current_profile.opacity)

        def _on_opacity(value: float) -> None:
            if tm is not None:
                tm.update_opacity(value)

        opacity_slider.valueChanged.connect(_on_opacity)
        opacity_layout.addWidget(opacity_slider)
        layout.addWidget(opacity_box)

        # Plane opacity slider (shared with SC mode)
        if self._plane_visual is not None:
            plane_box = QtWidgets.QGroupBox("Slice opacity")
            plane_layout = QtWidgets.QVBoxLayout(plane_box)
            plane_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
            plane_slider.setRange(0.0, 1.0)
            plane_slider.setSingleStep(0.05)
            plane_slider.setDecimals(2)
            plane_slider.setValue(self._initial_plane_opacity)

            def _on_plane_opacity(value: float) -> None:
                self._controller.update_appearance_field(
                    self._plane_visual.id, "opacity", value
                )
                if self._plane_store is not None:
                    self._plane_store.colors = _make_plane_colors(value)
                    self._controller.reslice_visual(self._plane_visual.id)
                for vid in self._orient_3d_visual_ids:
                    self._controller.update_appearance_field(vid, "opacity", value)

            plane_slider.valueChanged.connect(_on_plane_opacity)
            plane_layout.addWidget(plane_slider)
            layout.addWidget(plane_box)

        return group

    # ------------------------------------------------------------------

    @property
    def window(self):
        return self._window

    def close_widgets(self) -> None:
        for cw in self._canvas_widgets.values():
            cw.close()
        if self._2d_clim is not None:
            self._2d_clim.close()
        if self._2d_colormap is not None:
            self._2d_colormap.close()
        if self._2d_lod_bias is not None:
            self._2d_lod_bias.close()
        if self._3d_clim is not None:
            self._3d_clim.close()
        if self._3d_colormap is not None:
            self._3d_colormap.close()
        if self._3d_render is not None:
            self._3d_render.close()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _box_faces_geometry(
    centre_zyx: np.ndarray,
    half_extents_zyx: np.ndarray,
    vertex_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    cz, cy, cx = float(centre_zyx[0]), float(centre_zyx[1]), float(centre_zyx[2])
    hz, hy, hx = (
        float(half_extents_zyx[0]),
        float(half_extents_zyx[1]),
        float(half_extents_zyx[2]),
    )
    z0, z1 = cz - hz, cz + hz
    y0, y1 = cy - hy, cy + hy
    x0, x1 = cx - hx, cx + hx

    positions = np.array(
        [
            [z0, y0, x0],
            [z0, y1, x0],
            [z0, y1, x1],
            [z0, y0, x1],  # Face 0: -Z
            [z1, y0, x0],
            [z1, y0, x1],
            [z1, y1, x1],
            [z1, y1, x0],  # Face 1: +Z
            [z0, y0, x0],
            [z0, y0, x1],
            [z1, y0, x1],
            [z1, y0, x0],  # Face 2: -Y
            [z0, y1, x0],
            [z1, y1, x0],
            [z1, y1, x1],
            [z0, y1, x1],  # Face 3: +Y
            [z0, y0, x0],
            [z1, y0, x0],
            [z1, y1, x0],
            [z0, y1, x0],  # Face 4: -X
            [z0, y0, x1],
            [z0, y1, x1],
            [z1, y1, x1],
            [z1, y0, x1],  # Face 5: +X
        ],
        dtype=np.float32,
    )

    base_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [8, 9, 10],
            [8, 10, 11],
            [12, 13, 14],
            [12, 14, 15],
            [16, 17, 18],
            [16, 18, 19],
            [20, 21, 22],
            [20, 22, 23],
        ],
        dtype=np.int32,
    )
    return positions, base_indices + vertex_offset


def _make_axis_set_geometry(
    axis_a: int,
    axis_b: int,
    axis_length: float,
    cube_side: float,
    prism_cross_section: float,
) -> tuple[np.ndarray, np.ndarray]:
    half_cube = cube_side / 2.0
    half_length = axis_length / 2.0
    half_cross = prism_cross_section / 2.0

    origin = np.zeros(3, dtype=np.float64)
    cube_half_extents = np.full(3, half_cube, dtype=np.float64)
    cube_positions, cube_indices = _box_faces_geometry(origin, cube_half_extents, 0)

    centre_a = np.zeros(3, dtype=np.float64)
    centre_a[axis_a] = half_cube + half_length
    half_extents_a = np.full(3, half_cross, dtype=np.float64)
    half_extents_a[axis_a] = half_length
    prism_a_positions, prism_a_indices = _box_faces_geometry(
        centre_a, half_extents_a, 24
    )

    centre_b = np.zeros(3, dtype=np.float64)
    centre_b[axis_b] = half_cube + half_length
    half_extents_b = np.full(3, half_cross, dtype=np.float64)
    half_extents_b[axis_b] = half_length
    prism_b_positions, prism_b_indices = _box_faces_geometry(
        centre_b, half_extents_b, 48
    )

    positions = np.concatenate([cube_positions, prism_a_positions, prism_b_positions])
    indices = np.concatenate([cube_indices, prism_a_indices, prism_b_indices])
    return positions, indices


def _make_axis_set_face_colors(
    axis_a_color_rgb: tuple[float, float, float],
    axis_b_color_rgb: tuple[float, float, float],
) -> np.ndarray:
    cube_color = np.array(_AXIS_3D_CUBE_COLOUR, dtype=np.float32)
    color_a = np.array([*axis_a_color_rgb, 1.0], dtype=np.float32)
    color_b = np.array([*axis_b_color_rgb, 1.0], dtype=np.float32)
    return np.concatenate(
        [
            np.tile(cube_color, (_N_FACES_PER_BOX, 1)),
            np.tile(color_a, (_N_FACES_PER_BOX, 1)),
            np.tile(color_b, (_N_FACES_PER_BOX, 1)),
        ]
    )


def _pad_positions(
    positions_3d: np.ndarray,
    spatial_axes: tuple[int, int, int],
    n_dims: int,
) -> np.ndarray:
    """Scatter 3-D local ZYX positions into an N-dim global positions array.

    The three local columns (0=Z, 1=Y, 2=X) are placed at the global axis
    indices given by ``spatial_axes``.  All other columns remain zero.
    """
    if n_dims == 3 and spatial_axes == (0, 1, 2):
        return positions_3d
    out = np.zeros((len(positions_3d), n_dims), dtype=positions_3d.dtype)
    out[:, spatial_axes[0]] = positions_3d[:, 0]
    out[:, spatial_axes[1]] = positions_3d[:, 1]
    out[:, spatial_axes[2]] = positions_3d[:, 2]
    return out


def _spatial_translation(
    z: float,
    y: float,
    x: float,
    spatial_axes: tuple[int, int, int],
    n_dims: int,
) -> tuple[float, ...]:
    """Build an N-dim translation vector with spatial values at global axis indices."""
    t = np.zeros(n_dims, dtype=np.float64)
    t[spatial_axes[0]] = z
    t[spatial_axes[1]] = y
    t[spatial_axes[2]] = x
    return tuple(float(v) for v in t)


def _make_axis_meshes(
    controller,
    vol_scene,
    initial_centre_zyx: np.ndarray,
    world_min_extent: float,
    *,
    spatial_axes: tuple[int, int, int],
    n_dims: int,
) -> tuple:
    from cellier.v2.data.mesh._mesh_memory_store import MeshMemoryStore
    from cellier.v2.transform import AffineTransform
    from cellier.v2.visuals._mesh_memory import MeshFlatAppearance

    color_z = _PLANE_COLOR_XY
    color_y = _PLANE_COLOR_XZ
    color_x = _PLANE_COLOR_YZ

    view_specifications = [
        ("xy_axis_set", 1, 2, color_y, color_x),
        ("xz_axis_set", 0, 2, color_z, color_x),
        ("yz_axis_set", 0, 1, color_z, color_y),
    ]

    axis_length = _AXIS_3D_LENGTH_FRACTION * world_min_extent
    cube_side = _AXIS_3D_CUBE_SIDE_FRACTION * world_min_extent
    prism_cross_section = _AXIS_3D_PRISM_CROSS_SECTION_FRACTION * world_min_extent

    initial_translation = _spatial_translation(
        float(initial_centre_zyx[0]),
        float(initial_centre_zyx[1]),
        float(initial_centre_zyx[2]),
        spatial_axes,
        n_dims,
    )
    initial_transform = AffineTransform.from_translation(initial_translation)

    axis_stores = []
    axis_visuals = []
    for view_name, axis_a, axis_b, color_a, color_b in view_specifications:
        positions_3d, indices = _make_axis_set_geometry(
            axis_a, axis_b, axis_length, cube_side, prism_cross_section
        )
        positions = _pad_positions(positions_3d, spatial_axes, n_dims)
        face_colors = _make_axis_set_face_colors(color_a, color_b)
        store = MeshMemoryStore(
            positions=positions, indices=indices, colors=face_colors, name=view_name
        )
        appearance = MeshFlatAppearance(
            color_mode="face",
            side="both",
            opacity=1.0,
            render_order=1,
            depth_test=True,
            depth_write=True,
            depth_compare="<=",
            transparency_mode="blend",
        )
        visual = controller.add_mesh(
            data=store,
            scene_id=vol_scene.id,
            appearance=appearance,
            name=view_name,
            transform=initial_transform,
        )
        axis_stores.append(store)
        axis_visuals.append(visual)

    return tuple(axis_visuals), tuple(axis_stores)


def _make_plane_positions(
    z_world: float,
    y_world: float,
    x_world: float,
    world_max_zyx: np.ndarray,
    *,
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    n_dims: int = 3,
) -> np.ndarray:
    wz = float(world_max_zyx[0])
    wy = float(world_max_zyx[1])
    wx = float(world_max_zyx[2])
    z, y, x = float(z_world), float(y_world), float(x_world)
    sz0, sz1, sz2 = spatial_axes

    positions = np.zeros((12, n_dims), dtype=np.float32)

    # XY plane (constant Z = z)
    positions[0:4, sz0] = z
    positions[0:4, sz1] = [0.0, wy, wy, 0.0]
    positions[0:4, sz2] = [0.0, 0.0, wx, wx]

    # XZ plane (constant Y = y)
    positions[4:8, sz0] = [0.0, wz, wz, 0.0]
    positions[4:8, sz1] = y
    positions[4:8, sz2] = [0.0, 0.0, wx, wx]

    # YZ plane (constant X = x)
    positions[8:12, sz0] = [0.0, wz, wz, 0.0]
    positions[8:12, sz1] = [0.0, 0.0, wy, wy]
    positions[8:12, sz2] = x

    return positions


def _make_plane_colors(opacity: float) -> np.ndarray:
    a = float(opacity)
    return np.array(
        [
            [*_PLANE_COLOR_XY, a],
            [*_PLANE_COLOR_XY, a],
            [*_PLANE_COLOR_XZ, a],
            [*_PLANE_COLOR_XZ, a],
            [*_PLANE_COLOR_YZ, a],
            [*_PLANE_COLOR_YZ, a],
        ],
        dtype=np.float32,
    )


def _make_plane_mesh(
    controller,
    vol_scene,
    z_world: float,
    y_world: float,
    x_world: float,
    world_max_zyx: np.ndarray,
    initial_opacity: float = 0.4,
    *,
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    n_dims: int = 3,
):
    from cellier.v2.data.mesh._mesh_memory_store import MeshMemoryStore
    from cellier.v2.visuals._mesh_memory import MeshFlatAppearance

    positions = _make_plane_positions(
        z_world,
        y_world,
        x_world,
        world_max_zyx,
        spatial_axes=spatial_axes,
        n_dims=n_dims,
    )
    colors = _make_plane_colors(initial_opacity)
    indices = np.array(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [8, 9, 10], [8, 10, 11]],
        dtype=np.int32,
    )

    store = MeshMemoryStore(
        positions=positions, indices=indices, colors=colors, name="slice_planes"
    )
    appearance = MeshFlatAppearance(
        color_mode="face", side="both", opacity=initial_opacity, wireframe=False
    )
    visual = controller.add_mesh(
        data=store, scene_id=vol_scene.id, appearance=appearance, name="slice_planes"
    )
    return store, visual


class _PlaneUpdater:
    def __init__(
        self,
        controller,
        plane_store,
        plane_visual,
        world_max_zyx,
        *,
        spatial_axes: tuple[int, int, int] = (0, 1, 2),
        n_dims: int = 3,
        channel_axis: int | None = None,
    ) -> None:
        self._id = uuid4()
        self._controller = controller
        self._plane_store = plane_store
        self._plane_visual = plane_visual
        self._world_max_zyx = world_max_zyx
        self._spatial_axes = spatial_axes
        self._n_dims = n_dims
        self._channel_axis = channel_axis
        self._ch_world: float = 0.0

        # Read initial slice positions from the N-dim positions array.
        positions = plane_store.positions
        sz0, sz1, sz2 = spatial_axes
        self._z_world = float(positions[0, sz0])  # XY plane vertex 0: Z
        self._y_world = float(positions[4, sz1])  # XZ plane vertex 4: Y
        self._x_world = float(positions[8, sz2])  # YZ plane vertex 8: X

    def _update(self) -> None:
        positions = _make_plane_positions(
            self._z_world,
            self._y_world,
            self._x_world,
            self._world_max_zyx,
            spatial_axes=self._spatial_axes,
            n_dims=self._n_dims,
        )
        if self._channel_axis is not None:
            positions[:, self._channel_axis] = self._ch_world
        self._plane_store.positions = positions
        self._controller.reslice_visual(self._plane_visual.id)

    def on_channel_changed(self, new_ch: int) -> None:
        self._ch_world = float(new_ch)
        # Only update the store positions; reslice_scene (triggered by
        # update_slice_indices on the vol scene) handles the actual reslice with
        # the correct channel dims, avoiding a conflicting reslice with stale dims.
        positions = _make_plane_positions(
            self._z_world,
            self._y_world,
            self._x_world,
            self._world_max_zyx,
            spatial_axes=self._spatial_axes,
            n_dims=self._n_dims,
        )
        if self._channel_axis is not None:
            positions[:, self._channel_axis] = self._ch_world
        self._plane_store.positions = positions

    def on_xy_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        sz0 = self._spatial_axes[0]
        if sz0 in slice_indices:
            self._z_world = float(slice_indices[sz0])
            self._update()

    def on_xz_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        sz1 = self._spatial_axes[1]
        if sz1 in slice_indices:
            self._y_world = float(slice_indices[sz1])
            self._update()

    def on_yz_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        sz2 = self._spatial_axes[2]
        if sz2 in slice_indices:
            self._x_world = float(slice_indices[sz2])
            self._update()


class _OrientationUpdater:
    def __init__(
        self,
        controller,
        xy_axis_visual,
        xz_axis_visual,
        yz_axis_visual,
        world_max_zyx: np.ndarray,
        *,
        xy_axis_store=None,
        xz_axis_store=None,
        yz_axis_store=None,
        spatial_axes: tuple[int, int, int] = (0, 1, 2),
        n_dims: int = 3,
        channel_axis: int | None = None,
    ):
        self._id = uuid4()
        self._controller = controller
        self._xy_axis_visual_id = xy_axis_visual.id
        self._xz_axis_visual_id = xz_axis_visual.id
        self._yz_axis_visual_id = yz_axis_visual.id
        self._xy_axis_store = xy_axis_store
        self._xz_axis_store = xz_axis_store
        self._yz_axis_store = yz_axis_store
        self._spatial_axes = spatial_axes
        self._n_dims = n_dims
        self._channel_axis = channel_axis
        self._ch_world: float = 0.0

        mid = world_max_zyx / 2.0
        self._z_world = float(mid[0])
        self._y_world = float(mid[1])
        self._x_world = float(mid[2])
        # N-dim centre vectors, one per 2D panel.
        self._xy_centre = self._make_centre(self._z_world, self._y_world, self._x_world)
        self._xz_centre = self._xy_centre.copy()
        self._yz_centre = self._xy_centre.copy()

    def _make_centre(self, z: float, y: float, x: float) -> np.ndarray:
        c = np.zeros(self._n_dims, dtype=np.float64)
        c[self._spatial_axes[0]] = z
        c[self._spatial_axes[1]] = y
        c[self._spatial_axes[2]] = x
        if self._channel_axis is not None:
            c[self._channel_axis] = self._ch_world
        return c

    def on_channel_changed(self, new_ch: int) -> None:
        self._ch_world = float(new_ch)
        if self._channel_axis is not None:
            self._xy_centre[self._channel_axis] = self._ch_world
            self._xz_centre[self._channel_axis] = self._ch_world
            self._yz_centre[self._channel_axis] = self._ch_world
            # Update channel column in each axis store so the slab filter keeps
            # the meshes visible. reslice_scene (from update_slice_indices on the
            # vol scene) handles the actual reslice — no reslice_visual here.
            for store in (
                self._xy_axis_store,
                self._xz_axis_store,
                self._yz_axis_store,
            ):
                if store is not None:
                    positions = store.positions.copy()
                    positions[:, self._channel_axis] = self._ch_world
                    store.positions = positions
        self._update_3d()

    def _update_3d(self) -> None:
        from cellier.v2.transform import AffineTransform

        for visual_id, centre_nd in zip(
            (self._xy_axis_visual_id, self._xz_axis_visual_id, self._yz_axis_visual_id),
            (self._xy_centre, self._xz_centre, self._yz_centre),
            strict=False,
        ):
            self._controller.set_visual_transform(
                visual_id,
                AffineTransform.from_translation(tuple(float(v) for v in centre_nd)),
                reslice=False,
            )

    def on_xy_camera_changed(self, event) -> None:
        p = event.camera_state.position
        # p[0] → X world, p[1] → Y world (canvas horizontal/vertical convention)
        self._xy_centre = self._make_centre(self._z_world, p[1], p[0])
        self._update_3d()

    def on_xz_camera_changed(self, event) -> None:
        p = event.camera_state.position
        # p[0] → X world, p[1] → Z world
        self._xz_centre = self._make_centre(p[1], self._y_world, p[0])
        self._update_3d()

    def on_yz_camera_changed(self, event) -> None:
        p = event.camera_state.position
        # p[0] → Y world, p[1] → Z world
        self._yz_centre = self._make_centre(p[1], p[0], self._x_world)
        self._update_3d()

    def on_xy_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        sz0 = self._spatial_axes[0]
        if sz0 in slice_indices:
            self._z_world = float(slice_indices[sz0])
            self._xy_centre[sz0] = self._z_world
        self._update_3d()

    def on_xz_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        sz1 = self._spatial_axes[1]
        if sz1 in slice_indices:
            self._y_world = float(slice_indices[sz1])
            self._xz_centre[sz1] = self._y_world
        self._update_3d()

    def on_yz_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        sz2 = self._spatial_axes[2]
        if sz2 in slice_indices:
            self._x_world = float(slice_indices[sz2])
            self._yz_centre[sz2] = self._x_world
        self._update_3d()


# ---------------------------------------------------------------------------
# Channel-axis sync
# ---------------------------------------------------------------------------


class _ChannelAxisSyncer:
    """Fans out channel-axis slice changes across all scenes in SC mode.

    update_slice_indices is a full replacement, so each target scene's current
    slice_indices is read, patched with the new channel value, and written back
    in full to preserve spatial slice positions.
    """

    def __init__(
        self,
        controller,
        xy_scene,
        xz_scene,
        yz_scene,
        vol_scene,
        channel_axis: int,
    ) -> None:
        self._id = uuid4()
        self._controller = controller
        self._xy_scene_id = xy_scene.id
        self._xz_scene_id = xz_scene.id
        self._yz_scene_id = yz_scene.id
        # Keyed by scene id so _propagate can read current slice_indices.
        self._all_scenes = {
            xy_scene.id: xy_scene,
            xz_scene.id: xz_scene,
            yz_scene.id: yz_scene,
            vol_scene.id: vol_scene,
        }
        self._channel_axis = channel_axis
        self._syncing = False
        self.enabled = True
        # Set after construction once the updaters are available.
        self.plane_updater: _PlaneUpdater | None = None
        self.orient_updater: _OrientationUpdater | None = None

    def _propagate(self, source_scene_id, event) -> None:
        if not self.enabled or self._syncing:
            return
        slice_indices = event.dims_state.selection.slice_indices
        if self._channel_axis not in slice_indices:
            return
        new_ch = slice_indices[self._channel_axis]
        self._syncing = True
        try:
            # Update mesh positions/transforms before the vol scene reslice so
            # that when reslice_scene fires, the stores already carry the correct
            # channel world-coordinate and the ±0.5 slab filter keeps them visible.
            if self.plane_updater is not None:
                self.plane_updater.on_channel_changed(new_ch)
            if self.orient_updater is not None:
                self.orient_updater.on_channel_changed(new_ch)
            for sid, scene in self._all_scenes.items():
                if sid == source_scene_id:
                    continue
                current = dict(scene.dims.selection.slice_indices)
                current[self._channel_axis] = new_ch
                self._controller.update_slice_indices(sid, current)
        finally:
            self._syncing = False

    def on_xy_dims_changed(self, event) -> None:
        self._propagate(self._xy_scene_id, event)

    def on_xz_dims_changed(self, event) -> None:
        self._propagate(self._xz_scene_id, event)

    def on_yz_dims_changed(self, event) -> None:
        self._propagate(self._yz_scene_id, event)


# ---------------------------------------------------------------------------
# Layer 1: ViewerModel builder
# ---------------------------------------------------------------------------


def build_ortho_viewer_model(
    zarr_uri: str,
    *,
    channel_axis: int | None = None,
    initial_mode: str = "single",
    perf: StartupPerfTracer | None = None,
) -> tuple:
    """Build a ViewerModel for the orthoviewer without constructing any Qt objects.

    Parameters
    ----------
    zarr_uri : str
        Path or URI to the OME-Zarr store.
    channel_axis : int or None, optional
        Axis index to treat as the channel dimension.  When ``None`` (default),
        the channel axis is auto-detected from the OME-Zarr axis metadata if
        present.
    initial_mode : str, optional
        ``"single"`` (default) or ``"multichannel"``.  When ``"multichannel"``,
        scenes are built with correct dims but **no SC visuals** — the caller is
        responsible for hot-adding multichannel visuals after controller creation.
    perf : StartupPerfTracer | None, optional
        Optional startup performance tracer used to record timing milestones
        while building the viewer model.

    Returns
    -------
    tuple[cellier.v2.viewer_model.ViewerModel, int | None]
        ``(viewer_model, effective_channel_axis)`` where ``effective_channel_axis``
        is the resolved channel axis index (auto-detected or user-supplied) or
        ``None`` when no channel axis is present.
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

    # Auto-detect channel axis from OME-Zarr axis metadata when not explicitly set.
    if channel_axis is None:
        effective_channel_axis: int | None = None
        for idx, ax in enumerate(ms.axes):
            if getattr(ax, "type", None) == "channel":
                effective_channel_axis = idx
                break
    else:
        effective_channel_axis = channel_axis

    # Full per-axis scale (includes channel axis when present).
    level_0_scale_full = np.array(
        ms.datasets[0].scale_transform.scale, dtype=np.float64
    )
    n_dims = len(data_store.level_shapes[0])

    # Spatial axis indices (all axes except the channel axis).
    if effective_channel_axis is None:
        spatial_indices: list[int] = list(range(n_dims))
    else:
        spatial_indices = [i for i in range(n_dims) if i != effective_channel_axis]
    spatial_ndim = len(spatial_indices)

    level_0_scale_spatial = level_0_scale_full[spatial_indices]
    vox_shape_spatial = np.array(
        [data_store.level_shapes[0][i] for i in spatial_indices], dtype=np.float64
    )
    world_extents_spatial = vox_shape_spatial * level_0_scale_spatial
    max_extent = float(world_extents_spatial.max())
    depth_range = (max(1.0, max_extent * 0.0001), max_extent * 10.0)

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
    spatial_label = "spatial" if effective_channel_axis is not None else "ZYX"
    table.add_row(
        f"scale ({spatial_label})", "  ".join(f"{v:.4g}" for v in level_0_scale_spatial)
    )
    table.add_row(
        f"world extents ({spatial_label})",
        "  ".join(f"{v:.4g}" for v in world_extents_spatial),
    )
    table.add_row("depth range", f"near={depth_range[0]:.2f}  far={depth_range[1]:.0f}")

    Console().print(table)
    _perf_mark(perf, "viewer.model.metadata_printed")

    # Coordinate system covers all axes (spatial + channel when present).
    cs = CoordinateSystem(name="world", axis_labels=tuple(data_store.axis_names))
    # Transform covers all axes; channel scale = 1.0 from the OME-Zarr metadata.
    voxel_to_world = AffineTransform.from_scale_and_translation(
        scale=tuple(level_0_scale_full)
    )

    initial_clim_max = _dtype_clim_max(data_store.dtype)
    world_max_spatial = (vox_shape_spatial - 1) * level_0_scale_spatial

    # Mid-world position keyed by global axis index.
    spatial_mid_world = {
        spatial_indices[j]: round(float(world_max_spatial[j]) / 2.0)
        for j in range(spatial_ndim)
    }

    coarsest_level = data_store.n_levels - 1
    slider_decimals = _dtype_decimals(data_store.dtype)
    vox_shape_full = np.array(data_store.level_shapes[0], dtype=np.float64)
    world_max_full = (vox_shape_full - 1) * level_0_scale_full
    axis_ranges = {i: (0.0, round(float(world_max_full[i]))) for i in range(n_dims)}
    excluded_axes: set[int] | None = (
        {effective_channel_axis} if effective_channel_axis is not None else None
    )
    n_channels = (
        int(data_store.level_shapes[0][effective_channel_axis])
        if effective_channel_axis is not None
        else 0
    )

    # Base slice_indices fixes the channel at 0 when a channel axis is present.
    base_channel_slice: dict[int, int] = (
        {effective_channel_axis: 0} if effective_channel_axis is not None else {}
    )

    # In single-channel mode, SC visuals are built eagerly.
    # In multichannel mode, scenes are built empty — the caller hot-adds MC visuals.
    build_sc_visuals = initial_mode == "single"

    common_2d_appearance = MultiscaleImageAppearance(
        color_map="viridis",
        clim=(0.0, initial_clim_max),
        lod_bias=1.0,
        force_level=None,
        frustum_cull=True,
        iso_threshold=0.2,
        render_mode="mip",
    )
    common_render_config = MultiscaleImageRenderConfig(
        block_size=32,
        gpu_budget_bytes=512 * 1024**2,
        gpu_budget_bytes_2d=64 * 1024**2,
    )

    def _make_2d_canvas() -> Canvas:
        return Canvas(
            cameras={
                "2d": OrthographicCamera(
                    near_clipping_plane=depth_range[0],
                    far_clipping_plane=depth_range[1],
                    controller=PanZoomCameraController(enabled=True),
                )
            }
        )

    def _make_2d_visual(name: str) -> MultiscaleImageVisual:
        return MultiscaleImageVisual(
            name=name,
            data_store_id=str(data_store.id),
            level_transforms=data_store.level_transforms,
            appearance=common_2d_appearance,
            render_config=common_render_config,
            transform=voxel_to_world,
        )

    # --- XY scene (display spatial axes 1 and 2 for 3-D, or 0 and 1 for 2-D) ---
    if spatial_ndim == 3:
        xy_displayed = (spatial_indices[1], spatial_indices[2])
        xy_slice = {
            **base_channel_slice,
            spatial_indices[0]: spatial_mid_world[spatial_indices[0]],
        }
    else:
        xy_displayed = (spatial_indices[0], spatial_indices[1])
        xy_slice = {**base_channel_slice}

    xy_visual: MultiscaleImageVisual | None = (
        _make_2d_visual("xy_volume") if build_sc_visuals else None
    )
    xy_canvas = _make_2d_canvas()
    xy_scene = Scene(
        name="xy",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=xy_displayed, slice_indices=xy_slice
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=[xy_visual] if xy_visual is not None else [],
        canvases={xy_canvas.id: xy_canvas},
    )

    # --- XZ scene (display spatial axes 0 and 2; blank for 2-D data) ---
    xz_canvas = _make_2d_canvas()
    if spatial_ndim == 3:
        xz_displayed = (spatial_indices[0], spatial_indices[2])
        xz_slice = {
            **base_channel_slice,
            spatial_indices[1]: spatial_mid_world[spatial_indices[1]],
        }
        xz_visual: MultiscaleImageVisual | None = (
            _make_2d_visual("xz_volume") if build_sc_visuals else None
        )
        xz_visuals = [xz_visual] if xz_visual is not None else []
    else:
        xz_displayed = tuple(spatial_indices)
        xz_slice = {**base_channel_slice}
        xz_visual = None
        xz_visuals = []

    xz_scene = Scene(
        name="xz",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=xz_displayed, slice_indices=xz_slice
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=xz_visuals,
        canvases={xz_canvas.id: xz_canvas},
    )

    # --- YZ scene (display spatial axes 0 and 1; blank for 2-D data) ---
    yz_canvas = _make_2d_canvas()
    if spatial_ndim == 3:
        yz_displayed = (spatial_indices[0], spatial_indices[1])
        yz_slice = {
            **base_channel_slice,
            spatial_indices[2]: spatial_mid_world[spatial_indices[2]],
        }
        yz_visual: MultiscaleImageVisual | None = (
            _make_2d_visual("yz_volume") if build_sc_visuals else None
        )
        yz_visuals = [yz_visual] if yz_visual is not None else []
    else:
        yz_displayed = tuple(spatial_indices)
        yz_slice = {**base_channel_slice}
        yz_visual = None
        yz_visuals = []

    yz_scene = Scene(
        name="yz",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=yz_displayed, slice_indices=yz_slice
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=yz_visuals,
        canvases={yz_canvas.id: yz_canvas},
    )

    # --- Vol scene (3-D; blank for 2-D spatial data or multichannel mode) ---
    vol_canvas = Canvas(
        cameras={
            "3d": PerspectiveCamera(
                fov=70.0,
                near_clipping_plane=depth_range[0],
                far_clipping_plane=depth_range[1],
                controller=OrbitCameraController(enabled=True),
            )
        }
    )
    if spatial_ndim == 3 and build_sc_visuals:
        vol_displayed = tuple(spatial_indices)
        vol_slice = {**base_channel_slice}
        vol_visual: MultiscaleImageVisual | None = MultiscaleImageVisual(
            name="vol_volume",
            data_store_id=str(data_store.id),
            level_transforms=data_store.level_transforms,
            appearance=MultiscaleImageAppearance(
                color_map="white",
                clim=(0.0, initial_clim_max),
                lod_bias=1.0,
                force_level=coarsest_level,
                frustum_cull=False,
                iso_threshold=initial_clim_max / 2.0,
                render_mode="iso",
            ),
            render_config=MultiscaleImageRenderConfig(
                block_size=32,
                gpu_budget_bytes=2048 * 1024**2,
                gpu_budget_bytes_2d=64 * 1024**2,
            ),
            transform=voxel_to_world,
        )
        vol_visual.aabb.enabled = True
        vol_visual.aabb.color = "#ff00ff"
        vol_visuals = [vol_visual]
    else:
        vol_displayed = tuple(spatial_indices)
        vol_slice = {**base_channel_slice}
        vol_visual = None
        vol_visuals = []

    vol_scene = Scene(
        name="vol",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=vol_displayed, slice_indices=vol_slice
            ),
        ),
        render_modes={"3d"},
        lighting="none",
        visuals=vol_visuals,
        canvases={vol_canvas.id: vol_canvas},
    )

    viewer_model = ViewerModel(
        data=DataManager(stores={data_store.id: data_store}),
        scenes={
            xy_scene.id: xy_scene,
            xz_scene.id: xz_scene,
            yz_scene.id: yz_scene,
            vol_scene.id: vol_scene,
        },
    )

    _perf_mark(perf, "viewer.model.ready", n_scenes=len(viewer_model.scenes))
    geometry = _WorldGeometry(
        spatial_indices=spatial_indices,
        spatial_ndim=spatial_ndim,
        world_max_spatial=world_max_spatial,
        spatial_mid_world=spatial_mid_world,
        voxel_to_world=voxel_to_world,
        axis_ranges=axis_ranges,
        excluded_axes=excluded_axes,
        initial_clim_max=initial_clim_max,
        clim_range=(0.0, initial_clim_max),
        slider_decimals=slider_decimals,
        n_channels=n_channels,
    )
    return viewer_model, effective_channel_axis, geometry


# ---------------------------------------------------------------------------
# Layer 2: Qt bootstrap helper  (imported from _utils)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Layer 3: Non-blocking show (for interactive / Jupyter use)
# ---------------------------------------------------------------------------


def orthoviewer(
    zarr_uri: str,
    theme: str = "dark",
    *,
    channel_axis: int | None = None,
) -> OmeZarrOrthoViewer:
    """Open an orthoviewer window without blocking.

    Intended for interactive use (Jupyter Lab, IPython). The Qt event loop
    must already be running or be startable via IPython's ``enable_gui``; this
    function sets that up automatically. For scripts use ``launch_orthoviewer``.

    Parameters
    ----------
    zarr_uri : str
        Path or URI to the OME-Zarr store.
    theme : str
        Registered theme name. Defaults to ``"dark"``.
        Use ``oz_viewer.theme.list_themes()`` to see available themes.
    channel_axis : int or None, optional
        Axis index to treat as the channel dimension, enabling multichannel
        mode.  ``None`` (default) uses single-channel mode.

    Returns
    -------
    OmeZarrOrthoViewer
        The viewer window object. Keep a reference to prevent garbage collection.
    """
    app = _ensure_qt_app()
    if app is None:
        raise RuntimeError(
            "No Qt event loop is running. "
            "Use launch_orthoviewer() for scripts, or run inside IPython/Jupyter."
        )

    return _build_and_show(zarr_uri, theme=theme, channel_axis=channel_axis)


# ---------------------------------------------------------------------------
# Layer 4: Private async core
# ---------------------------------------------------------------------------


async def _run_orthoviewer_async(
    zarr_uri: str,
    theme: str = "dark",
    *,
    channel_axis: int | None = None,
    perf: StartupPerfTracer | None = None,
) -> None:
    import asyncio as _asyncio

    from PySide6.QtWidgets import QApplication

    _asyncio.get_event_loop().set_exception_handler(_asyncio_exception_handler)
    _perf_mark(perf, "viewer.async.start", theme=theme)

    viewer = _build_and_show(
        zarr_uri, theme=theme, channel_axis=channel_axis, perf=perf
    )
    _perf_mark(perf, "viewer.async.build_complete")

    app = QApplication.instance()
    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    app.aboutToQuit.connect(viewer.close_widgets)
    await close_event.wait()


# ---------------------------------------------------------------------------
# Layer 5: Blocking launcher (for scripts and CLI)
# ---------------------------------------------------------------------------


def launch_orthoviewer(
    zarr_uri: str,
    theme: str = "dark",
    *,
    channel_axis: int | None = None,
    perf: StartupPerfTracer | None = None,
) -> None:
    """Open an orthoviewer window and block until it is closed.

    Creates a ``QApplication`` if one does not already exist, then runs the
    Qt + asyncio event loop via ``QtAsyncio``. Intended for scripts and the
    CLI. For interactive/Jupyter use, call ``orthoviewer()`` instead.

    Parameters
    ----------
    zarr_uri : str
        Path or URI to the OME-Zarr store.
    theme : str
        Registered theme name. Defaults to ``"dark"``.
        Use ``oz_viewer.theme.list_themes()`` to see available themes.
    channel_axis : int or None, optional
        Axis index to treat as the channel dimension, enabling multichannel
        mode.  ``None`` (default) uses single-channel mode.
    perf : StartupPerfTracer | None, optional
        Optional startup performance tracer used to record timing milestones
        during viewer launch and startup.
    """
    import sys

    # Pre-initialise fsspec's dedicated IO event loop *before* QtAsyncio
    # installs its Qt-integrated loop policy.  QtAsyncio.run() replaces the
    # global asyncio loop with a Qt-coupled one; if fsspec's background thread
    # ("fsspecIO") tries to start its own loop afterwards it sees that loop as
    # already running and raises RuntimeError.  Calling get_loop() here forces
    # fsspec to create and own a plain SelectorEventLoop in its worker thread
    # ahead of time, so the two loop systems never conflict.  This is safe and
    # cheap for local datasets too (the thread is created but never used).
    import fsspec.asyn as _fsspec_asyn
    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    _fsspec_asyn.get_loop()

    _perf_mark(perf, "viewer.launch.start", theme=theme)
    app = QApplication.instance() or QApplication([sys.argv[0]])  # noqa: F841
    _perf_mark(perf, "viewer.launch.qapp_ready")
    QtAsyncio.run(
        _run_orthoviewer_async(
            zarr_uri, theme=theme, channel_axis=channel_axis, perf=perf
        ),
        handle_sigint=True,
    )


# ---------------------------------------------------------------------------
# Shared builder (used by both orthoviewer and _run_orthoviewer_async)
# ---------------------------------------------------------------------------


def _build_and_show(
    zarr_uri: str,
    theme: str = "dark",
    *,
    channel_axis: int | None = None,
    perf: StartupPerfTracer | None = None,
) -> OmeZarrOrthoViewer:
    """Build the full viewer from a zarr URI and show the window."""
    from PySide6.QtWidgets import QApplication

    from oz_viewer.theme import apply_theme

    _perf_mark(perf, "viewer.build.start", theme=theme)
    apply_theme(QApplication.instance(), theme)
    _perf_mark(perf, "viewer.build.theme_applied")
    from cellier.v2.controller import CellierController
    from cellier.v2.gui._scene import QtCanvasWidget, QtDimsSliders
    from cellier.v2.render._config import (
        RenderManagerConfig,
        SlicingConfig,
        TemporalAccumulationConfig,
    )
    from cellier.v2.visuals._canvas_overlay import (
        CenteredAxes2D,
        CenteredAxes2DAppearance,
    )

    # User-supplied channel_axis drives the initial mode.
    # Auto-detected channel axes always start in single-channel mode with the
    # toggle button available.
    initial_mode = "multichannel" if channel_axis is not None else "single"

    viewer_model, effective_channel_axis, _geo = build_ortho_viewer_model(
        zarr_uri, channel_axis=channel_axis, initial_mode=initial_mode, perf=perf
    )
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

    xy_scene = controller.get_scene_by_name("xy")
    xz_scene = controller.get_scene_by_name("xz")
    yz_scene = controller.get_scene_by_name("yz")
    vol_scene = controller.get_scene_by_name("vol")
    scenes = {"xy": xy_scene, "xz": xz_scene, "yz": yz_scene, "vol": vol_scene}

    # Retrieve the first visual from each scene (None for empty scenes).
    def _first_visual(scene):
        return next(iter(scene.visuals), None)

    visuals = {
        "xy": _first_visual(xy_scene),
        "xz": _first_visual(xz_scene),
        "yz": _first_visual(yz_scene),
        "vol": _first_visual(vol_scene),
    }

    # Unpack geometry returned by build_ortho_viewer_model (no second zarr read).
    data_store = next(iter(viewer_model.data.stores.values()))
    spatial_indices = _geo.spatial_indices
    spatial_ndim = _geo.spatial_ndim
    world_max_spatial = _geo.world_max_spatial
    spatial_mid_world = _geo.spatial_mid_world
    voxel_to_world = _geo.voxel_to_world
    axis_ranges = _geo.axis_ranges
    excluded_axes = _geo.excluded_axes
    initial_clim_max = _geo.initial_clim_max
    clim_range = _geo.clim_range
    slider_decimals = _geo.slider_decimals
    n_channels = _geo.n_channels

    # ── Eagerly hot-add MC visuals when starting in multichannel mode ─────
    initial_multichannel_visual_ids: list | None = None
    initial_channel_appearances: dict | None = None
    if initial_mode == "multichannel" and effective_channel_axis is not None:
        from cellier.v2.visuals._channel_appearance import ChannelAppearance
        from cellier.v2.visuals._image import MultiscaleImageRenderConfig as _MC_RCfg

        colormaps = _DEFAULT_COLORMAPS
        initial_channel_appearances = {
            i: ChannelAppearance(
                color_map=colormaps[i % len(colormaps)],
                clim=(0.0, initial_clim_max),
            )
            for i in range(n_channels)
        }
        mc_render_config = _MC_RCfg(
            block_size=32,
            gpu_budget_bytes=512 * 1024**2,
            gpu_budget_bytes_2d=64 * 1024**2,
        )
        mc_scene_keys = ["xy"] + (["xz", "yz", "vol"] if spatial_ndim == 3 else [])
        initial_multichannel_visual_ids = []
        _mc_vol_visual_id: object = None
        for _mc_key in mc_scene_keys:
            _mc_scene = scenes[_mc_key]
            _mc_visual = controller.add_multichannel_image_multiscale(
                data=data_store,
                scene_id=_mc_scene.id,
                channel_axis=effective_channel_axis,
                channels=initial_channel_appearances,
                name=f"multichannel_{_mc_key}",
                render_config=mc_render_config,
                transform=voxel_to_world,
            )
            initial_multichannel_visual_ids.append(_mc_visual.id)
            if _mc_key == "vol":
                _mc_vol_visual_id = _mc_visual.id
        _perf_mark(perf, "viewer.build.mc_visuals_ready", n_channels=n_channels)

    # ── 3D overlay infrastructure (only for 3-D spatial data) ─────────────
    plane_visual = None
    plane_store = None
    transparency_manager = None
    mc_transparency_manager = None
    orient_3d_visual_ids: list = []
    axes_2d_overlay_ids: list = []
    orient_updater = None
    _INITIAL_PLANE_OPACITY = 1.0

    if spatial_ndim == 3:
        z_mid = spatial_mid_world[spatial_indices[0]]
        y_mid = spatial_mid_world[spatial_indices[1]]
        x_mid = spatial_mid_world[spatial_indices[2]]
        initial_centre_zyx = np.array([z_mid, y_mid, x_mid], dtype=np.float64)

        # Derive global spatial axis indices and coordinate-system dimensionality
        # from the vol scene dims — used to build N-dim mesh positions/transforms.
        spatial_axes = tuple(vol_scene.dims.selection.displayed_axes)
        n_dims_world = len(vol_scene.dims.coordinate_system.axis_labels)

        (
            (xy_axis_visual, xz_axis_visual, yz_axis_visual),
            (xy_axis_store, xz_axis_store, yz_axis_store),
        ) = _make_axis_meshes(
            controller=controller,
            vol_scene=vol_scene,
            initial_centre_zyx=initial_centre_zyx,
            world_min_extent=float(world_max_spatial.min()),
            spatial_axes=spatial_axes,
            n_dims=n_dims_world,
        )
        _perf_mark(perf, "viewer.build.axis_meshes_ready")
        orient_3d_visual_ids = [
            xy_axis_visual.id,
            xz_axis_visual.id,
            yz_axis_visual.id,
        ]

        plane_store, plane_visual = _make_plane_mesh(
            controller,
            vol_scene,
            z_mid,
            y_mid,
            x_mid,
            world_max_spatial,
            initial_opacity=_INITIAL_PLANE_OPACITY,
            spatial_axes=spatial_axes,
            n_dims=n_dims_world,
        )
        _perf_mark(perf, "viewer.build.plane_mesh_ready")

        plane_updater = _PlaneUpdater(
            controller=controller,
            plane_store=plane_store,
            plane_visual=plane_visual,
            world_max_zyx=world_max_spatial,
            spatial_axes=spatial_axes,
            n_dims=n_dims_world,
            channel_axis=effective_channel_axis,
        )
        controller.on_dims_changed(
            xy_scene.id, plane_updater.on_xy_dims_changed, owner_id=plane_updater._id
        )
        controller.on_dims_changed(
            xz_scene.id, plane_updater.on_xz_dims_changed, owner_id=plane_updater._id
        )
        controller.on_dims_changed(
            yz_scene.id, plane_updater.on_yz_dims_changed, owner_id=plane_updater._id
        )

        # Build the transparency manager using model-layer IDs only.
        _plane_id = plane_visual.id
        _axis_ids = [xy_axis_visual.id, xz_axis_visual.id, yz_axis_visual.id]
        if visuals.get("vol") is not None:
            transparency_manager = _VolTransparencyManager(
                controller,
                visuals["vol"].id,
                plane_visual_id=_plane_id,
                axis_visual_ids=_axis_ids,
            )
        elif _mc_vol_visual_id is not None:
            # Eager multichannel path: always locked to MIP.
            mc_transparency_manager = _VolTransparencyManager(
                controller,
                _mc_vol_visual_id,
                vol_is_multichannel=True,
                plane_visual_id=_plane_id,
                axis_visual_ids=_axis_ids,
                initial_mode="mip",
            )
            mc_transparency_manager.apply()

    def _canvas_view(scene_id):
        canvas_id = controller.get_canvas_ids(scene_id)[0]
        return controller.get_canvas_view(canvas_id)

    def _make_canvas_widget(scene, slider_style, non_displayed_sliders=None):
        canvas_view = _canvas_view(scene.id)
        axis_labels = dict(enumerate(scene.dims.coordinate_system.axis_labels))
        selection = scene.dims.selection
        dims_sliders = QtDimsSliders(
            scene_id=scene.id,
            axis_ranges=axis_ranges,
            axis_labels=axis_labels,
            initial_slice_indices=dict(getattr(selection, "slice_indices", {})),
            initial_displayed_axes=getattr(selection, "displayed_axes", ()),
            non_displayed_sliders=non_displayed_sliders,
        )
        dims_sliders.widget.setStyleSheet(slider_style)
        cw = QtCanvasWidget(canvas_view=canvas_view, dims_sliders=dims_sliders)
        controller.connect_widget(
            dims_sliders, subscription_specs=dims_sliders.subscription_specs()
        )
        return cw

    vol_axis_ranges = (
        axis_ranges
        if excluded_axes is None
        else {k: v for k, v in axis_ranges.items() if k not in excluded_axes}
    )
    vol_cw = QtCanvasWidget.from_scene_and_canvas(
        vol_scene, _canvas_view(vol_scene.id), axis_ranges=vol_axis_ranges
    )
    controller.connect_widget(
        vol_cw.dims_sliders, subscription_specs=vol_cw.dims_sliders.subscription_specs()
    )

    # In multichannel mode the channel slider is suppressed; in SC mode it is visible.
    _2d_non_displayed: set[int] | None = (
        {effective_channel_axis}
        if initial_mode == "multichannel" and effective_channel_axis is not None
        else None
    )
    canvas_widgets = {
        "xy": _make_canvas_widget(
            xy_scene, _SLIDER_STYLE_XY, non_displayed_sliders=_2d_non_displayed
        ),
        "xz": _make_canvas_widget(
            xz_scene, _SLIDER_STYLE_XZ, non_displayed_sliders=_2d_non_displayed
        ),
        "yz": _make_canvas_widget(
            yz_scene, _SLIDER_STYLE_YZ, non_displayed_sliders=_2d_non_displayed
        ),
        "vol": vol_cw,
    }

    channel_syncer: _ChannelAxisSyncer | None = None
    if effective_channel_axis is not None:
        channel_syncer = _ChannelAxisSyncer(
            controller=controller,
            xy_scene=xy_scene,
            xz_scene=xz_scene,
            yz_scene=yz_scene,
            vol_scene=vol_scene,
            channel_axis=effective_channel_axis,
        )
        controller.on_dims_changed(
            xy_scene.id,
            channel_syncer.on_xy_dims_changed,
            owner_id=channel_syncer._id,
        )
        controller.on_dims_changed(
            xz_scene.id,
            channel_syncer.on_xz_dims_changed,
            owner_id=channel_syncer._id,
        )
        controller.on_dims_changed(
            yz_scene.id,
            channel_syncer.on_yz_dims_changed,
            owner_id=channel_syncer._id,
        )
        if initial_mode == "multichannel":
            channel_syncer.enabled = False
    _perf_mark(perf, "viewer.build.canvas_widgets_ready")

    paint_tracker = None
    settled_timer = None
    if perf is not None and perf.enabled:
        from PySide6.QtCore import QEvent, QObject, QTimer

        paint_seen: set[str] = set()
        quiet_ms = 300
        total_canvases = len(canvas_widgets)

        settled_timer = QTimer()
        settled_timer.setSingleShot(True)
        settled_timer.setInterval(quiet_ms)

        def _on_settled() -> None:
            if len(paint_seen) == total_canvases:
                _perf_mark(perf, "viewer.canvas.startup_settled", quiet_ms=quiet_ms)
                perf.report_rich_table()

        settled_timer.timeout.connect(_on_settled)

        class _CanvasPaintTracker(QObject):
            def eventFilter(self, watched, event):
                if event.type() != QEvent.Type.Paint:
                    return False
                for scene_name, canvas_widget in canvas_widgets.items():
                    if watched is canvas_widget.widget and scene_name not in paint_seen:
                        paint_seen.add(scene_name)
                        _perf_mark(
                            perf,
                            "viewer.canvas.first_paint",
                            scene=scene_name,
                            n_seen=len(paint_seen),
                            n_total=total_canvases,
                        )
                        if len(paint_seen) == total_canvases:
                            _perf_mark(perf, "viewer.canvas.all_first_paint")
                        settled_timer.start()
                        break
                return False

        paint_tracker = _CanvasPaintTracker()
        for canvas_widget in canvas_widgets.values():
            canvas_widget.widget.installEventFilter(paint_tracker)

    # Screen-space 2D axis overlays (all panels regardless of spatial_ndim)
    xy_axes_overlay = controller.add_canvas_overlay_model(
        controller.get_canvas_ids(xy_scene.id)[0],
        CenteredAxes2D(
            name="xy_axes",
            axis_a_direction=(0.0, 1.0, 0.0),
            axis_a_label="Y",
            axis_b_direction=(1.0, 0.0, 0.0),
            axis_b_label="X",
            appearance=CenteredAxes2DAppearance(
                axis_a_color=(*_PLANE_COLOR_XZ, 1.0),
                axis_b_color=(*_PLANE_COLOR_YZ, 1.0),
                label_color=(1.0, 0.0, 1.0, 1.0),
            ),
        ),
    )
    xz_axes_overlay = controller.add_canvas_overlay_model(
        controller.get_canvas_ids(xz_scene.id)[0],
        CenteredAxes2D(
            name="xz_axes",
            axis_a_direction=(0.0, 1.0, 0.0),
            axis_a_label="Z",
            axis_b_direction=(1.0, 0.0, 0.0),
            axis_b_label="X",
            appearance=CenteredAxes2DAppearance(
                axis_a_color=(*_PLANE_COLOR_XY, 1.0),
                axis_b_color=(*_PLANE_COLOR_YZ, 1.0),
                label_color=(1.0, 0.0, 1.0, 1.0),
            ),
        ),
    )
    yz_axes_overlay = controller.add_canvas_overlay_model(
        controller.get_canvas_ids(yz_scene.id)[0],
        CenteredAxes2D(
            name="yz_axes",
            axis_a_direction=(0.0, 1.0, 0.0),
            axis_a_label="Z",
            axis_b_direction=(1.0, 0.0, 0.0),
            axis_b_label="Y",
            appearance=CenteredAxes2DAppearance(
                axis_a_color=(*_PLANE_COLOR_XY, 1.0),
                axis_b_color=(*_PLANE_COLOR_XZ, 1.0),
                label_color=(1.0, 0.0, 1.0, 1.0),
            ),
        ),
    )
    axes_2d_overlay_ids = [
        xy_axes_overlay.id,
        xz_axes_overlay.id,
        yz_axes_overlay.id,
    ]

    viewer = OmeZarrOrthoViewer(
        controller,
        scenes=scenes,
        visuals=visuals,
        canvas_widgets=canvas_widgets,
        clim_range=clim_range,
        slider_decimals=slider_decimals,
        axes_2d_overlay_ids=axes_2d_overlay_ids,
        orient_3d_visual_ids=orient_3d_visual_ids,
        plane_visual=plane_visual,
        plane_store=plane_store,
        initial_plane_opacity=_INITIAL_PLANE_OPACITY,
        transparency_manager=transparency_manager,
        mc_transparency_manager=mc_transparency_manager,
        channel_axis=effective_channel_axis,
        n_channels=n_channels,
        spatial_ndim=spatial_ndim,
        data_store=data_store,
        voxel_to_world=voxel_to_world,
        initial_mode=initial_mode,
        initial_multichannel_visual_ids=initial_multichannel_visual_ids,
        initial_channel_appearances=initial_channel_appearances,
        world_max_spatial=world_max_spatial,
        channel_syncer=channel_syncer,
    )
    if paint_tracker is not None:
        viewer._startup_perf_objects = (paint_tracker, settled_timer)
    viewer.window.show()
    _perf_mark(perf, "viewer.window.show_called")

    if spatial_ndim == 3:
        # 3D orientation overlay wiring
        orient_updater = _OrientationUpdater(
            controller=controller,
            xy_axis_visual=xy_axis_visual,
            xz_axis_visual=xz_axis_visual,
            yz_axis_visual=yz_axis_visual,
            xy_axis_store=xy_axis_store,
            xz_axis_store=xz_axis_store,
            yz_axis_store=yz_axis_store,
            world_max_zyx=world_max_spatial,
            spatial_axes=spatial_axes,
            n_dims=n_dims_world,
            channel_axis=effective_channel_axis,
        )
        if channel_syncer is not None:
            channel_syncer.plane_updater = plane_updater
            channel_syncer.orient_updater = orient_updater

        controller.on_camera_changed(
            xy_scene.id,
            orient_updater.on_xy_camera_changed,
            owner_id=orient_updater._id,
        )
        controller.on_camera_changed(
            xz_scene.id,
            orient_updater.on_xz_camera_changed,
            owner_id=orient_updater._id,
        )
        controller.on_camera_changed(
            yz_scene.id,
            orient_updater.on_yz_camera_changed,
            owner_id=orient_updater._id,
        )
        controller.on_dims_changed(
            xy_scene.id,
            orient_updater.on_xy_dims_changed,
            owner_id=orient_updater._id,
        )
        controller.on_dims_changed(
            xz_scene.id,
            orient_updater.on_xz_dims_changed,
            owner_id=orient_updater._id,
        )
        controller.on_dims_changed(
            yz_scene.id,
            orient_updater.on_yz_dims_changed,
            owner_id=orient_updater._id,
        )

    for scene in scenes.values():
        controller.fit_camera(scene.id)
        _perf_mark(perf, "viewer.scene.fit_camera", scene=scene.name)
        controller.reslice_scene(scene.id)
        _perf_mark(perf, "viewer.scene.reslice_requested", scene=scene.name)

    if transparency_manager is not None:
        transparency_manager.apply()
        _perf_mark(perf, "viewer.transparency.apply_done")

    if spatial_ndim == 3:
        # Seed 3D orientation with post-fit camera state
        def _seed_camera_event(scene_id):
            from cellier.v2.events._events import CameraChangedEvent

            canvas_view = controller.get_canvas_view(
                controller.get_canvas_ids(scene_id)[0]
            )
            camera_state = canvas_view.capture_camera_state()
            return CameraChangedEvent(
                source_id=canvas_view.canvas_id,
                scene_id=scene_id,
                camera_state=camera_state,
            )

        orient_updater.on_xy_camera_changed(_seed_camera_event(xy_scene.id))
        orient_updater.on_xz_camera_changed(_seed_camera_event(xz_scene.id))
        orient_updater.on_yz_camera_changed(_seed_camera_event(yz_scene.id))

    _perf_mark(perf, "viewer.build.done")
    return viewer
