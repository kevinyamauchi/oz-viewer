"""OME-Zarr orthoviewer: 4-panel viewer (XY, XZ, YZ, 3D)."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np

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


# ---------------------------------------------------------------------------
# Multi-visual control helpers
# ---------------------------------------------------------------------------


class _MultiVisualClimSlider:
    """Contrast-limits range slider that updates multiple visuals at once."""

    from psygnal import Signal

    changed = Signal(object)
    closed = Signal()

    def __init__(
        self,
        visual_ids: list,
        *,
        clim_range: tuple[float, float],
        initial_clim: tuple[float, float],
        decimals: int = 2,
        parent=None,
    ) -> None:
        from cellier.v2.events import AppearanceUpdateEvent
        from qtpy.QtCore import Qt
        from superqt import QLabeledDoubleRangeSlider

        self._id = uuid4()
        self._visual_ids = visual_ids
        self._AppearanceUpdateEvent = AppearanceUpdateEvent

        self._slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal, parent)
        self._slider.setRange(*clim_range)
        self._slider.setValue(initial_clim)
        self._slider.setDecimals(decimals)
        self._slider.valueChanged.connect(self._on_changed)

    def _on_changed(self, value: tuple[float, float]) -> None:
        for vid in self._visual_ids:
            self.changed.emit(
                self._AppearanceUpdateEvent(
                    source_id=self._id,
                    visual_id=vid,
                    field="clim",
                    value=value,
                )
            )

    @property
    def widget(self):
        return self._slider

    def close(self) -> None:
        self.closed.emit()


class _MultiVisualColormapCombo:
    """Colormap combo box that updates multiple visuals at once."""

    from psygnal import Signal

    changed = Signal(object)
    closed = Signal()

    def __init__(
        self,
        visual_ids: list,
        *,
        initial_colormap,
        parent=None,
    ) -> None:
        from cellier.v2.events import AppearanceUpdateEvent
        from superqt import QColormapComboBox

        self._id = uuid4()
        self._visual_ids = visual_ids
        self._AppearanceUpdateEvent = AppearanceUpdateEvent

        self._combo = QColormapComboBox(parent)
        self._combo.setCurrentColormap(initial_colormap)
        self._combo.currentColormapChanged.connect(self._on_changed)

    def _on_changed(self, colormap) -> None:
        for vid in self._visual_ids:
            self.changed.emit(
                self._AppearanceUpdateEvent(
                    source_id=self._id,
                    visual_id=vid,
                    field="color_map",
                    value=colormap,
                )
            )

    @property
    def widget(self):
        return self._combo

    def close(self) -> None:
        self.closed.emit()


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
        plane_visual=None,
        plane_store=None,
        gfx_vol_visual=None,
        initial_plane_opacity: float = 0.4,
        axes_2d_overlay_ids: list | None = None,
        orient_3d_visual_ids: list | None = None,
    ):
        from cellier.v2.gui.visuals._colormap import QtColormapComboBox
        from cellier.v2.gui.visuals._contrast_limits import QtClimRangeSlider
        from cellier.v2.gui.visuals._image import QtVolumeRenderControls
        from PySide6 import QtCore, QtWidgets

        self._controller = controller
        self._scenes = scenes
        self._visuals = visuals
        self._canvas_widgets = canvas_widgets

        xy_id = visuals["xy"].id
        xz_id = visuals["xz"].id
        yz_id = visuals["yz"].id
        vol_id = visuals["vol"].id

        self._2d_clim = _MultiVisualClimSlider(
            [xy_id, xz_id, yz_id],
            clim_range=clim_range,
            initial_clim=visuals["xy"].appearance.clim,
            decimals=slider_decimals,
        )
        controller.connect_widget(self._2d_clim)
        self._2d_colormap = _MultiVisualColormapCombo(
            [xy_id, xz_id, yz_id],
            initial_colormap=visuals["xy"].appearance.color_map,
        )
        controller.connect_widget(self._2d_colormap)

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
            self._3d_render, subscription_specs=self._3d_render.subscription_specs()
        )

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

        root_layout.addWidget(grid_widget, stretch=1)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        root_layout.addWidget(panel)

        group_2d = QtWidgets.QGroupBox("2D Rendering")
        layout_2d = QtWidgets.QVBoxLayout(group_2d)

        clim_2d_box = QtWidgets.QGroupBox("Contrast limits")
        QtWidgets.QVBoxLayout(clim_2d_box).addWidget(self._2d_clim.widget)
        layout_2d.addWidget(clim_2d_box)

        cmap_2d_box = QtWidgets.QGroupBox("Colormap")
        QtWidgets.QVBoxLayout(cmap_2d_box).addWidget(self._2d_colormap.widget)
        layout_2d.addWidget(cmap_2d_box)

        if axes_2d_overlay_ids:
            from PySide6.QtWidgets import QCheckBox

            axes_2d_cb = QCheckBox("Show orientation axes")
            axes_2d_cb.setChecked(True)

            def _on_axes_2d_toggled(checked: bool) -> None:
                for oid in axes_2d_overlay_ids:
                    controller.set_overlay_visible(oid, checked)

            axes_2d_cb.toggled.connect(_on_axes_2d_toggled)
            layout_2d.addWidget(axes_2d_cb)

        panel_layout.addWidget(group_2d)

        group_3d = QtWidgets.QGroupBox("3D Rendering")
        layout_3d = QtWidgets.QVBoxLayout(group_3d)

        clim_3d_box = QtWidgets.QGroupBox("Contrast limits")
        QtWidgets.QVBoxLayout(clim_3d_box).addWidget(self._3d_clim.widget)
        layout_3d.addWidget(clim_3d_box)

        cmap_3d_box = QtWidgets.QGroupBox("Colormap")
        QtWidgets.QVBoxLayout(cmap_3d_box).addWidget(self._3d_colormap.widget)
        layout_3d.addWidget(cmap_3d_box)

        render_3d_box = QtWidgets.QGroupBox("Render mode")
        QtWidgets.QVBoxLayout(render_3d_box).addWidget(self._3d_render.widget)
        layout_3d.addWidget(render_3d_box)

        if orient_3d_visual_ids:
            from cellier.v2.events import (
                AppearanceUpdateEvent as _AppearanceUpdateEvent,
            )
            from PySide6.QtWidgets import QCheckBox

            orient_3d_cb = QCheckBox("Show orientation axes")
            orient_3d_cb.setChecked(True)
            _orient_3d_bid = uuid4()

            def _on_orient_3d_toggled(checked: bool) -> None:
                for vid in orient_3d_visual_ids:
                    controller.incoming_events.emit(
                        _AppearanceUpdateEvent(
                            source_id=_orient_3d_bid,
                            visual_id=vid,
                            field="visible",
                            value=checked,
                        )
                    )

            orient_3d_cb.toggled.connect(_on_orient_3d_toggled)
            layout_3d.addWidget(orient_3d_cb)

        panel_layout.addWidget(group_3d)

        if plane_visual is not None or gfx_vol_visual is not None:
            from PySide6.QtCore import Qt
            from superqt import QLabeledDoubleSlider

            group_planes = QtWidgets.QGroupBox("Plane Overlay")
            layout_planes = QtWidgets.QVBoxLayout(group_planes)

            vol_opacity_label = QtWidgets.QLabel("Volume opacity")
            vol_opacity_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
            vol_opacity_slider.setRange(0.0, 1.0)
            vol_opacity_slider.setValue(1.0)
            vol_opacity_slider.setDecimals(2)

            def _on_vol_opacity_changed(value: float) -> None:
                if (
                    gfx_vol_visual is not None
                    and gfx_vol_visual.material_3d is not None
                ):
                    gfx_vol_visual.material_3d.opacity = value

            vol_opacity_slider.valueChanged.connect(_on_vol_opacity_changed)
            layout_planes.addWidget(vol_opacity_label)
            layout_planes.addWidget(vol_opacity_slider)

            plane_opacity_label = QtWidgets.QLabel("Plane opacity")
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

            plane_opacity_slider.valueChanged.connect(_on_plane_opacity_changed)
            layout_planes.addWidget(plane_opacity_label)
            layout_planes.addWidget(plane_opacity_slider)

            _on_plane_opacity_changed(initial_plane_opacity)

            panel_layout.addWidget(group_planes)

        panel_layout.addStretch()

    @property
    def window(self):
        return self._window

    def close_widgets(self) -> None:
        for cw in self._canvas_widgets.values():
            cw.close()
        self._2d_clim.close()
        self._2d_colormap.close()
        self._3d_clim.close()
        self._3d_colormap.close()
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


def _make_axis_meshes(
    controller,
    vol_scene,
    initial_centre_zyx: np.ndarray,
    world_min_extent: float,
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

    initial_translation = tuple(float(v) for v in initial_centre_zyx)
    initial_transform = AffineTransform.from_translation(initial_translation)

    axis_visuals = []
    for view_name, axis_a, axis_b, color_a, color_b in view_specifications:
        positions, indices = _make_axis_set_geometry(
            axis_a, axis_b, axis_length, cube_side, prism_cross_section
        )
        face_colors = _make_axis_set_face_colors(color_a, color_b)
        store = MeshMemoryStore(
            positions=positions, indices=indices, colors=face_colors, name=view_name
        )
        appearance = MeshFlatAppearance(
            color_mode="face",
            side="both",
            opacity=1.0,
            render_order=1,
            depth_test=False,
            depth_write=False,
            depth_compare="<=",
        )
        visual = controller.add_mesh(
            data=store,
            scene_id=vol_scene.id,
            appearance=appearance,
            name=view_name,
            transform=initial_transform,
        )
        axis_visuals.append(visual)

    return tuple(axis_visuals)


def _make_plane_positions(
    z_world: float,
    y_world: float,
    x_world: float,
    world_max_zyx: np.ndarray,
) -> np.ndarray:
    wz, wy, wx = (
        float(world_max_zyx[0]),
        float(world_max_zyx[1]),
        float(world_max_zyx[2]),
    )
    z, y, x = float(z_world), float(y_world), float(x_world)
    return np.array(
        [
            [z, 0, 0],
            [z, wy, 0],
            [z, wy, wx],
            [z, 0, wx],  # XY plane
            [0, y, 0],
            [wz, y, 0],
            [wz, y, wx],
            [0, y, wx],  # XZ plane
            [0, 0, x],
            [wz, 0, x],
            [wz, wy, x],
            [0, wy, x],  # YZ plane
        ],
        dtype=np.float32,
    )


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
):
    from cellier.v2.data.mesh._mesh_memory_store import MeshMemoryStore
    from cellier.v2.visuals._mesh_memory import MeshFlatAppearance

    positions = _make_plane_positions(z_world, y_world, x_world, world_max_zyx)
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
    def __init__(self, controller, plane_store, plane_visual, world_max_zyx) -> None:
        self._id = uuid4()
        self._controller = controller
        self._plane_store = plane_store
        self._plane_visual = plane_visual
        self._world_max_zyx = world_max_zyx

        positions = plane_store.positions
        self._z_world = float(positions[0, 0])
        self._y_world = float(positions[4, 1])
        self._x_world = float(positions[8, 2])

    def _update(self) -> None:
        self._plane_store.positions = _make_plane_positions(
            self._z_world, self._y_world, self._x_world, self._world_max_zyx
        )
        self._controller.reslice_visual(self._plane_visual.id)

    def on_xy_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        if 0 in slice_indices:
            self._z_world = float(slice_indices[0])
            self._update()

    def on_xz_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        if 1 in slice_indices:
            self._y_world = float(slice_indices[1])
            self._update()

    def on_yz_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        if 2 in slice_indices:
            self._x_world = float(slice_indices[2])
            self._update()


class _OrientationUpdater:
    def __init__(
        self,
        controller,
        xy_axis_visual,
        xz_axis_visual,
        yz_axis_visual,
        world_max_zyx: np.ndarray,
    ):
        self._id = uuid4()
        self._controller = controller
        self._xy_axis_visual_id = xy_axis_visual.id
        self._xz_axis_visual_id = xz_axis_visual.id
        self._yz_axis_visual_id = yz_axis_visual.id

        mid = world_max_zyx / 2.0
        self._xy_centre_zyx = mid.copy()
        self._xz_centre_zyx = mid.copy()
        self._yz_centre_zyx = mid.copy()
        self._z_world = float(mid[0])
        self._y_world = float(mid[1])
        self._x_world = float(mid[2])

    def _update_3d(self) -> None:
        from cellier.v2.transform import AffineTransform

        for visual_id, centre_zyx in zip(
            (self._xy_axis_visual_id, self._xz_axis_visual_id, self._yz_axis_visual_id),
            (self._xy_centre_zyx, self._xz_centre_zyx, self._yz_centre_zyx),
            strict=False,
        ):
            self._controller.set_visual_transform(
                visual_id,
                AffineTransform.from_translation(tuple(float(v) for v in centre_zyx)),
                reslice=False,
            )

    def on_xy_camera_changed(self, event) -> None:
        p = event.camera_state.position
        self._xy_centre_zyx = np.array([self._z_world, p[1], p[0]], dtype=np.float64)
        self._update_3d()

    def on_xz_camera_changed(self, event) -> None:
        p = event.camera_state.position
        self._xz_centre_zyx = np.array([p[1], self._y_world, p[0]], dtype=np.float64)
        self._update_3d()

    def on_yz_camera_changed(self, event) -> None:
        p = event.camera_state.position
        self._yz_centre_zyx = np.array([p[1], p[0], self._x_world], dtype=np.float64)
        self._update_3d()

    def on_xy_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        if 0 in slice_indices:
            self._z_world = float(slice_indices[0])
            self._xy_centre_zyx[0] = self._z_world
        self._update_3d()

    def on_xz_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        if 1 in slice_indices:
            self._y_world = float(slice_indices[1])
            self._xz_centre_zyx[1] = self._y_world
        self._update_3d()

    def on_yz_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        if 2 in slice_indices:
            self._x_world = float(slice_indices[2])
            self._yz_centre_zyx[2] = self._x_world
        self._update_3d()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dtype_clim_max(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def _dtype_decimals(dtype: np.dtype) -> int:
    return 0 if np.issubdtype(dtype, np.integer) else 2


# ---------------------------------------------------------------------------
# Layer 1: ViewerModel builder
# ---------------------------------------------------------------------------


def build_ortho_viewer_model(zarr_uri: str):
    """Build a ViewerModel for the orthoviewer without constructing any Qt objects.

    Parameters
    ----------
    zarr_uri : str
        Path or URI to the OME-Zarr store.

    Returns
    -------
    cellier.v2.viewer_model.ViewerModel
        Fully assembled model ready for ``CellierController.from_model``.
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
        ImageAppearance,
        MultiscaleImageRenderConfig,
        MultiscaleImageVisual,
    )

    print(f"Opening OME-Zarr store: {zarr_uri}")
    data_store = OMEZarrImageDataStore.from_path(zarr_uri)
    print(f"  {data_store.n_levels} levels found.")
    for i, shape in enumerate(data_store.level_shapes):
        print(f"  Level {i}: shape={shape}")
    print(f"  Axes:  {data_store.axis_names}")
    print(f"  Units: {data_store.axis_units}")

    group = yaozarrs.open_group(data_store.zarr_path)
    ome_image = group.ome_metadata()
    ms = ome_image.multiscales[data_store.multiscale_index]
    level_0_scale_zyx = np.array(ms.datasets[0].scale_transform.scale, dtype=np.float64)
    print(f"\n  Level-0 physical scale (ZYX): {level_0_scale_zyx}")

    vox_shape_zyx = np.array(data_store.level_shapes[0], dtype=np.float64)
    world_extents_zyx = vox_shape_zyx * level_0_scale_zyx
    max_extent = float(world_extents_zyx.max())
    depth_range = (max(1.0, max_extent * 0.0001), max_extent * 10.0)
    print(f"  World extents (ZYX): {world_extents_zyx}")
    print(f"  Depth range: near={depth_range[0]:.2f}  far={depth_range[1]:.0f}\n")

    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    voxel_to_world = AffineTransform.from_scale_and_translation(
        scale=tuple(level_0_scale_zyx)
    )

    initial_clim_max = _dtype_clim_max(data_store.dtype)
    world_max_zyx = (vox_shape_zyx - 1) * level_0_scale_zyx

    z_mid_world = round(float(world_max_zyx[0]) / 2.0)
    y_mid_world = round(float(world_max_zyx[1]) / 2.0)
    x_mid_world = round(float(world_max_zyx[2]) / 2.0)

    coarsest_level = data_store.n_levels - 1

    common_2d_appearance = ImageAppearance(
        color_map="grays",
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
        use_brick_shader=True,
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

    xy_visual = _make_2d_visual("xy_volume")
    xy_canvas = _make_2d_canvas()
    xy_scene = Scene(
        name="xy",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=(1, 2), slice_indices={0: z_mid_world}
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=[xy_visual],
        canvases={xy_canvas.id: xy_canvas},
    )

    xz_visual = _make_2d_visual("xz_volume")
    xz_canvas = _make_2d_canvas()
    xz_scene = Scene(
        name="xz",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=(0, 2), slice_indices={1: y_mid_world}
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=[xz_visual],
        canvases={xz_canvas.id: xz_canvas},
    )

    yz_visual = _make_2d_visual("yz_volume")
    yz_canvas = _make_2d_canvas()
    yz_scene = Scene(
        name="yz",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=(0, 1), slice_indices={2: x_mid_world}
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=[yz_visual],
        canvases={yz_canvas.id: yz_canvas},
    )

    vol_visual = MultiscaleImageVisual(
        name="vol_volume",
        data_store_id=str(data_store.id),
        level_transforms=data_store.level_transforms,
        appearance=ImageAppearance(
            color_map="grays",
            clim=(0.0, initial_clim_max),
            lod_bias=1.0,
            force_level=coarsest_level,
            frustum_cull=False,
            iso_threshold=0.2,
            render_mode="iso",
        ),
        render_config=MultiscaleImageRenderConfig(
            block_size=32,
            gpu_budget_bytes=2048 * 1024**2,
            gpu_budget_bytes_2d=64 * 1024**2,
            use_brick_shader=True,
        ),
        transform=voxel_to_world,
    )
    vol_visual.aabb.enabled = True
    vol_visual.aabb.color = "#ff00ff"

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
    vol_scene = Scene(
        name="vol",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(displayed_axes=(0, 1, 2), slice_indices={}),
        ),
        render_modes={"3d"},
        lighting="none",
        visuals=[vol_visual],
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

    return viewer_model


# ---------------------------------------------------------------------------
# Layer 2: Qt bootstrap helper
# ---------------------------------------------------------------------------


def _ensure_qt_app():
    """Return the active QApplication, creating one via IPython if needed.

    Returns None if not in an interactive environment and no QApplication
    exists — callers should raise a useful error in that case.
    """
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
# Layer 3: Non-blocking show (for interactive / Jupyter use)
# ---------------------------------------------------------------------------


def orthoviewer(zarr_uri: str, theme: str = "dark") -> OmeZarrOrthoViewer:
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

    return _build_and_show(zarr_uri, theme=theme)


# ---------------------------------------------------------------------------
# Layer 4: Private async core
# ---------------------------------------------------------------------------


def _asyncio_exception_handler(context: dict) -> None:
    """Custom asyncio exception handler that works around two PySide6 bugs.

    Bug 1: PySide6's default_exception_handler unconditionally accesses
    context['task'], but the asyncio spec makes 'task' optional, causing a
    KeyError that swallows the original exception message.

    Bug 2: PySide6's QtAsyncio routes CancelledError to the exception handler
    instead of letting it propagate as normal task cancellation. CancelledError
    is how cellier cancels stale chunk fetches when the slice position changes —
    it is expected and should be silently ignored.
    """
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


async def _run_orthoviewer_async(zarr_uri: str, theme: str = "dark") -> None:
    import asyncio as _asyncio

    from PySide6.QtWidgets import QApplication

    _asyncio.get_event_loop().set_exception_handler(_asyncio_exception_handler)

    viewer = _build_and_show(zarr_uri, theme=theme)

    app = QApplication.instance()
    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    app.aboutToQuit.connect(viewer.close_widgets)
    await close_event.wait()


# ---------------------------------------------------------------------------
# Layer 5: Blocking launcher (for scripts and CLI)
# ---------------------------------------------------------------------------


def launch_orthoviewer(zarr_uri: str, theme: str = "dark") -> None:
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
    """
    import sys

    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([sys.argv[0]])  # noqa: F841
    QtAsyncio.run(_run_orthoviewer_async(zarr_uri, theme=theme), handle_sigint=True)


# ---------------------------------------------------------------------------
# Shared builder (used by both orthoviewer and _run_orthoviewer_async)
# ---------------------------------------------------------------------------


def _build_and_show(zarr_uri: str, theme: str = "dark") -> OmeZarrOrthoViewer:
    """Build the full viewer from a zarr URI and show the window."""
    from PySide6.QtWidgets import QApplication

    from oz_viewer.theme import apply_theme

    apply_theme(QApplication.instance(), theme)
    import yaozarrs
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

    viewer_model = build_ortho_viewer_model(zarr_uri)

    controller = CellierController.from_model(
        viewer_model,
        render_config=RenderManagerConfig(
            slicing=SlicingConfig(batch_size=32, render_every=4),
            temporal=TemporalAccumulationConfig(enabled=False),
        ),
        widget_parent=None,
    )

    xy_scene = controller.get_scene_by_name("xy")
    xz_scene = controller.get_scene_by_name("xz")
    yz_scene = controller.get_scene_by_name("yz")
    vol_scene = controller.get_scene_by_name("vol")
    scenes = {"xy": xy_scene, "xz": xz_scene, "yz": yz_scene, "vol": vol_scene}

    # Retrieve visuals from the model scenes
    def _first_visual(scene):
        return next(iter(scene.visuals))

    visuals = {
        "xy": _first_visual(xy_scene),
        "xz": _first_visual(xz_scene),
        "yz": _first_visual(yz_scene),
        "vol": _first_visual(vol_scene),
    }

    # Rebuild world geometry parameters needed for widgets/overlays
    data_store = next(iter(viewer_model.data.stores.values()))
    group = yaozarrs.open_group(data_store.zarr_path)
    ome_image = group.ome_metadata()
    ms = ome_image.multiscales[data_store.multiscale_index]
    level_0_scale_zyx = np.array(ms.datasets[0].scale_transform.scale, dtype=np.float64)
    vox_shape_zyx = np.array(data_store.level_shapes[0], dtype=np.float64)
    world_max_zyx = (vox_shape_zyx - 1) * level_0_scale_zyx

    initial_clim_max = _dtype_clim_max(data_store.dtype)
    slider_decimals = _dtype_decimals(data_store.dtype)
    clim_range = (0.0, initial_clim_max)

    level0_shape = data_store.level_shapes[0]
    axis_ranges = {
        i: (0, round(float(world_max_zyx[i]))) for i in range(len(level0_shape))
    }

    z_mid_world = round(float(world_max_zyx[0]) / 2.0)
    y_mid_world = round(float(world_max_zyx[1]) / 2.0)
    x_mid_world = round(float(world_max_zyx[2]) / 2.0)

    # 3D axis-set mesh overlays
    initial_centre_zyx = np.array(
        [z_mid_world, y_mid_world, x_mid_world], dtype=np.float64
    )
    xy_axis_visual, xz_axis_visual, yz_axis_visual = _make_axis_meshes(
        controller=controller,
        vol_scene=vol_scene,
        initial_centre_zyx=initial_centre_zyx,
        world_min_extent=float(world_max_zyx.min()),
    )

    # Slice plane mesh overlay
    _INITIAL_PLANE_OPACITY = 1.0
    plane_store, plane_visual = _make_plane_mesh(
        controller,
        vol_scene,
        z_mid_world,
        y_mid_world,
        x_mid_world,
        world_max_zyx,
        initial_opacity=_INITIAL_PLANE_OPACITY,
    )

    plane_updater = _PlaneUpdater(
        controller=controller,
        plane_store=plane_store,
        plane_visual=plane_visual,
        world_max_zyx=world_max_zyx,
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

    scene_mgr = controller._render_manager._scenes[vol_scene.id]
    gfx_vol_visual = scene_mgr.get_visual(visuals["vol"].id)

    def _canvas_view(scene_id):
        canvas_id = controller.get_canvas_ids(scene_id)[0]
        return controller.get_canvas_view(canvas_id)

    def _make_canvas_widget(scene, slider_style):
        canvas_view = _canvas_view(scene.id)
        axis_labels = dict(enumerate(scene.dims.coordinate_system.axis_labels))
        selection = scene.dims.selection
        dims_sliders = QtDimsSliders(
            scene_id=scene.id,
            axis_ranges=axis_ranges,
            axis_labels=axis_labels,
            initial_slice_indices=dict(getattr(selection, "slice_indices", {})),
            initial_displayed_axes=getattr(selection, "displayed_axes", ()),
        )
        dims_sliders.widget.setStyleSheet(slider_style)
        cw = QtCanvasWidget(canvas_view=canvas_view, dims_sliders=dims_sliders)
        controller.connect_widget(
            dims_sliders, subscription_specs=dims_sliders.subscription_specs()
        )
        return cw

    vol_cw = QtCanvasWidget.from_scene_and_canvas(
        vol_scene, _canvas_view(vol_scene.id), axis_ranges=axis_ranges
    )
    controller.connect_widget(
        vol_cw.dims_sliders, subscription_specs=vol_cw.dims_sliders.subscription_specs()
    )

    canvas_widgets = {
        "xy": _make_canvas_widget(xy_scene, _SLIDER_STYLE_XY),
        "xz": _make_canvas_widget(xz_scene, _SLIDER_STYLE_XZ),
        "yz": _make_canvas_widget(yz_scene, _SLIDER_STYLE_YZ),
        "vol": vol_cw,
    }

    # Screen-space 2D axis overlays
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

    viewer = OmeZarrOrthoViewer(
        controller,
        scenes=scenes,
        visuals=visuals,
        canvas_widgets=canvas_widgets,
        clim_range=clim_range,
        slider_decimals=slider_decimals,
        plane_visual=plane_visual,
        plane_store=plane_store,
        gfx_vol_visual=gfx_vol_visual,
        initial_plane_opacity=_INITIAL_PLANE_OPACITY,
        axes_2d_overlay_ids=[
            xy_axes_overlay.id,
            xz_axes_overlay.id,
            yz_axes_overlay.id,
        ],
        orient_3d_visual_ids=[xy_axis_visual.id, xz_axis_visual.id, yz_axis_visual.id],
    )
    viewer.window.show()

    # 3D orientation overlay wiring
    orient_updater = _OrientationUpdater(
        controller=controller,
        xy_axis_visual=xy_axis_visual,
        xz_axis_visual=xz_axis_visual,
        yz_axis_visual=yz_axis_visual,
        world_max_zyx=world_max_zyx,
    )

    controller.on_camera_changed(
        xy_scene.id, orient_updater.on_xy_camera_changed, owner_id=orient_updater._id
    )
    controller.on_camera_changed(
        xz_scene.id, orient_updater.on_xz_camera_changed, owner_id=orient_updater._id
    )
    controller.on_camera_changed(
        yz_scene.id, orient_updater.on_yz_camera_changed, owner_id=orient_updater._id
    )
    controller.on_dims_changed(
        xy_scene.id, orient_updater.on_xy_dims_changed, owner_id=orient_updater._id
    )
    controller.on_dims_changed(
        xz_scene.id, orient_updater.on_xz_dims_changed, owner_id=orient_updater._id
    )
    controller.on_dims_changed(
        yz_scene.id, orient_updater.on_yz_dims_changed, owner_id=orient_updater._id
    )

    for scene in scenes.values():
        controller.fit_camera(scene.id)
        controller.reslice_scene(scene.id)

    # Seed 3D orientation with post-fit camera state
    def _seed_camera_event(scene_id):
        from cellier.v2.events._events import CameraChangedEvent

        canvas_view = controller.get_canvas_view(controller.get_canvas_ids(scene_id)[0])
        camera_state = canvas_view._capture_camera_state()
        return CameraChangedEvent(
            source_id=canvas_view._canvas_id,
            scene_id=scene_id,
            camera_state=camera_state,
        )

    orient_updater.on_xy_camera_changed(_seed_camera_event(xy_scene.id))
    orient_updater.on_xz_camera_changed(_seed_camera_event(xz_scene.id))
    orient_updater.on_yz_camera_changed(_seed_camera_event(yz_scene.id))

    return viewer
