"""Shared GUI widgets used by both the single-panel viewer and the orthoviewer."""

from __future__ import annotations

from uuid import uuid4

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
# Multi-visual control widgets
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

    def _on_visual_changed(self, event) -> None:
        if event.source_id == self._id:
            return
        if event.field_name != "clim":
            return
        self._slider.blockSignals(True)
        self._slider.setValue(event.new_value)
        self._slider.blockSignals(False)

    def subscription_specs(self) -> list:
        from cellier.v2.events import AppearanceChangedEvent, SubscriptionSpec

        if not self._visual_ids:
            return []
        return [
            SubscriptionSpec(
                event_type=AppearanceChangedEvent,
                handler=self._on_visual_changed,
                entity_id=self._visual_ids[0],
            )
        ]

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
        self._combo.addColormaps(_DEFAULT_COLORMAPS)
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

    def _on_visual_changed(self, event) -> None:
        if event.source_id == self._id:
            return
        if event.field_name != "color_map":
            return
        self._combo.blockSignals(True)
        self._combo.setCurrentColormap(event.new_value)
        self._combo.blockSignals(False)

    def subscription_specs(self) -> list:
        from cellier.v2.events import AppearanceChangedEvent, SubscriptionSpec

        if not self._visual_ids:
            return []
        return [
            SubscriptionSpec(
                event_type=AppearanceChangedEvent,
                handler=self._on_visual_changed,
                entity_id=self._visual_ids[0],
            )
        ]

    @property
    def widget(self):
        return self._combo

    def close(self) -> None:
        self.closed.emit()


class _MultiVisualLodBiasSlider:
    """LOD-bias slider that updates multiple visuals at once."""

    from psygnal import Signal

    changed = Signal(object)
    closed = Signal()

    def __init__(
        self,
        visual_ids: list,
        *,
        initial_lod_bias: float = 1.0,
        lod_range: tuple[float, float] = (1e-6, 5.0),
        decimals: int = 2,
        parent=None,
    ) -> None:
        from cellier.v2.events import AppearanceUpdateEvent
        from qtpy.QtCore import Qt
        from superqt import QLabeledDoubleSlider

        self._id = uuid4()
        self._visual_ids = visual_ids
        self._AppearanceUpdateEvent = AppearanceUpdateEvent

        self._slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent)
        self._slider.setRange(*lod_range)
        self._slider.setDecimals(decimals)
        self._slider.setValue(initial_lod_bias)

        # Fire only on release to avoid a reslice on every drag tick.
        self._slider.sliderReleased.connect(self._on_released)

    def _on_released(self) -> None:
        value = self._slider.value()
        for vid in self._visual_ids:
            self.changed.emit(
                self._AppearanceUpdateEvent(
                    source_id=self._id,
                    visual_id=vid,
                    field="lod_bias",
                    value=value,
                )
            )

    def _on_visual_changed(self, event) -> None:
        if event.source_id == self._id:
            return
        if event.field_name != "lod_bias":
            return
        self._slider.blockSignals(True)
        self._slider.setValue(event.new_value)
        self._slider.blockSignals(False)

    def subscription_specs(self) -> list:
        from cellier.v2.events import AppearanceChangedEvent, SubscriptionSpec

        if not self._visual_ids:
            return []
        return [
            SubscriptionSpec(
                event_type=AppearanceChangedEvent,
                handler=self._on_visual_changed,
                entity_id=self._visual_ids[0],
            )
        ]

    @property
    def widget(self):
        return self._slider

    def close(self) -> None:
        self.closed.emit()


# ---------------------------------------------------------------------------
# Per-channel control builders
# ---------------------------------------------------------------------------


def build_channel_group(
    ch_idx: int,
    ch_appearance,
    clim_range: tuple[float, float],
    slider_decimals: int,
):
    """Group for visibility, colormap, clim, and opacity controls for 1 channel."""
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt
    from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider
    from superqt.cmap import QColormapComboBox

    group = QtWidgets.QGroupBox(f"Channel {ch_idx}")
    layout = QtWidgets.QVBoxLayout(group)

    vis_cb = QtWidgets.QCheckBox("Visible")
    vis_cb.setChecked(ch_appearance.visible)
    vis_cb.stateChanged.connect(
        lambda state, _ch=ch_appearance: setattr(_ch, "visible", bool(state))
    )
    ch_appearance.events.visible.connect(
        lambda v, _cb=vis_cb: (
            _cb.blockSignals(True),
            _cb.setChecked(v),
            _cb.blockSignals(False),
        )
    )
    layout.addWidget(vis_cb)

    combo = QColormapComboBox()
    combo.addColormaps(_DEFAULT_COLORMAPS)
    combo.setCurrentColormap(ch_appearance.color_map)
    combo.currentColormapChanged.connect(
        lambda cmap, _ch=ch_appearance: setattr(_ch, "color_map", cmap)
    )
    ch_appearance.events.color_map.connect(lambda v, _c=combo: _c.setCurrentColormap(v))
    layout.addWidget(combo)

    clim_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
    clim_slider.setDecimals(slider_decimals)
    clim_slider.setRange(*clim_range)
    clim_slider.setValue(ch_appearance.clim)
    clim_slider.valueChanged.connect(
        lambda v, _ch=ch_appearance: setattr(_ch, "clim", tuple(v))
    )
    ch_appearance.events.clim.connect(
        lambda v, _s=clim_slider: (
            _s.blockSignals(True),
            _s.setValue(v),
            _s.blockSignals(False),
        )
    )
    layout.addWidget(clim_slider)

    opacity_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
    opacity_slider.setRange(0.0, 1.0)
    opacity_slider.setSingleStep(0.05)
    opacity_slider.setValue(ch_appearance.opacity)
    opacity_slider.valueChanged.connect(
        lambda v, _ch=ch_appearance: setattr(_ch, "opacity", v)
    )
    ch_appearance.events.opacity.connect(
        lambda v, _s=opacity_slider: (
            _s.blockSignals(True),
            _s.setValue(v),
            _s.blockSignals(False),
        )
    )
    layout.addWidget(opacity_slider)

    return group


def build_channel_list_widget(
    channel_appearances: dict,
    clim_range: tuple[float, float],
    slider_decimals: int,
):
    """Return a widget containing per-channel control groups.

    Uses a QScrollArea when there are more than 3 channels so the panel does
    not overflow; otherwise returns a plain container widget.
    """
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt

    use_scroll = len(channel_appearances) > 3

    container = QtWidgets.QWidget()
    container_layout = QtWidgets.QVBoxLayout(container)
    container_layout.setContentsMargins(0, 0, 0, 0)
    container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    for i, ch in channel_appearances.items():
        container_layout.addWidget(
            build_channel_group(i, ch, clim_range, slider_decimals)
        )

    if not use_scroll:
        return container

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(container)
    return scroll
