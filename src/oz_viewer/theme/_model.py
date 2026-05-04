"""Pydantic models for the oz-viewer theme system.

Provides three frozen models that form a strict hierarchy:

* :class:`PaletteColorGroup` — colors for one ``QPalette.ColorGroup``.
* :class:`ThemePalette` — active, inactive, and disabled groups.
* :class:`Theme` — a named theme containing one :class:`ThemePalette`.
"""

from __future__ import annotations

from cmap import Color  # noqa: TC002 — Pydantic resolves annotations at runtime
from pydantic import BaseModel, ConfigDict, model_validator


class PaletteColorGroup(BaseModel):
    """Color values for one ``QPalette.ColorGroup``.

    Each field maps directly to a ``QPalette.ColorRole`` (snake_case to
    CamelCase).  The derived roles ``Light``, ``Midlight``, ``Dark``,
    ``Mid``, and ``Shadow`` are intentionally absent; the Fusion style
    computes them automatically from ``window`` and ``button``.

    Attributes
    ----------
    window : Color
        Background of main windows, panels, and group boxes.
    window_text : Color
        Foreground text drawn directly on the window background.
    base : Color
        Background of input widgets (``QLineEdit``, ``QListWidget``, …).
    alternate_base : Color
        Alternating-row background in item views.
    text : Color
        Foreground text inside input widgets.
    bright_text : Color
        High-contrast foreground used for emphasis or warnings.
    button : Color
        Background of push buttons and related controls.
    button_text : Color
        Foreground text on buttons.
    highlight : Color
        Background of selected items and the active range in range sliders.
    highlighted_text : Color
        Foreground text of selected items.
    tool_tip_base : Color
        Background of tool-tip pop-ups.
    tool_tip_text : Color
        Foreground text of tool-tip pop-ups.
    link : Color
        Hyperlink text color.
    link_visited : Color or None
        Visited hyperlink text color.  Falls back to ``link`` when ``None``.
    """

    model_config = ConfigDict(frozen=True)

    window: Color
    window_text: Color
    base: Color
    alternate_base: Color
    text: Color
    bright_text: Color
    button: Color
    button_text: Color
    highlight: Color
    highlighted_text: Color
    tool_tip_base: Color
    tool_tip_text: Color
    link: Color
    link_visited: Color | None = None


class ThemePalette(BaseModel):
    """Complete ``QPalette`` specification with active, inactive, and disabled groups.

    Only ``active`` is required.  ``inactive`` mirrors ``active`` when omitted
    (preventing the focus-loss colour shift that occurs when only the Active
    group is set).  ``disabled`` is auto-derived when omitted: all roles copy
    ``active`` with text roles dimmed to 50 % alpha.

    Attributes
    ----------
    active : PaletteColorGroup
        Colors used when the widget's window has keyboard focus.
    inactive : PaletteColorGroup or None
        Colors used when the widget's window lacks focus.  Mirrors ``active``
        when ``None``.
    disabled : PaletteColorGroup or None
        Colors used for disabled widgets.  Auto-derived from ``active`` when
        ``None``.
    """

    model_config = ConfigDict(frozen=True)

    active: PaletteColorGroup
    inactive: PaletteColorGroup | None = None
    disabled: PaletteColorGroup | None = None

    @model_validator(mode="after")
    def _mirror_inactive(self) -> ThemePalette:
        """Mirror ``active`` into ``inactive`` when the latter is absent.

        Returns
        -------
        ThemePalette
            The validated model instance, with ``inactive`` guaranteed
            non-``None``.
        """
        if self.inactive is None:
            object.__setattr__(self, "inactive", self.active)
        return self

    def to_qpalette(self):
        """Convert this palette to a ``QPalette`` suitable for ``QApplication``.

        Returns
        -------
        PySide6.QtGui.QPalette
            A fully populated ``QPalette`` covering the Active, Inactive, and
            Disabled color groups.
        """
        from oz_viewer.theme._convert import theme_palette_to_qpalette

        return theme_palette_to_qpalette(self)


class Theme(BaseModel):
    """A named theme containing a single :class:`ThemePalette`.

    Dark and light variants are represented as separate ``Theme`` instances
    registered under distinct names (e.g. ``"dark"`` and
    ``"light"``).

    Attributes
    ----------
    name : str
        Human-readable identifier, also used as the registry key.
    palette : ThemePalette
        The color data for this theme.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    palette: ThemePalette
