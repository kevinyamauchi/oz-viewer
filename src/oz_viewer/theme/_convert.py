"""QPalette conversion helpers for the oz-viewer theme system.

Translates :class:`~oz_viewer.theme._model.ThemePalette` instances into
``PySide6.QtGui.QPalette`` objects ready for ``QApplication.setPalette``.

Notes
-----
``cmap.Color.rgba8`` returns an ``RGBA8`` named-tuple whose ``r``, ``g``,
``b`` fields are 0-255 integers but whose ``a`` field is a 0-1 float.
All colour construction in this module multiplies ``a`` by 255 before
passing it to ``QColor`` to avoid nearly-transparent rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cmap import Color
    from PySide6.QtGui import QPalette

    from oz_viewer.theme._model import PaletteColorGroup, ThemePalette


# Maps PaletteColorGroup field names to QPalette.ColorRole enum member names.
_ROLE_MAP: dict[str, str] = {
    "window": "Window",
    "window_text": "WindowText",
    "base": "Base",
    "alternate_base": "AlternateBase",
    "text": "Text",
    "bright_text": "BrightText",
    "button": "Button",
    "button_text": "ButtonText",
    "highlight": "Highlight",
    "highlighted_text": "HighlightedText",
    "tool_tip_base": "ToolTipBase",
    "tool_tip_text": "ToolTipText",
    "link": "Link",
    "link_visited": "LinkVisited",
}

# Text roles whose alpha is halved when building the Disabled color group.
_DISABLED_TEXT_ROLES: frozenset[str] = frozenset(
    {"window_text", "text", "button_text", "highlighted_text", "tool_tip_text"}
)


def _cmap_to_qcolor(color: Color):
    """Convert a ``cmap.Color`` to a fully-opaque ``QColor``.

    Parameters
    ----------
    color : cmap.Color
        Source color.  Any alpha information is preserved.

    Returns
    -------
    PySide6.QtGui.QColor
        Equivalent ``QColor`` with alpha in 0-255 integer range.
    """
    from PySide6.QtGui import QColor

    r8 = color.rgba8
    # rgba8.a is a 0-1 float; r/g/b are 0-255 integers.
    return QColor(r8.r, r8.g, r8.b, round(r8.a * 255))


def _apply_group(
    palette: QPalette,
    group: object,
    color_group: PaletteColorGroup,
    *,
    dimmed: bool = False,
) -> None:
    """Write one color group's roles into *palette*.

    Parameters
    ----------
    palette : PySide6.QtGui.QPalette
        The palette to mutate in-place.
    group : PySide6.QtGui.QPalette.ColorGroup
        Which color group to populate (Active, Inactive, or Disabled).
    color_group : PaletteColorGroup
        Source color values.
    dimmed : bool
        When ``True``, text roles are set to 50 % alpha to represent the
        disabled state.  Non-text roles are written at full opacity.
    """
    from PySide6.QtGui import QColor
    from PySide6.QtGui import QPalette as _QPalette

    for field_name, role_name in _ROLE_MAP.items():
        color: Color | None = getattr(color_group, field_name)
        if color is None:
            continue
        role = getattr(_QPalette.ColorRole, role_name)
        r8 = color.rgba8
        # rgba8.a is 0-1 float; multiply by 255 for QColor's 0-255 scale.
        a = round(r8.a * 255)
        if dimmed and field_name in _DISABLED_TEXT_ROLES:
            qc = QColor(r8.r, r8.g, r8.b, 128)
        else:
            qc = QColor(r8.r, r8.g, r8.b, a)
        palette.setColor(group, role, qc)


def theme_palette_to_qpalette(theme_palette: ThemePalette) -> QPalette:
    """Build a ``QPalette`` from a :class:`~oz_viewer.theme._model.ThemePalette`.

    Populates all three color groups.  When ``theme_palette.disabled`` is
    ``None``, the Disabled group is derived from ``active`` with text roles
    dimmed to 50 % alpha.

    Parameters
    ----------
    theme_palette : ThemePalette
        Validated, frozen theme palette model.

    Returns
    -------
    PySide6.QtGui.QPalette
        Fully populated palette ready for ``QApplication.setPalette``.
    """
    from PySide6.QtGui import QPalette as _QPalette

    pal = _QPalette()
    _apply_group(pal, _QPalette.ColorGroup.Active, theme_palette.active)
    _apply_group(pal, _QPalette.ColorGroup.Inactive, theme_palette.inactive)

    if theme_palette.disabled is not None:
        _apply_group(pal, _QPalette.ColorGroup.Disabled, theme_palette.disabled)
    else:
        _apply_group(
            pal,
            _QPalette.ColorGroup.Disabled,
            theme_palette.active,
            dimmed=True,
        )

    return pal
