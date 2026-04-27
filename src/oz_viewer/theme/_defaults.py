"""Built-in theme definitions as plain Python dicts.

Storing themes as dicts rather than JSON files avoids file I/O at import
time and removes the need for ``importlib.resources``.  Pydantic validation
into :class:`~oz_viewer.theme._model.Theme` objects happens lazily on first
access via :func:`~oz_viewer.theme._registry.get_theme`.

Color values follow the same rules as ``cmap.Color``: any CSS color name,
hex string, or ``rgb()``/``rgba()`` string is accepted.

Notes
-----
The dark palette is modelled on a mid-grey charcoal scheme.
The light palette mirrors the Qt Fusion default light appearance.
Both can be used as starting points in
``scripts/theme_picker.py`` and saved to custom JSON theme files.
"""

from __future__ import annotations

DEFAULT_DARK: dict = {
    "name": "dark",
    "palette": {
        "active": {
            "window": "#2d2d2d",
            "window_text": "#dcdcdc",
            "base": "#1e1e1e",
            "alternate_base": "#252525",
            "text": "#dcdcdc",
            "bright_text": "#ffffff",
            "button": "#3c3c3c",
            "button_text": "#dcdcdc",
            "highlight": "#2a82da",
            "highlighted_text": "#ffffff",
            "tool_tip_base": "#3c3c3c",
            "tool_tip_text": "#dcdcdc",
            "link": "#5aabff",
        }
    },
}

DEFAULT_LIGHT: dict = {
    "name": "light",
    "palette": {
        "active": {
            "window": "#f0f0f0",
            "window_text": "#1a1a1a",
            "base": "#ffffff",
            "alternate_base": "#f5f5f5",
            "text": "#1a1a1a",
            "bright_text": "#000000",
            "button": "#e0e0e0",
            "button_text": "#1a1a1a",
            "highlight": "#2a82da",
            "highlighted_text": "#ffffff",
            "tool_tip_base": "#ffffdc",
            "tool_tip_text": "#1a1a1a",
            "link": "#0057ae",
        }
    },
}
