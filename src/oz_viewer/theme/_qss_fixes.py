"""QSS fix registry for Fusion-style compatibility corrections.

When :func:`oz_viewer.theme.apply_theme` sets ``QStyle("Fusion")``, some
third-party and custom Qt widgets require explicit QSS geometry hints to
render correctly.  This module maintains a registry of named QSS snippets
that are concatenated and applied at the application level by
:func:`oz_viewer.theme.apply_theme`.

All color values in the built-in snippets use ``palette()`` references so
they resolve against whatever theme palette is active at render time—no
hex literals are hard-coded here.

Built-in fixes
--------------
slider
    Fixes ``QSlider`` groove height and active-range vertical alignment for
    both standard ``QSlider`` and superqt ``QLabeledDoubleRangeSlider`` under
    Fusion.

Adding a new fix
----------------
Call :func:`register_qss` *before* :func:`~oz_viewer.theme.apply_theme`::

    from oz_viewer.theme import register_qss, apply_theme

    register_qss("my-widget", "MyWidget::item { ... }")
    apply_theme(app, "dark")

Because the registry is module-level, fixes registered in library
initialisation code are automatically included without any extra wiring.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps a fix name to its QSS snippet.  Plain dict gives O(1) lookup and
# preserves insertion order, which keeps the concatenated output stable.
_fixes: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Built-in fix: QSlider / QLabeledDoubleRangeSlider
# ---------------------------------------------------------------------------

# Under Fusion, QSlider sub-elements must all be styled via QSS for any
# individual override to take effect.  This snippet establishes consistent
# groove geometry so superqt's custom active-range painting aligns with the
# groove rect.  All colours use palette() references and therefore adapt to
# whatever Theme palette is active.
_SLIDER_QSS: str = """\
QSlider::groove:horizontal {
    height: 6px;
    background: palette(mid);
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    height: 6px;
    background: palette(highlight);
    border-radius: 3px;
}
QSlider::add-page:horizontal {
    height: 6px;
    background: palette(mid);
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
    background: palette(button);
    border: 1px solid palette(shadow);
}
QSlider::handle:horizontal:hover {
    background: palette(light);
}"""

_fixes["slider"] = _SLIDER_QSS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_qss(name: str, qss: str) -> None:
    """Register a QSS snippet under *name*, replacing any existing entry.

    The snippet is included in the application-level stylesheet the next time
    :func:`oz_viewer.theme.apply_theme` is called.  Use ``palette()``
    references (e.g. ``palette(highlight)``) rather than literal hex values
    so the snippet adapts automatically to any theme.

    Parameters
    ----------
    name : str
        Unique identifier for the snippet.  Existing entries with the same
        name are silently replaced.
    qss : str
        Valid Qt Style Sheet text to register.
    """
    _fixes[name] = qss


def list_qss_fixes() -> list[str]:
    """Return the names of all currently registered QSS fixes.

    Returns
    -------
    list of str
        Fix names in registration order.
    """
    return list(_fixes.keys())


def get_fusion_stylesheet() -> str:
    """Return the full application-level stylesheet from all registered fixes.

    Snippets are joined with a blank line separator.  The result is intended
    to be passed directly to ``QApplication.setStyleSheet``.

    Returns
    -------
    str
        Concatenated QSS string for all registered fixes.
    """
    return "\n\n".join(_fixes.values())
