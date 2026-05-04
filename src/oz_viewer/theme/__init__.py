"""Theme system for oz-viewer.

Provides palette-based theming for Qt applications using the Fusion style.
The public API covers four concerns:

* **Theme registry** — :func:`register_theme`, :func:`get_theme`,
  :func:`list_themes`.
* **QSS fix registry** — :func:`register_qss`, :func:`list_qss_fixes`.
  Fixes are applied application-wide on every :func:`apply_theme` call to
  correct Fusion-style rendering quirks in third-party widgets such as
  superqt's ``QLabeledDoubleRangeSlider``.
* **File I/O** — :func:`load_theme_file`, :func:`save_theme_file`.
* **Application** — :func:`apply_theme`.

Quick start
-----------
Apply the built-in dark theme::

    from oz_viewer.theme import apply_theme

    apply_theme(app, "dark")

Register a custom theme from a JSON file::

    from oz_viewer.theme import load_theme_file, register_theme

    register_theme("my-theme", load_theme_file("my_theme.json"))
    apply_theme(app, "my-theme")

Register a custom QSS fix for a third-party widget::

    from oz_viewer.theme import register_qss, apply_theme

    register_qss("my-widget", "MyWidget::handle { ... }")
    apply_theme(app, "dark")

List available themes and fixes::

    from oz_viewer.theme import list_themes, list_qss_fixes

    print(list_themes())
    print(list_qss_fixes())
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

    from oz_viewer.theme._model import Theme

from oz_viewer.theme._qss_fixes import list_qss_fixes, register_qss
from oz_viewer.theme._registry import get_theme, list_themes, register_theme

__all__ = [
    "apply_theme",
    "get_theme",
    "list_qss_fixes",
    "list_themes",
    "load_theme_file",
    "register_qss",
    "register_theme",
    "save_theme_file",
]


def load_theme_file(path: str | Path) -> Theme:
    """Load and validate a :class:`~oz_viewer.theme._model.Theme` from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to a JSON file produced by :func:`save_theme_file` or
        ``scripts/theme_picker.py``.

    Returns
    -------
    Theme
        Validated, frozen theme model.
    """
    from oz_viewer.theme._model import Theme as _Theme

    return _Theme.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_theme_file(theme: Theme, path: str | Path) -> None:
    """Serialize *theme* to a JSON file.

    The output format is compatible with :func:`load_theme_file` and with the
    ``--load`` option of ``scripts/theme_picker.py``.

    Parameters
    ----------
    theme : Theme
        Theme to serialize.
    path : str or Path
        Destination file path.  Parent directories must already exist.
    """
    Path(path).write_text(theme.model_dump_json(indent=2), encoding="utf-8")


def apply_theme(app: QApplication, theme: str | Theme) -> None:
    """Apply *theme* to *app*, updating the palette and application stylesheet.

    Three things happen in order:

    1. ``app.setStyle("Fusion")`` — enables full palette respect across
       platforms.
    2. ``app.setPalette(...)`` — propagates semantic colors to every widget.
    3. ``app.setStyleSheet(...)`` — applies all registered QSS fixes (see
       :func:`register_qss`) to correct Fusion-style rendering quirks in
       third-party widgets.

    Because the stylesheet uses ``palette()`` references, colors in QSS fixes
    automatically reflect the palette set in step 2.

    Widget-level stylesheets (set via ``widget.setStyleSheet()``) take
    precedence over the application-level stylesheet and are unaffected by
    this call.

    Parameters
    ----------
    app : PySide6.QtWidgets.QApplication
        The running Qt application instance.
    theme : str or Theme
        A registered theme name (e.g. ``"dark"``) or a
        :class:`~oz_viewer.theme._model.Theme` object.

    Raises
    ------
    KeyError
        When *theme* is a string not found in the registry.
    TypeError
        When *theme* is neither a string nor a
        :class:`~oz_viewer.theme._model.Theme`.
    """
    from oz_viewer.theme._model import Theme as _Theme
    from oz_viewer.theme._qss_fixes import get_fusion_stylesheet

    if isinstance(theme, str):
        theme = get_theme(theme)

    if not isinstance(theme, _Theme):
        raise TypeError(f"Expected a Theme or theme name str, got {type(theme)!r}")

    app.setStyle("Fusion")
    app.setPalette(theme.palette.to_qpalette())
    app.setStyleSheet(get_fusion_stylesheet())
