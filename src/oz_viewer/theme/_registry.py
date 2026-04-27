"""Theme registry for oz-viewer.

Maintains a module-level dictionary that maps theme names to either a
validated :class:`~oz_viewer.theme._model.Theme` instance or a raw ``dict``
pending first-use validation.  The lazy-validation approach keeps import
cost near zero: no pydantic models are instantiated until a theme is actually
requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oz_viewer.theme._model import Theme

from oz_viewer.theme._defaults import DEFAULT_DARK, DEFAULT_LIGHT

# Values are either a validated Theme or a raw dict awaiting first-use
# validation.  The dict form is replaced by the validated Theme on first
# access so subsequent calls are a plain dict lookup.
_registry: dict[str, Theme | dict] = {
    "dark": DEFAULT_DARK,
    "light": DEFAULT_LIGHT,
}


def register_theme(name: str, theme: Theme) -> None:
    """Register *theme* under *name*, replacing any existing entry.

    Parameters
    ----------
    name : str
        Registry key used to retrieve the theme later.  An existing entry
        with the same name is silently replaced.
    theme : Theme
        Validated :class:`~oz_viewer.theme._model.Theme` instance.
    """
    _registry[name] = theme


def get_theme(name: str) -> Theme:
    """Return the :class:`~oz_viewer.theme._model.Theme` registered as *name*.

    Built-in themes are stored as raw dicts and validated into ``Theme``
    objects on first access.  The validated object is cached so subsequent
    calls for the same name incur no validation overhead.

    Parameters
    ----------
    name : str
        Registry key to look up.

    Returns
    -------
    Theme
        The validated, frozen theme model.

    Raises
    ------
    KeyError
        When *name* is not found in the registry.  The error message lists
        all available theme names.
    """
    try:
        entry = _registry[name]
    except KeyError:
        available = ", ".join(sorted(_registry))
        raise KeyError(
            f"Unknown theme {name!r}. Available themes: {available}"
        ) from None

    if isinstance(entry, dict):
        from oz_viewer.theme._model import Theme as _Theme

        theme = _Theme.model_validate(entry)
        _registry[name] = theme
        return theme

    return entry


def list_themes() -> list[str]:
    """Return the names of all registered themes in insertion order.

    Returns
    -------
    list of str
        Theme names in the order they were registered.
    """
    return list(_registry.keys())
