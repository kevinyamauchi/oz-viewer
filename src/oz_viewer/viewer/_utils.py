"""Shared utilities used by both the single-panel viewer and the orthoviewer."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oz_viewer._perf import StartupPerfTracer


def _dtype_clim_max(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def _dtype_decimals(dtype: np.dtype) -> int:
    return 0 if np.issubdtype(dtype, np.integer) else 2


def _perf_mark(perf: StartupPerfTracer | None, step: str, /, **fields: object) -> None:
    if perf is not None:
        perf.mark(step, **fields)


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
