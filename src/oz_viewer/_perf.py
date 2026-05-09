"""Performance logging utilities for opt-in startup diagnostics."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

_PERF_LOGGER_NAME = "oz_viewer.perf"
_ENV_VAR_PERF = "OZ_VIEWER_PERF"
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(slots=True)
class _PerfEvent:
    step: str
    elapsed_s: float
    delta_s: float
    fields: dict[str, object]


def perf_enabled_from_env() -> bool:
    """Return whether performance logging is enabled through an env var."""
    return os.getenv(_ENV_VAR_PERF, "").strip().lower() in _TRUTHY


def configure_perf_logging(
    *, enabled: bool, log_file: str | None = None
) -> logging.Logger:
    """Configure the dedicated performance logger.

    The logger is isolated from the default logging tree so users only see
    performance output when this explicit configuration is enabled.
    """
    logger = logging.getLogger(_PERF_LOGGER_NAME)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    if not enabled:
        logger.setLevel(logging.CRITICAL + 1)
        return logger

    logger.setLevel(logging.INFO)
    handler: logging.Handler
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(handler)
    return logger


@dataclass(slots=True)
class StartupPerfTracer:
    """Lightweight startup milestone tracer for perf diagnostics."""

    enabled: bool
    show_table: bool = False
    table_title: str = "Orthoviewer startup timings"
    run_id: str = field(default_factory=lambda: uuid4().hex[:8])
    _t0: float = field(default_factory=perf_counter)
    _last_elapsed_s: float = 0.0
    _events: list[_PerfEvent] = field(default_factory=list)
    _table_reported: bool = False

    def mark(self, step: str, /, **fields: object) -> None:
        """Emit a perf milestone with elapsed startup time."""
        if not self.enabled:
            return

        logger = logging.getLogger(_PERF_LOGGER_NAME)
        if not logger.isEnabledFor(logging.INFO):
            return

        elapsed = perf_counter() - self._t0
        delta = max(0.0, elapsed - self._last_elapsed_s)
        self._last_elapsed_s = elapsed
        self._events.append(
            _PerfEvent(step=step, elapsed_s=elapsed, delta_s=delta, fields=dict(fields))
        )

        suffix = ""
        if fields:
            details = " ".join(f"{key}={value}" for key, value in fields.items())
            suffix = f" {details}"
        logger.info("run=%s +%.3fs step=%s%s", self.run_id, elapsed, step, suffix)

    def report_rich_table(self, *, title: str | None = None) -> None:
        """Render a Rich table of step and cumulative startup timings."""
        if not self.enabled or not self.show_table or self._table_reported:
            return

        from rich.console import Console
        from rich.table import Table

        table = Table(title=title or self.table_title)
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("Step")
        table.add_column("Step (ms)", justify="right")
        table.add_column("Cumulative (ms)", justify="right")
        table.add_column("Details")

        for idx, event in enumerate(self._events, start=1):
            details = " ".join(
                f"{key}={value}" for key, value in sorted(event.fields.items())
            )
            table.add_row(
                str(idx),
                event.step,
                f"{event.delta_s * 1000:.1f}",
                f"{event.elapsed_s * 1000:.1f}",
                details,
            )

        Console(stderr=True).print(table)
        self._table_reported = True
