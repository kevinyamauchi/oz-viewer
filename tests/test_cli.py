"""Tests for oz-viewer CLI commands."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from oz_viewer._cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Stage 1 — validate command
# ---------------------------------------------------------------------------


def test_validate_valid_image(write_demo_ome):
    path = write_demo_ome("image")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0
    assert "✓ Valid OME-Zarr" in result.output


def test_validate_valid_plate(write_demo_ome):
    path = write_demo_ome("plate")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0
    assert "Plate" in result.output


def test_validate_verbose_image(write_demo_ome):
    path = write_demo_ome("image")
    result = runner.invoke(app, ["validate", "--verbose", str(path)])
    assert result.exit_code == 0
    assert "Image(" in result.output


def test_validate_verbose_plate(write_demo_ome):
    path = write_demo_ome("plate")
    result = runner.invoke(app, ["validate", "--verbose", str(path)])
    assert result.exit_code == 0
    assert "Plate(" in result.output
    assert "PlateDef(" in result.output


def _break_store(path: Path) -> None:
    """Remove the first level subdirectory to break a demo OME-Zarr store."""
    from yaozarrs import validate_zarr_store

    group = validate_zarr_store(str(path))
    meta = group.ome_metadata()
    level_path = meta.multiscales[0].datasets[-1].path
    shutil.rmtree(path / level_path)


def test_validate_invalid_store(write_demo_ome):
    path = write_demo_ome("image")
    _break_store(path)
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    assert "✗ Validation failed" in result.output


def test_validate_nonexistent_store(tmp_path):
    path = tmp_path / "nonexistent.zarr"
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 2


def test_validate_verbose_does_not_run_on_failure(write_demo_ome):
    path = write_demo_ome("image")
    _break_store(path)
    result = runner.invoke(app, ["validate", "--verbose", str(path)])
    assert result.exit_code == 1
    assert "Metadata Model" not in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # typer exits with 2 for no-args help; just check output content
    assert "validate" in result.output


# ---------------------------------------------------------------------------
# Stage 2 — ping command
# ---------------------------------------------------------------------------


def test_ping_local_store(write_demo_ome):
    path = write_demo_ome("image")
    result = runner.invoke(app, ["ping", str(path)])
    assert result.exit_code == 0
    assert "✓ Complete" in result.output


def test_ping_default_n_fetch(write_demo_ome):
    path = write_demo_ome("image")
    result = runner.invoke(app, ["ping", str(path)])
    assert result.exit_code == 0
    assert "5" in result.output


def test_ping_timeout(write_demo_ome):
    path = write_demo_ome("image")
    result = runner.invoke(app, ["ping", str(path), "--timeout", "0"])
    assert result.exit_code == 0
    assert "Timeouts" in result.output


def test_ping_invalid_store(tmp_path):
    path = tmp_path / "nonexistent.zarr"
    result = runner.invoke(app, ["ping", str(path)])
    assert result.exit_code == 2
