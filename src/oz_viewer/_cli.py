"""CLI entry point for oz-viewer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from oz_viewer._display import (
    make_console,
    make_ping_progress,
    print_error_panel,
    print_metadata_panel,
    print_ping_header,
    print_ping_results,
    print_success_panel,
)

app = typer.Typer(
    name="oz-viewer",
    help="Validate and inspect OME-Zarr stores.",
    no_args_is_help=True,
)


@app.command()
def validate(
    path: Annotated[str, typer.Argument(help="Path or URI to the OME-Zarr store.")],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Pretty-print the full metadata model after successful validation.",
        ),
    ] = False,
) -> None:
    """Validate an OME-Zarr store."""
    from yaozarrs import validate_zarr_store
    from yaozarrs._storage import StorageValidationError

    console = make_console()
    try:
        group = validate_zarr_store(path)
    except StorageValidationError as e:
        print_error_panel(path, e, console)
        raise typer.Exit(code=1) from None
    except ImportError as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(code=2) from None
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(code=2) from None

    print_success_panel(path, group, console)
    if verbose:
        print_metadata_panel(group.ome_metadata(), console)


@app.command()
def ping(
    path: Annotated[str, typer.Argument(help="Path or URI to the OME-Zarr store.")],
    n_fetch: Annotated[
        int,
        typer.Option(
            "--n-fetch",
            help="Number of chunk fetches to average over.",
            min=1,
        ),
    ] = 5,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Per-fetch timeout in seconds.",
            min=0.0,
        ),
    ] = 10.0,
) -> None:
    """Measure chunk fetch latency for an OME-Zarr store."""
    from yaozarrs import validate_zarr_store

    from oz_viewer._ping import build_chunk_info, run_fetches

    console = make_console()
    try:
        group = validate_zarr_store(path)
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(code=2) from None

    chunk_info = build_chunk_info(group, group.ome_metadata())
    print_ping_header(path, chunk_info, n_fetch, timeout, console)

    progress = make_ping_progress(console)
    with progress:
        task_id = progress.add_task("Fetching chunks…", total=n_fetch)
        result = run_fetches(chunk_info, n_fetch, timeout, progress, task_id)

    print_ping_results(chunk_info, result, console)


def _resolve_zarr_uri(path: str) -> str:
    """Convert a local path to a file:// URI; pass remote URIs through unchanged."""
    if "://" in path:
        return path
    local = Path(path)
    if not local.exists():
        typer.echo(f"Error: OME-Zarr store not found at '{local}'", err=True)
        raise typer.Exit(code=1)
    return f"file://{local.resolve()}"


@app.command()
def ortho(
    path: Annotated[
        str | None,
        typer.Argument(
            help="Path or URI to the OME-Zarr store (local path, s3://, gs://, https://).",
            show_default=False,
        ),
    ] = None,
    path_option: Annotated[
        str | None,
        typer.Option(
            "--path",
            help=(
                "Path or URI to the OME-Zarr store"
                " (alternative to positional argument)."
            ),
            show_default=False,
        ),
    ] = None,
    make_example: Annotated[
        bool,
        typer.Option(
            "--make-example",
            help="Create a synthetic anisotropic OME-Zarr and open it in the viewer.",
        ),
    ] = False,
) -> None:
    """Open an OME-Zarr store in the 4-panel orthoviewer."""
    from oz_viewer.viewer import launch_orthoviewer

    if make_example:
        from oz_viewer.data._blobs import make_example_zarr

        zarr_path = make_example_zarr()
        zarr_uri = f"file://{zarr_path}"
    else:
        raw = path or path_option
        if raw is None:
            typer.echo(
                "Error: provide a path as a positional argument, via --path, "
                "or use --make-example.",
                err=True,
            )
            raise typer.Exit(code=1)
        if path is not None and path_option is not None:
            typer.echo(
                "Error: provide the path as a positional argument or --path, not both.",
                err=True,
            )
            raise typer.Exit(code=1)
        zarr_uri = _resolve_zarr_uri(raw)

    launch_orthoviewer(zarr_uri)


if __name__ == "__main__":
    app()
