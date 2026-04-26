"""Rich terminal rendering for oz-viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table, box
from rich.theme import Theme

if TYPE_CHECKING:
    from oz_viewer._ping import ChunkInfo, FetchResult

OZ_THEME = Theme({"repr.attrib_name": "bold blue"})


def make_console() -> Console:
    """Return a Console with the oz-viewer theme."""
    return Console(theme=OZ_THEME)


def print_success_panel(path: str, group: Any, console: Console) -> None:
    """Print a green success panel for a valid OME-Zarr store.

    Parameters
    ----------
    path : str
        The store path that was validated.
    group : Any
        The validated ZarrGroup.
    console : Console
        Rich console to print to.
    """
    version = group.ome_version()
    meta_type = type(group.ome_metadata()).__name__
    content = (
        f"[bold green]✓ Valid OME-Zarr[/bold green]\n"
        f"Path    {path}\n"
        f"Version {version}\n"
        f"Type    {meta_type}"
    )
    console.print(
        Panel(content, title="oz-viewer validate", border_style="green", expand=False)
    )


def print_metadata_panel(meta: Any, console: Console) -> None:
    """Print the pydantic metadata model inside a blue panel.

    Parameters
    ----------
    meta : Any
        The OME metadata model instance.
    console : Console
        Rich console to print to.
    """
    console.print(
        Panel(
            Pretty(meta, indent_size=2, expand_all=True, indent_guides=True),
            title=f"[bold]Metadata Model[/bold]  [dim]{type(meta).__name__}[/dim]",
            border_style="blue",
        )
    )


def print_error_panel(path: str, error: Exception, console: Console) -> None:
    """Print a red validation error panel.

    Parameters
    ----------
    path : str
        The store path that failed validation.
    error : Exception
        The StorageValidationError raised.
    console : Console
        Rich console to print to.
    """
    content = f"[bold red]✗ Validation failed[/bold red]\nPath    {path}\n\n{error}"
    console.print(
        Panel(content, title="oz-viewer validate", border_style="red", expand=False)
    )


def print_ping_header(
    path: str,
    chunk_info: ChunkInfo,
    n_fetch: int,
    timeout: float,
    console: Console,
) -> None:
    """Print the ping header panel before fetching begins.

    Parameters
    ----------
    path : str
        The store path being pinged.
    chunk_info : ChunkInfo
        Chunk metadata for display.
    n_fetch : int
        Number of fetches to perform.
    timeout : float
        Per-fetch timeout in seconds.
    console : Console
        Rich console to print to.
    """
    chunk_line = (
        f"Chunk    {chunk_info.origin_key}"
        f"  ({chunk_info.ndim}D, {chunk_info.level_path})"
    )
    content = (
        f"Store    {path}\n"
        f"Driver   {chunk_info.protocol}\n"
        f"{chunk_line}\n"
        f"Fetches  {n_fetch}  \N{MULTIPLICATION SIGN}  timeout {timeout}s"
    )
    console.print(
        Panel(content, title="oz-viewer ping", border_style="blue", expand=False)
    )


def make_ping_progress(console: Console) -> Progress:
    """Return a Rich Progress instance bound to the shared console.

    Parameters
    ----------
    console : Console
        Rich console to bind the progress bar to.

    Returns
    -------
    Progress
        Configured progress bar (not yet started).
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )


def _human_bytes(n: int | float) -> str:
    """Format a byte count as a human-readable string.

    Parameters
    ----------
    n : int | float
        Number of bytes.

    Returns
    -------
    str
        Human-readable size string, e.g. ``"4.0 KB"`` or ``"1.2 MB"``.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def print_ping_results(
    chunk_info: ChunkInfo,
    result: FetchResult,
    console: Console,
) -> None:
    """Print the ping results table inside a panel.

    Parameters
    ----------
    chunk_info : ChunkInfo
        Chunk metadata.
    result : FetchResult
        Fetch timing results.
    console : Console
        Rich console to print to.
    """
    import statistics

    n_success = len(result.latencies)
    all_failed = n_success == 0
    has_issues = result.n_timeouts > 0 or result.n_errors > 0

    if all_failed:
        border_style = "red"
        title = "✗ Failed"
    elif has_issues:
        border_style = "yellow"
        title = "⚠ Complete with issues"
    else:
        border_style = "green"
        title = "✓ Complete"

    if all_failed:
        lines = [f"[bold red]All {result.n_attempted} fetch(es) failed.[/bold red]"]
        if result.n_timeouts > 0:
            lines.append(f"[yellow]Timeouts: {result.n_timeouts}[/yellow]")
        if result.n_errors > 0:
            lines.append(f"[red]Errors: {result.n_errors}[/red]")
        content = "\n".join(lines)
        console.print(
            Panel(content, title=title, border_style=border_style, expand=False),
        )
        return

    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Fetches attempted", str(result.n_attempted))
    table.add_row("Fetches completed", str(n_success))
    if result.n_timeouts > 0:
        table.add_row("Timeouts", f"[yellow]{result.n_timeouts}[/yellow]")
    if result.n_errors > 0:
        table.add_row("Errors", f"[red]{result.n_errors}[/red]")
    table.add_row("Chunk shape", str(chunk_info.chunk_shape))
    table.add_row("Dtype", chunk_info.dtype_str)
    table.add_row("Uncompressed size", _human_bytes(chunk_info.uncompressed_bytes))
    if result.compressed_bytes is not None:
        table.add_row("Compressed size", _human_bytes(result.compressed_bytes))
        ratio = chunk_info.uncompressed_bytes / result.compressed_bytes
        table.add_row(
            "Compression ratio",
            f"{ratio:.1f} \N{MULTIPLICATION SIGN}",
        )

    if n_success >= 1:
        mean = statistics.mean(result.latencies)
        table.add_row("Mean latency", f"{mean * 1000:.1f} ms")
        table.add_row("Min latency", f"{min(result.latencies) * 1000:.1f} ms")
        table.add_row("Max latency", f"{max(result.latencies) * 1000:.1f} ms")
    if n_success >= 2:
        stdev = statistics.stdev(result.latencies)
        table.add_row("Std dev", f"{stdev * 1000:.1f} ms")
    if n_success >= 1 and result.compressed_bytes is not None:
        throughput = result.compressed_bytes / mean
        table.add_row("Throughput", f"{_human_bytes(throughput)}/s")

    console.print(Panel(table, title=title, border_style=border_style))
