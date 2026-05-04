"""Chunk fetch logic for the ping command."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import fsspec
import numpy as np

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID


@dataclass(frozen=True)
class ChunkInfo:
    """All information needed to fetch and display a chunk.

    Attributes
    ----------
    origin_key : str
        e.g. ``"s1/c/0/0/0/0"``
    chunk_path : str
        Full filesystem path for ``fsspec.cat()``.
    store_root : str
        Store root without trailing slash.
    level_path : str
        Coarsest dataset path, e.g. ``"s1"``.
    ndim : int
        Number of array dimensions.
    chunk_shape : tuple[int, ...]
        Voxel dimensions.
    dtype_str : str
        e.g. ``"uint16"``
    uncompressed_bytes : int
        ``prod(chunk_shape) * dtype.itemsize``
    protocol : str
        Normalised fsspec driver, e.g. ``"file"``, ``"https"``.
    """

    origin_key: str
    chunk_path: str
    store_root: str
    level_path: str
    ndim: int
    chunk_shape: tuple[int, ...]
    dtype_str: str
    uncompressed_bytes: int
    protocol: str


@dataclass(frozen=True)
class FetchResult:
    """Results from a series of chunk fetches.

    Attributes
    ----------
    latencies : tuple[float, ...]
        Seconds, one per successful fetch.
    compressed_bytes : int | None
        From first successful fetch; ``None`` if all failed.
    n_attempted : int
        Total fetches attempted.
    n_timeouts : int
        Number of fetches that timed out.
    n_errors : int
        Number of fetches that errored (excluding timeouts).
    """

    latencies: tuple[float, ...]
    compressed_bytes: int | None
    n_attempted: int
    n_timeouts: int
    n_errors: int


def build_chunk_info(group: Any, meta: Any) -> ChunkInfo:
    """Build a ChunkInfo from an open ZarrGroup and its metadata.

    Parameters
    ----------
    group : Any
        Open ZarrGroup from yaozarrs.
    meta : Any
        OME metadata model (e.g. Image or Plate).

    Returns
    -------
    ChunkInfo
        Frozen dataclass with all fetch parameters.
    """
    coarsest = meta.multiscales[0].datasets[-1]
    array = group[coarsest.path]
    level_path = coarsest.path

    # Determine chunk key encoding prefix and separator
    array_meta = array._metadata
    chunk_key_encoding = array_meta.chunk_key_encoding
    if isinstance(chunk_key_encoding, dict):
        enc_name = chunk_key_encoding.get("name", "default")
        enc_cfg = chunk_key_encoding.get("configuration", {}) or {}
        sep = enc_cfg.get("separator", "/") if isinstance(enc_cfg, dict) else "/"
    else:
        enc_name = getattr(chunk_key_encoding, "name", "default")
        sep = "/"

    if enc_name == "default":
        prefix = "c" + sep
    else:
        prefix = ""
        sep = "."

    origin_key = f"{array._path}/{prefix}{sep.join(['0'] * array.ndim)}"

    # Strip the array path from the store path to get store root
    store_path = str(array.store_path)
    array_rel = array._path.lstrip("/")
    if store_path.endswith("/" + array_rel):
        store_root = store_path[: -(len(array_rel) + 1)]
    elif store_path.endswith(array_rel):
        store_root = store_path[: -len(array_rel)]
    else:
        store_root = store_path
    store_root = store_root.rstrip("/")

    chunk_path = store_root + "/" + origin_key

    fs, _ = fsspec.url_to_fs(store_root)
    protocol = fs.protocol
    if isinstance(protocol, tuple):
        protocol = protocol[0]

    chunk_shape = tuple(
        int(x) for x in array_meta.chunk_grid["configuration"]["chunk_shape"]
    )
    if hasattr(array_meta, "data_type"):
        dtype_str = str(array_meta.data_type)
    else:
        dtype_str = str(array.dtype)
    # Strip module prefix if present (e.g. "DataType.uint16" -> "uint16")
    if "." in dtype_str:
        dtype_str = dtype_str.split(".")[-1]

    uncompressed_bytes = int(np.prod(chunk_shape)) * np.dtype(dtype_str).itemsize

    return ChunkInfo(
        origin_key=origin_key,
        chunk_path=chunk_path,
        store_root=store_root,
        level_path=level_path,
        ndim=array.ndim,
        chunk_shape=chunk_shape,
        dtype_str=dtype_str,
        uncompressed_bytes=uncompressed_bytes,
        protocol=protocol,
    )


def run_fetches(
    chunk_info: ChunkInfo,
    n_fetch: int,
    timeout: float,
    progress: Progress,
    task_id: TaskID,
) -> FetchResult:
    """Fetch a chunk repeatedly and collect timing results.

    Parameters
    ----------
    chunk_info : ChunkInfo
        Chunk location and metadata.
    n_fetch : int
        Number of fetches to perform.
    timeout : float
        Per-fetch timeout in seconds.
    progress : Progress
        Rich Progress instance for advancing the bar.
    task_id : TaskID
        Task ID within the progress bar.

    Returns
    -------
    FetchResult
        Collected timing and error statistics.
    """
    fs, _ = fsspec.url_to_fs(chunk_info.store_root)
    latencies: list[float] = []
    compressed_bytes: int | None = None
    n_timeouts = 0
    n_errors = 0

    for _ in range(n_fetch):
        t_start = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future: Future[bytes] = executor.submit(fs.cat, chunk_info.chunk_path)
                data = future.result(timeout=timeout)
            t_end = time.perf_counter()
            latencies.append(t_end - t_start)
            if compressed_bytes is None:
                compressed_bytes = len(data)
        except FuturesTimeoutError:
            n_timeouts += 1
        except Exception:
            n_errors += 1
        finally:
            progress.advance(task_id)

    return FetchResult(
        latencies=tuple(latencies),
        compressed_bytes=compressed_bytes,
        n_attempted=n_fetch,
        n_timeouts=n_timeouts,
        n_errors=n_errors,
    )
