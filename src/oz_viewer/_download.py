"""Download an OME-Zarr store from S3 or HTTPS to a local directory."""

from __future__ import annotations

import asyncio
import itertools
import math
import sys
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import aiohttp

if TYPE_CHECKING:
    import tensorstore as ts
    import zarr
    from rich.console import Console
    from rich.progress import Progress, TaskID


def _meta_key(rel_path: str, node: zarr.Array | zarr.Group) -> str:
    """Return the metadata store key for a zarr node.

    Parameters
    ----------
    rel_path : str
        Path of the node relative to the store root.  Pass an empty string
        for the root node itself.
    node : zarr.Array or zarr.Group
        The opened zarr node whose metadata key is needed.

    Returns
    -------
    str
        Store key for the node's metadata file, e.g. ``"zarr.json"`` for v3,
        ``".zarray"`` or ``".zgroup"`` for v2.
    """
    import zarr

    fmt = node.metadata.zarr_format
    if fmt == 3:
        fname = "zarr.json"
    elif isinstance(node, zarr.Array):
        fname = ".zarray"
    else:
        fname = ".zgroup"
    return f"{rel_path}/{fname}" if rel_path else fname


def _child_paths_from_attrs(attrs: dict) -> list[tuple[str, str]]:
    """Inspect OME-Zarr attributes and return child node descriptors.

    Reads both the ``ome`` sub-key (spec v0.5) and the top-level attributes
    (older bioformats2raw / v0.4 stores) for maximum compatibility.

    Parameters
    ----------
    attrs : dict
        Attribute dictionary from a zarr Group node.

    Returns
    -------
    list of tuple[str, str]
        Each entry is ``(rel_path, hint)`` where *rel_path* is the child path
        relative to the current node and *hint* is one of ``"array"``,
        ``"group"``, or ``"maybe"`` (attempt to open; silently skip on 404).

    Notes
    -----
    Handled OME node types:

    * **Image / LabelImage** — ``multiscales.datasets[*].path`` → arrays;
      ``labels/<name>`` → groups.
    * **Bf2Raw** — ``series[*]`` → groups.
    * **Plate** — ``plate.wells[*].path`` → groups.
    * **Well** — ``well.images[*].path`` → groups.
    * **LabelsGroup** — ``labels[*]`` → groups.
    """
    ome = attrs.get("ome") or {}
    merged = {**attrs, **ome}

    results: list[tuple[str, str]] = []

    for ms in merged.get("multiscales", []):
        for ds in ms.get("datasets", []):
            path = ds.get("path", "")
            if path:
                results.append((path, "array"))
    if merged.get("multiscales"):
        results.append(("labels", "maybe"))

    for label_name in merged.get("labels", []):
        results.append((str(label_name), "group"))

    for series_path in merged.get("series", []):
        results.append((str(series_path).strip("/"), "group"))

    plate = merged.get("plate") or {}
    for well in plate.get("wells", []):
        path = well.get("path", "")
        if path:
            results.append((path, "group"))

    well_meta = merged.get("well") or {}
    for img in well_meta.get("images", []):
        path = img.get("path", "")
        if path:
            results.append((path, "group"))

    return results


def _array_chunk_keys(rel_path: str, array: zarr.Array) -> list[str]:
    """Compute every storage key for a zarr array's chunks.

    Parameters
    ----------
    rel_path : str
        Path of the array relative to the store root.  Pass an empty string
        for a root-level array.
    array : zarr.Array
        The opened zarr array.

    Returns
    -------
    list of str
        All store keys for the array's chunk objects.  Missing chunks
        (fill-value only) simply won't exist on the server; the transfer
        loop handles them gracefully.

    Notes
    -----
    * zarr v3 with default encoding: keys like ``c/0/1/2``.
    * zarr v3 with v2 encoding or zarr v2 arrays: keys like ``0.1.2``.
    """
    shape = array.shape
    chunk_shape = array.chunks
    fmt = array.metadata.zarr_format

    if fmt == 3:
        enc = array.metadata.chunk_key_encoding
        sep: str = enc.separator
        default_enc: bool = enc.name == "default"
    else:
        sep = getattr(array.metadata, "dimension_separator", ".") or "."
        default_enc = False

    if not shape:
        if fmt == 3 and default_enc:
            chunk_key = "c"
        elif fmt == 3:
            chunk_key = ""
        else:
            chunk_key = "0"
        return [f"{rel_path}/{chunk_key}" if rel_path else chunk_key]

    n_chunks = [
        max(1, math.ceil(s / c)) for s, c in zip(shape, chunk_shape, strict=False)
    ]

    keys: list[str] = []
    for indices in itertools.product(*[range(n) for n in n_chunks]):
        index_part = sep.join(str(i) for i in indices)
        chunk_key = f"c{sep}{index_part}" if (fmt == 3 and default_enc) else index_part
        full_key = f"{rel_path}/{chunk_key}" if rel_path else chunk_key
        keys.append(full_key)

    return keys


def _enumerate_keys_via_zarr(url: str, storage_options: dict) -> list[str]:
    """Walk the OME-Zarr hierarchy via BFS and return every store key.

    Child discovery is driven entirely by OME metadata read from individual
    ``zarr.json`` / ``.zgroup`` files — ``group.members()`` is never called.
    This makes it work over plain HTTPS where directory listing is impossible.

    Parameters
    ----------
    url : str
        Root URL of the OME-Zarr store.
    storage_options : dict
        Extra kwargs forwarded to :func:`zarr.open` (e.g. auth headers).
        Pass an empty dict for public stores.

    Returns
    -------
    list of str
        Every store key in the hierarchy (metadata files + chunk objects).

    Notes
    -----
    Intended to be called via :func:`asyncio.to_thread` from async code.
    """
    import zarr

    keys: list[str] = []
    queue: list[tuple[str, str]] = [(url.rstrip("/"), "")]
    visited: set[str] = set()

    while queue:
        node_url, key_prefix = queue.pop(0)

        if key_prefix in visited:
            continue
        visited.add(key_prefix)

        open_kwargs: dict = (
            {"storage_options": storage_options} if storage_options else {}
        )
        try:
            node = zarr.open(node_url, mode="r", **open_kwargs)
        except Exception:
            continue

        fmt = node.metadata.zarr_format
        keys.append(_meta_key(key_prefix, node))

        if isinstance(node, zarr.Array):
            keys.extend(_array_chunk_keys(key_prefix, node))
            if fmt == 2:
                keys.append(f"{key_prefix}/.zattrs" if key_prefix else ".zattrs")
            continue

        if fmt == 2:
            attr_key = f"{key_prefix}/.zattrs" if key_prefix else ".zattrs"
            meta_key = f"{key_prefix}/.zmetadata" if key_prefix else ".zmetadata"
            keys.append(attr_key)
            keys.append(meta_key)

        attrs = dict(node.attrs)
        for child_rel, hint in _child_paths_from_attrs(attrs):
            child_key = f"{key_prefix}/{child_rel}" if key_prefix else child_rel
            child_url = f"{node_url}/{child_rel}"
            if hint == "array":
                if child_key in visited:
                    continue
                visited.add(child_key)
                try:
                    arr = zarr.open_array(child_url, mode="r", **open_kwargs)
                    keys.append(_meta_key(child_key, arr))
                    keys.extend(_array_chunk_keys(child_key, arr))
                    if arr.metadata.zarr_format == 2:
                        keys.append(f"{child_key}/.zattrs" if child_key else ".zattrs")
                except Exception:
                    pass
            else:
                queue.append((child_url, child_key))

    return keys


def _enumerate_keys_via_s3fs(bucket: str, path: str, anon: bool) -> list[str]:
    """List every object under an S3 prefix using s3fs.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    path : str
        Object prefix path within the bucket (without leading slash).
    anon : bool
        ``True`` for anonymous (unsigned) access; ``False`` for credentialed
        access using the default AWS credential chain.

    Returns
    -------
    list of str
        Store-relative keys for every object found under the prefix (the full
        S3 object keys with the ``<bucket>/<path>/`` prefix stripped).

    Notes
    -----
    Intended to be called via :func:`asyncio.to_thread` from async code.
    Requires the optional ``s3fs`` package.
    """
    import s3fs

    fs = s3fs.S3FileSystem(anon=anon)
    root = f"{bucket}/{path.strip('/')}"
    all_paths: list[str] = fs.find(root, detail=False)
    prefix = root + "/"
    return [p.removeprefix(prefix) for p in all_paths if not p.endswith("/")]


async def enumerate_keys(url: str, scheme: str, anon: bool) -> list[str]:
    """Enumerate all store keys for an OME-Zarr store.

    Dispatches to an S3 listing (fast, complete) for ``s3://`` URLs or a
    zarr BFS hierarchy walk for ``https://`` / ``http://`` URLs.

    Parameters
    ----------
    url : str
        Root URL of the OME-Zarr store.
    scheme : str
        URL scheme, one of ``"s3"``, ``"https"``, or ``"http"``.
    anon : bool
        Anonymous S3 access.  Ignored for non-S3 URLs.

    Returns
    -------
    list of str
        Every store key (metadata files + chunk objects) in the store.
    """
    if scheme == "s3":
        print("Enumerating keys via S3 listing...", flush=True)
        parsed = urlparse(url)
        bucket = parsed.netloc
        path = parsed.path.lstrip("/")
        keys = await asyncio.to_thread(_enumerate_keys_via_s3fs, bucket, path, anon)
    else:
        print("Enumerating keys via zarr hierarchy walk...", flush=True)
        keys = await asyncio.to_thread(_enumerate_keys_via_zarr, url, {})

    print(f"  Found {len(keys):,} keys.", flush=True)
    return keys


async def _open_src_kvstore(url: str, scheme: str) -> ts.KvStore:
    """Open a read-only tensorstore KvStore for an S3 source.

    Parameters
    ----------
    url : str
        Root URL of the OME-Zarr store.  Must be an ``s3://`` URL.
    scheme : str
        URL scheme; must be ``"s3"``.

    Returns
    -------
    tensorstore.KvStore
        An open, read-only KvStore pointing at the S3 prefix.

    Raises
    ------
    AssertionError
        If *scheme* is not ``"s3"``.
    """
    import tensorstore as ts

    assert scheme == "s3", "_open_src_kvstore is S3-only"
    parsed = urlparse(url)
    bucket = parsed.netloc
    path = parsed.path.lstrip("/").rstrip("/") + "/"
    spec: dict = {"driver": "s3", "bucket": bucket, "path": path}
    return await ts.KvStore.open(spec)


async def _open_dst_kvstore(output_dir: Path) -> ts.KvStore:
    """Open a writable tensorstore KvStore for a local destination directory.

    Parameters
    ----------
    output_dir : Path
        Local directory that will receive the downloaded store.  The
        directory must already exist.

    Returns
    -------
    tensorstore.KvStore
        An open, writable KvStore backed by the local filesystem.
    """
    import tensorstore as ts

    path = str(output_dir.resolve()).rstrip("/") + "/"
    return await ts.KvStore.open({"driver": "file", "path": path})


async def _transfer_http(
    base_url: str,
    dst_kv: ts.KvStore,
    keys: list[str],
    concurrency: int,
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Transfer keys from an HTTPS/HTTP source to a local KvStore.

    Uses :mod:`aiohttp` for downloading; tensorstore's HTTP KvStore driver
    fails silently on some S3-compatible HTTPS endpoints so :mod:`aiohttp`
    is used for all HTTP/HTTPS transfers.

    Parameters
    ----------
    base_url : str
        Root URL of the OME-Zarr store (HTTPS or HTTP).
    dst_kv : tensorstore.KvStore
        Open writable KvStore for the local destination.
    keys : list of str
        Store keys to transfer.
    concurrency : int
        Maximum number of simultaneous requests.
    progress : rich.progress.Progress
        Active Rich progress instance to update.
    task_id : rich.progress.TaskID
        Task identifier within *progress* to advance.

    Raises
    ------
    RuntimeError
        If any key fails to transfer.
    """
    sem = asyncio.Semaphore(concurrency)
    failures: list[tuple[str, Exception]] = []
    total_bytes = 0
    base = base_url.rstrip("/")

    async with aiohttp.ClientSession() as session:

        async def _fetch_one(key: str) -> None:
            nonlocal total_bytes
            async with sem:
                try:
                    url = f"{base}/{key}"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            value = await resp.read()
                            await dst_kv.write(key, value)
                            total_bytes += len(value)
                            progress.update(
                                task_id,
                                advance=1,
                                description=(
                                    f"Downloading  {total_bytes / 1_000_000:.1f} MB"
                                ),
                            )
                        elif resp.status == 404:
                            progress.update(task_id, advance=1)
                        else:
                            raise RuntimeError(f"HTTP {resp.status} for {url}")
                except Exception as exc:
                    failures.append((key, exc))
                    progress.update(task_id, advance=1)

        await asyncio.gather(*(_fetch_one(k) for k in keys))

    if failures:
        n = len(failures)
        print(f"\n  ⚠  {n} key(s) failed to transfer:", file=sys.stderr)
        for key, exc in failures[:10]:
            print(f"     {key!r}: {exc}", file=sys.stderr)
        if n > 10:
            print(f"     … and {n - 10} more.", file=sys.stderr)
        raise RuntimeError(f"{n} key(s) failed during transfer.")


async def transfer(
    src_kv: ts.KvStore,
    dst_kv: ts.KvStore,
    keys: list[str],
    concurrency: int,
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Copy keys from an S3 KvStore to a local KvStore concurrently.

    Parameters
    ----------
    src_kv : tensorstore.KvStore
        Open read-only KvStore for the S3 source.
    dst_kv : tensorstore.KvStore
        Open writable KvStore for the local destination.
    keys : list of str
        Store keys to copy.
    concurrency : int
        Maximum number of simultaneous read/write pairs.
    progress : rich.progress.Progress
        Active Rich progress instance to update.
    task_id : rich.progress.TaskID
        Task identifier within *progress* to advance.

    Notes
    -----
    Missing keys (fill-value chunks absent from the store) are silently
    skipped — they need not be written locally either.

    Raises
    ------
    RuntimeError
        If any keys fail after a single attempt.
    """
    sem = asyncio.Semaphore(concurrency)
    failures: list[tuple[str, Exception]] = []
    total_bytes = 0

    async def _copy_one(key: str) -> None:
        nonlocal total_bytes
        async with sem:
            try:
                result = await src_kv.read(key)
                value = bytes(result.value) if result.value else b""
                if value:
                    nb = len(value)
                    await dst_kv.write(key, value)
                    total_bytes += nb
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Downloading  {total_bytes / 1_000_000:.1f} MB",
                    )
                else:
                    progress.update(task_id, advance=1)
            except Exception as exc:
                failures.append((key, exc))
                progress.update(task_id, advance=1)

    await asyncio.gather(*(_copy_one(k) for k in keys))

    if failures:
        n = len(failures)
        print(f"\n  ⚠  {n} key(s) failed to transfer:", file=sys.stderr)
        for key, exc in failures[:10]:
            print(f"     {key!r}: {exc}", file=sys.stderr)
        if n > 10:
            print(f"     … and {n - 10} more.", file=sys.stderr)
        raise RuntimeError(f"{n} key(s) failed during transfer.")


async def download(
    url: str,
    output_dir: Path,
    concurrency: int = 32,
    anon: bool = True,
    dry_run: bool = False,
    console: Console | None = None,
) -> None:
    """Download an OME-Zarr store to a local directory.

    The caller is responsible for creating *output_dir* and handling any
    overwrite logic before calling this function.  On success the directory
    contains a complete local copy of the store.

    Parameters
    ----------
    url : str
        Source URL — ``s3://``, ``https://``, or ``http://``.
    output_dir : Path
        Local destination directory.  Must already exist.
    concurrency : int, optional
        Maximum simultaneous chunk transfers, by default 32.
    anon : bool, optional
        Use anonymous S3 access, by default ``True``.
    dry_run : bool, optional
        Enumerate keys and probe the first 3, but do not write anything,
        by default ``False``.
    console : rich.console.Console, optional
        Console for progress output.  A default console is created when
        ``None``.

    Raises
    ------
    RuntimeError
        If any keys fail to transfer.
    ValueError
        If the URL scheme is not ``"s3"``, ``"https"``, or ``"http"``.
    """
    from rich.console import Console as RichConsole

    from oz_viewer._display import make_download_progress

    if console is None:
        console = RichConsole()

    parsed = urlparse(url.rstrip("/"))
    scheme = parsed.scheme.lower()

    if scheme not in ("s3", "https", "http"):
        raise ValueError(
            f"Unsupported URL scheme {scheme!r}. Use s3://, https://, or http://."
        )

    keys = await enumerate_keys(url, scheme, anon=anon)

    src_kv = await _open_src_kvstore(url, scheme) if scheme == "s3" else None
    dst_kv = await _open_dst_kvstore(output_dir)

    if dry_run:
        console.print("  Dry run — probing first 3 keys...", style="dim")
        async with aiohttp.ClientSession() as session:
            for probe_key in keys[:3]:
                if scheme == "s3" and src_kv is not None:
                    result = await src_kv.read(probe_key)
                    value = bytes(result.value) if result.value else b""
                else:
                    probe_url = f"{url.rstrip('/')}/{probe_key}"
                    async with session.get(probe_url) as resp:
                        value = await resp.read() if resp.status == 200 else b""
                status = f"{len(value):,} bytes" if value else "MISSING (0 bytes)"
                console.print(f"    {probe_key!r}: {status}", style="dim")
        console.print("  Dry run complete. No files written.", style="dim")
        return

    progress = make_download_progress(console)
    with progress:
        task_id = progress.add_task("Downloading", total=len(keys))
        if scheme != "s3":
            await _transfer_http(url, dst_kv, keys, concurrency, progress, task_id)
        else:
            assert src_kv is not None
            await transfer(src_kv, dst_kv, keys, concurrency, progress, task_id)
