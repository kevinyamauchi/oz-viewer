"""A viewer for ome-zarr images."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("oz-viewer")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Kevin Yamauchi"
__email__ = "kevin.yamauchi@gmail.com"
