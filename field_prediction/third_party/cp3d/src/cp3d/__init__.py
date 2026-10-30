"""CP3D v1-compatible Python API."""

from .api import (
    CP3DDataset,
    compress,
    compress_cube,
    decompress,
    decompress_cube,
    load,
    open_dataset,
    save,
)

__all__ = [
    "CP3DDataset",
    "compress",
    "compress_cube",
    "decompress",
    "decompress_cube",
    "load",
    "open_dataset",
    "save",
]
__version__ = "1.1.0"
