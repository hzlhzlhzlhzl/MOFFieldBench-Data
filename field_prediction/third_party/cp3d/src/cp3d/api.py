"""Public CP3D API and parallel directory workflows."""

from __future__ import annotations

import json
import os
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np

from . import _v1, _v2
from ._cube import FIELD_SETTINGS, classify_cube, read_cube, write_cube
from ._dpcm import decode_lorenzo, encode_lorenzo


def _worker_count(workers: int | None) -> int:
    if workers is None or workers == 0:
        # Large Cube jobs use substantial temporary memory. Two concurrent
        # fields is a conservative automatic setting; callers can raise it.
        return max(1, min(2, os.cpu_count() or 1))
    if workers < 0:
        raise ValueError("workers cannot be negative")
    return workers


def _convert(
    values: np.ndarray,
    backend: str,
    *,
    dtype: Any = None,
    device: Any = None,
):
    if backend == "numpy":
        if device is not None:
            raise ValueError("device is only valid for the torch backend")
        return values.astype(dtype, copy=False) if dtype is not None else values
    if backend == "torch":
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "Torch output requires PyTorch; install cp3d-mof[torch]"
            ) from error
        tensor = torch.from_numpy(values)
        if dtype is not None or device is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor
    raise ValueError("backend must be 'numpy' or 'torch'")


def _decode_bytes(
    payload: bytes, *, workers: int = 0
) -> tuple[np.ndarray, dict[str, object]]:
    if payload.startswith(_v2.MAGIC):
        return _v2.decode(payload, workers=workers)
    if payload.startswith(b"PK\x03\x04"):
        return _v1.decode(payload)
    raise ValueError("unrecognized CP3D file format")


class CP3DDataset:
    """Lazy reader for a v1 directory bundle (`*.cp3d.zip`)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as archive:
            self.members = tuple(
                sorted(
                    info.filename
                    for info in archive.infolist()
                    if info.filename.endswith(".cube.cp3d")
                    and not PurePosixPath(info.filename).name.startswith("._")
                )
            )
            self.manifest = (
                json.loads(archive.read("manifest.json"))
                if "manifest.json" in archive.namelist()
                else None
            )

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self) -> Iterator[str]:
        return iter(self.members)

    def __getitem__(self, member: str):
        return self.load(member)

    def load(
        self,
        member: str,
        *,
        backend: str = "numpy",
        dtype: Any = None,
        device: Any = None,
        return_metadata: bool = False,
        workers: int = 0,
    ):
        if member not in self.members:
            raise KeyError(member)
        with zipfile.ZipFile(self.path) as archive:
            payload = archive.read(member)
        values, metadata = _decode_bytes(payload, workers=workers)
        converted = _convert(values, backend, dtype=dtype, device=device)
        return (converted, metadata) if return_metadata else converted

    def load_all(
        self,
        *,
        backend: str = "numpy",
        dtype: Any = None,
        device: Any = None,
        return_metadata: bool = False,
        workers: int = 0,
    ) -> dict[str, Any]:
        count = _worker_count(workers)
        def one(member: str):
            return member, self.load(
                member,
                backend=backend,
                dtype=dtype,
                device=device,
                return_metadata=return_metadata,
                workers=1,
            )
        if count == 1:
            return dict(one(member) for member in self.members)
        with ThreadPoolExecutor(max_workers=count) as executor:
            return dict(executor.map(one, self.members))


def open_dataset(path: str | Path) -> CP3DDataset:
    return CP3DDataset(path)


def load(
    source: str | Path | bytes,
    *,
    backend: str = "numpy",
    dtype: Any = None,
    device: Any = None,
    return_metadata: bool = False,
    member: str | None = None,
    workers: int = 0,
):
    """Load a CP3D field as a NumPy array or optional Torch tensor.

    A directory bundle without `member=` returns a lazy :class:`CP3DDataset`.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.name.endswith(".cp3d.zip"):
            dataset = CP3DDataset(path)
            if member is None:
                return dataset
            return dataset.load(
                member,
                backend=backend,
                dtype=dtype,
                device=device,
                return_metadata=return_metadata,
                workers=workers,
            )
        payload = path.read_bytes()
    else:
        payload = source
    values, metadata = _decode_bytes(payload, workers=workers)
    converted = _convert(values, backend, dtype=dtype, device=device)
    return (converted, metadata) if return_metadata else converted


def save(
    values: np.ndarray,
    path: str | Path,
    *,
    metadata: dict[str, object],
    tolerance: float,
    quantity: str | None = None,
    unit: str | None = None,
    verify: bool = True,
) -> dict[str, object]:
    """Save an in-memory 3D array using the v1-compatible CP3D format."""
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values = values.detach().cpu().numpy()
    return _v1.save(
        path,
        np.asarray(values),
        metadata,
        tolerance,
        quantity=quantity,
        unit=unit,
        verify=verify,
    )


def compress_cube(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    tolerance: float | None = None,
    quantity: str | None = None,
    verify: bool = True,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = (
        Path(output_path)
        if output_path is not None
        else input_path.with_name(input_path.name + ".cp3d")
    )
    detected = quantity or classify_cube(input_path)
    if detected is None and tolerance is None:
        raise ValueError(
            "unrecognized Cube quantity; specify quantity= or tolerance="
        )
    detected = detected or "scalar_field"
    setting = FIELD_SETTINGS.get(detected, {"unit": "unknown", "tolerance": tolerance})
    selected_tolerance = float(
        tolerance if tolerance is not None else setting["tolerance"]
    )
    values, metadata = read_cube(input_path)
    started = time.perf_counter()
    result = save(
        values,
        output_path,
        metadata=metadata,
        tolerance=selected_tolerance,
        quantity=detected,
        unit=str(setting["unit"]),
        verify=verify,
    )
    result.update(
        {
            "input": str(input_path.resolve()),
            "input_bytes": input_path.stat().st_size,
            "quantity": detected,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return result


def _safe_relative(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return Path(*path.parts)


def _compress_directory(
    root: Path,
    output: Path,
    *,
    tolerance: float | None,
    quantity: str | None,
    verify: bool,
    workers: int,
) -> dict[str, object]:
    candidates = [
        path for path in sorted(root.rglob("*.cube")) if not path.name.startswith("._")
    ]
    count = _worker_count(workers)
    # Compile the Numba kernel once before worker threads enter it.
    warm_values = np.zeros((1, 1, 1), dtype=np.float32)
    warm_codes, warm_anchors = encode_lorenzo(warm_values, 1.0)
    decode_lorenzo(warm_codes, warm_anchors, 1.0)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cp3d-python-") as temporary:
        scratch = Path(temporary)

        def one(path: Path) -> dict[str, object]:
            relative = path.relative_to(root)
            target = scratch / Path(str(relative) + ".cp3d")
            try:
                result = compress_cube(
                    path,
                    target,
                    tolerance=tolerance,
                    quantity=quantity,
                    verify=verify,
                )
                return {
                    **result,
                    "status": "compressed",
                    "input": relative.as_posix(),
                    "member": relative.as_posix() + ".cp3d",
                    "_scratch": str(target),
                }
            except Exception as error:
                return {
                    "status": "error",
                    "input": relative.as_posix(),
                    "error": f"{type(error).__name__}: {error}",
                }

        if count == 1:
            results = [one(path) for path in candidates]
        else:
            with ThreadPoolExecutor(max_workers=count) as executor:
                results = list(executor.map(one, candidates))
        report = {
            "codec_version": _v1.V1_CODEC_VERSION,
            "python_package_version": "1.1.0",
            "source_root_name": root.name,
            "decode_verification": verify,
            "workers": count,
            "results": [
                {key: value for key, value in result.items() if key != "_scratch"}
                for result in results
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_name(output.name + ".tmp")
        with zipfile.ZipFile(
            temporary_output, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as bundle:
            for result in results:
                if result["status"] == "compressed":
                    bundle.write(result["_scratch"], result["member"])
            bundle.writestr(
                "manifest.json",
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            )
        os.replace(temporary_output, output)
    report["archive"] = str(output.resolve())
    report["archive_bytes"] = output.stat().st_size
    return report


def compress(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    tolerance: float | None = None,
    quantity: str | None = None,
    verify: bool = True,
    workers: int = 0,
) -> dict[str, object]:
    path = Path(input_path)
    if path.is_file():
        return compress_cube(
            path,
            output_path,
            tolerance=tolerance,
            quantity=quantity,
            verify=verify,
        )
    if not path.is_dir():
        raise FileNotFoundError(path)
    output = (
        Path(output_path)
        if output_path is not None
        else path.parent / f"{path.name}.cp3d.zip"
    )
    return _compress_directory(
        path,
        output,
        tolerance=tolerance,
        quantity=quantity,
        verify=verify,
        workers=workers,
    )


def decompress_cube(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    workers: int = 0,
) -> dict[str, object]:
    input_path = Path(input_path)
    if output_path is None:
        name = input_path.name
        output_path = input_path.with_name(
            name[: -len(".cp3d")] if name.endswith(".cp3d") else name + ".cube"
        )
    output_path = Path(output_path)
    started = time.perf_counter()
    values, metadata = load(
        input_path, return_metadata=True, workers=workers
    )
    write_cube(output_path, values, metadata)
    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "shape": list(values.shape),
        "input_bytes": input_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _decompress_bundle(
    archive_path: Path, output_root: Path, *, workers: int
) -> dict[str, object]:
    dataset = CP3DDataset(archive_path)
    count = _worker_count(workers)
    started = time.perf_counter()

    def one(member: str) -> dict[str, object]:
        relative = _safe_relative(member)
        output = output_root / Path(str(relative)[: -len(".cp3d")])
        try:
            values, metadata = dataset.load(member, return_metadata=True, workers=1)
            write_cube(output, values, metadata)
            return {
                "status": "decompressed",
                "member": member,
                "output": str(output.relative_to(output_root)),
                "shape": list(values.shape),
            }
        except Exception as error:
            return {
                "status": "error",
                "member": member,
                "error": f"{type(error).__name__}: {error}",
            }

    if count == 1:
        results = [one(member) for member in dataset]
    else:
        with ThreadPoolExecutor(max_workers=count) as executor:
            results = list(executor.map(one, dataset))
    report = {
        "archive": str(archive_path.resolve()),
        "output_root": str(output_root.resolve()),
        "workers": count,
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "decompression_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def decompress(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    workers: int = 0,
) -> dict[str, object]:
    path = Path(input_path)
    if path.name.endswith(".cp3d.zip"):
        output = (
            Path(output_path)
            if output_path is not None
            else path.parent / f"{path.stem}_unzip"
        )
        return _decompress_bundle(path, output, workers=workers)
    return decompress_cube(path, output_path, workers=workers)
