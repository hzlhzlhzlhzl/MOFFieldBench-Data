"""Read and write the original CP3D v1 NPZ format."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import BinaryIO

import numpy as np

from ._dpcm import (
    decode_lorenzo,
    dense_anchors,
    encode_lorenzo,
    error_stats,
    sparse_anchors,
    zigzag_decode,
    zigzag_encode,
)


V1_CODEC_VERSION = "cube-dpcm-zip-v1"


def decode(source: str | Path | BinaryIO | bytes) -> tuple[np.ndarray, dict[str, object]]:
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with np.load(source, allow_pickle=False) as archive:
        required = {"codes", "anchor_positions", "anchor_values", "metadata_json"}
        if not required.issubset(archive.files):
            raise ValueError("not a CP3D v1 archive")
        stored_codes = archive["codes"]
        positions = archive["anchor_positions"]
        anchor_values = archive["anchor_values"]
        metadata = json.loads(str(archive["metadata_json"]))
    encoding = metadata["storage_encoding"]
    if encoding["type"] != "closed_loop_dpcm_lorenzo_3d":
        raise ValueError(f"unsupported v1 encoding: {encoding['type']}")
    codes = (
        zigzag_decode(stored_codes)
        if encoding.get("code_mapping") == "zigzag_int16_to_uint16"
        else stored_codes.astype(np.int16, copy=False)
    )
    step = float(encoding.get("step", encoding.get("step_hartree")))
    anchors = dense_anchors(codes.shape, positions, anchor_values)
    return decode_lorenzo(codes, anchors, step), metadata


def encode_bytes(
    values: np.ndarray,
    metadata: dict[str, object],
    tolerance: float,
    *,
    quantity: str | None = None,
    unit: str | None = None,
) -> tuple[bytes, dict[str, object]]:
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("CP3D requires a three-dimensional array")
    step = 2.0 * tolerance
    codes, anchor_grid = encode_lorenzo(values, step)
    positions, anchor_values = sparse_anchors(anchor_grid, codes)
    restored = decode_lorenzo(codes, anchor_grid, step)
    stats = error_stats(values, restored)
    encoding: dict[str, object] = {
        "type": "closed_loop_dpcm_lorenzo_3d",
        "step": step,
        "original_shape": list(values.shape),
        "anchor_count": int(positions.size),
        "code_dtype": "int16",
        "codec_version": V1_CODEC_VERSION,
        "predictor": "lorenzo3d",
        "tolerance": tolerance,
        "code_mapping": "zigzag_int16_to_uint16",
        "error_vs_input_float32": stats,
    }
    if quantity is not None:
        encoding["quantity"] = quantity
    if unit is not None:
        encoding["unit"] = unit
    output_metadata = {
        **metadata,
        "shape": list(values.shape),
        "storage_encoding": encoding,
    }
    if quantity is not None:
        output_metadata["scalar_quantity"] = quantity
    if unit is not None:
        output_metadata["scalar_unit"] = unit
    output = io.BytesIO()
    np.savez_compressed(
        output,
        codes=zigzag_encode(codes),
        anchor_positions=positions,
        anchor_values=anchor_values,
        metadata_json=np.array(
            json.dumps(output_metadata, ensure_ascii=False, separators=(",", ":"))
        ),
    )
    return output.getvalue(), output_metadata


def save(
    path: str | Path,
    values: np.ndarray,
    metadata: dict[str, object],
    tolerance: float,
    *,
    quantity: str | None = None,
    unit: str | None = None,
    verify: bool = True,
) -> dict[str, object]:
    path = Path(path)
    payload, output_metadata = encode_bytes(
        values, metadata, tolerance, quantity=quantity, unit=unit
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    if verify:
        restored, _ = decode(path)
        expected = output_metadata["storage_encoding"]["error_vs_input_float32"]
        actual = error_stats(np.asarray(values, dtype=np.float32), restored)
        if not np.isclose(actual["max_abs"], expected["max_abs"], rtol=0, atol=1e-12):
            raise RuntimeError("CP3D v1 independent decode verification failed")
    return {
        "output": str(path.resolve()),
        "output_bytes": path.stat().st_size,
        "shape": list(values.shape),
        "tolerance": tolerance,
        "anchor_count": output_metadata["storage_encoding"]["anchor_count"],
        "error": output_metadata["storage_encoding"]["error_vs_input_float32"],
        "verified": verify,
    }
