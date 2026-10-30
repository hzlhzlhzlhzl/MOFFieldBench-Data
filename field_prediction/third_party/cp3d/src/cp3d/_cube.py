"""Cube text parsing, metadata and writing."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np


FIELD_SETTINGS = {
    "electrostatic_potential": {"unit": "Hartree", "tolerance": 1e-4},
    "electron_density": {"unit": "electron/Bohr^3", "tolerance": 1e-6},
    "spin_density": {"unit": "electron/Bohr^3", "tolerance": 1e-6},
}


def classify_cube(path: Path) -> str | None:
    with path.open("rb") as handle:
        comments = (handle.readline() + handle.readline()).decode(
            "utf-8", errors="replace"
        ).upper()
    name = path.name.upper()
    if "SPIN_DENSITY" in name or "SPIN DENSITY" in comments:
        return "spin_density"
    if "ELECTRON_DENSITY" in name or "ELECTRON DENSITY" in comments:
        return "electron_density"
    if "V_HARTREE" in name or "HARTREE POTENTIAL" in comments:
        return "electrostatic_potential"
    return None


def _read_header(handle: BinaryIO) -> tuple[dict[str, object], int]:
    comments = [
        handle.readline().decode("utf-8").rstrip("\r\n"),
        handle.readline().decode("utf-8").rstrip("\r\n"),
    ]
    atom_fields = handle.readline().decode("ascii").split()
    if len(atom_fields) < 4:
        raise ValueError("invalid Cube atom-count/origin line")
    signed_atom_count = int(atom_fields[0])
    atom_count = abs(signed_atom_count)
    origin = [float(value) for value in atom_fields[1:4]]
    shape: list[int] = []
    axes: list[list[float]] = []
    for _ in range(3):
        fields = handle.readline().decode("ascii").split()
        if len(fields) < 4:
            raise ValueError("invalid Cube grid-axis line")
        shape.append(abs(int(fields[0])))
        axes.append([float(value) for value in fields[1:4]])
    atoms: list[list[float | int]] = []
    for _ in range(atom_count):
        fields = handle.readline().decode("ascii").split()
        if len(fields) < 5:
            raise ValueError("invalid Cube atom record")
        atoms.append([int(fields[0]), *[float(value) for value in fields[1:5]]])
    orbital_ids: list[int] | None = None
    if signed_atom_count < 0:
        fields = handle.readline().decode("ascii").split()
        if not fields:
            raise ValueError("missing Cube orbital-ID record")
        orbital_count = int(fields[0])
        orbital_ids = [int(value) for value in fields[1:]]
        while len(orbital_ids) < orbital_count:
            orbital_ids.extend(
                int(value) for value in handle.readline().decode("ascii").split()
            )
        orbital_ids = orbital_ids[:orbital_count]
    metadata: dict[str, object] = {
        "format": "cube-float32-v1",
        "comments": comments,
        "shape": shape,
        "origin": origin,
        "axes": axes,
        "atoms": atoms,
        "cube_value_order": "C (third grid axis varies fastest)",
        "dtype": "float32",
        "signed_atom_count": signed_atom_count,
    }
    if orbital_ids is not None:
        metadata["orbital_ids"] = orbital_ids
    return metadata, handle.tell()


def read_cube(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    path = Path(path)
    # np.fromfile(..., sep=" ") is unexpectedly slow for large ASCII Cube files.
    # Reading once and using fromstring keeps parsing in NumPy's C loop.
    raw = path.read_bytes()
    handle = io.BytesIO(raw)
    metadata, data_offset = _read_header(handle)
    values = np.fromstring(
        raw[data_offset:].decode("ascii"), dtype=np.float32, sep=" "
    )
    shape = tuple(int(value) for value in metadata["shape"])
    expected = int(np.prod(shape, dtype=np.int64))
    if values.size != expected:
        raise ValueError(
            f"Cube declares {shape} ({expected:,} values), found {values.size:,}"
        )
    metadata["source_file"] = str(path.resolve())
    return values.reshape(shape), metadata


def parse_header_bytes(header: bytes) -> dict[str, object]:
    metadata, _ = _read_header(io.BytesIO(header))
    metadata["cube_header_text"] = header.decode("utf-8", errors="replace")
    return metadata


def write_cube(
    path: str | Path, values: np.ndarray, metadata: dict[str, object]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(value) for value in metadata["shape"])
    if values.shape != shape:
        raise ValueError(f"array shape {values.shape} does not match metadata {shape}")
    with path.open("w", encoding="ascii", newline="\n") as handle:
        raw_header = metadata.get("cube_header_text")
        if raw_header is not None:
            handle.write(str(raw_header))
            if not str(raw_header).endswith("\n"):
                handle.write("\n")
        else:
            comments = list(metadata.get("comments", ["CP3D reconstructed Cube", "SCALAR FIELD"]))
            while len(comments) < 2:
                comments.append("SCALAR FIELD")
            atoms = list(metadata["atoms"])
            origin = list(metadata["origin"])
            signed_atom_count = int(metadata.get("signed_atom_count", len(atoms)))
            handle.write(f"{comments[0]}\n{comments[1]}\n")
            handle.write(
                f"{signed_atom_count:5d}{origin[0]:13.6f}{origin[1]:13.6f}{origin[2]:13.6f}\n"
            )
            for count, vector in zip(shape, metadata["axes"]):
                handle.write(
                    f"{count:5d}{vector[0]:13.6f}{vector[1]:13.6f}{vector[2]:13.6f}\n"
                )
            for atom in atoms:
                handle.write(
                    f"{int(atom[0]):5d}{float(atom[1]):13.6f}{float(atom[2]):13.6f}"
                    f"{float(atom[3]):13.6f}{float(atom[4]):13.6f}\n"
                )
            if signed_atom_count < 0:
                orbital_ids = list(metadata.get("orbital_ids", []))
                handle.write(
                    f"{len(orbital_ids):5d}"
                    + "".join(f"{int(value):5d}" for value in orbital_ids)
                    + "\n"
                )
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        for start in range(0, flat.size, 6):
            handle.write(
                "".join(f" {float(value):13.5E}" for value in flat[start : start + 6])
            )
            handle.write("\n")
