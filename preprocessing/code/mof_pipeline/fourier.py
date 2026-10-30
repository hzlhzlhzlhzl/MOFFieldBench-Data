#!/usr/bin/env python3
"""Fourier downsample a periodic CP2K/Gaussian CUBE scalar field.

This script adapts the official pyRho Fourier interpolation/compression
implementation accompanying:

    Shen et al., "A representation-independent electronic charge density
    database for crystalline materials", Scientific Data 9, 661 (2022).

Official implementation:
    https://github.com/materialsproject/pyrho
    pyrho.utils.interpolate_fourier

The default ``auto`` backend uses the installed official ``mp-pyrho`` package.
If it is unavailable, it uses a bundled algorithm-compatible implementation of
``pyrho.utils.interpolate_fourier`` (FFT -> Fourier corner crop/pad -> IFFT with
magnitude rescaling). Use ``--backend pyrho`` to require the official package.

The input may be a CP2K CUBE file even if its extension is .txt. For the user's
example with a 225 x 225 x 225 grid, the default factor 3 produces 75^3.

Examples
--------
# Strictly use the official pyRho package:
pip install mp-pyrho
python fourier_downsample_cp2k_cube.py HARTREE.cube HARTREE_75.cube \
    --shape 75 75 75 --field-kind esp --backend pyrho

# Electron density, preserving the integrated density after optional clipping:
python fourier_downsample_cp2k_cube.py DENSITY.cube DENSITY_75.cube \
    --factor 3 --field-kind density

# Optional ESP convention operations (not enabled by default):
python fourier_downsample_cp2k_cube.py HARTREE.cube HARTREE_75.cube \
    --factor 3 --field-kind esp --flip-sign --zero-mean-esp

Notes
-----
* The lattice and origin remain unchanged; only grid dimensions and voxel
  vectors change.
* Fourier resampling treats the single cell as periodic in all three lattice
  directions. No explicit supercell is needed.
* Do not use ``--flip-sign`` or ``--zero-mean-esp`` unless those conventions
  are desired for the intended benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


@dataclass
class CubeData:
    """Parsed scalar CUBE data and its header metadata."""

    comment1: str
    comment2: str
    natoms: int
    origin: np.ndarray  # shape (3,), in the input CUBE coordinate unit
    signed_counts: np.ndarray  # shape (3,); sign encodes the CUBE unit convention
    voxel_vectors: np.ndarray  # shape (3, 3), rows are voxel step vectors
    atom_lines: list[str]
    values: np.ndarray  # shape (Nx, Ny, Nz)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(abs(x)) for x in self.signed_counts)

    @property
    def lattice(self) -> np.ndarray:
        """Return lattice vectors as rows, in the same unit as the CUBE header."""
        return np.abs(self.signed_counts)[:, None] * self.voxel_vectors

    @property
    def coordinate_unit(self) -> str:
        signs = np.sign(self.signed_counts)
        if np.all(signs > 0):
            return "bohr"
        if np.all(signs < 0):
            return "angstrom"
        return "mixed/unknown"


def _parse_floats(tokens: Sequence[str], expected: int, context: str) -> np.ndarray:
    if len(tokens) < expected:
        raise ValueError(f"{context}: expected at least {expected} values, got {len(tokens)}")
    try:
        return np.asarray([float(x) for x in tokens[:expected]], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{context}: failed to parse numeric values") from exc


def read_cube(path: Path) -> CubeData:
    """Read a standard single-field Gaussian/CP2K CUBE file.

    The function intentionally rejects negative atom counts (orbital CUBE files)
    because this script targets one ED or ESP scalar field per file.
    """
    with path.open("rb") as handle:
        comment1_b = handle.readline()
        comment2_b = handle.readline()
        if not comment1_b or not comment2_b:
            raise ValueError("File is too short to be a CUBE file")
        comment1 = comment1_b.decode("utf-8", errors="replace").rstrip("\r\n")
        comment2 = comment2_b.decode("utf-8", errors="replace").rstrip("\r\n")

        atom_header = handle.readline().decode("utf-8", errors="replace").split()
        if len(atom_header) < 4:
            raise ValueError("Invalid CUBE atom-count/origin line")
        natoms = int(atom_header[0])
        if natoms < 0:
            raise NotImplementedError(
                "Negative atom-count/orbital CUBE files are not supported by this single-field script"
            )
        origin = _parse_floats(atom_header[1:], 3, "CUBE origin")

        signed_counts: list[int] = []
        voxel_vectors: list[np.ndarray] = []
        for axis in range(3):
            axis_tokens = handle.readline().decode("utf-8", errors="replace").split()
            if len(axis_tokens) < 4:
                raise ValueError(f"Invalid CUBE axis line {axis + 1}")
            signed_count = int(axis_tokens[0])
            if signed_count == 0:
                raise ValueError(f"Grid count on axis {axis + 1} cannot be zero")
            signed_counts.append(signed_count)
            voxel_vectors.append(_parse_floats(axis_tokens[1:], 3, f"axis {axis + 1}"))

        atom_lines: list[str] = []
        for atom_idx in range(natoms):
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Unexpected EOF while reading atom line {atom_idx + 1}")
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if len(line.split()) < 5:
                raise ValueError(f"Invalid atom line {atom_idx + 1}: {line!r}")
            atom_lines.append(line)

        shape = tuple(abs(x) for x in signed_counts)
        expected = math.prod(shape)

        # np.fromfile starts at the current file position and avoids constructing
        # a second huge Python string for an 11-million-value CUBE file.
        values_flat = np.fromfile(handle, dtype=np.float64, count=expected, sep=" ")
        if values_flat.size != expected:
            raise ValueError(
                f"CUBE contains {values_flat.size:,} scalar values; expected {expected:,} "
                f"for grid {shape}"
            )

        extra = np.fromfile(handle, dtype=np.float64, count=1, sep=" ")
        if extra.size:
            raise ValueError("CUBE contains extra scalar values after the expected grid")

    values = values_flat.reshape(shape, order="C")
    return CubeData(
        comment1=comment1,
        comment2=comment2,
        natoms=natoms,
        origin=origin,
        signed_counts=np.asarray(signed_counts, dtype=np.int64),
        voxel_vectors=np.asarray(voxel_vectors, dtype=np.float64),
        atom_lines=atom_lines,
        values=values,
    )


def _official_compatible_pad_arr(arr_in: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Algorithm-compatible copy of pyRho's official ``pad_arr`` logic.

    It keeps/crops Fourier coefficients at the corners of NumPy's unshifted FFT
    layout. This supports both upsampling and downsampling, including odd sizes
    such as 225 -> 75.
    """
    target = tuple(int(x) for x in shape)
    if len(target) != arr_in.ndim:
        raise ValueError("Target shape dimensionality does not match the input array")
    if any(x <= 0 for x in target):
        raise ValueError("All target dimensions must be positive")

    dimensions = arr_in.shape
    boundaries = [
        (
            int(np.ceil(min(i_dim, j_dim) + 1) / 2.0),
            int(np.floor(min(i_dim, j_dim)) / 2.0),
        )
        for i_dim, j_dim in zip(dimensions, target)
    ]

    def get_slice(idig: str, idim: int) -> slice:
        if idig == "0":
            return slice(0, boundaries[idim][0])
        if idig == "1":
            return slice(-boundaries[idim][1], None) if boundaries[idim][1] else slice(0, 0)
        raise ValueError("Binary digit must be 0 or 1")

    dim = arr_in.ndim
    arr_out = np.zeros(target, dtype=arr_in.dtype)
    for corner_index in range(2**dim):
        bits = format(corner_index, f"#0{dim + 2}b")[-dim:]
        slices = tuple(get_slice(bit, idim) for idim, bit in enumerate(bits))
        arr_out[slices] = arr_in[slices]
    return arr_out


def official_compatible_interpolate_fourier(
    arr_in: np.ndarray, shape: Sequence[int]
) -> np.ndarray:
    """Match the official ``pyrho.utils.interpolate_fourier`` algorithm."""
    target = tuple(int(x) for x in shape)
    fft_res = np.fft.fftn(arr_in)
    fft_res = _official_compatible_pad_arr(fft_res, target)
    results = np.fft.ifftn(fft_res) * np.size(fft_res) / np.size(arr_in)
    if not np.iscomplexobj(arr_in):
        return np.real(results)
    return results


def get_fourier_backend(name: str) -> tuple[Callable[[np.ndarray, Sequence[int]], np.ndarray], str]:
    """Resolve the requested official or compatible Fourier backend."""
    if name in {"auto", "pyrho"}:
        try:
            from pyrho.utils import interpolate_fourier as pyrho_interpolate_fourier

            def official(arr: np.ndarray, shape: Sequence[int]) -> np.ndarray:
                return np.asarray(pyrho_interpolate_fourier(arr, list(shape)))

            return official, "official mp-pyrho:pyrho.utils.interpolate_fourier"
        except ImportError:
            if name == "pyrho":
                raise RuntimeError(
                    "Official mp-pyrho is not installed. Run: pip install mp-pyrho\n"
                    "Or use --backend compatible for the bundled algorithm-compatible implementation."
                ) from None

    if name in {"auto", "compatible"}:
        return official_compatible_interpolate_fourier, "bundled pyRho-compatible implementation"

    raise ValueError(f"Unknown backend: {name}")


def write_cube(path: Path, cube: CubeData, values: np.ndarray, note: str) -> None:
    """Write a downsampled scalar field while preserving the original cell."""
    target_shape = tuple(int(x) for x in values.shape)
    lattice = cube.lattice
    new_voxel_vectors = lattice / np.asarray(target_shape, dtype=np.float64)[:, None]

    original_signs = np.sign(cube.signed_counts).astype(int)
    new_signed_counts = original_signs * np.asarray(target_shape, dtype=int)

    comment2 = cube.comment2
    if note:
        suffix = f" | {note}"
        # CUBE comments are free-form. Keep lines manageable.
        comment2 = (comment2 + suffix)[:240]

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(cube.comment1 + "\n")
        handle.write(comment2 + "\n")
        handle.write(
            f"{cube.natoms:5d}"
            f"{cube.origin[0]:13.6f}{cube.origin[1]:13.6f}{cube.origin[2]:13.6f}\n"
        )
        for signed_count, vector in zip(new_signed_counts, new_voxel_vectors):
            handle.write(
                f"{int(signed_count):5d}"
                f"{vector[0]:13.6f}{vector[1]:13.6f}{vector[2]:13.6f}\n"
            )
        for atom_line in cube.atom_lines:
            handle.write(atom_line + "\n")

        flat = np.asarray(values, dtype=np.float64).ravel(order="C")
        for start in range(0, flat.size, 6):
            chunk = flat[start : start + 6]
            handle.write("".join(f"{value:13.5E}" for value in chunk) + "\n")


def relative_error(new: float, old: float) -> float:
    denom = max(abs(old), np.finfo(np.float64).tiny)
    return abs(new - old) / denom


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Periodically Fourier-downsample a CP2K/Gaussian CUBE file using the official "
            "pyRho algorithm."
        )
    )
    parser.add_argument("input", type=Path, help="Input CUBE file (extension may be .cube or .txt)")
    parser.add_argument("output", type=Path, help="Output downsampled CUBE file")

    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--shape",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        help="Explicit target grid shape, e.g. --shape 75 75 75",
    )
    target.add_argument(
        "--factor",
        type=int,
        default=3,
        help="Integer downsampling factor for every axis (default: 3)",
    )
    target.add_argument(
        "--ratio",
        type=float,
        default=None,
        help=(
            "Floating-point downsampling ratio for every axis. The target grid is "
            "computed as round(N / ratio) for each axis."
        ),
    )

    parser.add_argument(
        "--backend",
        choices=("auto", "pyrho", "compatible"),
        default="auto",
        help=(
            "auto: use official mp-pyrho when installed, otherwise compatible fallback; "
            "pyrho: require official package; compatible: use bundled matching algorithm"
        ),
    )
    parser.add_argument(
        "--field-kind",
        choices=("generic", "density", "esp"),
        default="generic",
        help="Controls physical checks and optional post-processing",
    )
    parser.add_argument(
        "--flip-sign",
        action="store_true",
        help="Multiply the input field by -1 before resampling (use only when required by convention)",
    )
    parser.add_argument(
        "--zero-mean-esp",
        action="store_true",
        help="Subtract the cell mean before and after resampling; only valid with --field-kind esp",
    )
    parser.add_argument(
        "--clip-negative-density",
        action="store_true",
        help=(
            "After resampling density, clip negative ringing to zero and rescale to preserve the "
            "original cell integral; disabled by default"
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for a JSON quality-control report",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Add --overwrite to replace it.")
    if args.zero_mean_esp and args.field_kind != "esp":
        raise ValueError("--zero-mean-esp requires --field-kind esp")
    if args.clip_negative_density and args.field_kind != "density":
        raise ValueError("--clip-negative-density requires --field-kind density")

    print(f"Reading {args.input} ...", flush=True)
    cube = read_cube(args.input)
    original_shape = cube.shape

    if args.shape is not None:
        target_shape = tuple(args.shape)
    elif args.ratio is not None:
        ratio = float(args.ratio)
        if ratio <= 0.0:
            raise ValueError("--ratio must be positive")
        target_shape = tuple(max(1, int(round(dim / ratio))) for dim in original_shape)
    else:
        factor = int(args.factor)
        if factor <= 0:
            raise ValueError("--factor must be a positive integer")
        if any(dim % factor != 0 for dim in original_shape):
            raise ValueError(
                f"Input shape {original_shape} is not exactly divisible by factor {factor}. "
                "Use --shape NX NY NZ explicitly."
            )
        target_shape = tuple(dim // factor for dim in original_shape)

    if any(dim <= 0 for dim in target_shape):
        raise ValueError("All target dimensions must be positive")
    if any(new > old for new, old in zip(target_shape, original_shape)):
        print(
            "Warning: at least one target dimension is larger than the input; this is Fourier "
            "upsampling rather than pure downsampling.",
            file=sys.stderr,
        )

    field = np.asarray(cube.values, dtype=np.float64)
    if args.flip_sign:
        field = -field

    mean_before_raw = float(np.mean(field))
    min_before = float(np.min(field))
    max_before = float(np.max(field))

    if args.zero_mean_esp:
        field = field - np.mean(field)

    mean_before = float(np.mean(field))
    lattice_volume = float(abs(np.linalg.det(cube.lattice)))
    integral_before = mean_before * lattice_volume

    fourier_resample, backend_label = get_fourier_backend(args.backend)
    print(
        f"Fourier resampling {original_shape} -> {target_shape} with {backend_label} ...",
        flush=True,
    )
    downsampled = np.asarray(fourier_resample(field, target_shape), dtype=np.float64)

    if args.zero_mean_esp:
        # Remove tiny roundoff residual; the zero Fourier mode should already be preserved.
        downsampled = downsampled - np.mean(downsampled)

    negative_fraction_before_clip = float(np.mean(downsampled < 0.0))
    clipped = False
    if args.clip_negative_density:
        downsampled = np.maximum(downsampled, 0.0)
        integral_after_clip = float(np.mean(downsampled) * lattice_volume)
        if abs(integral_after_clip) <= np.finfo(np.float64).tiny:
            raise ValueError("Cannot renormalize clipped density because its integral is zero")
        downsampled *= integral_before / integral_after_clip
        clipped = True

    mean_after = float(np.mean(downsampled))
    integral_after = mean_after * lattice_volume
    min_after = float(np.min(downsampled))
    max_after = float(np.max(downsampled))

    new_voxel_vectors = cube.lattice / np.asarray(target_shape)[:, None]
    old_step_lengths = np.linalg.norm(cube.voxel_vectors, axis=1)
    new_step_lengths = np.linalg.norm(new_voxel_vectors, axis=1)

    note = f"Fourier {original_shape[0]}x{original_shape[1]}x{original_shape[2]} to " \
           f"{target_shape[0]}x{target_shape[1]}x{target_shape[2]} via pyRho"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_cube(args.output, cube, downsampled, note=note)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "backend": backend_label,
        "field_kind": args.field_kind,
        "coordinate_unit": cube.coordinate_unit,
        "original_shape": list(original_shape),
        "target_shape": list(target_shape),
        "requested_ratio": float(args.ratio) if args.ratio is not None else None,
        "actual_axiswise_ratios": [
            float(old) / float(new) for old, new in zip(original_shape, target_shape)
        ],
        "lattice_vectors_rows": cube.lattice.tolist(),
        "lattice_volume_coordinate_unit_cubed": lattice_volume,
        "old_voxel_step_lengths": old_step_lengths.tolist(),
        "new_voxel_step_lengths": new_step_lengths.tolist(),
        "flip_sign": bool(args.flip_sign),
        "zero_mean_esp": bool(args.zero_mean_esp),
        "clip_negative_density": clipped,
        "mean_before_raw": mean_before_raw,
        "mean_before_resampling": mean_before,
        "mean_after_resampling": mean_after,
        "mean_relative_error": relative_error(mean_after, mean_before),
        "integral_before": integral_before,
        "integral_after": integral_after,
        "integral_relative_error": relative_error(integral_after, integral_before),
        "min_before": min_before,
        "max_before": max_before,
        "min_after": min_after,
        "max_after": max_after,
        "negative_fraction_after_fourier_before_optional_clip": negative_fraction_before_clip,
    }

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
