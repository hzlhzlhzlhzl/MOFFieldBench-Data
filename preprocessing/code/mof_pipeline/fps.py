#!/usr/bin/env python3
"""
Generate nested periodic-FPS and ordinary-FPS point clouds from a MOF PLY file.

Tailored to PLY files such as density_gt_0.03(1).ply, whose header contains:
    comment origin ...
    comment cell_a ...
    comment cell_b ...
    comment cell_c ...
and vertex fields:
    x y z fx fy fz density esp

The script computes ONE progressive FPS ordering up to the largest requested size
for each method, then writes every smaller point cloud as a prefix. Thus the
2048-point cloud is contained in the 4096-point cloud, etc.

Recommended command (exact ordinary FPS + fast validated periodic image search):

    python mof_periodic_nonperiodic_fps.py \
        --input "density_gt_0.03(1).ply" \
        --output-dir fps_outputs \
        --device cuda \
        --periodic-mode validated8

Strict periodic distances can be requested with --periodic-mode images27, but
this is about 27 times more expensive per FPS update and is normally unnecessary
when validated8 passes for the supplied lattice.

Dependencies:
    numpy
    torch

For the supplied ~520k-point cloud and a maximum sample size of 40960, CUDA is
strongly recommended. Exact FPS has O(M*K) complexity. CPU execution is valid
but can be very slow.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


PLY_TO_NUMPY = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass
class PlyData:
    path: Path
    format_name: str
    vertex_count: int
    property_types: List[Tuple[str, str]]
    vertices: np.ndarray
    header_lines: List[str]
    comments: List[str]
    origin: np.ndarray
    cell: np.ndarray


@dataclass
class FpsResult:
    indices: np.ndarray
    metrics: Dict[int, Dict[str, float]]
    elapsed_seconds: float
    unique_count: int


def normalize_sizes(
    requested_sizes: Sequence[int],
    point_count: int,
    size_policy: str,
) -> List[int]:
    unique_sizes = sorted({int(v) for v in requested_sizes if int(v) > 0})
    if not unique_sizes:
        raise ValueError("No positive sample sizes were provided.")

    if size_policy == "error":
        if max(unique_sizes) > point_count:
            raise ValueError(
                f"Maximum requested size {max(unique_sizes)} exceeds point count {point_count}."
            )
        return unique_sizes

    return unique_sizes


def expand_indices_to_size(base_indices: np.ndarray, target_size: int) -> np.ndarray:
    if target_size <= len(base_indices):
        return base_indices[:target_size].copy()
    repeats = int(math.ceil(target_size / len(base_indices)))
    tiled = np.tile(base_indices, repeats)
    return tiled[:target_size].copy()


def parse_vector_comment(comments: Iterable[str], key: str) -> np.ndarray | None:
    prefix = key + " "
    for comment in comments:
        if comment.startswith(prefix):
            values = comment[len(prefix):].split()
            if len(values) != 3:
                raise ValueError(f"Malformed PLY comment: 'comment {comment}'")
            return np.asarray([float(v) for v in values], dtype=np.float64)
    return None


def read_ply(path: Path) -> PlyData:
    """Read a vertex-only ASCII or binary-little-endian PLY file."""
    with path.open("rb") as handle:
        header_raw: List[bytes] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("Unexpected end of file before 'end_header'.")
            header_raw.append(line)
            if line.strip() == b"end_header":
                break
        data_offset = handle.tell()

    header_lines = [line.decode("ascii").rstrip("\r\n") for line in header_raw]
    if not header_lines or header_lines[0] != "ply":
        raise ValueError("Input is not a PLY file.")

    format_name = ""
    vertex_count = None
    comments: List[str] = []
    property_types: List[Tuple[str, str]] = []
    active_element = None
    other_elements: List[str] = []

    for line in header_lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            format_name = parts[1]
        elif parts[0] == "comment":
            comments.append(line[len("comment "):])
        elif parts[0] == "element":
            active_element = parts[1]
            count = int(parts[2])
            if active_element == "vertex":
                vertex_count = count
            elif count != 0:
                other_elements.append(active_element)
        elif parts[0] == "property" and active_element == "vertex":
            if parts[1] == "list":
                raise ValueError("List-valued vertex properties are not supported.")
            ply_type, name = parts[1], parts[2]
            if ply_type not in PLY_TO_NUMPY:
                raise ValueError(f"Unsupported PLY scalar type: {ply_type}")
            property_types.append((ply_type, name))

    if vertex_count is None:
        raise ValueError("No vertex element found in PLY header.")
    if other_elements:
        raise ValueError(
            "This script writes sampled vertex-only PLY files. The input contains "
            f"additional non-empty elements: {other_elements}"
        )
    if format_name not in {"binary_little_endian", "ascii"}:
        raise ValueError(
            f"Unsupported PLY format '{format_name}'. Use binary_little_endian or ascii."
        )

    endian = "<" if format_name == "binary_little_endian" else "="
    dtype = np.dtype([(name, endian + PLY_TO_NUMPY[ply_type]) for ply_type, name in property_types])

    if format_name == "binary_little_endian":
        with path.open("rb") as handle:
            handle.seek(data_offset)
            vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)
    else:
        matrix = np.loadtxt(path, skiprows=len(header_lines), max_rows=vertex_count)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape != (vertex_count, len(property_types)):
            raise ValueError(
                f"ASCII PLY shape mismatch: got {matrix.shape}, expected "
                f"({vertex_count}, {len(property_types)})."
            )
        vertices = np.empty(vertex_count, dtype=dtype)
        for column, (_, name) in enumerate(property_types):
            vertices[name] = matrix[:, column]

    if len(vertices) != vertex_count:
        raise ValueError(f"Read {len(vertices)} vertices, expected {vertex_count}.")

    origin = parse_vector_comment(comments, "origin")
    cell_a = parse_vector_comment(comments, "cell_a")
    cell_b = parse_vector_comment(comments, "cell_b")
    cell_c = parse_vector_comment(comments, "cell_c")

    if cell_a is None or cell_b is None or cell_c is None:
        metadata_path = path.parent / "DONE.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            origin = np.asarray(metadata["origin"], dtype=np.float64)
            lattice = np.asarray(metadata["lattice_vectors_rows"], dtype=np.float64)
            cell_a, cell_b, cell_c = lattice
            comments.extend(
                [
                    "origin " + " ".join(map(str, origin)),
                    "cell_a " + " ".join(map(str, cell_a)),
                    "cell_b " + " ".join(map(str, cell_b)),
                    "cell_c " + " ".join(map(str, cell_c)),
                    "coordinate_unit " + str(metadata.get("coordinate_unit", "unknown")),
                ]
            )
    if origin is None:
        origin = np.zeros(3, dtype=np.float64)
    if cell_a is None or cell_b is None or cell_c is None:
        raise ValueError(
            "Periodic FPS requires PLY comments 'cell_a', 'cell_b', and 'cell_c'."
        )
    cell = np.vstack([cell_a, cell_b, cell_c]).astype(np.float64)
    if abs(np.linalg.det(cell)) < 1e-10:
        raise ValueError("The lattice matrix is singular.")

    return PlyData(
        path=path,
        format_name=format_name,
        vertex_count=vertex_count,
        property_types=property_types,
        vertices=vertices,
        header_lines=header_lines,
        comments=comments,
        origin=origin,
        cell=cell,
    )


def extract_coordinates(data: PlyData) -> Tuple[np.ndarray, np.ndarray, float]:
    names = set(data.vertices.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY must contain x, y, and z vertex properties.")

    cart = np.column_stack(
        [data.vertices["x"], data.vertices["y"], data.vertices["z"]]
    ).astype(np.float32, copy=False)

    if {"fx", "fy", "fz"}.issubset(names):
        frac = np.column_stack(
            [data.vertices["fx"], data.vertices["fy"], data.vertices["fz"]]
        ).astype(np.float64, copy=False)
    elif {"frac_x", "frac_y", "frac_z"}.issubset(names):
        frac = np.column_stack(
            [
                data.vertices["frac_x"],
                data.vertices["frac_y"],
                data.vertices["frac_z"],
            ]
        ).astype(np.float64, copy=False)
    else:
        frac = (cart.astype(np.float64) - data.origin) @ np.linalg.inv(data.cell)

    frac = np.mod(frac, 1.0)

    # Compare the provided/derived fractional coordinates with Cartesian coordinates
    # under periodic equivalence. Tiny floating-point negatives near zero must not be
    # interpreted as points one full lattice vector away.
    cart_frac_raw = (cart.astype(np.float64) - data.origin) @ np.linalg.inv(data.cell)
    delta_frac = frac - cart_frac_raw
    delta_frac -= np.round(delta_frac)
    residual_cart = delta_frac @ data.cell
    max_residual = float(np.max(np.linalg.norm(residual_cart, axis=1)))

    return cart, frac.astype(np.float32), max_residual


def make_sampled_header(
    source: PlyData,
    sample_count: int,
    method: str,
    periodic_mode: str,
    seed: int,
    unique_count: int | None = None,
    repeated_count: int | None = None,
) -> List[str]:
    output: List[str] = []
    inserted_sampling_comments = False
    existing_comments = {
        line[len("comment "):].split(maxsplit=1)[0]
        for line in source.header_lines
        if line.startswith("comment ") and line[len("comment "):].strip()
    }
    for line in source.header_lines:
        if line.startswith("element vertex "):
            if not inserted_sampling_comments:
                metadata_comments = [
                    ("origin", source.origin),
                    ("cell_a", source.cell[0]),
                    ("cell_b", source.cell[1]),
                    ("cell_c", source.cell[2]),
                ]
                for key, vector in metadata_comments:
                    if key not in existing_comments:
                        output.append(
                            f"comment {key} " + " ".join(f"{value:.12g}" for value in vector)
                        )
                if "coordinate_unit" not in existing_comments:
                    unit = next((c.split(maxsplit=1)[1] for c in source.comments if c.startswith("coordinate_unit ")), "unknown")
                    output.append(f"comment coordinate_unit {unit}")
                output.extend(
                    [
                        f"comment sampling_method {method}",
                        f"comment sampling_size {sample_count}",
                        f"comment source_vertex_count {source.vertex_count}",
                        f"comment fps_seed {seed}",
                        f"comment periodic_distance_mode {periodic_mode}",
                    ]
                )
                if unique_count is not None:
                    output.append(f"comment unique_sample_count {unique_count}")
                if repeated_count is not None:
                    output.append(f"comment repeated_point_count {repeated_count}")
                inserted_sampling_comments = True
            output.append(f"element vertex {sample_count}")
        else:
            output.append(line)
    return output


def write_sampled_ply(
    output_path: Path,
    source: PlyData,
    indices: np.ndarray,
    method: str,
    periodic_mode: str,
    seed: int,
    unique_count: int | None = None,
    repeated_count: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled = source.vertices[indices]
    header = make_sampled_header(
        source=source,
        sample_count=len(indices),
        method=method,
        periodic_mode=periodic_mode,
        seed=seed,
        unique_count=unique_count,
        repeated_count=repeated_count,
    )

    if source.format_name == "binary_little_endian":
        with output_path.open("wb") as handle:
            handle.write(("\n".join(header) + "\n").encode("ascii"))
            sampled.tofile(handle)
    else:
        names = sampled.dtype.names or ()
        with output_path.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("\n".join(header) + "\n")
            for row in sampled:
                handle.write(" ".join(str(row[name].item()) for name in names) + "\n")


def qform_components(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    metric: torch.Tensor,
) -> torch.Tensor:
    """Squared Cartesian length of fractional displacement (u,v,w)."""
    return (
        metric[0, 0] * u * u
        + metric[1, 1] * v * v
        + metric[2, 2] * w * w
        + (2.0 * metric[0, 1]) * u * v
        + (2.0 * metric[0, 2]) * u * w
        + (2.0 * metric[1, 2]) * v * w
    )


def periodic_distance2_wrapped(
    frac: torch.Tensor,
    center: torch.Tensor,
    metric: torch.Tensor,
) -> torch.Tensor:
    """
    Fast component-wrapped minimum image distance.

    Exact for orthogonal cells. For oblique cells, use validated8 or images27.
    """
    delta = frac - center
    delta = delta - torch.floor(delta + 0.5)
    return qform_components(delta[:, 0], delta[:, 1], delta[:, 2], metric)


def periodic_distance2_validated8(
    frac: torch.Tensor,
    center: torch.Tensor,
    metric: torch.Tensor,
) -> torch.Tensor:
    """
    Minimum over 8 sign-directed images after wrapping into [-0.5, 0.5)^3.

    For the supplied MOF lattice, this agrees with a full 27-image search in the
    validation performed by this script. The validation is repeated at runtime.
    """
    delta = frac - center
    delta = delta - torch.floor(delta + 0.5)

    base_u, base_v, base_w = delta[:, 0], delta[:, 1], delta[:, 2]
    alt_u = torch.where(base_u >= 0.0, base_u - 1.0, base_u + 1.0)
    alt_v = torch.where(base_v >= 0.0, base_v - 1.0, base_v + 1.0)
    alt_w = torch.where(base_w >= 0.0, base_w - 1.0, base_w + 1.0)

    best = torch.full_like(base_u, torch.inf)
    for bu, bv, bw in itertools.product((0, 1), repeat=3):
        u = alt_u if bu else base_u
        v = alt_v if bv else base_v
        w = alt_w if bw else base_w
        d2 = qform_components(u, v, w, metric)
        best = torch.minimum(best, d2)
    return best


def periodic_distance2_images27(
    frac: torch.Tensor,
    center: torch.Tensor,
    metric: torch.Tensor,
) -> torch.Tensor:
    """Strict minimum over the 27 neighboring lattice images."""
    delta = frac - center
    delta = delta - torch.floor(delta + 0.5)
    base_u, base_v, base_w = delta[:, 0], delta[:, 1], delta[:, 2]

    best = torch.full_like(base_u, torch.inf)
    for su, sv, sw in itertools.product((-1.0, 0.0, 1.0), repeat=3):
        d2 = qform_components(base_u + su, base_v + sv, base_w + sw, metric)
        best = torch.minimum(best, d2)
    return best


def numpy_qform(x: np.ndarray, metric: np.ndarray) -> np.ndarray:
    return np.einsum("ni,ij,nj->n", x, metric, x, optimize=True)


def validate_eight_image_search(
    cell: np.ndarray,
    seed: int,
    random_count: int = 200_000,
    tolerance: float = 1e-9,
) -> Tuple[bool, float, int]:
    """Empirically compare the 8-image shortcut with a full 27-image search."""
    rng = np.random.default_rng(seed)
    random_delta = rng.uniform(-0.5, 0.5, size=(random_count, 3))

    grid_axis = np.linspace(-0.5, 0.5, 17, endpoint=False) + (0.5 / 17.0)
    grid = np.stack(np.meshgrid(grid_axis, grid_axis, grid_axis, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3)
    delta = np.vstack([random_delta, grid])

    metric = cell @ cell.T
    min27 = np.full(len(delta), np.inf, dtype=np.float64)
    for shift in itertools.product((-1.0, 0.0, 1.0), repeat=3):
        candidate = delta + np.asarray(shift, dtype=np.float64)
        min27 = np.minimum(min27, numpy_qform(candidate, metric))

    sign_shift = np.where(delta >= 0.0, -1.0, 1.0)
    min8 = np.full(len(delta), np.inf, dtype=np.float64)
    for bits in itertools.product((0.0, 1.0), repeat=3):
        candidate = delta + sign_shift * np.asarray(bits, dtype=np.float64)
        min8 = np.minimum(min8, numpy_qform(candidate, metric))

    difference = np.abs(min27 - min8)
    max_difference = float(difference.max())
    mismatch_count = int(np.count_nonzero(difference > tolerance))
    return mismatch_count == 0, max_difference, mismatch_count


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return device


def progressive_fps(
    cart: np.ndarray,
    frac: np.ndarray,
    cell: np.ndarray,
    sizes: Sequence[int],
    method: str,
    periodic_mode: str,
    device: torch.device,
    start_index: int,
    progress_every: int,
) -> FpsResult:
    """Compute one exact progressive FPS ordering up to the available unique count."""
    point_count = len(cart)
    requested_max_samples = max(sizes)
    unique_max_samples = min(requested_max_samples, point_count)

    dtype = torch.float32
    cart_t = torch.as_tensor(cart, dtype=dtype, device=device)
    frac_t = torch.as_tensor(frac, dtype=dtype, device=device)
    cell_t = torch.as_tensor(cell, dtype=dtype, device=device)
    metric_t = cell_t @ cell_t.T

    selected = np.empty(unique_max_samples, dtype=np.int64)
    selected[0] = start_index
    min_distance2 = torch.full(
        (point_count,), torch.inf, dtype=dtype, device=device
    )
    selected_mask = torch.zeros(point_count, dtype=torch.bool, device=device)
    requested_sizes = set(int(v) for v in sizes if int(v) <= unique_max_samples)
    metrics: Dict[int, Dict[str, float]] = {}

    if method == "periodic":
        if periodic_mode == "wrapped":
            distance_function = periodic_distance2_wrapped
        elif periodic_mode == "validated8":
            distance_function = periodic_distance2_validated8
        elif periodic_mode == "images27":
            distance_function = periodic_distance2_images27
        else:
            raise ValueError(f"Unknown periodic mode: {periodic_mode}")
    elif method == "nonperiodic":
        distance_function = None
    else:
        raise ValueError(f"Unknown FPS method: {method}")

    print(
        f"[{method}] points={point_count:,}, requested_max={requested_max_samples:,}, "
        f"unique_max={unique_max_samples:,}, "
        f"device={device}, start_index={start_index}"
    )
    start_time = time.perf_counter()

    with torch.inference_mode():
        for step in range(unique_max_samples):
            current_index = int(selected[step])

            if method == "nonperiodic":
                delta = cart_t - cart_t[current_index]
                distance2 = torch.sum(delta * delta, dim=1)
            else:
                distance2 = distance_function(
                    frac_t, frac_t[current_index], metric_t
                )

            min_distance2 = torch.minimum(min_distance2, distance2)
            selected_mask[current_index] = True

            sample_count = step + 1
            need_next = sample_count < unique_max_samples
            need_metric = sample_count in requested_sizes
            need_progress = (
                sample_count == 1
                or sample_count == unique_max_samples
                or sample_count % progress_every == 0
            )

            if need_next or need_metric or need_progress:
                max_value, max_index = torch.max(min_distance2, dim=0)
            else:
                max_value = max_index = None

            if need_metric:
                mean_distance = float(torch.sqrt(min_distance2).mean().item())
                coverage_radius = math.sqrt(max(float(max_value.item()), 0.0))
                metrics[sample_count] = {
                    "coverage_radius": coverage_radius,
                    "mean_nearest_distance": mean_distance,
                }

            if need_progress:
                elapsed = time.perf_counter() - start_time
                coverage = math.sqrt(max(float(max_value.item()), 0.0))
                rate = sample_count / max(elapsed, 1e-9)
                eta = (unique_max_samples - sample_count) / max(rate, 1e-9)
                print(
                    f"[{method}] {sample_count:>6,}/{unique_max_samples:,}  "
                    f"coverage={coverage:.5f} (input unit)  elapsed={elapsed/60:.1f} min  "
                    f"ETA≈{eta/60:.1f} min",
                    flush=True,
                )

            if need_next:
                # Exclude selected rows even when all remaining distances are zero.
                scores = min_distance2.masked_fill(selected_mask, -1.0)
                selected[step + 1] = int(torch.argmax(scores).item())

    elapsed_seconds = time.perf_counter() - start_time
    del cart_t, frac_t, cell_t, metric_t, min_distance2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if requested_max_samples > unique_max_samples:
        for size in sizes:
            if size > unique_max_samples:
                metrics[size] = {
                    "coverage_radius": 0.0,
                    "mean_nearest_distance": 0.0,
                }

    return FpsResult(
        indices=selected,
        metrics=metrics,
        elapsed_seconds=elapsed_seconds,
        unique_count=unique_max_samples,
    )


def parse_sizes(text: str) -> List[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("Sample sizes must be positive integers.")
    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate periodic and non-periodic FPS samples from a MOF PLY point cloud."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input PLY path.")
    parser.add_argument(
        "--input-extra",
        type=Path,
        default=None,
        help="Optional second compatible PLY whose vertices are concatenated before FPS.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Output filename stem; defaults to the primary input stem.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("fps_outputs"), help="Output directory."
    )
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=parse_sizes("2048,4096,8192,10240,12288,16384,20480,40960"),
        help="Comma-separated sample sizes.",
    )
    parser.add_argument(
        "--methods",
        choices=("both", "periodic", "nonperiodic"),
        default="both",
        help="Which FPS variants to run.",
    )
    parser.add_argument(
        "--periodic-mode",
        choices=("validated8", "images27", "wrapped"),
        default="validated8",
        help=(
            "Periodic distance: validated8 is fast and checked against 27 images for "
            "the current lattice; images27 is strict but much slower; wrapped is the "
            "fast component-wise minimum image and is exact only for orthogonal cells."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument("--seed", type=int, default=20260726, help="FPS start-point seed.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional explicit initial point index. Overrides --seed.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=256,
        help="Print progress every this many selected points.",
    )
    parser.add_argument(
        "--validation-random-count",
        type=int,
        default=200_000,
        help="Random displacement count used to validate validated8 against images27.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output PLY files.",
    )
    parser.add_argument(
        "--size-policy",
        choices=("clip", "error"),
        default="clip",
        help=(
            "How to handle requested FPS sizes larger than the input point count. "
            "'clip' repeats the FPS index order to pad oversized samples; "
            "'error' keeps the old strict behavior."
        ),
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    requested_sizes: List[int] = args.sizes

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")

    data = read_ply(input_path)
    if args.input_extra is not None:
        extra_path = args.input_extra.expanduser().resolve()
        extra = read_ply(extra_path)
        if data.property_types != extra.property_types:
            raise ValueError("Primary and extra PLY properties do not match.")
        if not np.allclose(data.origin, extra.origin) or not np.allclose(data.cell, extra.cell):
            raise ValueError("Primary and extra PLY cell metadata do not match.")
        data.vertices = np.concatenate([data.vertices, extra.vertices])
        data.vertex_count = len(data.vertices)
    if data.vertex_count == 0:
        raise ValueError("Cannot run FPS on an empty point cloud")
    cart, frac, residual = extract_coordinates(data)
    if not np.isfinite(cart).all() or not np.isfinite(frac).all():
        raise ValueError("Point coordinates must be finite")
    device = choose_device(args.device)

    print(f"Input: {input_path}")
    print(f"Vertices: {data.vertex_count:,}")
    print(f"Properties: {', '.join(data.vertices.dtype.names or ())}")
    print("Lattice matrix (rows are a, b, c; input coordinate unit):")
    print(data.cell)
    print(f"Wrapped Cartesian/fractional consistency residual: {residual:.3e} (input unit)")
    sizes = normalize_sizes(
        requested_sizes=requested_sizes,
        point_count=data.vertex_count,
        size_policy=args.size_policy,
    )
    print(f"Requested sample sizes: {requested_sizes}")
    if sizes != requested_sizes:
        print(f"Adjusted sample sizes:  {sizes}")

    rng = np.random.default_rng(args.seed)
    if args.start_index is None:
        start_index = int(rng.integers(0, data.vertex_count))
    else:
        start_index = int(args.start_index)
        if not 0 <= start_index < data.vertex_count:
            raise ValueError("--start-index is outside the input point range.")

    if args.periodic_mode == "validated8" and args.methods in {"both", "periodic"}:
        print("Validating 8-image periodic search against strict 27-image search...")
        valid, max_difference, mismatches = validate_eight_image_search(
            data.cell,
            seed=args.seed,
            random_count=args.validation_random_count,
        )
        print(
            f"Validation: valid={valid}, mismatches={mismatches}, "
            f"max squared-distance difference={max_difference:.3e}"
        )
        if not valid:
            raise RuntimeError(
                "validated8 did not match images27 for this lattice. Rerun with "
                "--periodic-mode images27."
            )

    if device.type == "cpu" and max(sizes) >= 10_000:
        print(
            "WARNING: exact FPS with this many input/output points is O(M*K) and can "
            "be very slow on CPU. A CUDA-enabled PyTorch installation is recommended.",
            file=sys.stderr,
        )

    methods = (
        ["periodic", "nonperiodic"]
        if args.methods == "both"
        else [args.methods]
    )

    results: Dict[str, FpsResult] = {}
    stem = args.output_stem or input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        result = progressive_fps(
            cart=cart,
            frac=frac,
            cell=data.cell,
            sizes=sizes,
            method=method,
            periodic_mode=args.periodic_mode,
            device=device,
            start_index=start_index,
            progress_every=args.progress_every,
        )
        results[method] = result

        method_dir = output_dir / f"{method}_fps"
        method_dir.mkdir(parents=True, exist_ok=True)

        index_path = method_dir / f"{stem}_{method}_fps_order_{max(sizes)}.npy"
        np.save(index_path, result.indices)

        for size in sizes:
            output_path = method_dir / f"{stem}_{method}_fps_{size}.ply"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_path}. Use --overwrite to replace it."
                )
            sampled_indices = expand_indices_to_size(result.indices, size)
            write_sampled_ply(
                output_path=output_path,
                source=data,
                indices=sampled_indices,
                method=f"{method}_fps",
                periodic_mode=args.periodic_mode if method == "periodic" else "none",
                seed=args.seed,
                unique_count=min(size, result.unique_count),
                repeated_count=max(0, size - result.unique_count),
            )
            print(f"Wrote {output_path}")

    manifest = {
        "input": str(input_path),
        "input_extra": str(args.input_extra) if args.input_extra else None,
        "coordinate_unit": next((c.split(maxsplit=1)[1] for c in data.comments if c.startswith("coordinate_unit ")), "unknown"),
        "input_vertex_count": data.vertex_count,
        "properties": list(data.vertices.dtype.names or ()),
        "origin": data.origin.tolist(),
        "cell_rows": data.cell.tolist(),
        "requested_sizes": requested_sizes,
        "sizes": sizes,
        "size_policy": args.size_policy,
        "seed": args.seed,
        "start_index": start_index,
        "device": str(device),
        "periodic_mode": args.periodic_mode,
        "methods": {},
    }

    csv_rows: List[Dict[str, object]] = []
    for method, result in results.items():
        manifest["methods"][method] = {
            "elapsed_seconds": result.elapsed_seconds,
            "metrics": result.metrics,
        }
        for size in sizes:
            metric = result.metrics[size]
            csv_rows.append(
                {
                    "method": method,
                    "sample_size": size,
                    "unique_sample_count": min(size, result.unique_count),
                    "repeated_point_count": max(0, size - result.unique_count),
                    "coverage_radius": metric["coverage_radius"],
                    "mean_nearest_distance": metric[
                        "mean_nearest_distance"
                    ],
                    "elapsed_total_seconds": result.elapsed_seconds,
                }
            )

    manifest_path = output_dir / "fps_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    metrics_path = output_dir / "fps_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote metrics:  {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
