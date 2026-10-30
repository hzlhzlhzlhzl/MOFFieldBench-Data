"""Experimental read-only support for the C++ CP3D7V2 container."""

from __future__ import annotations

import json
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange

from ._cube import parse_header_bytes
from ._dpcm import decode_lorenzo, dense_anchors, zigzag_decode


MAGIC = b"CP3D7V2\x00"
FREQUENCIES = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=np.int8,
)


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def unpack(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        if self.offset + size > len(self.data):
            raise ValueError("truncated CP3D v2 file")
        values = struct.unpack_from("<" + fmt, self.data, self.offset)
        self.offset += size
        return values[0] if len(values) == 1 else values

    def read(self, size: int) -> bytes:
        if self.offset + size > len(self.data):
            raise ValueError("truncated CP3D v2 payload")
        value = self.data[self.offset : self.offset + size]
        self.offset += size
        return value


@dataclass
class Chunk:
    raw_size: int
    compressed_size: int
    checksum: int
    payload: bytes = b""


def _inflate(chunk: Chunk) -> bytes:
    raw = zlib.decompress(chunk.payload)
    if len(raw) != chunk.raw_size:
        raise ValueError("CP3D v2 decompressed size mismatch")
    if zlib.crc32(raw) & 0xFFFFFFFF != chunk.checksum:
        raise ValueError("CP3D v2 CRC mismatch")
    return raw


def _basis() -> np.ndarray:
    x = np.arange(7, dtype=np.float64)
    result = np.empty((2, 7), dtype=np.float32)
    result[0] = np.sqrt(1.0 / 7.0)
    result[1] = (
        np.sqrt(2.0 / 7.0) * np.cos(np.pi * (2.0 * x + 1.0) / 14.0)
    ).astype(np.float32)
    return result


@njit(cache=True, nogil=True, parallel=True)
def _reconstruct_numba(
    coefficients: np.ndarray,
    basis: np.ndarray,
    frequencies: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    bx_count: int,
    by_count: int,
    bz_count: int,
) -> np.ndarray:
    predictor = np.empty((nx, ny, nz), dtype=np.float32)
    block_count = bx_count * by_count * bz_count
    for block in prange(block_count):
        bz = block % bz_count
        q = block // bz_count
        by = q % by_count
        bx = q // by_count
        for x in range(7):
            gx = bx * 7 + x
            if gx >= nx:
                continue
            for y in range(7):
                gy = by * 7 + y
                if gy >= ny:
                    continue
                for z in range(7):
                    gz = bz * 7 + z
                    if gz >= nz:
                        continue
                    value = np.float32(0.0)
                    for index in range(8):
                        u = frequencies[index, 0]
                        v = frequencies[index, 1]
                        w = frequencies[index, 2]
                        value += (
                            coefficients[block, index]
                            * basis[u, x]
                            * basis[v, y]
                            * basis[w, z]
                        )
                    predictor[gx, gy, gz] = value
    return predictor


def _reconstruct_predictor(
    coefficient_bits: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
    block_shape = tuple((size + 6) // 7 for size in shape)
    block_count = int(np.prod(block_shape, dtype=np.int64))
    coefficients = coefficient_bits.view(np.float16).astype(np.float32)
    if coefficients.shape != (block_count, 8):
        raise ValueError("invalid CP3D v2 DCT coefficient shape")
    basis = _basis()
    return _reconstruct_numba(
        coefficients,
        basis,
        FREQUENCIES,
        shape[0],
        shape[1],
        shape[2],
        block_shape[0],
        block_shape[1],
        block_shape[2],
    )


def decode(
    source: str | Path | bytes, *, workers: int = 0
) -> tuple[np.ndarray, dict[str, object]]:
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    reader = Reader(data)
    if reader.read(8) != MAGIC:
        raise ValueError("not a CP3D7V2 file")
    if reader.unpack("I") != 2:
        raise ValueError("unsupported CP3D v2 format version")
    shape = tuple(int(value) for value in reader.unpack("III"))
    block_size, keep = reader.unpack("HH")
    reader.unpack("I")
    tolerance, step = reader.unpack("dd")
    header_size, block_count, anchor_count = reader.unpack("QQQ")
    if block_size != 7 or keep != 8:
        raise ValueError("unsupported experimental CP3D v2 DCT layout")
    stream_chunks: list[list[Chunk]] = []
    for _ in range(4):
        chunk_count = reader.unpack("I")
        if not 1 <= chunk_count <= 1024:
            raise ValueError("invalid CP3D v2 chunk count")
        chunks = [
            Chunk(*reader.unpack("QQI"))
            for _ in range(chunk_count)
        ]
        stream_chunks.append(chunks)
    kept_indices = reader.unpack("8H")
    if tuple(kept_indices) != (0, 1, 7, 49, 8, 50, 56, 57):
        raise ValueError("unsupported CP3D v2 frequency order")
    anchor_positions = np.empty(anchor_count, dtype=np.uint64)
    anchor_values = np.empty(anchor_count, dtype=np.float32)
    for index in range(anchor_count):
        anchor_positions[index] = reader.unpack("Q")
        anchor_values[index] = reader.unpack("f")
    cube_header = reader.read(header_size)
    all_chunks = [chunk for chunks in stream_chunks for chunk in chunks]
    for chunk in all_chunks:
        chunk.payload = reader.read(chunk.compressed_size)
    if reader.offset != len(data):
        raise ValueError("trailing bytes in CP3D v2 file")
    count = workers if workers > 0 else min(32, len(all_chunks))
    if count <= 1:
        inflated = [_inflate(chunk) for chunk in all_chunks]
    else:
        with ThreadPoolExecutor(max_workers=count) as executor:
            inflated = list(executor.map(_inflate, all_chunks))
    iterator = iter(inflated)
    streams = [b"".join(next(iterator) for _ in chunks) for chunks in stream_chunks]

    dct_low = np.frombuffer(streams[0], dtype=np.uint8)
    dct_high = np.frombuffer(streams[1], dtype=np.uint8)
    deltas = (dct_low.astype(np.uint16) | (dct_high.astype(np.uint16) << 8)).reshape(
        8, block_count
    )
    bits_frequency_major = np.cumsum(
        deltas.astype(np.uint64), axis=1, dtype=np.uint64
    ).astype(np.uint16)
    coefficient_bits = np.ascontiguousarray(bits_frequency_major.T)
    predictor = _reconstruct_predictor(coefficient_bits, shape)

    code_low = np.frombuffer(streams[2], dtype=np.uint8)
    code_high = np.frombuffer(streams[3], dtype=np.uint8)
    unsigned_codes = code_low.astype(np.uint16) | (code_high.astype(np.uint16) << 8)
    codes = zigzag_decode(unsigned_codes).reshape(shape)
    anchors = dense_anchors(shape, anchor_positions, anchor_values)
    residual = decode_lorenzo(codes, anchors, step)
    values = (predictor + residual).astype(np.float32)
    metadata = parse_header_bytes(cube_header)
    metadata.update(
        {
            "format": "cp3d7v2-experimental",
            "shape": list(shape),
            "storage_encoding": {
                "format": "CP3D7V2",
                "type": "block_dct_predictor_plus_closed_loop_dpcm_lorenzo_3d",
                "block_size": 7,
                "dct_coefficients_per_block": 8,
                "tolerance": tolerance,
                "step": step,
                "anchor_count": int(anchor_count),
                "read_only_in_python_package": True,
            },
        }
    )
    return values, metadata
