"""Closed-loop 3D Lorenzo DPCM used by the CP3D v1 format."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


SENTINEL = np.int16(-32768)


@njit(cache=True, nogil=True)
def _prediction(values: np.ndarray, i: int, j: int, k: int) -> float:
    prediction = 0.0
    if i > 0:
        prediction += float(values[i - 1, j, k])
    if j > 0:
        prediction += float(values[i, j - 1, k])
    if k > 0:
        prediction += float(values[i, j, k - 1])
    if i > 0 and j > 0:
        prediction -= float(values[i - 1, j - 1, k])
    if i > 0 and k > 0:
        prediction -= float(values[i - 1, j, k - 1])
    if j > 0 and k > 0:
        prediction -= float(values[i, j - 1, k - 1])
    if i > 0 and j > 0 and k > 0:
        prediction += float(values[i - 1, j - 1, k - 1])
    return prediction


@njit(cache=True, nogil=True)
def encode_lorenzo(values: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = values.shape
    codes = np.zeros(values.shape, dtype=np.int16)
    anchors = np.zeros(values.shape, dtype=np.float32)
    reconstructed = np.empty(values.shape, dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                prediction = _prediction(reconstructed, i, j, k)
                code = int(np.rint((float(values[i, j, k]) - prediction) / step))
                if (i == 0 and j == 0 and k == 0) or code <= -32768 or code > 32767:
                    codes[i, j, k] = SENTINEL
                    reconstructed[i, j, k] = np.float32(values[i, j, k])
                    anchors[i, j, k] = reconstructed[i, j, k]
                else:
                    codes[i, j, k] = np.int16(code)
                    reconstructed[i, j, k] = np.float32(prediction + code * step)
    return codes, anchors


@njit(cache=True, nogil=True)
def decode_lorenzo(codes: np.ndarray, anchors: np.ndarray, step: float) -> np.ndarray:
    nx, ny, nz = codes.shape
    reconstructed = np.empty(codes.shape, dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                prediction = _prediction(reconstructed, i, j, k)
                code = codes[i, j, k]
                if code == SENTINEL:
                    reconstructed[i, j, k] = np.float32(anchors[i, j, k])
                else:
                    reconstructed[i, j, k] = np.float32(prediction + int(code) * step)
    return reconstructed


def zigzag_encode(codes: np.ndarray) -> np.ndarray:
    signed = codes.astype(np.int32)
    return ((signed << 1) ^ (signed >> 15)).astype(np.uint16)


def zigzag_decode(codes: np.ndarray) -> np.ndarray:
    unsigned = codes.astype(np.int32)
    return ((unsigned >> 1) ^ -(unsigned & 1)).astype(np.int16)


def sparse_anchors(anchor_grid: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.flatnonzero(codes.reshape(-1) == SENTINEL).astype(np.uint32)
    values = anchor_grid.reshape(-1)[positions].astype(np.float32)
    return positions, values


def dense_anchors(
    shape: tuple[int, ...], positions: np.ndarray, values: np.ndarray
) -> np.ndarray:
    anchors = np.zeros(int(np.prod(shape, dtype=np.int64)), dtype=np.float32)
    anchors[positions] = values
    return anchors.reshape(shape)


def error_stats(reference: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    reference_flat = reference.reshape(-1)
    restored_flat = restored.reshape(-1)
    count = reference_flat.size
    maximum = 0.0
    absolute_sum = 0.0
    squared_sum = 0.0
    for start in range(0, count, 2_000_000):
        stop = min(start + 2_000_000, count)
        difference = (
            reference_flat[start:stop].astype(np.float64)
            - restored_flat[start:stop].astype(np.float64)
        )
        absolute = np.abs(difference)
        maximum = max(maximum, float(absolute.max(initial=0.0)))
        absolute_sum += float(absolute.sum(dtype=np.float64))
        squared_sum += float(np.dot(difference, difference))
    return {
        "max_abs": maximum,
        "mean_abs": absolute_sum / count,
        "rmse": math.sqrt(squared_sum / count),
    }
