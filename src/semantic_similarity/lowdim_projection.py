from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .sparse_ops import EPS


@dataclass(frozen=True)
class LowDimProjectionResult:
    z_i: np.ndarray
    z_t: np.ndarray
    z: np.ndarray
    metadata: dict[str, Any]


def _row_l2_normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
    return (arr / np.maximum(norms, EPS)).astype(np.float32)


def build_shared_linear_tanh_projection(
    *,
    x_i: np.ndarray,
    x_t: np.ndarray,
    z_dim: int,
    seed: int,
) -> LowDimProjectionResult:
    xi = np.asarray(x_i, dtype=np.float32)
    xt = np.asarray(x_t, dtype=np.float32)

    if xi.ndim != 2 or xt.ndim != 2:
        raise RuntimeError("X_I and X_T must be 2D for low-dim projection")
    if xi.shape[0] != xt.shape[0]:
        raise RuntimeError(f"X_I/X_T row mismatch: {xi.shape[0]} != {xt.shape[0]}")
    if xi.shape[1] != xt.shape[1]:
        raise RuntimeError(
            "shared_linear_tanh requires X_I and X_T to have the same final dimension"
        )
    if z_dim <= 0:
        raise RuntimeError("z_dim must be > 0")

    d_in = int(xi.shape[1])
    rng = np.random.default_rng(seed)
    w_z = rng.normal(loc=0.0, scale=1.0 / np.sqrt(float(max(d_in, 1))), size=(d_in, z_dim)).astype(
        np.float32
    )
    b_z = np.zeros((z_dim,), dtype=np.float32)

    z_i = _row_l2_normalize(np.tanh(xi @ w_z + b_z))
    z_t = _row_l2_normalize(np.tanh(xt @ w_z + b_z))
    z = _row_l2_normalize((z_i + z_t) * np.float32(0.5))

    return LowDimProjectionResult(
        z_i=z_i,
        z_t=z_t,
        z=z,
        metadata={
            "projection_mode": "shared_linear_tanh",
            "fusion_mode": "arithmetic_mean",
            "pseudo_dims": {
                "x_i": int(xi.shape[1]),
                "x_t": int(xt.shape[1]),
                "z": int(z_dim),
            },
        },
    )
