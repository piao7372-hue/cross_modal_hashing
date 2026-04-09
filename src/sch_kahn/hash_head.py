from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HashHeadOutput:
    v_i: np.ndarray
    v_t: np.ndarray
    b_i: np.ndarray
    b_t: np.ndarray
    d_hash: int


class SharedHashHead:
    """Shared-parameter hash head for SCH-KANH forward-only v2."""

    def __init__(self, *, d_in: int, d_hash: int, seed: int) -> None:
        if d_in <= 0:
            raise ValueError("d_in must be > 0")
        if d_hash <= 0:
            raise ValueError("d_hash must be > 0")

        self.d_in = int(d_in)
        self.d_hash = int(d_hash)

        rng = np.random.default_rng(seed)
        scale = np.float32(1.0 / np.sqrt(max(self.d_in, 1)))
        self.w_hash = (rng.standard_normal((self.d_in, self.d_hash)).astype(np.float32) * scale)
        self.b_hash = np.zeros((self.d_hash,), dtype=np.float32)

    @staticmethod
    def _to_binary(v: np.ndarray) -> np.ndarray:
        # Locked rule: >= 0 -> +1, < 0 -> -1
        return np.where(v >= 0.0, 1, -1).astype(np.int8, copy=False)

    def forward(self, zout_i: np.ndarray, zout_t: np.ndarray) -> HashHeadOutput:
        zi = np.asarray(zout_i, dtype=np.float32)
        zt = np.asarray(zout_t, dtype=np.float32)
        if zi.ndim != 2 or zt.ndim != 2:
            raise RuntimeError("Zout_I and Zout_T must be 2D")
        if zi.shape != zt.shape:
            raise RuntimeError(f"Zout shape mismatch: {zi.shape} != {zt.shape}")
        if int(zi.shape[1]) != self.d_in:
            raise RuntimeError(f"Hash head input dim mismatch: expected {self.d_in}, got {zi.shape[1]}")

        # Locked formula:
        # V_I = tanh(Zout_I @ W_hash + b_hash)
        # V_T = tanh(Zout_T @ W_hash + b_hash)
        v_i = np.tanh((zi @ self.w_hash) + self.b_hash).astype(np.float32, copy=False)
        v_t = np.tanh((zt @ self.w_hash) + self.b_hash).astype(np.float32, copy=False)
        b_i = self._to_binary(v_i)
        b_t = self._to_binary(v_t)

        return HashHeadOutput(v_i=v_i, v_t=v_t, b_i=b_i, b_t=b_t, d_hash=self.d_hash)
