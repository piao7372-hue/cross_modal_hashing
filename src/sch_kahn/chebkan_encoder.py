from __future__ import annotations

import numpy as np


class ChebKANEncoder:
    """Forward-only ChebKAN encoder with deterministic seeded weights."""

    def __init__(self, *, d_in: int, d_model: int, order_k: int, seed: int) -> None:
        if d_in <= 0:
            raise ValueError("d_in must be > 0")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if order_k < 1:
            raise ValueError("order_k must be >= 1")

        self.d_in = int(d_in)
        self.d_model = int(d_model)
        self.order_k = int(order_k)

        rng = np.random.default_rng(seed)
        scale = np.float32(1.0 / np.sqrt(max(self.d_in, 1)))
        self.weights: list[np.ndarray] = [
            (rng.standard_normal((self.d_in, self.d_model)).astype(np.float32) * scale)
            for _ in range(self.order_k + 1)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 2:
            raise RuntimeError(f"Encoder input must be 2D, got shape {x_arr.shape}")
        if int(x_arr.shape[1]) != self.d_in:
            raise RuntimeError(f"Encoder input dim mismatch: expected {self.d_in}, got {x_arr.shape[1]}")

        t0 = np.ones_like(x_arr, dtype=np.float32)
        t1 = x_arr

        out = t0 @ self.weights[0]
        out = out + (t1 @ self.weights[1])

        prev2 = t0
        prev1 = t1
        for k in range(2, self.order_k + 1):
            tk = (2.0 * x_arr * prev1) - prev2
            out = out + (tk @ self.weights[k])
            prev2 = prev1
            prev1 = tk
        return out.astype(np.float32, copy=False)
