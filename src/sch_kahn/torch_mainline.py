from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _seeded_randn(*, shape: tuple[int, ...], seed: int, scale: float) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randn(shape, generator=gen, dtype=torch.float32) * float(scale)


@dataclass(frozen=True)
class TorchHashHeadOutput:
    v_i: torch.Tensor
    v_t: torch.Tensor
    b_i: torch.Tensor
    b_t: torch.Tensor


@dataclass(frozen=True)
class TorchMainlineOutput:
    zout_i: torch.Tensor
    zout_t: torch.Tensor
    v_i: torch.Tensor
    v_t: torch.Tensor
    b_i: torch.Tensor
    b_t: torch.Tensor


class ShadowChebKANEncoder(nn.Module):
    """Torch shadow of the frozen encoder, used only for trainer smoke."""

    def __init__(self, *, d_in: int, d_model: int, order_k: int, seed: int) -> None:
        super().__init__()
        if d_in <= 0 or d_model <= 0:
            raise ValueError("d_in and d_model must be > 0")
        if order_k < 1:
            raise ValueError("order_k must be >= 1")

        self.d_in = int(d_in)
        self.d_model = int(d_model)
        self.order_k = int(order_k)
        scale = 1.0 / max(self.d_in, 1) ** 0.5
        self.weights = nn.ParameterList(
            [
                nn.Parameter(
                    _seeded_randn(shape=(self.d_in, self.d_model), seed=seed + k, scale=scale)
                )
                for k in range(self.order_k + 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise RuntimeError(f"Encoder input must be 2D, got shape {tuple(x.shape)}")
        if int(x.shape[1]) != self.d_in:
            raise RuntimeError(f"Encoder input dim mismatch: expected {self.d_in}, got {x.shape[1]}")

        x32 = x.to(dtype=torch.float32)
        t0 = torch.ones_like(x32, dtype=torch.float32)
        t1 = x32

        out = t0 @ self.weights[0]
        out = out + (t1 @ self.weights[1])

        prev2 = t0
        prev1 = t1
        for k in range(2, self.order_k + 1):
            tk = (2.0 * x32 * prev1) - prev2
            out = out + (tk @ self.weights[k])
            prev2 = prev1
            prev1 = tk
        return out


class ShadowSharedHashHead(nn.Module):
    """Torch shadow of the frozen hash head, used only for trainer smoke."""

    def __init__(self, *, d_in: int, d_hash: int, seed: int) -> None:
        super().__init__()
        if d_in <= 0 or d_hash <= 0:
            raise ValueError("d_in and d_hash must be > 0")

        self.d_in = int(d_in)
        self.d_hash = int(d_hash)
        scale = 1.0 / max(self.d_in, 1) ** 0.5
        self.w_hash = nn.Parameter(_seeded_randn(shape=(self.d_in, self.d_hash), seed=seed, scale=scale))
        self.b_hash = nn.Parameter(torch.zeros((self.d_hash,), dtype=torch.float32))

    @staticmethod
    def _derive_binary_target(v: torch.Tensor) -> torch.Tensor:
        # Trainer smoke rule: B is derived from V only and is not an independent learnable branch.
        return torch.where(v >= 0.0, torch.ones_like(v), -torch.ones_like(v)).detach()

    def forward(self, zout_i: torch.Tensor, zout_t: torch.Tensor) -> TorchHashHeadOutput:
        if zout_i.ndim != 2 or zout_t.ndim != 2:
            raise RuntimeError("Zout_I and Zout_T must be 2D")
        if tuple(zout_i.shape) != tuple(zout_t.shape):
            raise RuntimeError(f"Zout shape mismatch: {tuple(zout_i.shape)} != {tuple(zout_t.shape)}")
        if int(zout_i.shape[1]) != self.d_in:
            raise RuntimeError(f"Hash head input dim mismatch: expected {self.d_in}, got {zout_i.shape[1]}")

        v_i = torch.tanh((zout_i @ self.w_hash) + self.b_hash)
        v_t = torch.tanh((zout_t @ self.w_hash) + self.b_hash)
        b_i = self._derive_binary_target(v_i)
        b_t = self._derive_binary_target(v_t)
        return TorchHashHeadOutput(v_i=v_i, v_t=v_t, b_i=b_i, b_t=b_t)


class _TorchTrainingMainlineBase(nn.Module):
    def __init__(self, *, d_in: int, d_model: int, order_k: int, d_hash: int, seed: int) -> None:
        super().__init__()
        self.encoder_i = ShadowChebKANEncoder(d_in=d_in, d_model=d_model, order_k=order_k, seed=seed)
        self.encoder_t = ShadowChebKANEncoder(d_in=d_in, d_model=d_model, order_k=order_k, seed=seed + 101)
        self.hash_head = ShadowSharedHashHead(d_in=d_model, d_hash=d_hash, seed=seed + 202)

    def forward(self, x_i: torch.Tensor, x_t: torch.Tensor) -> TorchMainlineOutput:
        zout_i = self.encoder_i(x_i)
        zout_t = self.encoder_t(x_t)
        hash_out = self.hash_head(zout_i, zout_t)
        return TorchMainlineOutput(
            zout_i=zout_i,
            zout_t=zout_t,
            v_i=hash_out.v_i,
            v_t=hash_out.v_t,
            b_i=hash_out.b_i,
            b_t=hash_out.b_t,
        )


class TrainerSmokeShadowMainline(_TorchTrainingMainlineBase):
    """Trainer-smoke-only differentiable shadow path. It must not replace the frozen numpy mainline."""


class SchKanhTrainingMainline(_TorchTrainingMainlineBase):
    """Formal training mainline v1 torch path for S_final-only supervision."""
