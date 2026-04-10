from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy import sparse
from torch import nn


def _as_float32_matrix(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim != 2:
        raise RuntimeError(f"{name} must be 2D")
    return value.to(dtype=torch.float32)


def pairwise_cosine_rows(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    left = _as_float32_matrix(a, name="a")
    right = _as_float32_matrix(b, name="b")
    if int(left.shape[1]) != int(right.shape[1]):
        raise RuntimeError(f"Cosine dim mismatch: {left.shape[1]} != {right.shape[1]}")

    left_unit = F.normalize(left, p=2.0, dim=1, eps=eps)
    right_unit = F.normalize(right, p=2.0, dim=1, eps=eps)
    return left_unit @ right_unit.transpose(0, 1)


def loss_sem(v_i: torch.Tensor, v_t: torch.Tensor, s_final_batch: sparse.csr_matrix) -> torch.Tensor:
    left = _as_float32_matrix(v_i, name="V_I")
    right = _as_float32_matrix(v_t, name="V_T")
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError(f"V_I shape {tuple(left.shape)} != V_T shape {tuple(right.shape)}")
    if not isinstance(s_final_batch, sparse.csr_matrix):
        raise RuntimeError("torch loss_sem requires batch supervision as CSR sparse matrix")
    if s_final_batch.shape != (int(left.shape[0]), int(left.shape[0])):
        raise RuntimeError(
            f"S_final batch shape {s_final_batch.shape} does not match batch rows {int(left.shape[0])}"
        )

    target = torch.from_numpy(s_final_batch.toarray()).to(device=left.device, dtype=torch.float32)
    pred = pairwise_cosine_rows(left, right)
    return torch.mean((pred - target) ** 2)


def loss_q(v_i: torch.Tensor, v_t: torch.Tensor, b_i: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
    left = _as_float32_matrix(v_i, name="V_I")
    right = _as_float32_matrix(v_t, name="V_T")
    code_i = _as_float32_matrix(b_i, name="B_I")
    code_t = _as_float32_matrix(b_t, name="B_T")
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError(f"V_I shape {tuple(left.shape)} != V_T shape {tuple(right.shape)}")
    if tuple(code_i.shape) != tuple(left.shape) or tuple(code_t.shape) != tuple(right.shape):
        raise RuntimeError("B_I/B_T shapes must match V_I/V_T")
    return 0.5 * (((left - code_i) ** 2).mean() + ((right - code_t) ** 2).mean())


def loss_bal(v_i: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
    left = _as_float32_matrix(v_i, name="V_I")
    right = _as_float32_matrix(v_t, name="V_T")
    if int(left.shape[1]) != int(right.shape[1]):
        raise RuntimeError(f"V bit dim mismatch: {left.shape[1]} != {right.shape[1]}")
    codes = torch.cat([left, right], dim=0)
    bit_means = torch.mean(codes, dim=0)
    return torch.mean(bit_means ** 2)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        ctx.lambda_grl = float(lambda_grl)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return (-ctx.lambda_grl * grad_output), None


def grad_reverse(x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
    return GradientReversalFunction.apply(x, float(lambda_grl))


class GrlDomainHead(nn.Module):
    def __init__(self, *, d_in: int, lambda_grl: float, seed: int) -> None:
        super().__init__()
        if d_in <= 0:
            raise ValueError("d_in must be > 0")
        self.d_in = int(d_in)
        self.lambda_grl = float(lambda_grl)
        self.linear = nn.Linear(self.d_in, 1)
        with torch.no_grad():
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            scale = 1.0 / max(self.d_in, 1) ** 0.5
            init = torch.randn((1, self.d_in), generator=gen, dtype=torch.float32) * scale
            self.linear.weight.copy_(init)
            self.linear.bias.zero_()

    def forward_logits(self, zout: torch.Tensor) -> torch.Tensor:
        x = _as_float32_matrix(zout, name="Zout")
        return self.linear(grad_reverse(x, self.lambda_grl)).squeeze(1)


def loss_grl(zout_i: torch.Tensor, zout_t: torch.Tensor, domain_head: GrlDomainHead) -> torch.Tensor:
    left = _as_float32_matrix(zout_i, name="Zout_I")
    right = _as_float32_matrix(zout_t, name="Zout_T")
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError(f"Zout_I shape {tuple(left.shape)} != Zout_T shape {tuple(right.shape)}")

    logits_i = domain_head.forward_logits(left)
    logits_t = domain_head.forward_logits(right)
    labels_i = torch.zeros_like(logits_i, dtype=torch.float32)
    labels_t = torch.ones_like(logits_t, dtype=torch.float32)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(logits_i, labels_i)
        + F.binary_cross_entropy_with_logits(logits_t, labels_t)
    )


@dataclass(frozen=True)
class WeightedLossTerms:
    loss_sem: torch.Tensor
    loss_q: torch.Tensor
    loss_bal: torch.Tensor
    loss_grl: torch.Tensor
    total_loss: torch.Tensor
