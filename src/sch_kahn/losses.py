from __future__ import annotations

import numpy as np
from scipy import sparse


def _as_float32_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"{name} must be 2D")
    return arr


def pairwise_cosine_rows(a: np.ndarray, b: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    left = _as_float32_matrix(a, name="a")
    right = _as_float32_matrix(b, name="b")
    if left.shape[1] != right.shape[1]:
        raise RuntimeError(f"Cosine dim mismatch: {left.shape[1]} != {right.shape[1]}")

    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_unit = left / np.maximum(left_norm, eps)
    right_unit = right / np.maximum(right_norm, eps)
    return (left_unit @ right_unit.T).astype(np.float32, copy=False)


def loss_sem(v_i: np.ndarray, v_t: np.ndarray, s_final_batch: sparse.csr_matrix) -> float:
    left = _as_float32_matrix(v_i, name="V_I")
    right = _as_float32_matrix(v_t, name="V_T")
    if left.shape != right.shape:
        raise RuntimeError(f"V_I shape {left.shape} != V_T shape {right.shape}")
    if not isinstance(s_final_batch, sparse.csr_matrix):
        raise RuntimeError("loss_sem requires batch supervision as CSR sparse matrix")
    if s_final_batch.shape != (left.shape[0], left.shape[0]):
        raise RuntimeError(
            f"S_final batch shape {s_final_batch.shape} does not match batch rows {left.shape[0]}"
        )
    if s_final_batch.dtype != np.float32:
        raise RuntimeError(f"S_final batch dtype must be float32, got {s_final_batch.dtype}")

    # Dense conversion is allowed only after batch slicing. Global S_final densify is blocked upstream.
    target = s_final_batch.toarray().astype(np.float32, copy=False)
    pred = pairwise_cosine_rows(left, right)
    diff = pred - target
    return float(np.mean(diff * diff, dtype=np.float32))


def loss_q(v_i: np.ndarray, v_t: np.ndarray, b_i: np.ndarray, b_t: np.ndarray) -> float:
    left = _as_float32_matrix(v_i, name="V_I")
    right = _as_float32_matrix(v_t, name="V_T")
    code_i = np.asarray(b_i)
    code_t = np.asarray(b_t)
    if left.shape != right.shape:
        raise RuntimeError(f"V_I shape {left.shape} != V_T shape {right.shape}")
    if code_i.shape != left.shape or code_t.shape != right.shape:
        raise RuntimeError("B_I/B_T shapes must match V_I/V_T")
    if not np.issubdtype(code_i.dtype, np.integer) or not np.issubdtype(code_t.dtype, np.integer):
        raise RuntimeError("B_I/B_T must be integer typed")
    if not np.isin(code_i, np.array([-1, 1], dtype=code_i.dtype)).all():
        raise RuntimeError("B_I values must be in {-1, +1}")
    if not np.isin(code_t, np.array([-1, 1], dtype=code_t.dtype)).all():
        raise RuntimeError("B_T values must be in {-1, +1}")

    diff_i = left - code_i.astype(np.float32, copy=False)
    diff_t = right - code_t.astype(np.float32, copy=False)
    return float(0.5 * (np.mean(diff_i * diff_i, dtype=np.float32) + np.mean(diff_t * diff_t, dtype=np.float32)))


def loss_bal(v_i: np.ndarray, v_t: np.ndarray) -> float:
    left = _as_float32_matrix(v_i, name="V_I")
    right = _as_float32_matrix(v_t, name="V_T")
    if left.shape[1] != right.shape[1]:
        raise RuntimeError(f"V bit dim mismatch: {left.shape[1]} != {right.shape[1]}")
    codes = np.concatenate([left, right], axis=0)
    bit_means = np.mean(codes, axis=0, dtype=np.float32)
    return float(np.mean(bit_means * bit_means, dtype=np.float32))


def loss_grl(
    zout_i: np.ndarray,
    zout_t: np.ndarray,
    *,
    domain_head_weight: np.ndarray | None = None,
    domain_head_bias: float = 0.0,
) -> float:
    left = _as_float32_matrix(zout_i, name="Zout_I")
    right = _as_float32_matrix(zout_t, name="Zout_T")
    if left.shape != right.shape:
        raise RuntimeError(f"Zout_I shape {left.shape} != Zout_T shape {right.shape}")

    if domain_head_weight is None:
        weight = np.full((left.shape[1],), 1.0 / np.sqrt(max(left.shape[1], 1)), dtype=np.float32)
    else:
        weight = np.asarray(domain_head_weight, dtype=np.float32)
    if weight.ndim != 1 or weight.shape[0] != left.shape[1]:
        raise RuntimeError(
            f"domain_head_weight must be 1D with dim {left.shape[1]}, got shape {weight.shape}"
        )

    logits_i = (left @ weight) + np.float32(domain_head_bias)
    logits_t = (right @ weight) + np.float32(domain_head_bias)
    labels_i = np.zeros_like(logits_i, dtype=np.float32)
    labels_t = np.ones_like(logits_t, dtype=np.float32)

    def _bce_with_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.maximum(logits, 0.0) - (logits * labels) + np.log1p(np.exp(-np.abs(logits)))

    loss_i = _bce_with_logits(logits_i.astype(np.float32, copy=False), labels_i)
    loss_t = _bce_with_logits(logits_t.astype(np.float32, copy=False), labels_t)
    return float(0.5 * (np.mean(loss_i, dtype=np.float32) + np.mean(loss_t, dtype=np.float32)))
