from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .semantic_supervision_reader import SemanticSupervision


def _normalize_sample_indices(sample_indices: np.ndarray | list[int], rows: int) -> np.ndarray:
    idx = np.asarray(sample_indices)
    if idx.ndim != 1:
        raise RuntimeError("batch sample_index must be 1D")
    if idx.size <= 0:
        raise RuntimeError("batch sample_index must be non-empty")
    if not np.issubdtype(idx.dtype, np.integer):
        raise RuntimeError("batch sample_index must be integer typed")
    idx64 = idx.astype(np.int64, copy=False)
    if np.any(idx64 < 0) or np.any(idx64 >= rows):
        raise RuntimeError(f"batch sample_index out of range for rows={rows}")
    if np.unique(idx64).size != idx64.size:
        raise RuntimeError("batch sample_index must not contain duplicates")
    return idx64


def _slice_rows(arr: np.ndarray, sample_indices: np.ndarray, *, name: str, expected_rows: int) -> np.ndarray:
    value = np.asarray(arr)
    if value.ndim != 2:
        raise RuntimeError(f"{name} must be 2D")
    if int(value.shape[0]) != expected_rows:
        raise RuntimeError(f"{name} rows mismatch: expected {expected_rows}, got {value.shape[0]}")
    return value[sample_indices]


@dataclass(frozen=True)
class LossBatchInputs:
    sample_indices: np.ndarray
    zout_i: np.ndarray
    zout_t: np.ndarray
    v_i: np.ndarray
    v_t: np.ndarray
    b_i: np.ndarray
    b_t: np.ndarray
    s_final_batch: sparse.csr_matrix


def build_loss_batch_inputs(
    *,
    sample_indices: np.ndarray | list[int],
    zout_i: np.ndarray,
    zout_t: np.ndarray,
    v_i: np.ndarray,
    v_t: np.ndarray,
    b_i: np.ndarray,
    b_t: np.ndarray,
    supervision: SemanticSupervision,
) -> LossBatchInputs:
    idx = _normalize_sample_indices(sample_indices, rows=supervision.rows)

    batch_zout_i = _slice_rows(zout_i, idx, name="Zout_I", expected_rows=supervision.rows)
    batch_zout_t = _slice_rows(zout_t, idx, name="Zout_T", expected_rows=supervision.rows)
    batch_v_i = _slice_rows(v_i, idx, name="V_I", expected_rows=supervision.rows)
    batch_v_t = _slice_rows(v_t, idx, name="V_T", expected_rows=supervision.rows)
    batch_b_i = _slice_rows(b_i, idx, name="B_I", expected_rows=supervision.rows)
    batch_b_t = _slice_rows(b_t, idx, name="B_T", expected_rows=supervision.rows)
    s_final_batch = supervision.s_final.slice_batch(idx)

    batch_rows = int(idx.size)
    if s_final_batch.shape != (batch_rows, batch_rows):
        raise RuntimeError(f"S_final batch shape mismatch: got {s_final_batch.shape}, expected {(batch_rows, batch_rows)}")

    return LossBatchInputs(
        sample_indices=idx,
        zout_i=batch_zout_i,
        zout_t=batch_zout_t,
        v_i=batch_v_i,
        v_t=batch_v_t,
        b_i=batch_b_i,
        b_t=batch_b_t,
        s_final_batch=s_final_batch,
    )
