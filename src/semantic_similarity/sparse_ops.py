from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

EPS = 1.0e-12


@dataclass(frozen=True)
class SupportSelection:
    support: sparse.csr_matrix
    selected_positions: np.ndarray



def row_l2_norms(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    norms = np.linalg.norm(np.asarray(x, dtype=np.float32), ord=2, axis=1)
    return np.maximum(norms, eps).astype(np.float32)



def cosine_topk_sparse(x: np.ndarray, k: int, block_rows: int, eps: float = EPS) -> sparse.csr_matrix:
    n = int(x.shape[0])
    if k <= 0:
        raise ValueError("k must be > 0")
    k_eff = min(k, n)
    x_arr = np.asarray(x, dtype=np.float32)
    norms = row_l2_norms(x_arr, eps=eps)

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    vals_list: list[np.ndarray] = []

    for start in range(0, n, block_rows):
        end = min(start + block_rows, n)
        xb = x_arr[start:end]
        sims = xb @ x_arr.T
        denom = norms[start:end, None] * norms[None, :]
        sims = sims / np.maximum(denom, eps)

        if k_eff == n:
            top_idx = np.tile(np.arange(n, dtype=np.int32), (end - start, 1))
            top_vals = sims.astype(np.float32)
        else:
            part = np.argpartition(sims, kth=n - k_eff, axis=1)[:, -k_eff:]
            part_vals = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(part_vals, axis=1)[:, ::-1]
            top_idx = np.take_along_axis(part, order, axis=1).astype(np.int32)
            top_vals = np.take_along_axis(part_vals, order, axis=1).astype(np.float32)

        row_idx = np.arange(start, end, dtype=np.int32)[:, None]
        rows_list.append(np.repeat(row_idx, k_eff, axis=1).reshape(-1))
        cols_list.append(top_idx.reshape(-1))
        vals_list.append(top_vals.reshape(-1))

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    m = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    m.sum_duplicates()
    m.sort_indices()
    return m



def build_candidate_union(s_i_topk: sparse.csr_matrix, s_t_topk: sparse.csr_matrix) -> sparse.csr_matrix:
    n = int(s_i_topk.shape[0])
    c_i = s_i_topk.copy()
    c_t = s_t_topk.copy()
    c_i.data = np.ones_like(c_i.data, dtype=np.float32)
    c_t.data = np.ones_like(c_t.data, dtype=np.float32)
    cand = (c_i + c_t).tocsr()
    cand.data = np.ones_like(cand.data, dtype=np.float32)
    cand = (cand + sparse.eye(n, format="csr", dtype=np.float32)).tocsr()
    cand.sum_duplicates()
    cand.data = np.ones_like(cand.data, dtype=np.float32)
    cand.sort_indices()
    cand.eliminate_zeros()
    return cand



def compute_pair_cosines_on_support(
    x_left: np.ndarray,
    x_right: np.ndarray,
    support: sparse.csr_matrix,
    left_norms: np.ndarray,
    right_norms: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    indptr = support.indptr
    indices = support.indices
    values = np.empty(indices.shape[0], dtype=np.float32)

    x_l = np.asarray(x_left, dtype=np.float32)
    x_r = np.asarray(x_right, dtype=np.float32)

    for i in range(support.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        js = indices[start:end]
        vec_i = x_l[i]
        num = x_r[js] @ vec_i
        den = np.maximum(left_norms[i] * right_norms[js], eps)
        values[start:end] = num / den
    return values



def compute_relation_norms(x: np.ndarray, gram: np.ndarray, block_rows: int, eps: float = EPS) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    n = int(x_arr.shape[0])
    out = np.empty(n, dtype=np.float32)
    g = np.asarray(gram, dtype=np.float32)
    for start in range(0, n, block_rows):
        end = min(start + block_rows, n)
        xb = x_arr[start:end]
        proj = xb @ g
        vals = np.sum(proj * xb, axis=1)
        out[start:end] = np.sqrt(np.maximum(vals, eps)).astype(np.float32)
    return out



def stable_sigmoid(x: np.ndarray, clip_value: float) -> np.ndarray:
    z = np.clip(x, -clip_value, clip_value)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)



def select_topk_support_by_values(
    support: sparse.csr_matrix,
    values: np.ndarray,
    k_final: int,
) -> SupportSelection:
    n = int(support.shape[0])
    indptr = support.indptr
    indices = support.indices

    out_rows: list[np.ndarray] = []
    out_cols: list[np.ndarray] = []
    out_pos: list[np.ndarray] = []

    for i in range(n):
        start, end = indptr[i], indptr[i + 1]
        row_cols = indices[start:end]
        row_vals = values[start:end]
        m = row_cols.shape[0]
        keep = min(k_final, m)

        if keep == m:
            selected_local = np.arange(m, dtype=np.int32)
        else:
            selected_local = np.argpartition(row_vals, kth=m - keep)[-keep:].astype(np.int32)

        self_candidates = np.where(row_cols == i)[0]
        if self_candidates.size == 0:
            raise RuntimeError(f"Row {i} has no self-loop in candidate support")
        self_pos = int(self_candidates[0])

        selected_set = set(int(x) for x in selected_local.tolist())
        selected_set.add(self_pos)
        if len(selected_set) > keep:
            if keep == 1:
                selected = np.asarray([self_pos], dtype=np.int32)
            else:
                others = [p for p in selected_set if p != self_pos]
                others = sorted(others, key=lambda p: float(row_vals[p]), reverse=True)
                chosen = [self_pos] + others[: keep - 1]
                selected = np.asarray(chosen, dtype=np.int32)
        else:
            selected = np.asarray(sorted(selected_set), dtype=np.int32)

        global_pos = start + selected
        cols = row_cols[selected]

        out_rows.append(np.full(cols.shape[0], i, dtype=np.int32))
        out_cols.append(cols.astype(np.int32))
        out_pos.append(global_pos.astype(np.int64))

    rows = np.concatenate(out_rows)
    cols = np.concatenate(out_cols)
    pos = np.concatenate(out_pos)

    out_support = sparse.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.float32), (rows, cols)),
        shape=support.shape,
        dtype=np.float32,
    )
    out_support.sum_duplicates()
    out_support.sort_indices()
    return SupportSelection(support=out_support, selected_positions=pos)



def row_softmax_on_support(
    support: sparse.csr_matrix,
    values: np.ndarray,
    tau: float,
    clip_value: float,
    eps: float = EPS,
) -> np.ndarray:
    if tau <= 0:
        raise ValueError("tau must be > 0")
    indptr = support.indptr
    out = np.empty(values.shape[0], dtype=np.float32)
    for i in range(support.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        raw = values[start:end] / tau
        raw = np.clip(raw, -clip_value, clip_value)
        shifted = raw - float(np.max(raw))
        expv = np.exp(shifted)
        denom = np.maximum(float(np.sum(expv)), eps)
        out[start:end] = (expv / denom).astype(np.float32)
    return out



def add_self_loop_and_row_normalize(
    support: sparse.csr_matrix,
    values: np.ndarray,
    lambda_self_loop: float,
    eps: float = EPS,
) -> np.ndarray:
    indptr = support.indptr
    indices = support.indices
    out = values.astype(np.float32, copy=True)
    for i in range(support.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        row_cols = indices[start:end]
        row_vals = out[start:end]
        self_candidates = np.where(row_cols == i)[0]
        if self_candidates.size == 0:
            raise RuntimeError(f"Row {i} missing self-loop before lambda injection")
        row_vals[self_candidates[0]] += np.float32(lambda_self_loop)
        denom = np.maximum(float(np.sum(row_vals)), eps)
        out[start:end] = (row_vals / denom).astype(np.float32)
    return out



def csr_from_support(support: sparse.csr_matrix, values: np.ndarray) -> sparse.csr_matrix:
    m = sparse.csr_matrix((values.astype(np.float32), support.indices, support.indptr), shape=support.shape)
    m.sort_indices()
    m.eliminate_zeros()
    return m



def symmetrize_arithmetic_mean(m: sparse.csr_matrix) -> sparse.csr_matrix:
    sym = ((m + m.T) * 0.5).tocsr()
    sym.sum_duplicates()
    sym.sort_indices()
    return sym



def symmetric_normalize(m: sparse.csr_matrix, eps: float = EPS) -> sparse.csr_matrix:
    deg = np.asarray(m.sum(axis=1)).reshape(-1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, eps))
    d_inv = sparse.diags(inv_sqrt.astype(np.float32), format="csr")
    out = (d_inv @ m @ d_inv).tocsr()
    out.sum_duplicates()
    out.sort_indices()
    out.eliminate_zeros()
    return out
