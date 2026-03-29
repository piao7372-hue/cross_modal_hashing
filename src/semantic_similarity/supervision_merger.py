from __future__ import annotations

from scipy import sparse



def merge_supervision(s_high: sparse.csr_matrix, s_pseudo: sparse.csr_matrix, beta: float) -> sparse.csr_matrix:
    if s_high.shape != s_pseudo.shape:
        raise RuntimeError(f"S_high shape {s_high.shape} != S_pseudo shape {s_pseudo.shape}")
    if beta < 0.0 or beta > 1.0:
        raise RuntimeError("beta must be in [0, 1]")
    out = (beta * s_high + (1.0 - beta) * s_pseudo).tocsr().astype("float32")
    out.sum_duplicates()
    out.sort_indices()
    out.eliminate_zeros()
    return out
