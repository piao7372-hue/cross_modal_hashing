from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .config import RuntimeConfig, SemanticConfig, TopKConfig
from .sparse_ops import (
    EPS,
    SupportSelection,
    add_self_loop_and_row_normalize,
    build_candidate_union,
    compute_pair_cosines_on_support,
    compute_relation_norms,
    cosine_topk_sparse,
    csr_from_support,
    row_l2_norms,
    row_softmax_on_support,
    select_topk_support_by_values,
    stable_sigmoid,
    symmetrize_arithmetic_mean,
    symmetric_normalize,
)


@dataclass(frozen=True)
class GraphBuildResult:
    matrices: dict[str, sparse.csr_matrix]
    stats: dict[str, float | int | str]



def _build_relation_cosines_s2(
    x_i: np.ndarray,
    x_t: np.ndarray,
    support: sparse.csr_matrix,
    runtime: RuntimeConfig,
    alpha1: float,
    alpha2: float,
    alpha3: float,
) -> np.ndarray:
    g_i = (x_i.T @ x_i).astype(np.float32)
    g_t = (x_t.T @ x_t).astype(np.float32)
    c_ti = (x_t.T @ x_i).astype(np.float32)
    c_it = c_ti.T.astype(np.float32)

    rel_norm_i = compute_relation_norms(x_i, g_i, runtime.block_rows, eps=EPS)
    rel_norm_t = compute_relation_norms(x_t, g_t, runtime.block_rows, eps=EPS)

    indptr = support.indptr
    indices = support.indices
    out = np.empty(indices.shape[0], dtype=np.float32)

    # cos(A, B) is explicitly fixed as row-wise relation-vector cosine in global sample_index space:
    # [cos(A,B)]_ij = cosine(A[i,:], B[j,:]).
    for i in range(support.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        js = indices[start:end]

        xi = x_i[i]
        xt = x_t[i]

        v_ii = xi @ g_i
        v_tt = xt @ g_t
        v_ti = xt @ c_ti
        v_it = xi @ c_it

        num_ii = x_i[js] @ v_ii
        num_tt = x_t[js] @ v_tt
        num_ti = x_i[js] @ v_ti
        num_it = x_t[js] @ v_it

        c_ii = num_ii / np.maximum(rel_norm_i[i] * rel_norm_i[js], EPS)
        c_tt = num_tt / np.maximum(rel_norm_t[i] * rel_norm_t[js], EPS)
        c_ti_vals = num_ti / np.maximum(rel_norm_t[i] * rel_norm_i[js], EPS)
        c_it_vals = num_it / np.maximum(rel_norm_i[i] * rel_norm_t[js], EPS)

        out[start:end] = (
            alpha3 * (alpha1 * c_ii + (1.0 - alpha1) * c_tt)
            + (1.0 - alpha3) * (alpha2 * c_ti_vals + (1.0 - alpha2) * c_it_vals)
        ).astype(np.float32)

    return out



def build_graph_matrices(
    x_i: np.ndarray,
    x_t: np.ndarray,
    topk_cfg: TopKConfig,
    semantic_cfg: SemanticConfig,
    runtime_cfg: RuntimeConfig,
) -> GraphBuildResult:
    n = int(x_i.shape[0])
    if int(x_t.shape[0]) != n:
        raise RuntimeError("X_I and X_T row count mismatch")

    if semantic_cfg.softmax_domain == "full_row":
        if not runtime_cfg.dense_debug_mode or n > runtime_cfg.dense_debug_max_rows:
            raise RuntimeError(
                "softmax_domain=full_row is only allowed in dense debug mode with small N"
            )
        raise RuntimeError("full_row mode is reserved for debug only and is not enabled in this build")

    p = semantic_cfg.params
    if p.tau <= 0:
        raise RuntimeError("tau must be > 0")

    xi = np.asarray(x_i, dtype=np.float32)
    xt = np.asarray(x_t, dtype=np.float32)

    norm_xi = row_l2_norms(xi)
    norm_xt = row_l2_norms(xt)

    s_i_topk = cosine_topk_sparse(xi, k=topk_cfg.k_candidate, block_rows=runtime_cfg.block_rows)
    s_t_topk = cosine_topk_sparse(xt, k=topk_cfg.k_candidate, block_rows=runtime_cfg.block_rows)

    support_candidate = build_candidate_union(s_i_topk, s_t_topk)

    s_i_vals = compute_pair_cosines_on_support(xi, xi, support_candidate, norm_xi, norm_xi)
    s_t_vals = compute_pair_cosines_on_support(xt, xt, support_candidate, norm_xt, norm_xt)

    s1_vals = (p.alpha1 * s_i_vals + (1.0 - p.alpha1) * s_t_vals).astype(np.float32)
    s2_vals = _build_relation_cosines_s2(
        x_i=xi,
        x_t=xt,
        support=support_candidate,
        runtime=runtime_cfg,
        alpha1=p.alpha1,
        alpha2=p.alpha2,
        alpha3=p.alpha3,
    )

    gate = stable_sigmoid(s2_vals, clip_value=runtime_cfg.exp_clip_value)
    s_fused_vals = np.tanh(s1_vals + s2_vals * gate).astype(np.float32)

    selected: SupportSelection = select_topk_support_by_values(
        support=support_candidate,
        values=s_fused_vals,
        k_final=topk_cfg.k_final,
    )
    support_final = selected.support
    pos = selected.selected_positions

    s_i_final = s_i_vals[pos]
    s_t_final = s_t_vals[pos]
    s1_final = s1_vals[pos]
    s2_final = s2_vals[pos]
    s_fused_final = s_fused_vals[pos]

    s_i_mat = csr_from_support(support_final, s_i_final)
    s_t_mat = csr_from_support(support_final, s_t_final)
    s1_mat = csr_from_support(support_final, s1_final)
    s2_mat = csr_from_support(support_final, s2_final)
    s_fused_mat = csr_from_support(support_final, s_fused_final)

    softmax_vals = row_softmax_on_support(
        support=support_final,
        values=s_fused_final,
        tau=p.tau,
        clip_value=runtime_cfg.exp_clip_value,
    )
    s_high_vals = add_self_loop_and_row_normalize(
        support=support_final,
        values=softmax_vals,
        lambda_self_loop=p.lambda_self_loop,
    )
    s_high_mat = csr_from_support(support_final, s_high_vals)

    # exp is stabilized by clipping to avoid overflow on extreme values.
    a_vals = np.exp(np.clip(s_fused_final / p.tau, -runtime_cfg.exp_clip_value, runtime_cfg.exp_clip_value)).astype(
        np.float32
    )
    a_dir = csr_from_support(support_final, a_vals)

    a_sym = symmetrize_arithmetic_mean(a_dir)
    a_tilde = (a_sym + p.lambda_self_loop * sparse.eye(n, format="csr", dtype=np.float32)).tocsr()
    s_graph_mat = symmetric_normalize(a_tilde)

    out: dict[str, sparse.csr_matrix] = {
        "S_I": s_i_mat,
        "S_T": s_t_mat,
        "S1": s1_mat,
        "S2": s2_mat,
        "S_fused": s_fused_mat,
        "S_high": s_high_mat,
        # Keep S_graph available for compatibility paths; not part of current default formal outputs.
        "S_graph": s_graph_mat,
    }

    stats: dict[str, float | int | str] = {
        "rows": n,
        "dim": int(x_i.shape[1]),
        "candidate_nnz": int(support_candidate.nnz),
        "final_nnz": int(support_final.nnz),
        "s_graph_nnz": int(s_graph_mat.nnz),
        "softmax_domain": semantic_cfg.softmax_domain,
        "symmetrization_rule": semantic_cfg.symmetrization_rule,
    }
    return GraphBuildResult(matrices=out, stats=stats)
