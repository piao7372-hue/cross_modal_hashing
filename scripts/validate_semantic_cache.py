from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semantic_similarity import load_semantic_similarity_config  # noqa: E402

S2_MIN_STD = 1.0e-8
S2_MIN_MAX_ABS = 1.0e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate semantic cache contract and math properties.")
    parser.add_argument("--dataset", required=True, choices=["nuswide", "mirflickr25k", "mscoco"])
    parser.add_argument("--semantic-set-id", required=True)
    parser.add_argument("--config", default="configs/semantic_similarity.yaml")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sample_index_file(path: Path) -> tuple[int, str]:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            actual = obj.get("sample_index")
            if actual != count:
                raise RuntimeError(
                    f"sample_index mismatch at row {count} in {path}: expected {count}, got {actual}"
                )
            count += 1
    return count, _sha256_file(path)


def _row_sum_stats(m: sparse.csr_matrix) -> dict[str, float]:
    row_sum = np.asarray(m.sum(axis=1)).reshape(-1)
    return {
        "min": float(np.min(row_sum)),
        "max": float(np.max(row_sum)),
        "mean": float(np.mean(row_sum)),
        "std": float(np.std(row_sum)),
    }


def _density(m: sparse.csr_matrix) -> float:
    n = m.shape[0]
    if n == 0:
        return 0.0
    return float(m.nnz) / float(n * n)


def _avg_degree(m: sparse.csr_matrix) -> float:
    n = m.shape[0]
    if n == 0:
        return 0.0
    return float(m.nnz) / float(n)


def _symmetry_error(m: sparse.csr_matrix) -> float:
    diff = (m - m.T).tocsr()
    if diff.nnz == 0:
        return 0.0
    return float(np.max(np.abs(diff.data)))


def _max_abs_sparse(m: sparse.csr_matrix) -> float:
    if m.nnz == 0:
        return 0.0
    return float(np.max(np.abs(m.data)))


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, int):
        return value
    raise RuntimeError(f"{field_name} must be int")


def _require_str(value: Any, field_name: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise RuntimeError(f"{field_name} must be non-empty string")


def _validate_s2_nondegenerate(s2: sparse.csr_matrix, expected_n: int) -> dict[str, float | int]:
    if s2.shape != (expected_n, expected_n):
        raise RuntimeError(f"S2 shape mismatch: expected ({expected_n}, {expected_n}), got {s2.shape}")
    if s2.nnz <= 0:
        raise RuntimeError("S2 is empty (nnz=0), considered degenerate")

    data = np.asarray(s2.data, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        raise RuntimeError("S2 contains non-finite values")

    std_val = float(np.std(data))
    max_abs_val = float(np.max(np.abs(data)))
    min_val = float(np.min(data))
    max_val = float(np.max(data))

    if max_abs_val <= S2_MIN_MAX_ABS:
        raise RuntimeError(
            f"S2 max abs is too small ({max_abs_val}); expected > {S2_MIN_MAX_ABS} for non-degeneracy"
        )
    if std_val <= S2_MIN_STD:
        raise RuntimeError(
            f"S2 std is too small ({std_val}); expected > {S2_MIN_STD} for non-degeneracy"
        )
    if max_val <= min_val:
        raise RuntimeError("S2 value range is collapsed (max <= min), considered degenerate")

    return {
        "nnz": int(s2.nnz),
        "std": std_val,
        "max_abs": max_abs_val,
        "min": min_val,
        "max": max_val,
    }


def _validate_two_stage_topk_edge_counts(
    *,
    meta: dict[str, Any],
    n: int,
    s_high: sparse.csr_matrix,
) -> dict[str, int]:
    k_candidate = _require_int(meta.get("k_candidate"), "meta.k_candidate")
    k_final = _require_int(meta.get("k_final"), "meta.k_final")
    if k_candidate < k_final:
        raise RuntimeError(f"k_candidate ({k_candidate}) must be >= k_final ({k_final})")
    if k_final <= 0:
        raise RuntimeError("k_final must be > 0")

    stats_obj = meta.get("stats")
    if not isinstance(stats_obj, dict):
        raise RuntimeError("meta.stats must be a mapping")
    candidate_nnz = _require_int(stats_obj.get("candidate_nnz"), "meta.stats.candidate_nnz")
    final_nnz_meta = _require_int(stats_obj.get("final_nnz"), "meta.stats.final_nnz")
    final_nnz_actual = int(s_high.nnz)

    if final_nnz_meta != final_nnz_actual:
        raise RuntimeError(
            f"final_nnz mismatch: meta.stats.final_nnz={final_nnz_meta}, S_high.nnz={final_nnz_actual}"
        )

    k_eff_candidate = min(k_candidate, n)
    k_eff_final = min(k_final, n)

    candidate_nnz_min = n * k_eff_candidate
    candidate_nnz_max = n * min(n, (2 * k_eff_candidate) + 1)
    if candidate_nnz < candidate_nnz_min or candidate_nnz > candidate_nnz_max:
        raise RuntimeError(
            "candidate_nnz out of theoretical bounds for stage-1 union support: "
            f"{candidate_nnz} not in [{candidate_nnz_min}, {candidate_nnz_max}]"
        )

    final_nnz_min = n
    final_nnz_max = n * k_eff_final
    if final_nnz_actual < final_nnz_min or final_nnz_actual > final_nnz_max:
        raise RuntimeError(
            "final_nnz out of theoretical bounds for stage-2 top-k support: "
            f"{final_nnz_actual} not in [{final_nnz_min}, {final_nnz_max}]"
        )

    if candidate_nnz < final_nnz_actual:
        raise RuntimeError(
            f"candidate_nnz ({candidate_nnz}) must be >= final_nnz ({final_nnz_actual})"
        )

    row_nnz = np.diff(s_high.indptr)
    if row_nnz.shape[0] != n:
        raise RuntimeError("S_high row nnz shape mismatch")
    if np.any(row_nnz <= 0):
        raise RuntimeError("S_high has rows without edges after stage-2 top-k")
    if np.any(row_nnz > k_eff_final):
        max_row_nnz = int(np.max(row_nnz))
        raise RuntimeError(
            f"S_high row nnz exceeds k_final bound: max_row_nnz={max_row_nnz}, k_eff_final={k_eff_final}"
        )

    diag_high = s_high.diagonal()
    if diag_high.shape[0] != n:
        raise RuntimeError("S_high diagonal length mismatch")
    if np.any(diag_high <= 0):
        raise RuntimeError("S_high diagonal must be strictly positive (self-loop required)")

    return {
        "k_candidate": int(k_candidate),
        "k_final": int(k_final),
        "candidate_nnz": int(candidate_nnz),
        "final_nnz_meta": int(final_nnz_meta),
        "final_nnz_actual": int(final_nnz_actual),
        "candidate_nnz_min": int(candidate_nnz_min),
        "candidate_nnz_max": int(candidate_nnz_max),
        "final_nnz_min": int(final_nnz_min),
        "final_nnz_max": int(final_nnz_max),
        "row_nnz_min": int(np.min(row_nnz)),
        "row_nnz_max": int(np.max(row_nnz)),
    }


def _validate_formal_pseudo_mainline(
    *,
    meta: dict[str, Any],
    s_high: sparse.csr_matrix,
    s_pseudo: sparse.csr_matrix,
    s_final: sparse.csr_matrix,
) -> dict[str, Any]:
    if meta.get("pseudo_source_mode") != "spectral_clustering":
        raise RuntimeError("formal pseudo mainline requires pseudo_source_mode=spectral_clustering")
    if meta.get("z_source") != "fused_projection_from_x_i_x_t":
        raise RuntimeError("formal pseudo mainline requires z_source=fused_projection_from_x_i_x_t")
    if meta.get("projection_mode") != "shared_linear_tanh":
        raise RuntimeError("formal pseudo mainline requires projection_mode=shared_linear_tanh")
    if meta.get("fusion_mode") != "arithmetic_mean":
        raise RuntimeError("formal pseudo mainline requires fusion_mode=arithmetic_mean")
    if meta.get("affinity_builder") != "sparse_knn_cosine":
        raise RuntimeError("formal pseudo mainline requires affinity_builder=sparse_knn_cosine")
    if meta.get("laplacian_type") != "symmetric_normalized":
        raise RuntimeError("formal pseudo mainline requires laplacian_type=symmetric_normalized")
    if meta.get("clustering_method") != "spectral_kmeans":
        raise RuntimeError("formal pseudo mainline requires clustering_method=spectral_kmeans")
    if meta.get("s_pseudo_storage") != "csr_sparse_on_pseudo_support":
        raise RuntimeError("formal pseudo mainline requires s_pseudo_storage=csr_sparse_on_pseudo_support")
    if meta.get("s_final_storage") != "csr_sparse_on_union_support":
        raise RuntimeError("formal pseudo mainline requires s_final_storage=csr_sparse_on_union_support")

    pseudo_dims = meta.get("pseudo_dims")
    if not isinstance(pseudo_dims, dict):
        raise RuntimeError("formal pseudo mainline requires meta.pseudo_dims")
    dim_x_i = _require_int(pseudo_dims.get("x_i"), "meta.pseudo_dims.x_i")
    dim_x_t = _require_int(pseudo_dims.get("x_t"), "meta.pseudo_dims.x_t")
    dim_z = _require_int(pseudo_dims.get("z"), "meta.pseudo_dims.z")
    if dim_x_i != dim_x_t:
        raise RuntimeError("shared_linear_tanh mainline requires meta.pseudo_dims.x_i == meta.pseudo_dims.x_t")
    if dim_z <= 0:
        raise RuntimeError("meta.pseudo_dims.z must be > 0")

    affinity_k = _require_int(meta.get("affinity_k"), "meta.affinity_k")
    n_clusters = _require_int(meta.get("n_clusters"), "meta.n_clusters")
    pseudo_seed = _require_int(meta.get("pseudo_seed"), "meta.pseudo_seed")
    if n_clusters <= 1 or n_clusters >= s_high.shape[0]:
        raise RuntimeError("formal pseudo mainline requires 1 < n_clusters < rows")
    if meta.get("entrypoints", {}).get("supervision_target") != "S_final":
        raise RuntimeError("formal pseudo mainline must set entrypoints.supervision_target = S_final")

    if s_pseudo.nnz <= 0:
        raise RuntimeError("formal pseudo mainline requires non-empty S_pseudo")
    if not np.all(np.isfinite(s_pseudo.data)):
        raise RuntimeError("S_pseudo contains non-finite values")
    if np.any(np.abs(s_pseudo.data - 1.0) > 1.0e-6):
        raise RuntimeError("formal pseudo mainline requires S_pseudo data values to be 1 on pseudo support")
    diag = s_pseudo.diagonal()
    if diag.shape[0] != s_pseudo.shape[0] or np.any(diag <= 0):
        raise RuntimeError("formal pseudo mainline requires positive self-loop on every S_pseudo row")

    n = s_pseudo.shape[0]
    k_eff = min(affinity_k, n)
    pseudo_nnz_max = n * min(n, (2 * k_eff) + 1)
    if s_pseudo.nnz > pseudo_nnz_max:
        raise RuntimeError(
            f"S_pseudo.nnz exceeds sparse pseudo-support bound: {s_pseudo.nnz} > {pseudo_nnz_max}"
        )
    if s_final.nnz > int(s_high.nnz + s_pseudo.nnz):
        raise RuntimeError(
            f"S_final.nnz exceeds union-support upper bound: {s_final.nnz} > {int(s_high.nnz + s_pseudo.nnz)}"
        )

    pseudo_label_count = _require_int(meta.get("pseudo_label_count"), "meta.pseudo_label_count")
    if pseudo_label_count <= 0 or pseudo_label_count > n_clusters:
        raise RuntimeError("meta.pseudo_label_count must be in [1, n_clusters]")

    return {
        "path_validation": "formal_mainline_pass",
        "affinity_k": int(affinity_k),
        "n_clusters": int(n_clusters),
        "pseudo_seed": int(pseudo_seed),
        "pseudo_nnz_max": int(pseudo_nnz_max),
        "pseudo_label_count": int(pseudo_label_count),
        "dim_x_i": int(dim_x_i),
        "dim_x_t": int(dim_x_t),
        "dim_z": int(dim_z),
    }


def _validate_compat_external_matrix(meta: dict[str, Any]) -> dict[str, Any]:
    if meta.get("pseudo_source_mode") != "external_matrix":
        raise RuntimeError("compat pseudo path requires pseudo_source_mode=external_matrix")
    _require_str(meta.get("external_matrix_path"), "meta.external_matrix_path")
    if meta.get("entrypoints", {}).get("supervision_target") != "S_final":
        raise RuntimeError("compat external_matrix path must set entrypoints.supervision_target = S_final")
    return {
        "path_validation": "compat_external_matrix_pass",
    }


def main() -> int:
    args = parse_args()
    cfg = load_semantic_similarity_config(PROJECT_ROOT, (PROJECT_ROOT / args.config).resolve())

    semantic_dir = cfg.resolve_semantic_cache_dir(dataset=args.dataset, semantic_set_id=args.semantic_set_id)
    if not semantic_dir.exists():
        raise FileNotFoundError(f"Semantic cache directory not found: {semantic_dir}")

    meta_path = semantic_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    pipeline_mode = meta.get("pipeline_mode")
    if pipeline_mode not in {"high_only", "with_pseudo"}:
        raise RuntimeError("meta.pipeline_mode must be `high_only` or `with_pseudo`")

    lineage = meta.get("lineage", {})
    if not isinstance(lineage, dict):
        raise RuntimeError("meta.lineage must be a mapping")

    feature_cache_dir_raw = lineage.get("feature_cache_dir")
    if not isinstance(feature_cache_dir_raw, str) or not feature_cache_dir_raw:
        raise RuntimeError("meta.lineage.feature_cache_dir is required")
    feature_cache_dir = Path(feature_cache_dir_raw)

    sample_index_path = feature_cache_dir / "sample_index.jsonl"
    if not sample_index_path.exists():
        raise FileNotFoundError(f"Upstream sample_index missing: {sample_index_path}")

    upstream_rows, upstream_sample_index_hash = _validate_sample_index_file(sample_index_path)

    if meta.get("sample_index_hash") != upstream_sample_index_hash:
        raise RuntimeError("sample_index_hash mismatch between semantic cache and feature cache")
    if lineage.get("sample_index_hash") != upstream_sample_index_hash:
        raise RuntimeError("lineage.sample_index_hash mismatch")

    formal_required = ["S2.npz", "S_high.npz"]
    for name in formal_required:
        if not (semantic_dir / name).exists():
            raise FileNotFoundError(f"Missing required matrix file: {semantic_dir / name}")

    optional_intermediate = ["S_I.npz", "S_T.npz", "S1.npz", "S_fused.npz"]
    for name in optional_intermediate:
        path = semantic_dir / name
        if not path.exists():
            continue
        mat = sparse.load_npz(path).tocsr().astype(np.float32)
        if mat.shape[0] != mat.shape[1]:
            raise RuntimeError(f"{name} must be square")

    s_high = sparse.load_npz(semantic_dir / "S_high.npz").tocsr().astype(np.float32)
    s2 = sparse.load_npz(semantic_dir / "S2.npz").tocsr().astype(np.float32)
    s_graph_path = semantic_dir / "S_graph.npz"
    has_s_graph = s_graph_path.exists()
    s_graph = sparse.load_npz(s_graph_path).tocsr().astype(np.float32) if has_s_graph else None

    n = int(s_high.shape[0])
    if s_high.shape[0] != s_high.shape[1]:
        raise RuntimeError("S_high must be square")
    if has_s_graph and s_graph is not None:
        if s_graph.shape[0] != s_graph.shape[1]:
            raise RuntimeError("S_graph must be square")
        if s_graph.shape != s_high.shape:
            raise RuntimeError(f"S_graph shape {s_graph.shape} != S_high shape {s_high.shape}")
    if n != upstream_rows:
        raise RuntimeError(f"N mismatch: semantic={n}, upstream feature sample_index rows={upstream_rows}")

    s2_stats = _validate_s2_nondegenerate(s2=s2, expected_n=n)
    topk_edge_stats = _validate_two_stage_topk_edge_counts(meta=meta, n=n, s_high=s_high)

    if n > cfg.runtime.dense_debug_max_rows:
        matrix_format = meta.get("output", {}).get("matrix_format")
        if matrix_format != "csr_npz":
            raise RuntimeError("Large dataset must use csr_npz output format")

    row_sums_high = np.asarray(s_high.sum(axis=1)).reshape(-1)
    max_row_sum_res = float(np.max(np.abs(row_sums_high - 1.0)))
    if max_row_sum_res > cfg.validation.row_sum_atol:
        raise RuntimeError(
            f"S_high row-sum residual too large: {max_row_sum_res} > {cfg.validation.row_sum_atol}"
        )

    sym_err = 0.0
    if has_s_graph and s_graph is not None:
        sym_err = _symmetry_error(s_graph)
        if sym_err > cfg.validation.symmetry_atol:
            raise RuntimeError(f"S_graph symmetry error too large: {sym_err} > {cfg.validation.symmetry_atol}")

        min_graph_val = float(np.min(s_graph.data)) if s_graph.nnz > 0 else 0.0
        if min_graph_val < -cfg.validation.symmetry_atol:
            raise RuntimeError(f"S_graph has negative values below tolerance: min={min_graph_val}")

        diag = s_graph.diagonal()
        if diag.shape[0] != n:
            raise RuntimeError("S_graph diagonal length mismatch")
        if np.any(diag <= 0):
            raise RuntimeError("S_graph diagonal must be strictly positive for all rows (self-loop required)")

    if pipeline_mode == "with_pseudo":
        s_pseudo_path = semantic_dir / "S_pseudo.npz"
        s_final_path = semantic_dir / "S_final.npz"
        if not s_pseudo_path.exists() or not s_final_path.exists():
            raise RuntimeError("with_pseudo mode requires both S_pseudo.npz and S_final.npz")

        s_pseudo = sparse.load_npz(s_pseudo_path).tocsr().astype(np.float32)
        s_final = sparse.load_npz(s_final_path).tocsr().astype(np.float32)
        if s_pseudo.shape != s_high.shape or s_final.shape != s_high.shape:
            raise RuntimeError("S_pseudo/S_final shape mismatch against S_high")

        beta = float(meta.get("params", {}).get("beta"))
        combo = (beta * s_high + (1.0 - beta) * s_pseudo).tocsr()
        diff = (s_final - combo).tocsr()
        formula_err = _max_abs_sparse(diff)
        if formula_err > cfg.validation.formula_atol:
            raise RuntimeError(
                f"S_final formula mismatch: max_abs_err={formula_err} > {cfg.validation.formula_atol}"
            )

        if meta.get("entrypoints", {}).get("supervision_target") != "S_final":
            raise RuntimeError("with_pseudo mode must set entrypoints.supervision_target = S_final")
        pseudo_source_mode = meta.get("pseudo_source_mode")
        if pseudo_source_mode == "spectral_clustering":
            pseudo_path_stats = _validate_formal_pseudo_mainline(
                meta=meta,
                s_high=s_high,
                s_pseudo=s_pseudo,
                s_final=s_final,
            )
        elif pseudo_source_mode == "external_matrix":
            pseudo_path_stats = _validate_compat_external_matrix(meta)
        else:
            raise RuntimeError("with_pseudo mode requires pseudo_source_mode to be spectral_clustering or external_matrix")
    else:
        if meta.get("entrypoints", {}).get("supervision_target") != "unavailable":
            raise RuntimeError("high_only mode must set entrypoints.supervision_target = unavailable")
        pseudo_path_stats = {"path_validation": "high_only_no_supervision_target"}

    propagation_graph_entry = meta.get("entrypoints", {}).get("propagation_graph")
    if has_s_graph:
        if propagation_graph_entry != "S_graph":
            raise RuntimeError("entrypoints.propagation_graph must be S_graph when S_graph.npz exists")
    else:
        if propagation_graph_entry not in {"unavailable", None}:
            raise RuntimeError(
                "entrypoints.propagation_graph must be unavailable when S_graph.npz is not saved"
            )

    result: dict[str, Any] = {
        "dataset": args.dataset,
        "semantic_set_id": args.semantic_set_id,
        "pipeline_mode": pipeline_mode,
        "rows": n,
        "density": {
            "S_high": _density(s_high),
        },
        "avg_degree": {
            "S_high": _avg_degree(s_high),
        },
        "row_sum_stats": {
            "S_high": _row_sum_stats(s_high),
        },
        "s2_stats": s2_stats,
        "topk_edge_stats": topk_edge_stats,
        "checks": {
            "sample_index_hash_match": True,
            "N_match": True,
            "s2_nondegenerate_ok": True,
            "two_stage_topk_edge_counts_ok": True,
            "s_high_row_sum_ok": True,
            "optional_intermediate_shape_ok": True,
            "s_graph_present": has_s_graph,
            "s_graph_symmetry_ok": True if has_s_graph else "skipped",
            "s_graph_nonnegative_ok": True if has_s_graph else "skipped",
            "s_graph_self_loop_ok": True if has_s_graph else "skipped",
            "pseudo_path_validation": pseudo_path_stats["path_validation"],
        },
    }
    if pipeline_mode == "with_pseudo":
        result["density"]["S_pseudo"] = _density(s_pseudo)
        result["density"]["S_final"] = _density(s_final)
        result["avg_degree"]["S_pseudo"] = _avg_degree(s_pseudo)
        result["avg_degree"]["S_final"] = _avg_degree(s_final)
        result["row_sum_stats"]["S_final"] = _row_sum_stats(s_final)
        result["pseudo_path_validation"] = pseudo_path_stats
    if has_s_graph and s_graph is not None:
        result["density"]["S_graph"] = _density(s_graph)
        result["avg_degree"]["S_graph"] = _avg_degree(s_graph)
        result["symmetry_error"] = {"S_graph": sym_err}
        result["row_sum_stats"]["S_graph"] = _row_sum_stats(s_graph)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
