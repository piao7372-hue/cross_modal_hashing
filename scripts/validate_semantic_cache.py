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

    required_always = ["S_high.npz", "S_graph.npz"]
    for name in required_always:
        if not (semantic_dir / name).exists():
            raise FileNotFoundError(f"Missing required matrix file: {semantic_dir / name}")

    s_high = sparse.load_npz(semantic_dir / "S_high.npz").tocsr().astype(np.float32)
    s_graph = sparse.load_npz(semantic_dir / "S_graph.npz").tocsr().astype(np.float32)

    n = int(s_high.shape[0])
    if s_high.shape[0] != s_high.shape[1]:
        raise RuntimeError("S_high must be square")
    if s_graph.shape[0] != s_graph.shape[1]:
        raise RuntimeError("S_graph must be square")
    if s_graph.shape != s_high.shape:
        raise RuntimeError(f"S_graph shape {s_graph.shape} != S_high shape {s_high.shape}")
    if n != upstream_rows:
        raise RuntimeError(f"N mismatch: semantic={n}, upstream feature sample_index rows={upstream_rows}")

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
    else:
        if meta.get("entrypoints", {}).get("supervision_target") != "unavailable":
            raise RuntimeError("high_only mode must set entrypoints.supervision_target = unavailable")

    if meta.get("entrypoints", {}).get("propagation_graph") != "S_graph":
        raise RuntimeError("entrypoints.propagation_graph must be S_graph")

    result: dict[str, Any] = {
        "dataset": args.dataset,
        "semantic_set_id": args.semantic_set_id,
        "pipeline_mode": pipeline_mode,
        "rows": n,
        "density": {
            "S_high": _density(s_high),
            "S_graph": _density(s_graph),
        },
        "avg_degree": {
            "S_high": _avg_degree(s_high),
            "S_graph": _avg_degree(s_graph),
        },
        "symmetry_error": {
            "S_graph": sym_err,
        },
        "row_sum_stats": {
            "S_high": _row_sum_stats(s_high),
            "S_graph": _row_sum_stats(s_graph),
        },
        "checks": {
            "sample_index_hash_match": True,
            "N_match": True,
            "s_high_row_sum_ok": True,
            "s_graph_symmetry_ok": True,
            "s_graph_nonnegative_ok": True,
            "s_graph_self_loop_ok": True,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
