from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


def resolve_semantic_cache_dir(*, processed_root: Path, dataset: str, semantic_set_id: str) -> Path:
    return processed_root / dataset / "semantic_cache" / semantic_set_id


def _require_mapping(obj: dict[str, Any], key: str) -> dict[str, Any]:
    value = obj.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"semantic meta `{key}` must be mapping")
    return value


def _require_str(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if isinstance(value, str) and value:
        return value
    raise RuntimeError(f"semantic meta `{key}` must be non-empty string")


def _require_int(obj: dict[str, Any], key: str) -> int:
    value = obj.get(key)
    if isinstance(value, int):
        return value
    raise RuntimeError(f"semantic meta `{key}` must be int")


def _extract_rows(meta: dict[str, Any]) -> int:
    value = meta.get("rows")
    if isinstance(value, int):
        return value
    stats = meta.get("stats")
    if isinstance(stats, dict):
        nested = stats.get("rows")
        if isinstance(nested, int):
            return nested
    raise RuntimeError("semantic meta must provide rows at meta.rows or meta.stats.rows")


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


@dataclass(frozen=True)
class SemanticSupervisionMatrix:
    matrix_name: str
    source_path: Path
    rows: int
    _csr: sparse.csr_matrix

    @property
    def shape(self) -> tuple[int, int]:
        return self._csr.shape

    @property
    def nnz(self) -> int:
        return int(self._csr.nnz)

    @property
    def dtype(self) -> np.dtype:
        return self._csr.dtype

    def toarray(self) -> np.ndarray:
        raise RuntimeError("Global densify of S_final is prohibited; slice batch supervision from CSR instead")

    def todense(self) -> np.ndarray:
        raise RuntimeError("Global densify of S_final is prohibited; slice batch supervision from CSR instead")

    def slice_batch(self, sample_indices: np.ndarray | list[int]) -> sparse.csr_matrix:
        idx = _normalize_sample_indices(sample_indices, rows=self.rows)
        batch = self._csr[idx][:, idx]
        if not isinstance(batch, sparse.csr_matrix):
            batch = batch.tocsr()
        if batch.dtype != np.float32:
            raise RuntimeError(f"S_final batch dtype must stay float32, got {batch.dtype}")
        batch.sum_duplicates()
        batch.sort_indices()
        batch.eliminate_zeros()
        return batch


@dataclass(frozen=True)
class SemanticSupervision:
    semantic_cache_dir: Path
    dataset: str
    feature_set_id: str
    semantic_set_id: str
    pipeline_mode: str
    sample_index_hash: str
    rows: int
    meta: dict[str, Any]
    s_final: SemanticSupervisionMatrix


def load_semantic_supervision(
    semantic_cache_dir: Path,
    *,
    expected_sample_index_hash: str | None = None,
    expected_rows: int | None = None,
    expected_feature_set_id: str | None = None,
) -> SemanticSupervision:
    if not semantic_cache_dir.exists():
        raise FileNotFoundError(f"Semantic cache directory not found: {semantic_cache_dir}")

    meta_path = semantic_cache_dir / "meta.json"
    s_final_path = semantic_cache_dir / "S_final.npz"
    for required in [meta_path, s_final_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required semantic supervision file: {required}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    pipeline_mode = _require_str(meta, "pipeline_mode")
    if pipeline_mode != "with_pseudo":
        raise RuntimeError("semantic supervision requires meta.pipeline_mode = `with_pseudo`")

    entrypoints = _require_mapping(meta, "entrypoints")
    if entrypoints.get("supervision_target") != "S_final":
        raise RuntimeError("semantic supervision requires entrypoints.supervision_target = `S_final`")

    role_map = meta.get("role_map")
    if isinstance(role_map, dict) and "S_final" in role_map and role_map.get("S_final") != "supervision_target":
        raise RuntimeError("semantic role_map.S_final must equal `supervision_target` when role_map.S_final exists")

    dataset = _require_str(meta, "dataset")
    feature_set_id = _require_str(meta, "feature_set_id")
    semantic_set_id = _require_str(meta, "semantic_set_id")
    sample_index_hash = _require_str(meta, "sample_index_hash")
    rows = _extract_rows(meta)

    matrix = sparse.load_npz(s_final_path)
    if not isinstance(matrix, sparse.csr_matrix):
        raise RuntimeError(f"S_final must be stored as CSR sparse matrix, got {type(matrix).__name__}")
    if matrix.dtype != np.float32:
        raise RuntimeError(f"S_final dtype must be float32, got {matrix.dtype}")
    if matrix.shape != (rows, rows):
        raise RuntimeError(f"S_final shape {matrix.shape} != ({rows}, {rows}) from meta.rows")
    if not np.all(np.isfinite(matrix.data)):
        raise RuntimeError("S_final contains non-finite values")

    if expected_sample_index_hash is not None and sample_index_hash != expected_sample_index_hash:
        raise RuntimeError(
            "semantic sample_index_hash mismatch: "
            f"expected {expected_sample_index_hash}, got {sample_index_hash}"
        )
    if expected_rows is not None and rows != expected_rows:
        raise RuntimeError(f"semantic rows mismatch: expected {expected_rows}, got {rows}")
    if expected_feature_set_id is not None and feature_set_id != expected_feature_set_id:
        raise RuntimeError(
            f"semantic feature_set_id mismatch: expected {expected_feature_set_id}, got {feature_set_id}"
        )

    return SemanticSupervision(
        semantic_cache_dir=semantic_cache_dir,
        dataset=dataset,
        feature_set_id=feature_set_id,
        semantic_set_id=semantic_set_id,
        pipeline_mode=pipeline_mode,
        sample_index_hash=sample_index_hash,
        rows=rows,
        meta=meta,
        s_final=SemanticSupervisionMatrix(
            matrix_name="S_final",
            source_path=s_final_path,
            rows=rows,
            _csr=matrix,
        ),
    )
