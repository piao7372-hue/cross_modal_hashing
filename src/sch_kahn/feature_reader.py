from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureInputs:
    feature_cache_dir: Path
    feature_set_id: str
    x_i: np.ndarray
    x_t: np.ndarray
    rows: int
    dim: int
    sample_index_hash: str
    feature_meta: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_and_hash_sample_index(path: Path, expected_rows: int) -> str:
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
    if count != expected_rows:
        raise RuntimeError(f"sample_index rows mismatch: expected {expected_rows}, got {count}")
    return _sha256_file(path)


def _extract_meta_sample_index_hash(meta: dict[str, Any]) -> str | None:
    value = meta.get("sample_index_hash")
    if isinstance(value, str) and value:
        return value
    lineage = meta.get("lineage")
    if isinstance(lineage, dict):
        v2 = lineage.get("sample_index_hash")
        if isinstance(v2, str) and v2:
            return v2
    return None


def load_feature_cache_inputs(feature_cache_dir: Path) -> FeatureInputs:
    if not feature_cache_dir.exists():
        raise FileNotFoundError(f"Feature cache directory not found: {feature_cache_dir}")

    meta_path = feature_cache_dir / "meta.json"
    x_i_path = feature_cache_dir / "X_I.npy"
    x_t_path = feature_cache_dir / "X_T.npy"
    sample_index_path = feature_cache_dir / "sample_index.jsonl"

    for required in [meta_path, x_i_path, x_t_path, sample_index_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required feature-cache file: {required}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    x_i = np.load(x_i_path).astype(np.float32, copy=False)
    x_t = np.load(x_t_path).astype(np.float32, copy=False)
    if x_i.ndim != 2 or x_t.ndim != 2:
        raise RuntimeError("X_I and X_T must be 2D")
    if x_i.shape != x_t.shape:
        raise RuntimeError(f"X_I shape {x_i.shape} != X_T shape {x_t.shape}")

    rows, dim = int(x_i.shape[0]), int(x_i.shape[1])

    # Rule: prefer hash from feature meta; recompute from sample_index.jsonl when needed.
    hash_from_meta = _extract_meta_sample_index_hash(meta)
    hash_recomputed = _validate_and_hash_sample_index(sample_index_path, expected_rows=rows)
    if hash_from_meta is None:
        sample_index_hash = hash_recomputed
    else:
        if hash_from_meta != hash_recomputed:
            raise RuntimeError(
                "feature meta sample_index_hash mismatch against recomputed sample_index hash"
            )
        sample_index_hash = hash_from_meta

    feature_set_id = str(meta.get("feature_set_id") or feature_cache_dir.name)

    return FeatureInputs(
        feature_cache_dir=feature_cache_dir,
        feature_set_id=feature_set_id,
        x_i=x_i,
        x_t=x_t,
        rows=rows,
        dim=dim,
        sample_index_hash=sample_index_hash,
        feature_meta=meta,
    )
