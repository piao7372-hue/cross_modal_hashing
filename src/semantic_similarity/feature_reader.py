from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureCacheInputs:
    feature_cache_dir: Path
    feature_set_id: str
    x_i: np.ndarray
    x_t: np.ndarray
    rows: int
    dim: int
    sample_index_path: Path
    sample_index_hash: str
    feature_meta: dict[str, Any]
    lineage: dict[str, Any]



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



def load_feature_cache_inputs(feature_cache_dir: Path) -> FeatureCacheInputs:
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

    x_i = np.load(x_i_path, mmap_mode="r")
    x_t = np.load(x_t_path, mmap_mode="r")
    if x_i.ndim != 2 or x_t.ndim != 2:
        raise RuntimeError("X_I and X_T must be 2D")
    if x_i.shape != x_t.shape:
        raise RuntimeError(f"X_I shape {x_i.shape} != X_T shape {x_t.shape}")

    rows, dim = int(x_i.shape[0]), int(x_i.shape[1])
    sample_index_hash = _validate_and_hash_sample_index(sample_index_path, expected_rows=rows)

    manifest_obj = meta.get("manifest", {}) if isinstance(meta.get("manifest"), dict) else {}
    lineage = {
        "feature_set_id": meta.get("feature_set_id"),
        "feature_contract_version": meta.get("contract_version"),
        "manifest_path": manifest_obj.get("path"),
        "manifest_sha256": manifest_obj.get("sha256"),
        "manifest_rows": manifest_obj.get("rows"),
        "sample_index_basis": meta.get("sample_index_contract", {}).get("basis"),
        "sample_index_hash": sample_index_hash,
        "feature_cache_dir": feature_cache_dir.as_posix(),
    }

    feature_set_id = str(meta.get("feature_set_id") or feature_cache_dir.name)

    return FeatureCacheInputs(
        feature_cache_dir=feature_cache_dir,
        feature_set_id=feature_set_id,
        x_i=x_i,
        x_t=x_t,
        rows=rows,
        dim=dim,
        sample_index_path=sample_index_path,
        sample_index_hash=sample_index_hash,
        feature_meta=meta,
        lineage=lineage,
    )
