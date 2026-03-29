from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .config import CacheConfig


@dataclass(frozen=True)
class CachePaths:
    output_dir: Path
    x_i_path: Path
    x_t_path: Path
    x_i_unique_path: Path
    row_to_image_idx_path: Path
    sample_index_path: Path
    image_index_path: Path
    meta_path: Path


def build_cache_paths(output_dir: Path, cache_cfg: CacheConfig) -> CachePaths:
    return CachePaths(
        output_dir=output_dir,
        x_i_path=output_dir / "X_I.npy",
        x_t_path=output_dir / "X_T.npy",
        x_i_unique_path=output_dir / "X_I_unique.npy",
        row_to_image_idx_path=output_dir / "row_to_image_idx.npy",
        sample_index_path=output_dir / cache_cfg.sample_index_filename,
        image_index_path=output_dir / "image_index.jsonl",
        meta_path=output_dir / cache_cfg.meta_filename,
    )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Feature cache output already exists: {output_dir}. Use --overwrite to replace."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def create_feature_memmap(path: Path, rows: int, dim: int) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, dim))


def create_index_memmap(path: Path, rows: int) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(rows,))


def write_jsonl(fp: TextIO, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_meta(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
