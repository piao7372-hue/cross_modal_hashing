from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from feature_extraction import load_feature_extraction_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate feature cache contract.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["nuswide", "mirflickr25k", "mscoco"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--feature-set-id",
        required=True,
        help="Feature set id under data/processed/<dataset>/feature_cache/",
    )
    parser.add_argument(
        "--config",
        default="configs/feature_extraction.yaml",
        help="Path to feature extraction config.",
    )
    parser.add_argument(
        "--l2-sample-size",
        type=int,
        default=2048,
        help="How many rows to sample for L2 norm checks per matrix.",
    )
    parser.add_argument(
        "--l2-tolerance",
        type=float,
        default=1.0e-3,
        help="Absolute tolerance for ||x||2 close to 1.",
    )
    return parser.parse_args()


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _validate_sample_index(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            expected = count
            actual = obj.get("sample_index")
            if actual != expected:
                raise RuntimeError(
                    f"sample_index mismatch at row {count}: expected {expected}, got {actual}"
                )
            count += 1
    return count


def _sample_l2_residuals(matrix: np.ndarray, sample_size: int) -> tuple[float, float]:
    rows = matrix.shape[0]
    if rows == 0:
        return 0.0, 0.0
    if sample_size >= rows:
        indices = np.arange(rows, dtype=np.int64)
    else:
        step = rows / sample_size
        indices = np.asarray([int(math.floor(i * step)) for i in range(sample_size)], dtype=np.int64)
    selected = np.asarray(matrix[indices], dtype=np.float32)
    norms = np.linalg.norm(selected, ord=2, axis=1)
    residuals = np.abs(norms - 1.0)
    return float(np.max(residuals)), float(np.mean(residuals))


def main() -> int:
    args = parse_args()
    cfg = load_feature_extraction_config(PROJECT_ROOT, (PROJECT_ROOT / args.config).resolve())
    output_dir = cfg.resolve_output_dir(dataset=args.dataset, feature_set_id=args.feature_set_id)
    if not output_dir.exists():
        raise FileNotFoundError(f"Feature cache directory not found: {output_dir}")

    meta_path = output_dir / cfg.cache.meta_filename
    sample_index_path = output_dir / cfg.cache.sample_index_filename
    x_i_path = output_dir / "X_I.npy"
    x_t_path = output_dir / "X_T.npy"
    row_to_image_idx_path = output_dir / "row_to_image_idx.npy"

    for required in [meta_path, sample_index_path, x_i_path, x_t_path, row_to_image_idx_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required output file: {required}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("contract_version") != "feature_cache_v1":
        raise RuntimeError("Unexpected contract_version")
    if meta.get("sample_index_contract", {}).get("row_index_used_as_position") is not False:
        raise RuntimeError("row_index_used_as_position must be false")

    rows_from_sample_index = _validate_sample_index(sample_index_path)
    rows_by_count = _count_jsonl_rows(sample_index_path)
    if rows_from_sample_index != rows_by_count:
        raise RuntimeError("Sample index row count mismatch")

    x_i = np.load(x_i_path, mmap_mode="r")
    x_t = np.load(x_t_path, mmap_mode="r")
    row_to_image_idx = np.load(row_to_image_idx_path, mmap_mode="r")

    if x_i.ndim != 2 or x_t.ndim != 2:
        raise RuntimeError("X_I/X_T must be 2D matrices")
    if x_i.shape != x_t.shape:
        raise RuntimeError(f"X_I shape {x_i.shape} != X_T shape {x_t.shape}")
    if x_i.shape[0] != rows_from_sample_index:
        raise RuntimeError(
            f"Row count mismatch: sample_index={rows_from_sample_index}, X_I={x_i.shape[0]}"
        )
    if row_to_image_idx.shape != (rows_from_sample_index,):
        raise RuntimeError("row_to_image_idx shape mismatch")
    if np.min(row_to_image_idx) < 0:
        raise RuntimeError("row_to_image_idx must be non-negative")

    x_i_unique_path = output_dir / "X_I_unique.npy"
    x_i_unique = np.load(x_i_unique_path, mmap_mode="r") if x_i_unique_path.exists() else None
    if x_i_unique is not None:
        max_idx = int(np.max(row_to_image_idx))
        if max_idx >= x_i_unique.shape[0]:
            raise RuntimeError("row_to_image_idx out of range for X_I_unique")

    max_res_i, mean_res_i = _sample_l2_residuals(x_i, args.l2_sample_size)
    max_res_t, mean_res_t = _sample_l2_residuals(x_t, args.l2_sample_size)
    if max_res_i > args.l2_tolerance:
        raise RuntimeError(f"X_I L2 residual too large: {max_res_i}")
    if max_res_t > args.l2_tolerance:
        raise RuntimeError(f"X_T L2 residual too large: {max_res_t}")

    result: dict[str, Any] = {
        "dataset": args.dataset,
        "feature_set_id": args.feature_set_id,
        "output_dir": output_dir.as_posix(),
        "rows": int(rows_from_sample_index),
        "dim": int(x_i.shape[1]),
        "l2_residual_max": {"X_I": max_res_i, "X_T": max_res_t},
        "l2_residual_mean": {"X_I": mean_res_i, "X_T": mean_res_t},
        "has_unique_image_cache": bool(x_i_unique is not None),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
