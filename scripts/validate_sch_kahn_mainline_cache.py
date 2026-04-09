from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sch_kahn import load_sch_kahn_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SCH-KANH mainline forward-only v2 cache (Zout->V->B).")
    parser.add_argument("--dataset", required=True, choices=["nuswide", "mirflickr25k", "mscoco"])
    parser.add_argument("--sch-set-id", required=True)
    parser.add_argument("--config", default="configs/sch_kahn_mainline.yaml")
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


def _require_str(meta: dict[str, Any], key: str) -> str:
    v = meta.get(key)
    if isinstance(v, str) and v:
        return v
    raise RuntimeError(f"meta.{key} must be non-empty string")


def _require_int(meta: dict[str, Any], key: str) -> int:
    v = meta.get(key)
    if isinstance(v, int):
        return v
    raise RuntimeError(f"meta.{key} must be int")


def _optional_str(meta: dict[str, Any], key: str) -> str | None:
    v = meta.get(key)
    if v is None:
        return None
    if isinstance(v, str) and v:
        return v
    raise RuntimeError(f"meta.{key} must be non-empty string when provided")


def main() -> int:
    args = parse_args()
    cfg = load_sch_kahn_config(PROJECT_ROOT, (PROJECT_ROOT / args.config).resolve())

    out_dir = cfg.resolve_sch_kahn_cache_dir(dataset=args.dataset, sch_set_id=args.sch_set_id)
    if not out_dir.exists():
        raise FileNotFoundError(f"SCH-KANH cache directory not found: {out_dir}")

    z_i_path = out_dir / "Zout_I.npy"
    z_t_path = out_dir / "Zout_T.npy"
    v_i_path = out_dir / "V_I.npy"
    v_t_path = out_dir / "V_T.npy"
    b_i_path = out_dir / "B_I.npy"
    b_t_path = out_dir / "B_T.npy"
    meta_path = out_dir / "meta.json"
    for required in [z_i_path, z_t_path, v_i_path, v_t_path, b_i_path, b_t_path, meta_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    z_i = np.load(z_i_path, mmap_mode="r")
    z_t = np.load(z_t_path, mmap_mode="r")
    v_i = np.load(v_i_path, mmap_mode="r")
    v_t = np.load(v_t_path, mmap_mode="r")
    b_i = np.load(b_i_path, mmap_mode="r")
    b_t = np.load(b_t_path, mmap_mode="r")

    if z_i.ndim != 2 or z_t.ndim != 2:
        raise RuntimeError("Zout_I and Zout_T must be 2D")
    if z_i.shape != z_t.shape:
        raise RuntimeError(f"Zout shape mismatch: {z_i.shape} != {z_t.shape}")
    if z_i.dtype != np.float32 or z_t.dtype != np.float32:
        raise RuntimeError(f"Zout dtype must be float32, got {z_i.dtype} and {z_t.dtype}")
    if not np.isfinite(z_i).all() or not np.isfinite(z_t).all():
        raise RuntimeError("Zout contains non-finite values")

    if v_i.ndim != 2 or v_t.ndim != 2:
        raise RuntimeError("V_I and V_T must be 2D")
    if v_i.shape != v_t.shape:
        raise RuntimeError(f"V shape mismatch: {v_i.shape} != {v_t.shape}")
    if int(v_i.shape[0]) != int(z_i.shape[0]):
        raise RuntimeError(f"V rows mismatch against Zout rows: {v_i.shape[0]} != {z_i.shape[0]}")
    if v_i.dtype != np.float32 or v_t.dtype != np.float32:
        raise RuntimeError(f"V dtype must be float32, got {v_i.dtype} and {v_t.dtype}")
    if not np.isfinite(v_i).all() or not np.isfinite(v_t).all():
        raise RuntimeError("V contains non-finite values")

    if b_i.ndim != 2 or b_t.ndim != 2:
        raise RuntimeError("B_I and B_T must be 2D")
    if b_i.shape != b_t.shape:
        raise RuntimeError(f"B shape mismatch: {b_i.shape} != {b_t.shape}")
    if b_i.shape != v_i.shape:
        raise RuntimeError(f"B shape must equal V shape: {b_i.shape} != {v_i.shape}")
    if b_i.dtype != np.int8 or b_t.dtype != np.int8:
        raise RuntimeError(f"B dtype must be int8, got {b_i.dtype} and {b_t.dtype}")
    if not np.isin(b_i, np.array([-1, 1], dtype=np.int8)).all():
        raise RuntimeError("B_I values must be in {-1, +1}")
    if not np.isin(b_t, np.array([-1, 1], dtype=np.int8)).all():
        raise RuntimeError("B_T values must be in {-1, +1}")
    expected_b_i = np.where(v_i >= 0.0, 1, -1).astype(np.int8, copy=False)
    expected_b_t = np.where(v_t >= 0.0, 1, -1).astype(np.int8, copy=False)
    if not np.array_equal(b_i, expected_b_i):
        raise RuntimeError("B_I does not match binarize rule sign_ge_zero_to_pos1")
    if not np.array_equal(b_t, expected_b_t):
        raise RuntimeError("B_T does not match binarize rule sign_ge_zero_to_pos1")

    # Required minimal fields.
    input_source = _require_str(meta, "input_source")
    sample_index_hash = _require_str(meta, "sample_index_hash")
    feature_set_id = _require_str(meta, "feature_set_id")
    sch_set_id = _require_str(meta, "sch_set_id")
    stop_at = _require_str(meta, "stop_at")
    rows = _require_int(meta, "rows")
    dtype = _require_str(meta, "dtype")
    device = _optional_str(meta, "device")
    seed = _require_int(meta, "seed")

    if sch_set_id != args.sch_set_id:
        raise RuntimeError(f"meta.sch_set_id mismatch: expected {args.sch_set_id}, got {sch_set_id}")
    if rows != int(z_i.shape[0]):
        raise RuntimeError(f"meta.rows mismatch: expected {z_i.shape[0]}, got {rows}")

    dim_obj = meta.get("dim")
    if not isinstance(dim_obj, dict):
        raise RuntimeError("meta.dim must be mapping with keys `in`, `zout`, `v`, and `b`")
    dim_in = dim_obj.get("in")
    dim_zout = dim_obj.get("zout")
    dim_v = dim_obj.get("v")
    dim_b = dim_obj.get("b")
    if not isinstance(dim_in, int) or not isinstance(dim_zout, int) or not isinstance(dim_v, int) or not isinstance(dim_b, int):
        raise RuntimeError("meta.dim.in/meta.dim.zout/meta.dim.v/meta.dim.b must be int")
    if dim_zout != int(z_i.shape[1]):
        raise RuntimeError(f"meta.dim.zout mismatch: expected {z_i.shape[1]}, got {dim_zout}")
    if dim_v != int(v_i.shape[1]):
        raise RuntimeError(f"meta.dim.v mismatch: expected {v_i.shape[1]}, got {dim_v}")
    if dim_b != int(b_i.shape[1]):
        raise RuntimeError(f"meta.dim.b mismatch: expected {b_i.shape[1]}, got {dim_b}")

    # Boundary checks explicitly required in this phase.
    if input_source != "feature_cache_x_i_x_t":
        raise RuntimeError("meta.input_source must be `feature_cache_x_i_x_t`")
    graph_side = meta.get("graph_side", {})
    if not isinstance(graph_side, dict):
        raise RuntimeError("meta.graph_side must be mapping")
    if graph_side.get("mode") != "disabled":
        raise RuntimeError("meta.graph_side.mode must be `disabled`")
    if stop_at != "b":
        raise RuntimeError("meta.stop_at must be `b`")

    hash_head = meta.get("hash_head")
    if not isinstance(hash_head, dict):
        raise RuntimeError("meta.hash_head must be mapping")
    if hash_head.get("enabled") is not True:
        raise RuntimeError("meta.hash_head.enabled must be true")
    if hash_head.get("shared_params") is not True:
        raise RuntimeError("meta.hash_head.shared_params must be true")
    if hash_head.get("binarize_rule") != "sign_ge_zero_to_pos1":
        raise RuntimeError("meta.hash_head.binarize_rule must be sign_ge_zero_to_pos1")
    d_hash = hash_head.get("d_hash")
    if not isinstance(d_hash, int):
        raise RuntimeError("meta.hash_head.d_hash must be int")
    if d_hash != dim_v or d_hash != dim_b:
        raise RuntimeError("meta.hash_head.d_hash must equal dim.v and dim.b")

    semantic_inputs_used = meta.get("semantic_inputs_used")
    if not isinstance(semantic_inputs_used, list):
        raise RuntimeError("meta.semantic_inputs_used must be list")
    banned = {"S2", "S_high", "S_graph"}
    used_banned = [x for x in semantic_inputs_used if isinstance(x, str) and x in banned]
    if used_banned:
        raise RuntimeError(f"forward-only default must not consume semantic matrices: found {used_banned}")

    lineage = meta.get("lineage")
    if not isinstance(lineage, dict):
        raise RuntimeError("meta.lineage must be mapping")
    feature_cache_dir = lineage.get("feature_cache_dir")
    if not isinstance(feature_cache_dir, str) or not feature_cache_dir:
        raise RuntimeError("meta.lineage.feature_cache_dir is required")
    if lineage.get("feature_set_id") != feature_set_id:
        raise RuntimeError("meta.lineage.feature_set_id mismatch")
    if lineage.get("sample_index_hash") != sample_index_hash:
        raise RuntimeError("meta.lineage.sample_index_hash mismatch")

    sample_index_path = Path(feature_cache_dir) / "sample_index.jsonl"
    if not sample_index_path.exists():
        raise FileNotFoundError(f"Upstream sample_index missing: {sample_index_path}")
    upstream_rows, upstream_hash = _validate_sample_index_file(sample_index_path)
    if upstream_rows != rows:
        raise RuntimeError(f"rows mismatch against upstream sample_index: {rows} != {upstream_rows}")
    if upstream_hash != sample_index_hash:
        raise RuntimeError("sample_index_hash mismatch against upstream sample_index")

    result = {
        "dataset": args.dataset,
        "sch_set_id": args.sch_set_id,
        "feature_set_id": feature_set_id,
        "rows": rows,
        "dim": {"in": dim_in, "zout": dim_zout, "v": dim_v, "b": dim_b},
        "dtype": dtype,
        "seed": seed,
        "checks": {
            "files_exist": True,
            "shape_ok": True,
            "dtype_ok": True,
            "finite_ok": True,
            "b_value_domain_ok": True,
            "b_sign_rule_ok": True,
            "input_source_ok": True,
            "graph_side_disabled_ok": True,
            "stop_at_ok": True,
            "hash_head_meta_ok": True,
            "semantic_inputs_default_unused_ok": True,
            "sample_index_hash_match_ok": True,
        },
    }
    if device is not None:
        result["device"] = device
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
