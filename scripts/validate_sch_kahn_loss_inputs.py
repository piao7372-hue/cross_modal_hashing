from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sch_kahn.loss_input_builder import build_loss_batch_inputs  # noqa: E402
from sch_kahn.losses import loss_bal, loss_grl, loss_q, loss_sem  # noqa: E402
from sch_kahn.semantic_supervision_reader import (  # noqa: E402
    load_semantic_supervision,
    resolve_semantic_cache_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SCH-KANH loss-side semantic supervision smoke path.")
    parser.add_argument("--config", default="configs/sch_kahn_loss_semantic.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--feature-set-id", default=None)
    parser.add_argument("--sch-set-id", default=None)
    parser.add_argument("--semantic-set-id", default=None)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML root object in {path}")
    return data


def _require_mapping(obj: dict[str, Any], key: str) -> dict[str, Any]:
    value = obj.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"`{key}` must be mapping")
    return value


def _require_str(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if isinstance(value, str) and value:
        return value
    raise RuntimeError(f"`{key}` must be non-empty string")


def _load_sch_cache(
    *,
    processed_root: Path,
    dataset: str,
    sch_set_id: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    sch_dir = processed_root / dataset / "sch_kahn_cache" / sch_set_id
    if not sch_dir.exists():
        raise FileNotFoundError(f"SCH-KANH cache directory not found: {sch_dir}")

    meta_path = sch_dir / "meta.json"
    required_arrays = {
        "zout_i": sch_dir / "Zout_I.npy",
        "zout_t": sch_dir / "Zout_T.npy",
        "v_i": sch_dir / "V_I.npy",
        "v_t": sch_dir / "V_T.npy",
        "b_i": sch_dir / "B_I.npy",
        "b_t": sch_dir / "B_T.npy",
    }
    for required in [meta_path, *required_arrays.values()]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required SCH-KANH loss input file: {required}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    arrays = {key: np.load(path) for key, path in required_arrays.items()}
    rows = int(meta.get("rows"))
    for key, arr in arrays.items():
        if arr.ndim != 2:
            raise RuntimeError(f"{key} must be 2D")
        if int(arr.shape[0]) != rows:
            raise RuntimeError(f"{key} rows mismatch: expected {rows}, got {arr.shape[0]}")

    return meta, arrays


def _assert_signature_exact(fn: Any, *, expected_names: list[str]) -> bool:
    actual = [param.name for param in inspect.signature(fn).parameters.values()]
    return actual == expected_names


def _build_bad_entrypoint_fixture(*, semantic_dir: Path) -> Path:
    tmp_dir = PROJECT_ROOT / "outputs" / "tmp_bad_semantic_entrypoint_for_loss_validation"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    with (semantic_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["entrypoints"]["supervision_target"] = "S_high"

    shutil.copy2(semantic_dir / "S_final.npz", tmp_dir / "S_final.npz")
    with (tmp_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return tmp_dir


def main() -> int:
    args = parse_args()
    cfg = _load_yaml((PROJECT_ROOT / args.config).resolve())

    processed_root = PROJECT_ROOT / _require_str(_require_mapping(cfg, "output"), "processed_root")
    smoke = _require_mapping(cfg, "smoke")

    dataset = args.dataset or _require_str(smoke, "dataset")
    feature_set_id = args.feature_set_id or _require_str(smoke, "feature_set_id")
    sch_set_id = args.sch_set_id or _require_str(smoke, "sch_set_id")
    semantic_set_id = args.semantic_set_id or _require_str(smoke, "semantic_set_id")
    sample_indices = smoke.get("sample_indices")
    if not isinstance(sample_indices, list) or not sample_indices:
        raise RuntimeError("smoke.sample_indices must be non-empty list")

    sch_meta, arrays = _load_sch_cache(processed_root=processed_root, dataset=dataset, sch_set_id=sch_set_id)
    semantic_dir = resolve_semantic_cache_dir(
        processed_root=processed_root,
        dataset=dataset,
        semantic_set_id=semantic_set_id,
    )
    supervision = load_semantic_supervision(
        semantic_dir,
        expected_sample_index_hash=_require_str(sch_meta, "sample_index_hash"),
        expected_rows=int(sch_meta.get("rows")),
        expected_feature_set_id=feature_set_id,
    )
    batch = build_loss_batch_inputs(
        sample_indices=sample_indices,
        zout_i=arrays["zout_i"],
        zout_t=arrays["zout_t"],
        v_i=arrays["v_i"],
        v_t=arrays["v_t"],
        b_i=arrays["b_i"],
        b_t=arrays["b_t"],
        supervision=supervision,
    )

    sem_value = loss_sem(batch.v_i, batch.v_t, batch.s_final_batch)
    q_value = loss_q(batch.v_i, batch.v_t, batch.b_i, batch.b_t)
    bal_value = loss_bal(batch.v_i, batch.v_t)
    grl_value = loss_grl(batch.zout_i, batch.zout_t)

    zero_target = batch.s_final_batch.copy()
    zero_target.data = np.zeros_like(zero_target.data, dtype=np.float32)
    zero_target.eliminate_zeros()
    sem_value_zero = loss_sem(batch.v_i, batch.v_t, zero_target)

    global_dense_blocked = False
    try:
        supervision.s_final.toarray()
    except RuntimeError:
        global_dense_blocked = True

    bad_entrypoint_dir = _build_bad_entrypoint_fixture(semantic_dir=semantic_dir)
    s_high_fallback_blocked = False
    try:
        load_semantic_supervision(
            bad_entrypoint_dir,
            expected_sample_index_hash=supervision.sample_index_hash,
            expected_rows=supervision.rows,
            expected_feature_set_id=supervision.feature_set_id,
        )
    except RuntimeError:
        s_high_fallback_blocked = True
    finally:
        if bad_entrypoint_dir.exists():
            shutil.rmtree(bad_entrypoint_dir)

    checks = {
        "semantic_read_ok": True,
        "batch_slice_ok": batch.s_final_batch.shape == (len(sample_indices), len(sample_indices)),
        "s_final_batch_is_csr": batch.s_final_batch.format == "csr",
        "s_final_batch_dtype_ok": str(batch.s_final_batch.dtype) == "float32",
        "loss_sem_signature_ok": _assert_signature_exact(
            loss_sem,
            expected_names=["v_i", "v_t", "s_final_batch"],
        ),
        "loss_sem_depends_on_s_final_batch": abs(sem_value - sem_value_zero) > 1.0e-8,
        "loss_q_signature_ok": _assert_signature_exact(
            loss_q,
            expected_names=["v_i", "v_t", "b_i", "b_t"],
        ),
        "loss_bal_signature_ok": _assert_signature_exact(
            loss_bal,
            expected_names=["v_i", "v_t"],
        ),
        "loss_grl_signature_ok": _assert_signature_exact(
            loss_grl,
            expected_names=["zout_i", "zout_t", "domain_head_weight", "domain_head_bias"],
        ),
        "loss_grl_zout_only": _assert_signature_exact(
            loss_grl,
            expected_names=["zout_i", "zout_t", "domain_head_weight", "domain_head_bias"],
        ),
        "s_high_fallback_blocked": s_high_fallback_blocked,
        "global_dense_blocked": global_dense_blocked,
    }
    for key, passed in checks.items():
        if not passed:
            raise RuntimeError(f"Smoke validation failed at check: {key}")

    result = {
        "dataset": dataset,
        "feature_set_id": feature_set_id,
        "sch_set_id": sch_set_id,
        "semantic_set_id": semantic_set_id,
        "sample_index_hash": supervision.sample_index_hash,
        "batch_rows": int(batch.sample_indices.size),
        "loss_values": {
            "loss_sem": sem_value,
            "loss_q": q_value,
            "loss_bal": bal_value,
            "loss_grl": grl_value,
        },
        "checks": checks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
