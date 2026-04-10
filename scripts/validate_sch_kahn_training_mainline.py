from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sch_kahn.semantic_supervision_reader import load_semantic_supervision  # noqa: E402
from sch_kahn.trainer_config import load_training_mainline_config  # noqa: E402
from sch_kahn.trainer_step import (  # noqa: E402
    _load_training_checkpoint,
    build_training_mainline_context,
    run_training_mainline,
)

REQUIRED_METRIC_FIELDS = [
    "epoch",
    "global_step",
    "loss_sem",
    "loss_q",
    "loss_bal",
    "loss_grl",
    "total_loss",
    "lr",
]

REQUIRED_CHECKPOINT_FIELDS = [
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "global_step",
    "feature_set_id",
    "semantic_set_id",
    "sample_index_hash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SCH-KANH formal training mainline v1.")
    parser.add_argument("--config", default="configs/sch_kahn_training_mainline.yaml")
    parser.add_argument(
        "--keep-run-dir",
        action="store_true",
        help="Keep the temporary validation run directory after validation succeeds.",
    )
    return parser.parse_args()


def _build_bad_entrypoint_fixture(*, semantic_dir: Path) -> Path:
    tmp_dir = PROJECT_ROOT / "outputs" / "tmp_bad_semantic_entrypoint_for_training_mainline_validation"
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


def _load_metrics(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise RuntimeError(f"metrics row {line_no} must be JSON object")
            rows.append(obj)
    if not rows:
        raise RuntimeError("train_metrics.jsonl must contain at least one row")
    return rows


def _assert_required_metric_fields(rows: list[dict[str, object]]) -> bool:
    required = set(REQUIRED_METRIC_FIELDS)
    return all(required.issubset(set(row.keys())) for row in rows)


def _assert_run_artifacts(run_dir: Path) -> tuple[Path, Path, Path]:
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    metrics_path = run_dir / "train_metrics.jsonl"
    checkpoint_path = run_dir / "last.pt"
    for path in [config_snapshot_path, metrics_path, checkpoint_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required training artifact: {path}")
    return config_snapshot_path, metrics_path, checkpoint_path


def _load_checkpoint_payload(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint payload at {path} must be dict")
    return payload


def _assert_required_checkpoint_fields(payload: dict[str, object]) -> bool:
    return all(key in payload for key in REQUIRED_CHECKPOINT_FIELDS)


def _assert_s_final_only(cfg_path: Path) -> bool:
    cfg = load_training_mainline_config(PROJECT_ROOT, cfg_path)
    context = build_training_mainline_context(cfg)

    bad_entrypoint_dir = _build_bad_entrypoint_fixture(semantic_dir=context.supervision.semantic_cache_dir)
    try:
        load_semantic_supervision(
            bad_entrypoint_dir,
            expected_sample_index_hash=context.supervision.sample_index_hash,
            expected_rows=context.supervision.rows,
            expected_feature_set_id=context.supervision.feature_set_id,
        )
    except RuntimeError:
        return True
    finally:
        if bad_entrypoint_dir.exists():
            shutil.rmtree(bad_entrypoint_dir)
    return False


def main() -> int:
    args = parse_args()
    cfg_path = (PROJECT_ROOT / args.config).resolve()
    cfg = load_training_mainline_config(PROJECT_ROOT, cfg_path)

    run_name = f"{cfg.train.run_name}_validator_{uuid.uuid4().hex[:8]}"
    fresh_result = run_training_mainline(cfg, run_name=run_name, stop_after_epochs=1)
    resume_result = run_training_mainline(cfg, run_name=run_name, resume=True)

    run_dir = Path(fresh_result.run_dir)
    run_dir_exists = run_dir.exists()
    config_snapshot_path, metrics_path, checkpoint_path = _assert_run_artifacts(run_dir)
    metric_rows = _load_metrics(metrics_path)
    checkpoint_payload = _load_checkpoint_payload(checkpoint_path)

    checkpoint_context = build_training_mainline_context(cfg)
    checkpoint_start_epoch, checkpoint_global_step = _load_training_checkpoint(
        path=checkpoint_path,
        context=checkpoint_context,
    )

    checks = {
        "config_load_ok": cfg.train.run_name != "",
        "run_dir_exists": run_dir_exists,
        "config_snapshot_exists": config_snapshot_path.exists(),
        "train_metrics_exists": metrics_path.exists(),
        "last_checkpoint_exists": checkpoint_path.exists(),
        "metrics_required_fields_present": _assert_required_metric_fields(metric_rows),
        "checkpoint_required_fields_present": _assert_required_checkpoint_fields(checkpoint_payload),
        "supervision_target_is_s_final_only": _assert_s_final_only(cfg_path),
        "resume_epoch_grew": int(resume_result.completed_epoch) > int(fresh_result.completed_epoch),
        "resume_global_step_grew": int(resume_result.global_step) > int(fresh_result.global_step),
        "checkpoint_epoch_matches_resume": int(checkpoint_payload["epoch"]) == int(resume_result.completed_epoch),
        "checkpoint_global_step_matches_resume": int(checkpoint_payload["global_step"]) == int(resume_result.global_step),
        "checkpoint_loader_reads_current_last_pt": checkpoint_start_epoch == int(resume_result.completed_epoch)
        and checkpoint_global_step == int(resume_result.global_step),
    }

    for key, passed in checks.items():
        if not passed:
            raise RuntimeError(f"Training mainline validation failed at check: {key}")

    result = {
        "config_path": str(cfg_path),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "fresh_result": fresh_result.to_dict(),
        "resume_result": resume_result.to_dict(),
        "metrics_rows": len(metric_rows),
        "metrics_first_row": metric_rows[0],
        "metrics_last_row": metric_rows[-1],
        "checkpoint_summary": {
            key: checkpoint_payload[key]
            for key in REQUIRED_CHECKPOINT_FIELDS
            if key in checkpoint_payload and key not in {"model_state_dict", "optimizer_state_dict"}
        },
        "checks": checks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not args.keep_run_dir and run_dir.exists():
        shutil.rmtree(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
