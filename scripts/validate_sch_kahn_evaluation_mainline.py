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

from sch_kahn.evaluation_mainline import (  # noqa: E402
    load_evaluation_mainline_config,
    load_feature_cache_light,
    run_evaluation_mainline,
)
from sch_kahn.evaluation_protocol import load_frozen_evaluation_protocol  # noqa: E402
from sch_kahn.torch_losses import GrlDomainHead  # noqa: E402
from sch_kahn.torch_mainline import SchKanhTrainingMainline  # noqa: E402

REQUIRED_METRIC_FIELDS = [
    "i2t_map",
    "t2i_map",
    "avg_map",
    "query_count",
    "database_count",
    "bit_length",
    "epoch",
    "global_step",
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
    parser = argparse.ArgumentParser(description="Validate SCH-KANH formal evaluation mainline v1.")
    parser.add_argument("--config", default="configs/sch_kahn_evaluation_mainline.yaml")
    parser.add_argument("--keep-run-dir", action="store_true")
    parser.add_argument("--keep-checkpoint-fixture", action="store_true")
    return parser.parse_args()


def _checkpoint_fixture_path() -> Path:
    return PROJECT_ROOT / "outputs" / "tmp_mscoco_eval_checkpoint_fixture_last.pt"


def _build_checkpoint_fixture(cfg_path: Path) -> Path:
    cfg = load_evaluation_mainline_config(PROJECT_ROOT, cfg_path)
    feature_inputs = load_feature_cache_light(cfg.resolve_feature_cache_dir())

    model = SchKanhTrainingMainline(
        d_in=feature_inputs.dim,
        d_model=cfg.model.d_model,
        order_k=cfg.model.order_k,
        d_hash=cfg.model.d_hash,
        seed=cfg.runtime.seed,
    )
    domain_head = GrlDomainHead(
        d_in=cfg.model.d_model,
        lambda_grl=1.0,
        seed=cfg.runtime.seed + 303,
    )
    optimizer = torch.optim.SGD(list(model.parameters()) + list(domain_head.parameters()), lr=0.001)

    payload = {
        "model_state_dict": {
            "training_mainline": model.state_dict(),
            "domain_head": domain_head.state_dict(),
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 1,
        "global_step": 8,
        "feature_set_id": cfg.evaluation.feature_set_id,
        "semantic_set_id": cfg.evaluation.semantic_set_id,
        "sample_index_hash": feature_inputs.sample_index_hash,
        "config_key_fields": {
            "training_mainline": {
                "order_k": cfg.model.order_k,
                "d_model": cfg.model.d_model,
                "d_hash": cfg.model.d_hash,
            }
        },
    }
    checkpoint_path = _checkpoint_fixture_path()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _load_metrics(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError("evaluation metrics payload must be JSON object")
    return obj


def _assert_required_metric_fields(metrics: dict[str, object]) -> bool:
    return all(field in metrics for field in REQUIRED_METRIC_FIELDS)


def _assert_required_checkpoint_fields(payload: dict[str, object]) -> bool:
    return all(field in payload for field in REQUIRED_CHECKPOINT_FIELDS)


def _assert_run_artifacts(run_dir: Path) -> tuple[Path, Path, Path]:
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    protocol_snapshot_path = run_dir / "protocol_snapshot.yaml"
    metrics_path = run_dir / "metrics.json"
    for required in [config_snapshot_path, protocol_snapshot_path, metrics_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required evaluation artifact: {required}")
    return config_snapshot_path, protocol_snapshot_path, metrics_path


def main() -> int:
    args = parse_args()
    cfg_path = (PROJECT_ROOT / args.config).resolve()
    cfg = load_evaluation_mainline_config(PROJECT_ROOT, cfg_path)
    protocol = load_frozen_evaluation_protocol(PROJECT_ROOT, cfg.protocol.config_path)
    feature_inputs = load_feature_cache_light(cfg.resolve_feature_cache_dir())

    checkpoint_path = _build_checkpoint_fixture(cfg_path)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    run_name = f"{cfg.evaluation.run_name}_validator_{uuid.uuid4().hex[:8]}"
    result = run_evaluation_mainline(
        cfg,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        max_image_groups=4,
    )

    run_dir = Path(result.run_dir)
    config_snapshot_path, protocol_snapshot_path, metrics_path = _assert_run_artifacts(run_dir)
    metrics = _load_metrics(metrics_path)

    query_count = metrics.get("query_count")
    database_count = metrics.get("database_count")
    selection_summary = metrics.get("selection_summary")

    checks = {
        "config_load_ok": cfg.evaluation.dataset == "mscoco",
        "protocol_load_ok": protocol.protocol_id == "mscoco_val2014_unique_image_caption_same_image_v1",
        "run_dir_exists": run_dir.exists(),
        "config_snapshot_exists": config_snapshot_path.exists(),
        "protocol_snapshot_exists": protocol_snapshot_path.exists(),
        "metrics_exists": metrics_path.exists(),
        "metrics_required_fields_present": _assert_required_metric_fields(metrics),
        "checkpoint_required_fields_present": _assert_required_checkpoint_fields(checkpoint_payload),
        "checkpoint_lineage_matches_feature_cache": checkpoint_payload["feature_set_id"] == cfg.evaluation.feature_set_id
        and checkpoint_payload["semantic_set_id"] == cfg.evaluation.semantic_set_id
        and checkpoint_payload["sample_index_hash"] == feature_inputs.sample_index_hash,
        "protocol_is_val2014_only": protocol.scope.allowed_splits == ("val2014",)
        and set(protocol.scope.forbidden_splits) == {"train2014", "test2014"},
        "protocol_same_image_relevance": protocol.relevance.rule == "same_split_and_image_id",
        "protocol_empty_text_forbidden": protocol.empty_text.allow_in_query is False
        and protocol.empty_text.allow_in_database is False,
        "selection_summary_split_ok": isinstance(selection_summary, dict)
        and selection_summary.get("allowed_split") == "val2014",
        "selection_summary_max_image_groups_ok": isinstance(selection_summary, dict)
        and selection_summary.get("max_image_groups") == 4,
        "selection_summary_excluded_empty_text_ok": isinstance(selection_summary, dict)
        and isinstance(selection_summary.get("excluded_empty_text_rows"), int),
        "query_count_present_and_positive": isinstance(query_count, dict)
        and int(query_count.get("i2t", 0)) > 0
        and int(query_count.get("t2i", 0)) > 0,
        "database_count_present_and_positive": isinstance(database_count, dict)
        and int(database_count.get("i2t", 0)) > 0
        and int(database_count.get("t2i", 0)) > 0,
        "bit_length_matches_model": int(metrics["bit_length"]) == cfg.model.d_hash,
        "epoch_global_step_match_checkpoint": int(metrics["epoch"]) == int(checkpoint_payload["epoch"])
        and int(metrics["global_step"]) == int(checkpoint_payload["global_step"]),
        "maps_are_bounded": 0.0 <= float(metrics["i2t_map"]) <= 1.0
        and 0.0 <= float(metrics["t2i_map"]) <= 1.0
        and 0.0 <= float(metrics["avg_map"]) <= 1.0,
    }

    for key, passed in checks.items():
        if not passed:
            raise RuntimeError(f"Evaluation mainline validation failed at check: {key}")

    summary = {
        "config_path": str(cfg_path),
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "result": result.to_dict(),
        "metrics": metrics,
        "checks": checks,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.keep_run_dir and run_dir.exists():
        shutil.rmtree(run_dir)
    if not args.keep_checkpoint_fixture and checkpoint_path.exists():
        checkpoint_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
