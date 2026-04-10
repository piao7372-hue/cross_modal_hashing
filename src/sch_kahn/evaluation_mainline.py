from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .evaluation_protocol import FrozenEvaluationProtocol, load_frozen_evaluation_protocol
from .torch_mainline import SchKanhTrainingMainline


@dataclass(frozen=True)
class EvaluationOutputConfig:
    processed_root: Path
    run_root: Path


@dataclass(frozen=True)
class EvaluationRuntimeConfig:
    device: str
    dtype: str
    seed: int


@dataclass(frozen=True)
class EvaluationProtocolRef:
    config_path: Path


@dataclass(frozen=True)
class EvaluationModelConfig:
    order_k: int
    d_model: int
    d_hash: int


@dataclass(frozen=True)
class EvaluationRunConfig:
    dataset: str
    feature_set_id: str
    semantic_set_id: str
    batch_size: int
    run_name: str


@dataclass(frozen=True)
class EvaluationMainlineConfig:
    project_root: Path
    output: EvaluationOutputConfig
    runtime: EvaluationRuntimeConfig
    protocol: EvaluationProtocolRef
    model: EvaluationModelConfig
    evaluation: EvaluationRunConfig

    def resolve_feature_cache_dir(self) -> Path:
        return self.output.processed_root / self.evaluation.dataset / "feature_cache" / self.evaluation.feature_set_id

    def resolve_evaluation_run_dir(self, *, run_name: str | None = None) -> Path:
        actual_run_name = run_name or self.evaluation.run_name
        return self.output.run_root / self.evaluation.dataset / actual_run_name

    def config_snapshot(self, *, checkpoint_path: Path, run_name: str | None = None) -> dict[str, Any]:
        actual_run_name = run_name or self.evaluation.run_name
        return {
            "output": {
                "processed_root": str(self.output.processed_root),
                "run_root": str(self.output.run_root),
            },
            "runtime": {
                "device": self.runtime.device,
                "dtype": self.runtime.dtype,
                "seed": self.runtime.seed,
            },
            "protocol": {
                "config_path": str(self.protocol.config_path),
            },
            "model": {
                "order_k": self.model.order_k,
                "d_model": self.model.d_model,
                "d_hash": self.model.d_hash,
            },
            "evaluation": {
                "dataset": self.evaluation.dataset,
                "feature_set_id": self.evaluation.feature_set_id,
                "semantic_set_id": self.evaluation.semantic_set_id,
                "batch_size": self.evaluation.batch_size,
                "run_name": actual_run_name,
                "checkpoint_path": str(checkpoint_path),
            },
        }


@dataclass(frozen=True)
class FeatureCacheLightInputs:
    feature_cache_dir: Path
    feature_set_id: str
    rows: int
    dim: int
    sample_index_hash: str
    sample_index_path: Path
    x_i_path: Path
    x_t_path: Path
    meta: dict[str, Any]


@dataclass(frozen=True)
class CaptionUnit:
    sample_index: int
    split: str
    image_id: str
    caption_ann_id: str

    @property
    def image_key(self) -> tuple[str, str]:
        return (self.split, self.image_id)


@dataclass(frozen=True)
class ImageGroup:
    split: str
    image_id: str
    representative_sample_index: int
    caption_sample_indices: tuple[int, ...]

    @property
    def image_key(self) -> tuple[str, str]:
        return (self.split, self.image_id)


@dataclass(frozen=True)
class EvaluationSlice:
    caption_units: tuple[CaptionUnit, ...]
    image_groups: tuple[ImageGroup, ...]
    selected_sample_indices: tuple[int, ...]
    excluded_non_val_rows: int
    excluded_empty_text_rows: int
    limited_image_group_count: int | None


@dataclass(frozen=True)
class EvaluationRunResult:
    run_dir: str
    metrics_path: str
    protocol_snapshot_path: str
    config_snapshot_path: str
    dataset: str
    feature_set_id: str
    semantic_set_id: str
    sample_index_hash: str
    checkpoint_path: str
    protocol_id: str
    i2t_map: float
    t2i_map: float
    avg_map: float
    query_count: dict[str, int]
    database_count: dict[str, int]
    bit_length: int
    epoch: int
    global_step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "metrics_path": self.metrics_path,
            "protocol_snapshot_path": self.protocol_snapshot_path,
            "config_snapshot_path": self.config_snapshot_path,
            "dataset": self.dataset,
            "feature_set_id": self.feature_set_id,
            "semantic_set_id": self.semantic_set_id,
            "sample_index_hash": self.sample_index_hash,
            "checkpoint_path": self.checkpoint_path,
            "protocol_id": self.protocol_id,
            "i2t_map": self.i2t_map,
            "t2i_map": self.t2i_map,
            "avg_map": self.avg_map,
            "query_count": self.query_count,
            "database_count": self.database_count,
            "bit_length": self.bit_length,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML root mapping: {path}")
    return data


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"`{key}` must be mapping")
    return value


def _require_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if isinstance(value, str) and value:
        return value
    raise RuntimeError(f"`{key}` must be non-empty string")


def _require_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, int):
        return value
    raise RuntimeError(f"`{key}` must be int")


def load_evaluation_mainline_config(project_root: Path, config_path: Path) -> EvaluationMainlineConfig:
    raw = _load_yaml(config_path)

    output_raw = _require_mapping(raw, "output")
    runtime_raw = _require_mapping(raw, "runtime")
    protocol_raw = _require_mapping(raw, "protocol")
    model_raw = _require_mapping(raw, "model")
    evaluation_raw = _require_mapping(raw, "evaluation")

    cfg = EvaluationMainlineConfig(
        project_root=project_root,
        output=EvaluationOutputConfig(
            processed_root=project_root / _require_str(output_raw, "processed_root"),
            run_root=project_root / _require_str(output_raw, "run_root"),
        ),
        runtime=EvaluationRuntimeConfig(
            device=_require_str(runtime_raw, "device"),
            dtype=_require_str(runtime_raw, "dtype"),
            seed=_require_int(runtime_raw, "seed"),
        ),
        protocol=EvaluationProtocolRef(
            config_path=(project_root / _require_str(protocol_raw, "config_path")).resolve(),
        ),
        model=EvaluationModelConfig(
            order_k=_require_int(model_raw, "order_k"),
            d_model=_require_int(model_raw, "d_model"),
            d_hash=_require_int(model_raw, "d_hash"),
        ),
        evaluation=EvaluationRunConfig(
            dataset=_require_str(evaluation_raw, "dataset"),
            feature_set_id=_require_str(evaluation_raw, "feature_set_id"),
            semantic_set_id=_require_str(evaluation_raw, "semantic_set_id"),
            batch_size=_require_int(evaluation_raw, "batch_size"),
            run_name=_require_str(evaluation_raw, "run_name"),
        ),
    )

    if cfg.runtime.dtype != "float32":
        raise RuntimeError("evaluation mainline requires runtime.dtype = `float32`")
    if cfg.evaluation.dataset != "mscoco":
        raise RuntimeError("evaluation mainline v1 is frozen to dataset `mscoco`")
    if cfg.evaluation.batch_size <= 0:
        raise RuntimeError("evaluation.batch_size must be > 0")
    if cfg.model.order_k < 1 or cfg.model.d_model <= 0 or cfg.model.d_hash <= 0:
        raise RuntimeError("evaluation model dimensions must be positive")
    return cfg


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
        raise RuntimeError(f"sample_index row count mismatch: expected {expected_rows}, got {count}")
    return _sha256_file(path)


def load_feature_cache_light(feature_cache_dir: Path) -> FeatureCacheLightInputs:
    meta_path = feature_cache_dir / "meta.json"
    x_i_path = feature_cache_dir / "X_I.npy"
    x_t_path = feature_cache_dir / "X_T.npy"
    sample_index_path = feature_cache_dir / "sample_index.jsonl"
    for required in [meta_path, x_i_path, x_t_path, sample_index_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required feature cache file: {required}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    x_i = np.load(x_i_path, mmap_mode="r")
    x_t = np.load(x_t_path, mmap_mode="r")
    if x_i.ndim != 2 or x_t.ndim != 2 or x_i.shape != x_t.shape:
        raise RuntimeError("feature cache X_I and X_T must be matching 2D arrays")

    sample_index_hash = _validate_and_hash_sample_index(sample_index_path, expected_rows=int(x_i.shape[0]))
    return FeatureCacheLightInputs(
        feature_cache_dir=feature_cache_dir,
        feature_set_id=str(meta.get("feature_set_id") or feature_cache_dir.name),
        rows=int(x_i.shape[0]),
        dim=int(x_i.shape[1]),
        sample_index_hash=sample_index_hash,
        sample_index_path=sample_index_path,
        x_i_path=x_i_path,
        x_t_path=x_t_path,
        meta=meta,
    )


def _parse_mscoco_id(sample_id: str) -> tuple[str, str, str]:
    parts = sample_id.split(":")
    if len(parts) != 4 or parts[0] != "mscoco":
        raise RuntimeError(f"Invalid MSCOCO sample id: {sample_id}")
    return parts[1], parts[2], parts[3]


def build_evaluation_slice(
    *,
    protocol: FrozenEvaluationProtocol,
    feature_inputs: FeatureCacheLightInputs,
    max_image_groups: int | None = None,
) -> EvaluationSlice:
    caption_units_all: list[CaptionUnit] = []
    excluded_non_val_rows = 0
    excluded_empty_text_rows = 0

    with feature_inputs.sample_index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sample_index = obj.get("sample_index")
            sample_id = obj.get("id")
            split = obj.get("split")
            caption_ann_id = obj.get("caption_ann_id")
            text_empty = obj.get("text_empty")
            if not isinstance(sample_index, int):
                raise RuntimeError("sample_index.jsonl row missing int sample_index")
            if not isinstance(sample_id, str) or not sample_id:
                raise RuntimeError("sample_index.jsonl row missing sample id")
            if not isinstance(split, str) or not split:
                raise RuntimeError("sample_index.jsonl row missing split")
            if not isinstance(caption_ann_id, str) or not caption_ann_id:
                raise RuntimeError("sample_index.jsonl row missing caption_ann_id")
            if not isinstance(text_empty, bool):
                raise RuntimeError("sample_index.jsonl row missing bool text_empty")

            parsed_split, image_id, parsed_caption_ann_id = _parse_mscoco_id(sample_id)
            if parsed_split != split or parsed_caption_ann_id != caption_ann_id:
                raise RuntimeError("MSCOCO sample identity mismatch between id, split, and caption_ann_id")

            if split in protocol.scope.forbidden_splits:
                excluded_non_val_rows += 1
                continue
            if split not in protocol.scope.allowed_splits:
                excluded_non_val_rows += 1
                continue
            if text_empty:
                excluded_empty_text_rows += 1
                continue

            caption_units_all.append(
                CaptionUnit(
                    sample_index=sample_index,
                    split=split,
                    image_id=image_id,
                    caption_ann_id=caption_ann_id,
                )
            )

    ordered_keys: list[tuple[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for caption_unit in caption_units_all:
        if caption_unit.image_key not in seen_keys:
            seen_keys.add(caption_unit.image_key)
            ordered_keys.append(caption_unit.image_key)

    if max_image_groups is not None:
        if max_image_groups <= 0:
            raise RuntimeError("max_image_groups must be > 0 when provided")
        selected_keys = set(ordered_keys[:max_image_groups])
        caption_units = tuple(c for c in caption_units_all if c.image_key in selected_keys)
    else:
        caption_units = tuple(caption_units_all)

    grouped: dict[tuple[str, str], list[int]] = {}
    representative: dict[tuple[str, str], int] = {}
    limited_keys: list[tuple[str, str]] = []
    for caption_unit in caption_units:
        key = caption_unit.image_key
        if key not in grouped:
            grouped[key] = []
            representative[key] = caption_unit.sample_index
            limited_keys.append(key)
        grouped[key].append(caption_unit.sample_index)

    image_groups = tuple(
        ImageGroup(
            split=key[0],
            image_id=key[1],
            representative_sample_index=representative[key],
            caption_sample_indices=tuple(grouped[key]),
        )
        for key in limited_keys
    )

    if not caption_units:
        raise RuntimeError("Evaluation slice is empty after protocol filtering")
    if not image_groups:
        raise RuntimeError("Evaluation slice produced zero image groups")

    selected_sample_indices = tuple(c.sample_index for c in caption_units)
    return EvaluationSlice(
        caption_units=caption_units,
        image_groups=image_groups,
        selected_sample_indices=selected_sample_indices,
        excluded_non_val_rows=excluded_non_val_rows,
        excluded_empty_text_rows=excluded_empty_text_rows,
        limited_image_group_count=max_image_groups,
    )


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested torch device `{name}` but CUDA is not available")
    return torch.device(name)


def _load_checkpoint_payload(
    *,
    checkpoint_path: Path,
    cfg: EvaluationMainlineConfig,
    feature_inputs: FeatureCacheLightInputs,
) -> tuple[dict[str, Any], int, int]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("Evaluation checkpoint payload must be dict")

    for key in [
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "global_step",
        "feature_set_id",
        "semantic_set_id",
        "sample_index_hash",
    ]:
        if key not in payload:
            raise RuntimeError(f"Evaluation checkpoint missing required field: {key}")

    if payload["feature_set_id"] != cfg.evaluation.feature_set_id:
        raise RuntimeError("Checkpoint feature_set_id does not match evaluation config")
    if payload["semantic_set_id"] != cfg.evaluation.semantic_set_id:
        raise RuntimeError("Checkpoint semantic_set_id does not match evaluation config")
    if payload["sample_index_hash"] != feature_inputs.sample_index_hash:
        raise RuntimeError("Checkpoint sample_index_hash does not match feature cache sample_index hash")

    key_fields = payload.get("config_key_fields")
    if isinstance(key_fields, dict):
        training_mainline = key_fields.get("training_mainline")
        if isinstance(training_mainline, dict):
            if training_mainline.get("order_k") != cfg.model.order_k:
                raise RuntimeError("Checkpoint order_k does not match evaluation model config")
            if training_mainline.get("d_model") != cfg.model.d_model:
                raise RuntimeError("Checkpoint d_model does not match evaluation model config")
            if training_mainline.get("d_hash") != cfg.model.d_hash:
                raise RuntimeError("Checkpoint d_hash does not match evaluation model config")

    epoch = payload.get("epoch")
    global_step = payload.get("global_step")
    if not isinstance(epoch, int) or not isinstance(global_step, int):
        raise RuntimeError("Checkpoint epoch/global_step must be int")
    return payload, epoch, global_step


def run_checkpoint_inference(
    *,
    cfg: EvaluationMainlineConfig,
    feature_inputs: FeatureCacheLightInputs,
    checkpoint_path: Path,
    selected_sample_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    payload, _, _ = _load_checkpoint_payload(checkpoint_path=checkpoint_path, cfg=cfg, feature_inputs=feature_inputs)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise RuntimeError("Checkpoint model_state_dict must be mapping")
    training_state = model_state.get("training_mainline")
    if not isinstance(training_state, dict):
        raise RuntimeError("Checkpoint model_state_dict.training_mainline must be mapping")

    device = _resolve_device(cfg.runtime.device)
    model = SchKanhTrainingMainline(
        d_in=feature_inputs.dim,
        d_model=cfg.model.d_model,
        order_k=cfg.model.order_k,
        d_hash=cfg.model.d_hash,
        seed=cfg.runtime.seed,
    ).to(device=device)
    model.load_state_dict(training_state)
    model.eval()

    x_i = np.load(feature_inputs.x_i_path, mmap_mode="r")
    x_t = np.load(feature_inputs.x_t_path, mmap_mode="r")
    sample_indices = np.asarray(selected_sample_indices, dtype=np.int64)

    b_i_batches: list[np.ndarray] = []
    b_t_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, sample_indices.size, cfg.evaluation.batch_size):
            end = min(start + cfg.evaluation.batch_size, sample_indices.size)
            batch_idx = sample_indices[start:end]
            x_i_batch = torch.from_numpy(np.asarray(x_i[batch_idx], dtype=np.float32)).to(device=device)
            x_t_batch = torch.from_numpy(np.asarray(x_t[batch_idx], dtype=np.float32)).to(device=device)
            output = model(x_i_batch, x_t_batch)
            b_i_batches.append(output.b_i.detach().cpu().numpy().astype(np.int8, copy=False))
            b_t_batches.append(output.b_t.detach().cpu().numpy().astype(np.int8, copy=False))

    b_i = np.concatenate(b_i_batches, axis=0)
    b_t = np.concatenate(b_t_batches, axis=0)
    if b_i.shape != b_t.shape:
        raise RuntimeError("B_I/B_T shape mismatch during evaluation inference")
    if not np.isin(b_i, np.array([-1, 1], dtype=np.int8)).all():
        raise RuntimeError("Evaluation B_I values must stay in {-1, +1}")
    if not np.isin(b_t, np.array([-1, 1], dtype=np.int8)).all():
        raise RuntimeError("Evaluation B_T values must stay in {-1, +1}")
    return b_i, b_t


def _average_precision_from_scores(scores: np.ndarray, positive_indices: np.ndarray) -> float:
    if positive_indices.size <= 0:
        raise RuntimeError("Each evaluation query must have at least one positive target")
    order = np.argsort(-scores, kind="stable")
    hits = np.isin(order, positive_indices, assume_unique=False)
    hit_positions = np.flatnonzero(hits)
    if hit_positions.size != positive_indices.size:
        raise RuntimeError("Positive target count mismatch during AP computation")
    precision_at_hits = np.cumsum(hits, dtype=np.int64)[hit_positions] / (hit_positions + 1)
    return float(np.mean(precision_at_hits))


def compute_map(
    *,
    query_codes: np.ndarray,
    database_codes: np.ndarray,
    positives: list[np.ndarray],
    query_block_size: int = 32,
) -> float:
    if query_codes.ndim != 2 or database_codes.ndim != 2:
        raise RuntimeError("query_codes and database_codes must be 2D")
    if query_codes.shape[1] != database_codes.shape[1]:
        raise RuntimeError("query_codes and database_codes bit dimensions must match")
    if len(positives) != int(query_codes.shape[0]):
        raise RuntimeError("positives list length must equal query count")

    db = database_codes.astype(np.int16, copy=False)
    aps: list[float] = []
    for start in range(0, int(query_codes.shape[0]), query_block_size):
        end = min(start + query_block_size, int(query_codes.shape[0]))
        q_block = query_codes[start:end].astype(np.int16, copy=False)
        score_block = q_block @ db.T
        for local_index in range(end - start):
            aps.append(_average_precision_from_scores(score_block[local_index], positives[start + local_index]))
    return float(np.mean(np.asarray(aps, dtype=np.float64)))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_evaluation_mainline(
    cfg: EvaluationMainlineConfig,
    *,
    checkpoint_path: Path,
    run_name: str | None = None,
    max_image_groups: int | None = None,
) -> EvaluationRunResult:
    protocol = load_frozen_evaluation_protocol(cfg.project_root, cfg.protocol.config_path)
    if protocol.dataset != cfg.evaluation.dataset:
        raise RuntimeError("Evaluation config dataset does not match frozen protocol dataset")

    feature_inputs = load_feature_cache_light(cfg.resolve_feature_cache_dir())
    if feature_inputs.feature_set_id != cfg.evaluation.feature_set_id:
        raise RuntimeError("Feature cache feature_set_id mismatch against evaluation config")

    _, epoch, global_step = _load_checkpoint_payload(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        feature_inputs=feature_inputs,
    )

    eval_slice = build_evaluation_slice(
        protocol=protocol,
        feature_inputs=feature_inputs,
        max_image_groups=max_image_groups,
    )
    b_i_rows, b_t_rows = run_checkpoint_inference(
        cfg=cfg,
        feature_inputs=feature_inputs,
        checkpoint_path=checkpoint_path,
        selected_sample_indices=eval_slice.selected_sample_indices,
    )

    sample_index_to_local = {
        sample_index: local_index for local_index, sample_index in enumerate(eval_slice.selected_sample_indices)
    }
    caption_keys = [caption.image_key for caption in eval_slice.caption_units]
    caption_key_to_indices: dict[tuple[str, str], list[int]] = {}
    for caption_index, key in enumerate(caption_keys):
        caption_key_to_indices.setdefault(key, []).append(caption_index)

    image_db_codes = np.stack(
        [b_i_rows[sample_index_to_local[group.representative_sample_index]] for group in eval_slice.image_groups],
        axis=0,
    )
    caption_db_codes = b_t_rows

    image_keys = [group.image_key for group in eval_slice.image_groups]
    image_key_to_db_index = {key: idx for idx, key in enumerate(image_keys)}

    i2t_positives = [
        np.asarray(caption_key_to_indices[group.image_key], dtype=np.int64) for group in eval_slice.image_groups
    ]
    t2i_positives = [
        np.asarray([image_key_to_db_index[caption.image_key]], dtype=np.int64) for caption in eval_slice.caption_units
    ]

    i2t_map = compute_map(query_codes=image_db_codes, database_codes=caption_db_codes, positives=i2t_positives)
    t2i_map = compute_map(query_codes=caption_db_codes, database_codes=image_db_codes, positives=t2i_positives)
    avg_map = float((i2t_map + t2i_map) / 2.0)
    bit_length = int(image_db_codes.shape[1])

    actual_run_name = run_name or cfg.evaluation.run_name
    run_dir = cfg.resolve_evaluation_run_dir(run_name=actual_run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    protocol_snapshot_path = run_dir / "protocol_snapshot.yaml"
    metrics_path = run_dir / "metrics.json"

    _write_yaml(config_snapshot_path, cfg.config_snapshot(checkpoint_path=checkpoint_path, run_name=actual_run_name))
    _write_yaml(protocol_snapshot_path, protocol.snapshot())
    metrics_payload = {
        "dataset": cfg.evaluation.dataset,
        "feature_set_id": cfg.evaluation.feature_set_id,
        "semantic_set_id": cfg.evaluation.semantic_set_id,
        "sample_index_hash": feature_inputs.sample_index_hash,
        "protocol_id": protocol.protocol_id,
        "checkpoint_path": str(checkpoint_path),
        "epoch": epoch,
        "global_step": global_step,
        "i2t_map": i2t_map,
        "t2i_map": t2i_map,
        "avg_map": avg_map,
        "query_count": {
            "i2t": int(image_db_codes.shape[0]),
            "t2i": int(caption_db_codes.shape[0]),
        },
        "database_count": {
            "i2t": int(caption_db_codes.shape[0]),
            "t2i": int(image_db_codes.shape[0]),
        },
        "bit_length": bit_length,
        "selection_summary": {
            "allowed_split": protocol.scope.allowed_splits[0],
            "image_group_count": int(len(eval_slice.image_groups)),
            "caption_count": int(len(eval_slice.caption_units)),
            "excluded_non_val_rows": int(eval_slice.excluded_non_val_rows),
            "excluded_empty_text_rows": int(eval_slice.excluded_empty_text_rows),
            "max_image_groups": eval_slice.limited_image_group_count,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(metrics_path, metrics_payload)

    return EvaluationRunResult(
        run_dir=str(run_dir),
        metrics_path=str(metrics_path),
        protocol_snapshot_path=str(protocol_snapshot_path),
        config_snapshot_path=str(config_snapshot_path),
        dataset=cfg.evaluation.dataset,
        feature_set_id=cfg.evaluation.feature_set_id,
        semantic_set_id=cfg.evaluation.semantic_set_id,
        sample_index_hash=feature_inputs.sample_index_hash,
        checkpoint_path=str(checkpoint_path),
        protocol_id=protocol.protocol_id,
        i2t_map=i2t_map,
        t2i_map=t2i_map,
        avg_map=avg_map,
        query_count={"i2t": int(image_db_codes.shape[0]), "t2i": int(caption_db_codes.shape[0])},
        database_count={"i2t": int(caption_db_codes.shape[0]), "t2i": int(image_db_codes.shape[0])},
        bit_length=bit_length,
        epoch=epoch,
        global_step=global_step,
    )
