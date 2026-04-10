from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainerRuntimeConfig:
    device: str
    dtype: str
    seed: int


@dataclass(frozen=True)
class ShadowMainlineConfig:
    order_k: int
    d_model: int
    d_hash: int


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float


@dataclass(frozen=True)
class LossWeightsConfig:
    sem: float
    q: float
    bal: float
    grl: float
    grl_lambda: float


@dataclass(frozen=True)
class SmokeRunConfig:
    dataset: str
    feature_set_id: str
    semantic_set_id: str
    batch_size: int
    sample_indices: list[int]


@dataclass(frozen=True)
class TrainerSmokeConfig:
    project_root: Path
    processed_root: Path
    runtime: TrainerRuntimeConfig
    shadow_mainline: ShadowMainlineConfig
    optimizer: OptimizerConfig
    loss_weights: LossWeightsConfig
    smoke: SmokeRunConfig

    def resolve_feature_cache_dir(self, *, dataset: str, feature_set_id: str) -> Path:
        return self.processed_root / dataset / "feature_cache" / feature_set_id

    def resolve_semantic_cache_dir(self, *, dataset: str, semantic_set_id: str) -> Path:
        return self.processed_root / dataset / "semantic_cache" / semantic_set_id


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root object in {path}: expected mapping")
    return data


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"`{key}` must be mapping")
    return value


def _as_str(value: Any, name: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"`{name}` must be non-empty string")


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, int):
        return value
    raise ValueError(f"`{name}` must be int")


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"`{name}` must be float")


def load_trainer_smoke_config(project_root: Path, config_path: Path) -> TrainerSmokeConfig:
    raw = _load_yaml(config_path)

    output_raw = _require_mapping(raw, "output")
    runtime_raw = _require_mapping(raw, "runtime")
    shadow_raw = _require_mapping(raw, "shadow_mainline")
    optimizer_raw = _require_mapping(raw, "optimizer")
    loss_raw = _require_mapping(raw, "loss_weights")
    smoke_raw = _require_mapping(raw, "smoke")

    processed_root = project_root / _as_str(output_raw.get("processed_root"), "output.processed_root")

    runtime = TrainerRuntimeConfig(
        device=_as_str(runtime_raw.get("device"), "runtime.device"),
        dtype=_as_str(runtime_raw.get("dtype"), "runtime.dtype"),
        seed=_as_int(runtime_raw.get("seed"), "runtime.seed"),
    )
    if runtime.dtype != "float32":
        raise ValueError("trainer smoke currently requires runtime.dtype = `float32`")

    shadow = ShadowMainlineConfig(
        order_k=_as_int(shadow_raw.get("order_k"), "shadow_mainline.order_k"),
        d_model=_as_int(shadow_raw.get("d_model"), "shadow_mainline.d_model"),
        d_hash=_as_int(shadow_raw.get("d_hash"), "shadow_mainline.d_hash"),
    )
    if shadow.order_k < 1:
        raise ValueError("shadow_mainline.order_k must be >= 1")
    if shadow.d_model <= 0 or shadow.d_hash <= 0:
        raise ValueError("shadow_mainline.d_model and shadow_mainline.d_hash must be > 0")

    optimizer = OptimizerConfig(
        name=_as_str(optimizer_raw.get("name"), "optimizer.name"),
        lr=_as_float(optimizer_raw.get("lr"), "optimizer.lr"),
    )
    if optimizer.name != "sgd":
        raise ValueError("trainer smoke currently requires optimizer.name = `sgd`")
    if optimizer.lr <= 0:
        raise ValueError("optimizer.lr must be > 0")

    loss_weights = LossWeightsConfig(
        sem=_as_float(loss_raw.get("sem"), "loss_weights.sem"),
        q=_as_float(loss_raw.get("q"), "loss_weights.q"),
        bal=_as_float(loss_raw.get("bal"), "loss_weights.bal"),
        grl=_as_float(loss_raw.get("grl"), "loss_weights.grl"),
        grl_lambda=_as_float(loss_raw.get("grl_lambda"), "loss_weights.grl_lambda"),
    )
    if loss_weights.grl_lambda <= 0:
        raise ValueError("loss_weights.grl_lambda must be > 0")

    sample_indices = smoke_raw.get("sample_indices")
    if not isinstance(sample_indices, list) or not sample_indices:
        raise ValueError("smoke.sample_indices must be non-empty list[int]")
    if not all(isinstance(x, int) for x in sample_indices):
        raise ValueError("smoke.sample_indices must contain only int values")

    smoke = SmokeRunConfig(
        dataset=_as_str(smoke_raw.get("dataset"), "smoke.dataset"),
        feature_set_id=_as_str(smoke_raw.get("feature_set_id"), "smoke.feature_set_id"),
        semantic_set_id=_as_str(smoke_raw.get("semantic_set_id"), "smoke.semantic_set_id"),
        batch_size=_as_int(smoke_raw.get("batch_size"), "smoke.batch_size"),
        sample_indices=sample_indices,
    )
    if smoke.batch_size <= 0:
        raise ValueError("smoke.batch_size must be > 0")
    if smoke.batch_size != len(smoke.sample_indices):
        raise ValueError("trainer smoke requires smoke.batch_size == len(smoke.sample_indices)")

    return TrainerSmokeConfig(
        project_root=project_root,
        processed_root=processed_root,
        runtime=runtime,
        shadow_mainline=shadow,
        optimizer=optimizer,
        loss_weights=loss_weights,
        smoke=smoke,
    )
