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
class TrainingMainlineModelConfig:
    order_k: int
    d_model: int
    d_hash: int


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float


@dataclass(frozen=True)
class SchedulerConfig:
    type: str


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
class TrainingRunConfig:
    dataset: str
    feature_set_id: str
    semantic_set_id: str
    batch_size: int
    num_epochs: int
    run_name: str


@dataclass(frozen=True)
class TrainingOutputConfig:
    processed_root: Path
    run_root: Path


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


@dataclass(frozen=True)
class TrainingMainlineConfig:
    project_root: Path
    output: TrainingOutputConfig
    runtime: TrainerRuntimeConfig
    training_mainline: TrainingMainlineModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss_weights: LossWeightsConfig
    train: TrainingRunConfig

    @property
    def processed_root(self) -> Path:
        return self.output.processed_root

    def resolve_feature_cache_dir(self, *, dataset: str, feature_set_id: str) -> Path:
        return self.output.processed_root / dataset / "feature_cache" / feature_set_id

    def resolve_semantic_cache_dir(self, *, dataset: str, semantic_set_id: str) -> Path:
        return self.output.processed_root / dataset / "semantic_cache" / semantic_set_id

    def resolve_training_run_dir(self, *, run_name: str | None = None) -> Path:
        actual_run_name = run_name or self.train.run_name
        return self.output.run_root / self.train.dataset / actual_run_name

    def config_snapshot(self, *, run_name: str | None = None) -> dict[str, Any]:
        actual_run_name = run_name or self.train.run_name
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
            "training_mainline": {
                "order_k": self.training_mainline.order_k,
                "d_model": self.training_mainline.d_model,
                "d_hash": self.training_mainline.d_hash,
            },
            "optimizer": {
                "name": self.optimizer.name,
                "lr": self.optimizer.lr,
            },
            "scheduler": {
                "type": self.scheduler.type,
            },
            "loss_weights": {
                "sem": self.loss_weights.sem,
                "q": self.loss_weights.q,
                "bal": self.loss_weights.bal,
                "grl": self.loss_weights.grl,
                "grl_lambda": self.loss_weights.grl_lambda,
            },
            "train": {
                "dataset": self.train.dataset,
                "feature_set_id": self.train.feature_set_id,
                "semantic_set_id": self.train.semantic_set_id,
                "batch_size": self.train.batch_size,
                "num_epochs": self.train.num_epochs,
                "run_name": actual_run_name,
            },
        }

    def checkpoint_key_fields(self) -> dict[str, Any]:
        return {
            "runtime": {
                "dtype": self.runtime.dtype,
                "seed": self.runtime.seed,
            },
            "training_mainline": {
                "order_k": self.training_mainline.order_k,
                "d_model": self.training_mainline.d_model,
                "d_hash": self.training_mainline.d_hash,
            },
            "optimizer": {
                "name": self.optimizer.name,
                "lr": self.optimizer.lr,
            },
            "scheduler": {
                "type": self.scheduler.type,
            },
            "loss_weights": {
                "sem": self.loss_weights.sem,
                "q": self.loss_weights.q,
                "bal": self.loss_weights.bal,
                "grl": self.loss_weights.grl,
                "grl_lambda": self.loss_weights.grl_lambda,
            },
            "train": {
                "dataset": self.train.dataset,
                "feature_set_id": self.train.feature_set_id,
                "semantic_set_id": self.train.semantic_set_id,
                "batch_size": self.train.batch_size,
            },
        }


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


def load_training_mainline_config(project_root: Path, config_path: Path) -> TrainingMainlineConfig:
    raw = _load_yaml(config_path)

    output_raw = _require_mapping(raw, "output")
    runtime_raw = _require_mapping(raw, "runtime")
    training_mainline_raw = _require_mapping(raw, "training_mainline")
    optimizer_raw = _require_mapping(raw, "optimizer")
    scheduler_raw = _require_mapping(raw, "scheduler")
    loss_raw = _require_mapping(raw, "loss_weights")
    train_raw = _require_mapping(raw, "train")

    output = TrainingOutputConfig(
        processed_root=project_root / _as_str(output_raw.get("processed_root"), "output.processed_root"),
        run_root=project_root / _as_str(output_raw.get("run_root"), "output.run_root"),
    )

    runtime = TrainerRuntimeConfig(
        device=_as_str(runtime_raw.get("device"), "runtime.device"),
        dtype=_as_str(runtime_raw.get("dtype"), "runtime.dtype"),
        seed=_as_int(runtime_raw.get("seed"), "runtime.seed"),
    )
    if runtime.dtype != "float32":
        raise ValueError("training mainline requires runtime.dtype = `float32`")

    training_mainline = TrainingMainlineModelConfig(
        order_k=_as_int(training_mainline_raw.get("order_k"), "training_mainline.order_k"),
        d_model=_as_int(training_mainline_raw.get("d_model"), "training_mainline.d_model"),
        d_hash=_as_int(training_mainline_raw.get("d_hash"), "training_mainline.d_hash"),
    )
    if training_mainline.order_k < 1:
        raise ValueError("training_mainline.order_k must be >= 1")
    if training_mainline.d_model <= 0 or training_mainline.d_hash <= 0:
        raise ValueError("training_mainline.d_model and training_mainline.d_hash must be > 0")

    optimizer = OptimizerConfig(
        name=_as_str(optimizer_raw.get("name"), "optimizer.name"),
        lr=_as_float(optimizer_raw.get("lr"), "optimizer.lr"),
    )
    if optimizer.name != "sgd":
        raise ValueError("training mainline requires optimizer.name = `sgd`")
    if optimizer.lr <= 0:
        raise ValueError("optimizer.lr must be > 0")

    scheduler = SchedulerConfig(type=_as_str(scheduler_raw.get("type"), "scheduler.type"))
    if scheduler.type != "none":
        raise ValueError("training mainline requires scheduler.type = `none`")

    loss_weights = LossWeightsConfig(
        sem=_as_float(loss_raw.get("sem"), "loss_weights.sem"),
        q=_as_float(loss_raw.get("q"), "loss_weights.q"),
        bal=_as_float(loss_raw.get("bal"), "loss_weights.bal"),
        grl=_as_float(loss_raw.get("grl"), "loss_weights.grl"),
        grl_lambda=_as_float(loss_raw.get("grl_lambda"), "loss_weights.grl_lambda"),
    )
    if loss_weights.grl_lambda <= 0:
        raise ValueError("loss_weights.grl_lambda must be > 0")

    train = TrainingRunConfig(
        dataset=_as_str(train_raw.get("dataset"), "train.dataset"),
        feature_set_id=_as_str(train_raw.get("feature_set_id"), "train.feature_set_id"),
        semantic_set_id=_as_str(train_raw.get("semantic_set_id"), "train.semantic_set_id"),
        batch_size=_as_int(train_raw.get("batch_size"), "train.batch_size"),
        num_epochs=_as_int(train_raw.get("num_epochs"), "train.num_epochs"),
        run_name=_as_str(train_raw.get("run_name"), "train.run_name"),
    )
    if train.batch_size <= 0:
        raise ValueError("train.batch_size must be > 0")
    if train.num_epochs <= 0:
        raise ValueError("train.num_epochs must be > 0")

    return TrainingMainlineConfig(
        project_root=project_root,
        output=output,
        runtime=runtime,
        training_mainline=training_mainline,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_weights=loss_weights,
        train=train,
    )
