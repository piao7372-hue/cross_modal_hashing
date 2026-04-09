from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_DATASETS = ("nuswide", "mirflickr25k", "mscoco")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    dtype: str
    seed: int


@dataclass(frozen=True)
class EncoderConfig:
    order_k: int
    d_model: int


@dataclass(frozen=True)
class GraphSideConfig:
    mode: str


@dataclass(frozen=True)
class HashHeadConfig:
    enabled: bool
    d_hash: int
    shared_params: bool
    binarize_rule: str


@dataclass(frozen=True)
class MainlineConfig:
    input_baseline: str
    encoder: EncoderConfig
    hash_head: HashHeadConfig
    graph_side: GraphSideConfig
    stop_at: str


@dataclass(frozen=True)
class SchKanhConfig:
    project_root: Path
    processed_root: Path
    runtime: RuntimeConfig
    mainline: MainlineConfig

    def resolve_feature_cache_dir(self, dataset: str, feature_set_id: str) -> Path:
        return self.processed_root / dataset / "feature_cache" / feature_set_id

    def resolve_sch_kahn_cache_dir(self, dataset: str, sch_set_id: str) -> Path:
        return self.processed_root / dataset / "sch_kahn_cache" / sch_set_id


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
        raise ValueError(f"`{key}` must be a mapping")
    return value


def _as_str(value: Any, name: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"`{name}` must be non-empty string")


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, int):
        return value
    raise ValueError(f"`{name}` must be int")


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"`{name}` must be bool")


def load_sch_kahn_config(project_root: Path, config_path: Path) -> SchKanhConfig:
    raw = _load_yaml(config_path)

    output_raw = _require_mapping(raw, "output")
    runtime_raw = _require_mapping(raw, "runtime")
    mainline_raw = _require_mapping(raw, "mainline")
    encoder_raw = _require_mapping(mainline_raw, "encoder")
    hash_head_raw = _require_mapping(mainline_raw, "hash_head")
    graph_side_raw = _require_mapping(mainline_raw, "graph_side")

    processed_root = project_root / _as_str(output_raw.get("processed_root"), "output.processed_root")

    runtime = RuntimeConfig(
        device=_as_str(runtime_raw.get("device"), "runtime.device"),
        dtype=_as_str(runtime_raw.get("dtype"), "runtime.dtype"),
        seed=_as_int(runtime_raw.get("seed"), "runtime.seed"),
    )

    mainline = MainlineConfig(
        input_baseline=_as_str(mainline_raw.get("input_baseline"), "mainline.input_baseline"),
        encoder=EncoderConfig(
            order_k=_as_int(encoder_raw.get("order_k"), "mainline.encoder.order_k"),
            d_model=_as_int(encoder_raw.get("d_model"), "mainline.encoder.d_model"),
        ),
        hash_head=HashHeadConfig(
            enabled=_as_bool(hash_head_raw.get("enabled"), "mainline.hash_head.enabled"),
            d_hash=_as_int(hash_head_raw.get("d_hash"), "mainline.hash_head.d_hash"),
            shared_params=_as_bool(hash_head_raw.get("shared_params"), "mainline.hash_head.shared_params"),
            binarize_rule=_as_str(hash_head_raw.get("binarize_rule"), "mainline.hash_head.binarize_rule"),
        ),
        graph_side=GraphSideConfig(mode=_as_str(graph_side_raw.get("mode"), "mainline.graph_side.mode")),
        stop_at=_as_str(mainline_raw.get("stop_at"), "mainline.stop_at"),
    )

    if mainline.input_baseline != "feature_cache_x_i_x_t":
        raise ValueError("mainline.input_baseline must be `feature_cache_x_i_x_t`")
    if mainline.graph_side.mode != "disabled":
        raise ValueError("mainline.graph_side.mode must be `disabled` in this forward-only build")
    if mainline.stop_at != "b":
        raise ValueError("mainline.stop_at must be `b` in this forward-only v2 build")
    if mainline.encoder.order_k < 1:
        raise ValueError("mainline.encoder.order_k must be >= 1")
    if mainline.encoder.d_model <= 0:
        raise ValueError("mainline.encoder.d_model must be > 0")
    if not mainline.hash_head.enabled:
        raise ValueError("mainline.hash_head.enabled must be `true` in this forward-only v2 build")
    if mainline.hash_head.d_hash <= 0:
        raise ValueError("mainline.hash_head.d_hash must be > 0")
    if not mainline.hash_head.shared_params:
        raise ValueError("mainline.hash_head.shared_params must be `true` in this forward-only v2 build")
    if mainline.hash_head.binarize_rule != "sign_ge_zero_to_pos1":
        raise ValueError("mainline.hash_head.binarize_rule must be `sign_ge_zero_to_pos1`")
    if runtime.dtype != "float32":
        raise ValueError("runtime.dtype must be `float32` in this forward-only build")

    return SchKanhConfig(
        project_root=project_root,
        processed_root=processed_root,
        runtime=runtime,
        mainline=mainline,
    )
