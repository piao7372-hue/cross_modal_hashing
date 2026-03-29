from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_DATASETS = ("nuswide", "mirflickr25k", "mscoco")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    dtype: str
    deterministic: bool
    seed: int
    image_batch_size: int
    text_batch_size: int
    huggingface_cache_dir: Path


@dataclass(frozen=True)
class EncoderConfig:
    family: str
    model_name: str
    text_tokenizer: str
    text_max_length: int
    image_preprocess: dict[str, Any]
    normalize_type: str
    normalize_epsilon: float


@dataclass(frozen=True)
class CacheConfig:
    feature_cache_dirname: str
    save_row_aligned_image_matrix: bool
    save_unique_image_cache: str
    sample_index_filename: str
    meta_filename: str


@dataclass(frozen=True)
class ContractConfig:
    sample_index_basis: str
    forbid_row_index_as_position: bool
    empty_text_policy: str
    default_nus_input_view: str


@dataclass(frozen=True)
class DatasetViews:
    default_view: str
    views: dict[str, Path]


@dataclass(frozen=True)
class FeatureExtractionConfig:
    project_root: Path
    processed_root: Path
    runtime: RuntimeConfig
    encoder: EncoderConfig
    cache: CacheConfig
    contracts: ContractConfig
    data_views: dict[str, DatasetViews] = field(default_factory=dict)

    def supported_datasets(self) -> list[str]:
        return sorted(self.data_views.keys())

    def resolve_manifest_path(self, dataset: str, view: str | None = None) -> tuple[str, Path]:
        dataset_views = self.data_views.get(dataset)
        if dataset_views is None:
            raise ValueError(f"Unsupported dataset `{dataset}`")
        resolved_view = view or dataset_views.default_view
        manifest_path = dataset_views.views.get(resolved_view)
        if manifest_path is None:
            valid = ", ".join(sorted(dataset_views.views.keys()))
            raise ValueError(
                f"Unsupported view `{resolved_view}` for dataset `{dataset}`. Valid views: {valid}"
            )
        return resolved_view, manifest_path

    def resolve_output_dir(self, dataset: str, feature_set_id: str) -> Path:
        return self.processed_root / dataset / self.cache.feature_cache_dirname / feature_set_id


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


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, int):
        return value
    raise ValueError(f"`{name}` must be int")


def _as_str(value: Any, name: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"`{name}` must be non-empty string")


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"`{name}` must be bool")


def load_feature_extraction_config(project_root: Path, config_path: Path) -> FeatureExtractionConfig:
    raw = _load_yaml(config_path)

    output_raw = _require_mapping(raw, "output")
    runtime_raw = _require_mapping(raw, "runtime")
    encoder_raw = _require_mapping(raw, "encoder")
    cache_raw = _require_mapping(raw, "cache")
    contracts_raw = _require_mapping(raw, "contracts")
    data_views_raw = _require_mapping(raw, "data_views")

    processed_root = project_root / _as_str(output_raw.get("processed_root"), "output.processed_root")

    runtime = RuntimeConfig(
        device=_as_str(runtime_raw.get("device"), "runtime.device"),
        dtype=_as_str(runtime_raw.get("dtype"), "runtime.dtype"),
        deterministic=_as_bool(runtime_raw.get("deterministic"), "runtime.deterministic"),
        seed=_as_int(runtime_raw.get("seed"), "runtime.seed"),
        image_batch_size=_as_int(runtime_raw.get("image_batch_size"), "runtime.image_batch_size"),
        text_batch_size=_as_int(runtime_raw.get("text_batch_size"), "runtime.text_batch_size"),
        huggingface_cache_dir=project_root
        / _as_str(runtime_raw.get("huggingface_cache_dir"), "runtime.huggingface_cache_dir"),
    )

    normalize_raw = _require_mapping(encoder_raw, "normalize")
    encoder = EncoderConfig(
        family=_as_str(encoder_raw.get("family"), "encoder.family"),
        model_name=_as_str(encoder_raw.get("model_name"), "encoder.model_name"),
        text_tokenizer=_as_str(encoder_raw.get("text_tokenizer"), "encoder.text_tokenizer"),
        text_max_length=_as_int(encoder_raw.get("text_max_length"), "encoder.text_max_length"),
        image_preprocess=dict(_require_mapping(encoder_raw, "image_preprocess")),
        normalize_type=_as_str(normalize_raw.get("type"), "encoder.normalize.type"),
        normalize_epsilon=float(normalize_raw.get("epsilon", 1.0e-12)),
    )

    cache = CacheConfig(
        feature_cache_dirname=_as_str(cache_raw.get("feature_cache_dirname"), "cache.feature_cache_dirname"),
        save_row_aligned_image_matrix=_as_bool(
            cache_raw.get("save_row_aligned_image_matrix"), "cache.save_row_aligned_image_matrix"
        ),
        save_unique_image_cache=_as_str(cache_raw.get("save_unique_image_cache"), "cache.save_unique_image_cache"),
        sample_index_filename=_as_str(cache_raw.get("sample_index_filename"), "cache.sample_index_filename"),
        meta_filename=_as_str(cache_raw.get("meta_filename"), "cache.meta_filename"),
    )

    contracts = ContractConfig(
        sample_index_basis=_as_str(contracts_raw.get("sample_index_basis"), "contracts.sample_index_basis"),
        forbid_row_index_as_position=_as_bool(
            contracts_raw.get("forbid_row_index_as_position"), "contracts.forbid_row_index_as_position"
        ),
        empty_text_policy=_as_str(contracts_raw.get("empty_text_policy"), "contracts.empty_text_policy"),
        default_nus_input_view=_as_str(contracts_raw.get("default_nus_input_view"), "contracts.default_nus_input_view"),
    )

    parsed_views: dict[str, DatasetViews] = {}
    for dataset in SUPPORTED_DATASETS:
        if dataset not in data_views_raw:
            raise ValueError(f"Missing `data_views.{dataset}` in config")
        ds_obj = data_views_raw[dataset]
        if not isinstance(ds_obj, dict):
            raise ValueError(f"`data_views.{dataset}` must be mapping")
        default_view = _as_str(ds_obj.get("default_view"), f"data_views.{dataset}.default_view")
        views_obj = ds_obj.get("views")
        if not isinstance(views_obj, dict) or not views_obj:
            raise ValueError(f"`data_views.{dataset}.views` must be non-empty mapping")
        resolved: dict[str, Path] = {}
        for view_name, raw_path in views_obj.items():
            view_key = _as_str(view_name, f"data_views.{dataset}.views key")
            rel_path = _as_str(raw_path, f"data_views.{dataset}.views.{view_name}")
            resolved[view_key] = project_root / rel_path
        if default_view not in resolved:
            raise ValueError(
                f"default_view `{default_view}` not found in data_views.{dataset}.views"
            )
        parsed_views[dataset] = DatasetViews(default_view=default_view, views=resolved)

    return FeatureExtractionConfig(
        project_root=project_root,
        processed_root=processed_root,
        runtime=runtime,
        encoder=encoder,
        cache=cache,
        contracts=contracts,
        data_views=parsed_views,
    )
