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
    seed: int
    block_rows: int
    dense_debug_mode: bool
    dense_debug_max_rows: int
    exp_clip_value: float


@dataclass(frozen=True)
class TopKConfig:
    k_candidate: int
    k_final: int


@dataclass(frozen=True)
class SemanticParams:
    alpha1: float
    alpha2: float
    alpha3: float
    beta: float
    tau: float
    lambda_self_loop: float


@dataclass(frozen=True)
class SemanticConfig:
    formula_version: str
    pipeline_mode: str
    debug_save_intermediates: bool
    softmax_domain: str
    symmetrization_rule: str
    sparsification_strategy: str
    params: SemanticParams
    topk_by_dataset: dict[str, TopKConfig]


@dataclass(frozen=True)
class PseudoConfig:
    source_mode: str
    z_source: str
    clustering_method: str
    n_clusters: int | None
    pseudo_seed: int
    external_matrix_path: Path | None


@dataclass(frozen=True)
class ValidationConfig:
    row_sum_atol: float
    symmetry_atol: float
    formula_atol: float


@dataclass(frozen=True)
class SemanticSimilarityConfig:
    project_root: Path
    processed_root: Path
    runtime: RuntimeConfig
    semantic: SemanticConfig
    pseudo: PseudoConfig
    validation: ValidationConfig

    def resolve_feature_cache_dir(self, dataset: str, feature_set_id: str) -> Path:
        return self.processed_root / dataset / "feature_cache" / feature_set_id

    def resolve_semantic_cache_dir(self, dataset: str, semantic_set_id: str) -> Path:
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


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"`{name}` must be float")


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"`{name}` must be bool")


def _as_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise ValueError(f"`{name}` must be int or null")


def _as_optional_path(value: Any, name: str, project_root: Path) -> Path | None:
    if value is None:
        return None
    raw = _as_str(value, name)
    p = Path(raw)
    if not p.is_absolute():
        p = project_root / p
    return p


def _parse_topk_by_dataset(topk_raw: dict[str, Any]) -> dict[str, TopKConfig]:
    parsed: dict[str, TopKConfig] = {}
    for dataset in SUPPORTED_DATASETS:
        ds_obj = topk_raw.get(dataset)
        if not isinstance(ds_obj, dict):
            raise ValueError(f"Missing `semantic.topk.{dataset}` mapping")
        k_candidate = _as_int(ds_obj.get("k_candidate"), f"semantic.topk.{dataset}.k_candidate")
        k_final = _as_int(ds_obj.get("k_final"), f"semantic.topk.{dataset}.k_final")
        if k_candidate < k_final:
            raise ValueError(
                f"semantic.topk.{dataset}: k_candidate ({k_candidate}) must be >= k_final ({k_final})"
            )
        if k_final <= 0:
            raise ValueError(f"semantic.topk.{dataset}.k_final must be > 0")
        parsed[dataset] = TopKConfig(k_candidate=k_candidate, k_final=k_final)
    return parsed


def load_semantic_similarity_config(project_root: Path, config_path: Path) -> SemanticSimilarityConfig:
    raw = _load_yaml(config_path)

    output_raw = _require_mapping(raw, "output")
    runtime_raw = _require_mapping(raw, "runtime")
    semantic_raw = _require_mapping(raw, "semantic")
    pseudo_raw = _require_mapping(raw, "pseudo")
    validation_raw = _require_mapping(raw, "validation")

    processed_root = project_root / _as_str(output_raw.get("processed_root"), "output.processed_root")

    runtime = RuntimeConfig(
        device=_as_str(runtime_raw.get("device"), "runtime.device"),
        dtype=_as_str(runtime_raw.get("dtype"), "runtime.dtype"),
        seed=_as_int(runtime_raw.get("seed"), "runtime.seed"),
        block_rows=_as_int(runtime_raw.get("block_rows"), "runtime.block_rows"),
        dense_debug_mode=_as_bool(runtime_raw.get("dense_debug_mode"), "runtime.dense_debug_mode"),
        dense_debug_max_rows=_as_int(runtime_raw.get("dense_debug_max_rows"), "runtime.dense_debug_max_rows"),
        exp_clip_value=_as_float(runtime_raw.get("exp_clip_value"), "runtime.exp_clip_value"),
    )

    params_raw = _require_mapping(semantic_raw, "params")
    topk_raw = _require_mapping(semantic_raw, "topk")
    semantic = SemanticConfig(
        formula_version=_as_str(semantic_raw.get("formula_version"), "semantic.formula_version"),
        pipeline_mode=_as_str(semantic_raw.get("pipeline_mode"), "semantic.pipeline_mode"),
        debug_save_intermediates=_as_bool(
            semantic_raw.get("debug_save_intermediates"), "semantic.debug_save_intermediates"
        ),
        softmax_domain=_as_str(semantic_raw.get("softmax_domain"), "semantic.softmax_domain"),
        symmetrization_rule=_as_str(semantic_raw.get("symmetrization_rule"), "semantic.symmetrization_rule"),
        sparsification_strategy=_as_str(
            semantic_raw.get("sparsification_strategy"), "semantic.sparsification_strategy"
        ),
        params=SemanticParams(
            alpha1=_as_float(params_raw.get("alpha1"), "semantic.params.alpha1"),
            alpha2=_as_float(params_raw.get("alpha2"), "semantic.params.alpha2"),
            alpha3=_as_float(params_raw.get("alpha3"), "semantic.params.alpha3"),
            beta=_as_float(params_raw.get("beta"), "semantic.params.beta"),
            tau=_as_float(params_raw.get("tau"), "semantic.params.tau"),
            # Config keeps mathematical key `lambda`; code uses `lambda_self_loop` to avoid Python keyword.
            lambda_self_loop=_as_float(params_raw.get("lambda"), "semantic.params.lambda"),
        ),
        topk_by_dataset=_parse_topk_by_dataset(topk_raw),
    )

    if semantic.pipeline_mode not in {"high_only", "with_pseudo"}:
        raise ValueError("semantic.pipeline_mode must be `high_only` or `with_pseudo`")
    if semantic.softmax_domain not in {"candidate_subgraph", "full_row"}:
        raise ValueError("semantic.softmax_domain must be `candidate_subgraph` or `full_row`")
    if semantic.symmetrization_rule != "undirected_union_arithmetic_mean":
        raise ValueError("semantic.symmetrization_rule must be `undirected_union_arithmetic_mean`")
    if semantic.sparsification_strategy != "two_stage_topk":
        raise ValueError("semantic.sparsification_strategy must be `two_stage_topk`")
    if semantic.params.tau <= 0:
        raise ValueError("semantic.params.tau must be > 0")

    pseudo = PseudoConfig(
        source_mode=_as_str(pseudo_raw.get("source_mode"), "pseudo.source_mode"),
        z_source=_as_str(pseudo_raw.get("z_source"), "pseudo.z_source"),
        clustering_method=_as_str(pseudo_raw.get("clustering_method"), "pseudo.clustering_method"),
        n_clusters=_as_optional_int(pseudo_raw.get("n_clusters"), "pseudo.n_clusters"),
        pseudo_seed=_as_int(pseudo_raw.get("pseudo_seed"), "pseudo.pseudo_seed"),
        external_matrix_path=_as_optional_path(
            pseudo_raw.get("external_matrix_path"), "pseudo.external_matrix_path", project_root
        ),
    )

    validation = ValidationConfig(
        row_sum_atol=_as_float(validation_raw.get("row_sum_atol"), "validation.row_sum_atol"),
        symmetry_atol=_as_float(validation_raw.get("symmetry_atol"), "validation.symmetry_atol"),
        formula_atol=_as_float(validation_raw.get("formula_atol"), "validation.formula_atol"),
    )

    return SemanticSimilarityConfig(
        project_root=project_root,
        processed_root=processed_root,
        runtime=runtime,
        semantic=semantic,
        pseudo=pseudo,
        validation=validation,
    )

