from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    processed_root: Path
    outputs_root: Path
    dry_run_default: bool
    strict_validation: bool
    log_level: str
    drop_policies: dict[str, bool]


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    expected_total_rows: int
    expected_clean_rows_after_preconfirmed_ambiguous_drop: int
    canonical_sources: dict[str, str]
    disabled_sources: list[str]
    preconfirmed_ambiguous_ids: set[str]
    safe_duplicate_resolution_policy: str
    drop_ambiguous_duplicate_ids: bool
    filtering: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root object in {path}: expected mapping")
    return data


def load_runtime_config(project_root: Path, config_path: Path) -> RuntimeConfig:
    raw = _load_yaml(config_path)
    output = raw.get("output", {})
    run = raw.get("run", {})
    drop = raw.get("drop_policies", {})

    processed_root = project_root / str(output.get("processed_root", "data/processed"))
    outputs_root = project_root / str(output.get("outputs_root", "outputs"))
    return RuntimeConfig(
        processed_root=processed_root,
        outputs_root=outputs_root,
        dry_run_default=bool(run.get("dry_run_default", False)),
        strict_validation=bool(run.get("strict_validation", True)),
        log_level=str(run.get("log_level", "INFO")).upper(),
        drop_policies={
            "drop_missing_image": bool(drop.get("drop_missing_image", True)),
            "drop_parse_failure": bool(drop.get("drop_parse_failure", True)),
            "drop_ambiguous_duplicate_ids": bool(drop.get("drop_ambiguous_duplicate_ids", True)),
        },
    )


def load_dataset_config(dataset_config_path: Path) -> DatasetConfig:
    raw = _load_yaml(dataset_config_path)
    canonical_sources = raw.get("canonical_sources", {})
    if not isinstance(canonical_sources, dict):
        raise ValueError("`canonical_sources` must be a mapping")
    return DatasetConfig(
        dataset_name=str(raw.get("dataset_name", "nuswide")),
        expected_total_rows=int(raw["expected_total_rows"]),
        expected_clean_rows_after_preconfirmed_ambiguous_drop=int(
            raw["expected_clean_rows_after_preconfirmed_ambiguous_drop"]
        ),
        canonical_sources={str(k): str(v) for k, v in canonical_sources.items()},
        disabled_sources=[str(x) for x in raw.get("disabled_sources", [])],
        preconfirmed_ambiguous_ids={str(x) for x in raw.get("preconfirmed_ambiguous_ids", [])},
        safe_duplicate_resolution_policy=str(raw["safe_duplicate_resolution_policy"]),
        drop_ambiguous_duplicate_ids=bool(raw.get("drop_ambiguous_duplicate_ids", True)),
        filtering={
            str(k): v for k, v in (raw.get("filtering", raw.get("paper_alignment", {})) or {}).items()
        },
    )
