from __future__ import annotations

from dataclasses import dataclass, field
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
    expected_total_rows: int | None = None
    expected_clean_rows_after_preconfirmed_ambiguous_drop: int | None = None
    canonical_sources: dict[str, Any] = field(default_factory=dict)
    disabled_sources: list[str] = field(default_factory=list)
    preconfirmed_ambiguous_ids: set[str] = field(default_factory=set)
    safe_duplicate_resolution_policy: str = ""
    drop_ambiguous_duplicate_ids: bool = True
    filtering: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    non_canonical_sources: list[str] = field(default_factory=list)
    identity: dict[str, Any] = field(default_factory=dict)
    text: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    duplicate_policy: dict[str, Any] = field(default_factory=dict)
    drop_reasons: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    open_items: list[dict[str, Any] | str] = field(default_factory=list)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root object in {path}: expected mapping")
    return data


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        if text.upper().startswith("TO_BE_"):
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


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
    validation = raw.get("validation", {})
    if validation is None:
        validation = {}
    if not isinstance(validation, dict):
        raise ValueError("`validation` must be a mapping when provided")

    expected_total_rows = _optional_int(raw.get("expected_total_rows"))
    if expected_total_rows is None:
        expected_total_rows = _optional_int(validation.get("expected_total_rows"))

    expected_clean_rows = _optional_int(raw.get("expected_clean_rows_after_preconfirmed_ambiguous_drop"))
    if expected_clean_rows is None:
        expected_clean_rows = _optional_int(validation.get("expected_clean_rows"))

    return DatasetConfig(
        dataset_name=str(raw.get("dataset_name", "nuswide")),
        expected_total_rows=expected_total_rows,
        expected_clean_rows_after_preconfirmed_ambiguous_drop=expected_clean_rows,
        canonical_sources={str(k): v for k, v in canonical_sources.items()},
        disabled_sources=[str(x) for x in raw.get("disabled_sources", [])],
        preconfirmed_ambiguous_ids={str(x) for x in raw.get("preconfirmed_ambiguous_ids", [])},
        safe_duplicate_resolution_policy=str(raw.get("safe_duplicate_resolution_policy", "")),
        drop_ambiguous_duplicate_ids=bool(raw.get("drop_ambiguous_duplicate_ids", True)),
        filtering={
            str(k): v for k, v in (raw.get("filtering", raw.get("paper_alignment", {})) or {}).items()
        },
        raw={str(k): v for k, v in (raw.get("raw", {}) or {}).items()},
        non_canonical_sources=[str(x) for x in raw.get("non_canonical_sources", [])],
        identity={str(k): v for k, v in (raw.get("identity", {}) or {}).items()},
        text={str(k): v for k, v in (raw.get("text", {}) or {}).items()},
        labels={str(k): v for k, v in (raw.get("labels", {}) or {}).items()},
        validation={str(k): v for k, v in validation.items()},
        duplicate_policy={str(k): v for k, v in (raw.get("duplicate_policy", {}) or {}).items()},
        drop_reasons={str(k): v for k, v in (raw.get("drop_reasons", {}) or {}).items()},
        outputs={str(k): v for k, v in (raw.get("outputs", {}) or {}).items()},
        open_items=list(raw.get("open_items", []) or []),
    )
