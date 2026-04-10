from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvaluationScope:
    canonical_only: bool
    allowed_splits: tuple[str, ...]
    forbidden_splits: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationGrouping:
    image_unit: str
    text_unit: str


@dataclass(frozen=True)
class EvaluationRelevance:
    rule: str


@dataclass(frozen=True)
class EvaluationEmptyTextPolicy:
    allow_in_query: bool
    allow_in_database: bool


@dataclass(frozen=True)
class EvaluationMethodGuards:
    backbone: str
    supervision_target: str
    forbid_supervision_matrices: tuple[str, ...]
    grl_scope: str


@dataclass(frozen=True)
class EvaluationReport:
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class FrozenEvaluationProtocol:
    source_path: Path
    contract_version: str
    protocol_id: str
    dataset: str
    scope: EvaluationScope
    grouping: EvaluationGrouping
    relevance: EvaluationRelevance
    empty_text: EvaluationEmptyTextPolicy
    method_guards: EvaluationMethodGuards
    report: EvaluationReport

    def snapshot(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "protocol_id": self.protocol_id,
            "dataset": self.dataset,
            "scope": {
                "canonical_only": self.scope.canonical_only,
                "allowed_splits": list(self.scope.allowed_splits),
                "forbidden_splits": list(self.scope.forbidden_splits),
            },
            "grouping": {
                "image_unit": self.grouping.image_unit,
                "text_unit": self.grouping.text_unit,
            },
            "relevance": {
                "rule": self.relevance.rule,
            },
            "empty_text": {
                "allow_in_query": self.empty_text.allow_in_query,
                "allow_in_database": self.empty_text.allow_in_database,
            },
            "method_guards": {
                "backbone": self.method_guards.backbone,
                "supervision_target": self.method_guards.supervision_target,
                "forbid_supervision_matrices": list(self.method_guards.forbid_supervision_matrices),
                "grl_scope": self.method_guards.grl_scope,
            },
            "report": {
                "metrics": list(self.report.metrics),
            },
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation protocol file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Evaluation protocol root must be mapping: {path}")
    return data


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Evaluation protocol `{key}` must be mapping")
    return value


def _require_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if isinstance(value, str) and value:
        return value
    raise RuntimeError(f"Evaluation protocol `{key}` must be non-empty string")


def _require_bool(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if isinstance(value, bool):
        return value
    raise RuntimeError(f"Evaluation protocol `{key}` must be bool")


def _require_str_tuple(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise RuntimeError(f"Evaluation protocol `{key}` must be non-empty list[str]")
    return tuple(value)


def load_frozen_evaluation_protocol(project_root: Path, config_path: Path) -> FrozenEvaluationProtocol:
    path = config_path if config_path.is_absolute() else (project_root / config_path)
    raw = _load_yaml(path.resolve())

    scope_raw = _require_mapping(raw, "scope")
    grouping_raw = _require_mapping(raw, "grouping")
    relevance_raw = _require_mapping(raw, "relevance")
    empty_text_raw = _require_mapping(raw, "empty_text")
    method_guards_raw = _require_mapping(raw, "method_guards")
    report_raw = _require_mapping(raw, "report")

    protocol = FrozenEvaluationProtocol(
        source_path=path.resolve(),
        contract_version=_require_str(raw, "contract_version"),
        protocol_id=_require_str(raw, "protocol_id"),
        dataset=_require_str(raw, "dataset"),
        scope=EvaluationScope(
            canonical_only=_require_bool(scope_raw, "canonical_only"),
            allowed_splits=_require_str_tuple(scope_raw, "allowed_splits"),
            forbidden_splits=_require_str_tuple(scope_raw, "forbidden_splits"),
        ),
        grouping=EvaluationGrouping(
            image_unit=_require_str(grouping_raw, "image_unit"),
            text_unit=_require_str(grouping_raw, "text_unit"),
        ),
        relevance=EvaluationRelevance(rule=_require_str(relevance_raw, "rule")),
        empty_text=EvaluationEmptyTextPolicy(
            allow_in_query=_require_bool(empty_text_raw, "allow_in_query"),
            allow_in_database=_require_bool(empty_text_raw, "allow_in_database"),
        ),
        method_guards=EvaluationMethodGuards(
            backbone=_require_str(method_guards_raw, "backbone"),
            supervision_target=_require_str(method_guards_raw, "supervision_target"),
            forbid_supervision_matrices=_require_str_tuple(method_guards_raw, "forbid_supervision_matrices"),
            grl_scope=_require_str(method_guards_raw, "grl_scope"),
        ),
        report=EvaluationReport(metrics=_require_str_tuple(report_raw, "metrics")),
    )

    if protocol.contract_version != "sch_kahn_evaluation_protocol_v1":
        raise RuntimeError("Evaluation protocol contract_version must be `sch_kahn_evaluation_protocol_v1`")
    if protocol.protocol_id != "mscoco_val2014_unique_image_caption_same_image_v1":
        raise RuntimeError("Evaluation protocol_id must stay frozen to the approved MSCOCO val2014 v1 protocol")
    if protocol.dataset != "mscoco":
        raise RuntimeError("Evaluation protocol dataset must be `mscoco`")
    if protocol.scope.canonical_only is not True:
        raise RuntimeError("Evaluation protocol scope.canonical_only must be true")
    if protocol.scope.allowed_splits != ("val2014",):
        raise RuntimeError("Evaluation protocol allowed_splits must be exactly [`val2014`]")
    if set(protocol.scope.forbidden_splits) != {"train2014", "test2014"}:
        raise RuntimeError("Evaluation protocol forbidden_splits must be exactly `train2014` and `test2014`")
    if protocol.grouping.image_unit != "unique_image_first_sample_index":
        raise RuntimeError("Evaluation protocol image_unit must be `unique_image_first_sample_index`")
    if protocol.grouping.text_unit != "unique_caption_per_row":
        raise RuntimeError("Evaluation protocol text_unit must be `unique_caption_per_row`")
    if protocol.relevance.rule != "same_split_and_image_id":
        raise RuntimeError("Evaluation protocol relevance.rule must be `same_split_and_image_id`")
    if protocol.empty_text.allow_in_query or protocol.empty_text.allow_in_database:
        raise RuntimeError("Evaluation protocol empty_text policy must forbid empty text in query and database")
    if protocol.method_guards.backbone != "zout_to_v_to_b":
        raise RuntimeError("Evaluation protocol method_guards.backbone must be `zout_to_v_to_b`")
    if protocol.method_guards.supervision_target != "S_final":
        raise RuntimeError("Evaluation protocol method_guards.supervision_target must be `S_final`")
    if set(protocol.method_guards.forbid_supervision_matrices) != {"S_high", "S_graph"}:
        raise RuntimeError(
            "Evaluation protocol method_guards.forbid_supervision_matrices must be exactly `S_high` and `S_graph`"
        )
    if protocol.method_guards.grl_scope != "zout_only":
        raise RuntimeError("Evaluation protocol method_guards.grl_scope must be `zout_only`")
    if protocol.report.metrics != ("i2t_map", "t2i_map", "avg_map"):
        raise RuntimeError("Evaluation protocol report.metrics must be exactly i2t_map, t2i_map, avg_map")

    return protocol
