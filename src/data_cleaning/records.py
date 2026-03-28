from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CleanSample:
    id: str
    dataset_name: str
    row_index: int
    image_id: str
    image_path: str
    text_raw_tokens: list[str]
    text: str
    tags_1k: list[int]
    labels_81: list[int]
    notes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DroppedSample:
    id: str
    row_index: int
    image_id: str
    candidate_image_paths: list[str]
    drop_reason: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleaningStats:
    dataset_name: str
    expected_total_rows: int
    actual_total_rows: int
    expected_clean_rows_after_preconfirmed_ambiguous_drop: int
    actual_clean_rows: int
    clean_records_written: int
    dropped_records_written: int
    safe_duplicate_ids: list[str] = field(default_factory=list)
    ambiguous_duplicate_ids: list[str] = field(default_factory=list)
    safe_duplicate_resolution_policy: str = ""
    ambiguous_duplicate_drop_count: int = 0
    missing_image_count: int = 0
    parse_failure_count: int = 0
    drop_reason_breakdown: dict[str, int] = field(default_factory=dict)
    strict_validation: bool = True
    dry_run: bool = False
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Top10FilterStats:
    dataset_name: str
    canonical_clean_rows: int
    categories_with_positive: int
    top10_category_names: list[str]
    top10_category_indices: list[int]
    top10_category_counts: list[int]
    multi_label_policy: str
    filtered_rows_from_canonical_clean: int
    filtered_records_written: int
    dropped_rows_from_canonical_clean: int
    dropped_records_written: int
    ra_reported_target_rows: int
    difference_from_ra_target: int
    difference_reason: str
    dry_run: bool
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
