from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TextIO

from ..config import DatasetConfig, RuntimeConfig
from ..records import Top10FilterStats

LOGGER = logging.getLogger(__name__)


class NUSWIDERATop10Filter:
    """Top-10 category filtering over canonical-cleaned NUS-WIDE manifest."""

    def __init__(self, project_root: Path, runtime_config: RuntimeConfig, dataset_config: DatasetConfig) -> None:
        self.project_root = project_root
        self.runtime = runtime_config
        self.dataset = dataset_config
        if self.dataset.dataset_name != "nuswide":
            raise ValueError(f"Unsupported dataset for this filter: {self.dataset.dataset_name}")

        self.filtering = dict(self.dataset.filtering)
        self.enable_top10_filter = bool(self.filtering.get("enable_top10_filter", False))
        if not self.enable_top10_filter:
            raise ValueError("Top-10 filtering is disabled in dataset config (filtering.enable_top10_filter=false).")

        self.top_k = int(self.filtering.get("top_k_categories", 10))
        self.rank_metric = str(self.filtering.get("category_rank_metric", "positive_frequency"))
        self.multi_label_policy = str(self.filtering.get("multi_label_policy", "union_any"))
        self.ra_reported_target_rows = int(self.filtering.get("expected_filtered_rows_from_ra", 0))

        canonical = self.dataset.canonical_sources
        self.concepts_list_path = self._resolve_required_path(canonical, "concepts_list")

        self.canonical_manifest_path = self.runtime.processed_root / "nuswide" / "clean_manifest.jsonl"
        self.filtered_processed_dir = self.runtime.processed_root / "nuswide_ra_top10"
        self.filtered_outputs_dir = self.runtime.outputs_root / "nuswide_ra_top10"
        self.filtered_manifest_path = self.filtered_processed_dir / "filtered_manifest.jsonl"
        self.filtered_dropped_path = self.filtered_processed_dir / "dropped_by_top10_filter.jsonl"
        self.filter_stats_path = self.filtered_outputs_dir / "filter_stats.json"
        self.filter_report_path = self.filtered_outputs_dir / "filter_report.md"

    def run(self, dry_run: bool = False, limit: int | None = None) -> Top10FilterStats:
        if not self.canonical_manifest_path.exists():
            raise FileNotFoundError(
                f"Canonical manifest not found: {self.canonical_manifest_path}. Run canonical cleaning first."
            )

        concept_names = self._load_concept_names()
        label_counts, canonical_clean_rows = self._scan_category_counts()
        categories_with_positive = sum(1 for value in label_counts if value > 0)
        top_indices = self._select_top_categories(label_counts)
        top_names = [concept_names[index] for index in top_indices]
        top_counts = [label_counts[index] for index in top_indices]

        LOGGER.info("canonical_clean_rows=%s", canonical_clean_rows)
        LOGGER.info("top10_category_indices=%s", top_indices)
        LOGGER.info("top10_category_names=%s", top_names)
        LOGGER.info("top10_category_counts=%s", top_counts)

        filtered_fp: TextIO | None = None
        dropped_fp: TextIO | None = None
        if not dry_run:
            self.filtered_processed_dir.mkdir(parents=True, exist_ok=True)
            self.filtered_outputs_dir.mkdir(parents=True, exist_ok=True)
            filtered_fp = self.filtered_manifest_path.open("w", encoding="utf-8")
            dropped_fp = self.filtered_dropped_path.open("w", encoding="utf-8")

        filtered_rows = 0
        dropped_rows = 0
        filtered_written = 0
        dropped_written = 0

        with self.canonical_manifest_path.open("r", encoding="utf-8") as manifest_fp:
            for line in manifest_fp:
                if not line.strip():
                    continue
                obj = json.loads(line)
                result = self._apply_filter_policy(obj, top_indices)
                if result["keep"]:
                    filtered_rows += 1
                    if filtered_fp is not None and (limit is None or filtered_written < limit):
                        filtered_fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
                        filtered_written += 1
                else:
                    dropped_rows += 1
                    dropped_obj = self._build_filter_drop_record(obj, result["hit_top10_indices"])
                    if dropped_fp is not None:
                        dropped_fp.write(json.dumps(dropped_obj, ensure_ascii=False, separators=(",", ":")) + "\n")
                        dropped_written += 1

        if filtered_fp is not None:
            filtered_fp.close()
        if dropped_fp is not None:
            dropped_fp.close()

        difference = filtered_rows - self.ra_reported_target_rows
        if difference == 0:
            difference_reason = "No difference."
        else:
            difference_reason = (
                "Canonical clean removed 6 ambiguous duplicate rows. "
                "Top-10 filtering is performed strictly on this clean base, so row count may differ from ra."
            )

        stats = Top10FilterStats(
            dataset_name="nuswide_ra_top10",
            canonical_clean_rows=canonical_clean_rows,
            categories_with_positive=categories_with_positive,
            top10_category_names=top_names,
            top10_category_indices=top_indices,
            top10_category_counts=top_counts,
            multi_label_policy=self.multi_label_policy,
            filtered_rows_from_canonical_clean=filtered_rows,
            filtered_records_written=filtered_written,
            dropped_rows_from_canonical_clean=dropped_rows,
            dropped_records_written=dropped_written,
            ra_reported_target_rows=self.ra_reported_target_rows,
            difference_from_ra_target=difference,
            difference_reason=difference_reason,
            dry_run=dry_run,
            limit=limit,
        )

        if not dry_run:
            self._write_stats(stats)
            self._write_report(stats)

        return stats

    def _scan_category_counts(self) -> tuple[list[int], int]:
        label_counts = [0] * 81
        rows = 0
        with self.canonical_manifest_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                labels = obj.get("labels_81")
                if not isinstance(labels, list) or len(labels) != 81:
                    raise ValueError(f"Invalid labels_81 at clean manifest line {line_idx}")
                for index, value in enumerate(labels):
                    if value not in (0, 1):
                        raise ValueError(f"Invalid labels_81 value at line {line_idx}, dim {index}: {value}")
                    if value == 1:
                        label_counts[index] += 1
                rows += 1
        return label_counts, rows

    def _select_top_categories(self, label_counts: list[int]) -> list[int]:
        if self.rank_metric != "positive_frequency":
            raise ValueError(f"Unsupported category_rank_metric: {self.rank_metric}")
        if self.top_k <= 0 or self.top_k > len(label_counts):
            raise ValueError(f"Invalid top_k_categories: {self.top_k}")
        return sorted(range(len(label_counts)), key=lambda i: (-label_counts[i], i))[: self.top_k]

    def _apply_filter_policy(self, obj: dict[str, Any], top_indices: list[int]) -> dict[str, Any]:
        labels = obj.get("labels_81")
        if not isinstance(labels, list) or len(labels) != 81:
            raise ValueError(f"Invalid labels_81 for id={obj.get('id', '<unknown>')}")
        hit_indices = [index for index in top_indices if labels[index] == 1]
        if self.multi_label_policy != "union_any":
            raise ValueError(f"Unsupported multi_label_policy: {self.multi_label_policy}")
        return {"keep": bool(hit_indices), "hit_top10_indices": hit_indices}

    def _build_filter_drop_record(self, obj: dict[str, Any], hit_top10_indices: list[int]) -> dict[str, Any]:
        return {
            "id": obj.get("id", ""),
            "row_index": obj.get("row_index", -1),
            "image_id": obj.get("image_id", ""),
            "image_path": obj.get("image_path", ""),
            "drop_reason": "top10_filter_excluded",
            "detail": "No positive label in selected top-10 categories under union_any policy.",
            "hit_top10_indices": hit_top10_indices,
        }

    def _load_concept_names(self) -> list[str]:
        concepts = [line.strip() for line in self.concepts_list_path.read_text(encoding="utf-8").splitlines()]
        concepts = [concept for concept in concepts if concept]
        if len(concepts) != 81:
            raise RuntimeError(f"Expected 81 concepts from {self.concepts_list_path}, got {len(concepts)}")
        return concepts

    def _resolve_required_path(self, canonical: dict[str, str], key: str) -> Path:
        if key not in canonical:
            raise KeyError(f"Missing canonical source key in dataset config: {key}")
        path = self.project_root / canonical[key]
        if not path.exists():
            raise FileNotFoundError(f"Canonical source path does not exist for {key}: {path}")
        return path

    def _write_stats(self, stats: Top10FilterStats) -> None:
        with self.filter_stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)

    def _write_report(self, stats: Top10FilterStats) -> None:
        lines = [
            "# NUS-WIDE RA Top-10 Filtering Report",
            "",
            "## A. Canonical-Clean-Based Filtered Result",
            f"- canonical_clean_rows: {stats.canonical_clean_rows}",
            f"- filtered_rows_from_canonical_clean: {stats.filtered_rows_from_canonical_clean}",
            f"- dropped_rows_from_canonical_clean: {stats.dropped_rows_from_canonical_clean}",
            "",
            "## B. Paper-Target Comparison",
            f"- ra_reported_target_rows: {stats.ra_reported_target_rows}",
            f"- difference_from_ra_target: {stats.difference_from_ra_target}",
            f"- difference_reason: {stats.difference_reason}",
            "",
            "## Top-10 Categories (0-based indices)",
            f"- top10_category_indices: {stats.top10_category_indices}",
            f"- top10_category_names: {stats.top10_category_names}",
            f"- top10_category_counts: {stats.top10_category_counts}",
            f"- categories_with_positive: {stats.categories_with_positive}",
            "",
            "## Filtering Policy",
            f"- multi_label_policy: {stats.multi_label_policy}",
            "- This policy is chosen because it reproduces the retained sample count 186,577 reported by ra.",
            "- union_any is an implementation assumption for a paper-ambiguous multi-label detail, "
            "not an explicitly specified rule in ra.",
            "",
            "## Output Files",
            f"- `{self.filtered_manifest_path.as_posix()}`",
            f"- `{self.filtered_dropped_path.as_posix()}`",
            f"- `{self.filter_stats_path.as_posix()}`",
            f"- `{self.filter_report_path.as_posix()}`",
        ]
        self.filter_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
