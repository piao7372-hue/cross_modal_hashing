from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import TextIO

from ..base_cleaner import BaseCleaner
from ..config import DatasetConfig, RuntimeConfig
from ..exporters import (
    DatasetOutputPaths,
    build_dataset_output_paths,
    ensure_output_dirs,
    write_json,
    write_jsonl,
)
from ..records import CleaningStats, DroppedSample
from ..reporting import build_cleaning_report_markdown

LOGGER = logging.getLogger(__name__)

IMAGE_NAME_PATTERN = re.compile(r"^im(?P<id>\d+)\.jpg$")


class MIRFlickr25KCleaner(BaseCleaner):
    """Canonical MIRFlickr25K cleaner aligned to current policy draft."""

    def __init__(
        self,
        project_root: Path,
        runtime_config: RuntimeConfig,
        dataset_config: DatasetConfig,
    ) -> None:
        self.project_root = project_root
        self.runtime = runtime_config
        self.dataset = dataset_config
        if self.dataset.dataset_name != "mirflickr25k":
            raise ValueError(f"Unsupported dataset for this cleaner: {self.dataset.dataset_name}")

        self.strict = self.runtime.strict_validation

        canonical = self.dataset.canonical_sources
        self.images_dir = self._resolve_required_dir(canonical, "image_root")
        self.text_dir = self._resolve_required_dir(canonical, "text_root")
        self.labels_dir = self._resolve_required_dir(canonical, "label_root")
        self.image_glob = str(canonical.get("image_glob", "im*.jpg"))
        self.label_glob = str(canonical.get("label_glob", "*.txt"))
        self.label_exclude = {str(x) for x in (canonical.get("label_exclude", []) or [])}
        self.annotation_source_version = str(canonical.get("annotation_source_version", ""))

        self.output_paths: DatasetOutputPaths = build_dataset_output_paths(
            runtime=self.runtime,
            dataset_name="mirflickr25k",
        )

    def run(self, dry_run: bool = False, limit: int | None = None) -> CleaningStats:
        potential_labels, relevant_labels_r1, annotation_duplicate_rows = self._load_annotations()

        clean_fp: TextIO | None = None
        drop_fp: TextIO | None = None
        if not dry_run:
            ensure_output_dirs(self.output_paths)
            clean_fp = self.output_paths.clean_manifest_path.open("w", encoding="utf-8")
            drop_fp = self.output_paths.dropped_samples_path.open("w", encoding="utf-8")

        actual_total_rows = 0
        actual_clean_rows = 0
        clean_records_written = 0
        dropped_records_written = 0
        parse_failure_count = 0
        drop_reason_breakdown: Counter[str] = Counter()
        seen_sample_ids: set[int] = set()

        def write_clean(sample: dict[str, object]) -> None:
            nonlocal clean_records_written
            if dry_run:
                return
            if limit is not None and clean_records_written >= limit:
                return
            write_jsonl(clean_fp, sample)
            clean_records_written += 1

        def write_drop(sample: DroppedSample) -> None:
            nonlocal dropped_records_written
            if dry_run:
                return
            write_jsonl(drop_fp, sample.to_dict())
            dropped_records_written += 1

        for image_path in sorted(self.images_dir.glob(self.image_glob), key=lambda p: p.name):
            actual_total_rows += 1
            row_index = actual_total_rows
            image_rel = self._to_rel(image_path)

            match = IMAGE_NAME_PATTERN.match(image_path.name)
            if not match:
                drop_reason_breakdown["invalid_image_id"] += 1
                write_drop(
                    DroppedSample(
                        id=f"mirflickr25k:{row_index}",
                        row_index=row_index,
                        image_id="",
                        candidate_image_paths=[image_rel],
                        drop_reason="invalid_image_id",
                        detail=f"Image filename does not match `im<ID>.jpg`: {image_path.name}",
                    )
                )
                continue

            sample_id = int(match.group("id"))
            if sample_id in seen_sample_ids:
                raise RuntimeError(f"Primary alignment key conflict for sample_id={sample_id}")
            seen_sample_ids.add(sample_id)

            text_path = self.text_dir / f"tags{sample_id}.txt"
            if not text_path.exists():
                drop_reason_breakdown["missing_text_file"] += 1
                write_drop(
                    DroppedSample(
                        id=f"mirflickr25k:{sample_id}",
                        row_index=row_index,
                        image_id=str(sample_id),
                        candidate_image_paths=[image_rel],
                        drop_reason="missing_text_file",
                        detail=f"Missing canonical text file: {self._to_rel(text_path)}",
                    )
                )
                continue

            try:
                text_raw = text_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                parse_failure_count += 1
                drop_reason_breakdown["text_parse_failure"] += 1
                write_drop(
                    DroppedSample(
                        id=f"mirflickr25k:{sample_id}",
                        row_index=row_index,
                        image_id=str(sample_id),
                        candidate_image_paths=[image_rel],
                        drop_reason="text_parse_failure",
                        detail=str(exc),
                    )
                )
                continue

            text_empty = text_raw == ""
            text_raw_tokens = [line for line in text_raw.splitlines() if line]
            potential = sorted(potential_labels.get(sample_id, set()))
            relevant = sorted(relevant_labels_r1.get(sample_id, set()))
            notes = [
                f"annotation_source_version={self.annotation_source_version}",
                f"potential_labels={','.join(potential)}",
                f"relevant_labels_r1={','.join(relevant)}",
            ]

            clean: dict[str, object] = {
                "id": f"mirflickr25k:{sample_id}",
                "dataset_name": "mirflickr25k",
                "row_index": row_index,
                "image_id": str(sample_id),
                "image_path": image_rel,
                "text_raw_tokens": text_raw_tokens,
                "text": text_raw,
                "text_empty": text_empty,
                "tags_1k": [],
                "labels_81": [],
                "notes": notes,
            }
            actual_clean_rows += 1
            write_clean(clean)

        if clean_fp is not None:
            clean_fp.close()
        if drop_fp is not None:
            drop_fp.close()

        expected_total_rows = (
            self.dataset.expected_total_rows
            if self.dataset.expected_total_rows is not None
            else actual_total_rows
        )
        expected_clean_rows = (
            self.dataset.expected_clean_rows_after_preconfirmed_ambiguous_drop
            if self.dataset.expected_clean_rows_after_preconfirmed_ambiguous_drop is not None
            else actual_clean_rows
        )

        if self.strict:
            if self.dataset.expected_total_rows is not None and actual_total_rows != expected_total_rows:
                raise RuntimeError(
                    f"Row count mismatch: expected {expected_total_rows}, got {actual_total_rows}"
                )
            if (
                self.dataset.expected_clean_rows_after_preconfirmed_ambiguous_drop is not None
                and actual_clean_rows != expected_clean_rows
            ):
                raise RuntimeError(
                    f"Clean row mismatch: expected {expected_clean_rows}, got {actual_clean_rows}"
                )

        stats = CleaningStats(
            dataset_name="mirflickr25k",
            expected_total_rows=expected_total_rows,
            actual_total_rows=actual_total_rows,
            expected_clean_rows_after_preconfirmed_ambiguous_drop=expected_clean_rows,
            actual_clean_rows=actual_clean_rows,
            clean_records_written=clean_records_written,
            dropped_records_written=dropped_records_written,
            safe_duplicate_ids=[],
            ambiguous_duplicate_ids=[],
            safe_duplicate_resolution_policy="primary_key_conflict_hard_fail",
            ambiguous_duplicate_drop_count=0,
            missing_image_count=0,
            parse_failure_count=parse_failure_count,
            drop_reason_breakdown=dict(sorted(drop_reason_breakdown.items())),
            strict_validation=self.strict,
            dry_run=dry_run,
            limit=limit,
        )

        if not dry_run:
            self._write_stats(stats)
            self._write_report(stats, annotation_duplicate_rows)

        return stats

    def _load_annotations(self) -> tuple[dict[int, set[str]], dict[int, set[str]], int]:
        label_paths = [
            path
            for path in sorted(self.labels_dir.glob(self.label_glob), key=lambda p: p.name)
            if path.name not in self.label_exclude
        ]
        if not label_paths:
            raise RuntimeError(
                f"No annotation files matched at {self.labels_dir} with glob `{self.label_glob}`"
            )

        potential: dict[int, set[str]] = defaultdict(set)
        relevant: dict[int, set[str]] = defaultdict(set)
        duplicate_rows = 0

        for label_path in label_paths:
            label_name = label_path.stem
            target = relevant if label_name.endswith("_r1") else potential
            seen_in_file: set[int] = set()

            with label_path.open("r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f, start=1):
                    token = line.strip()
                    if token == "":
                        continue
                    if not token.isdigit():
                        raise RuntimeError(
                            f"Annotation parse failure at {label_path.name}:{line_idx} token=`{token}`"
                        )
                    sample_id = int(token)
                    if sample_id in seen_in_file:
                        duplicate_rows += 1
                        continue
                    seen_in_file.add(sample_id)
                    target[sample_id].add(label_name)

        return dict(potential), dict(relevant), duplicate_rows

    def _resolve_required_dir(self, canonical: dict[str, object], key: str) -> Path:
        if key not in canonical:
            raise KeyError(f"Missing canonical source key in dataset config: {key}")
        value = canonical[key]
        if not isinstance(value, str):
            raise ValueError(f"Canonical source `{key}` must be a string path")
        path = self.project_root / value
        if not path.exists():
            raise FileNotFoundError(f"Canonical source path does not exist for {key}: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Canonical source path for {key} is not a directory: {path}")
        return path

    def _write_stats(self, stats: CleaningStats) -> None:
        write_json(self.output_paths.clean_stats_path, stats.to_dict())

    def _write_report(self, stats: CleaningStats, annotation_duplicate_rows: int) -> None:
        canonical_sources = {str(k): str(v) for k, v in self.dataset.canonical_sources.items()}
        disabled_sources = [str(x) for x in self.dataset.non_canonical_sources]
        notes = [
            "MIRFlickr canonical text source is meta/tags_raw.",
            "MIRFlickr annotation source is mirflickr25k_annotations_v080 (version marker v080).",
            "Annotations use parallel namespaces: potential_labels and relevant_labels_r1.",
            f"annotation_duplicate_rows_deduplicated: {annotation_duplicate_rows}",
        ]
        report = build_cleaning_report_markdown(
            stats=stats,
            canonical_sources=canonical_sources,
            disabled_sources=disabled_sources,
            notes=notes,
        )
        self.output_paths.cleaning_report_path.write_text(report, encoding="utf-8")

    def _to_rel(self, path: Path) -> str:
        return path.relative_to(self.project_root).as_posix()
