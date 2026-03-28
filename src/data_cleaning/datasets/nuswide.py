from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
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
from ..records import CleanSample, CleaningStats, DroppedSample
from ..reporting import build_cleaning_report_markdown

LOGGER = logging.getLogger(__name__)

IMAGE_NAME_PATTERN = re.compile(r"^\d{4}_(\d+)\.jpg$")


@dataclass
class _MetadataRow:
    row_index: int
    image_id: str
    text_raw_tokens: list[str]
    text: str
    tags_81: list[int]
    tags_1k: list[int]
    labels_81: list[int]


class NUSWIDECleaner(BaseCleaner):
    """Canonical NUS-WIDE cleaner based on docs/nuswide_alignment_policy.md."""

    def __init__(
        self,
        project_root: Path,
        runtime_config: RuntimeConfig,
        dataset_config: DatasetConfig,
    ) -> None:
        self.project_root = project_root
        self.runtime = runtime_config
        self.dataset = dataset_config
        if self.dataset.dataset_name != "nuswide":
            raise ValueError(f"Unsupported dataset for this cleaner: {self.dataset.dataset_name}")

        self.strict = self.runtime.strict_validation

        canonical = self.dataset.canonical_sources
        self.all_tags_path = self._resolve_required_path(canonical, "all_tags")
        self.all_tags_81_path = self._resolve_required_path(canonical, "all_tags_81")
        self.all_tags_1k_path = self._resolve_required_path(canonical, "all_tags_1k")
        self.all_labels_dir = self._resolve_required_path(canonical, "all_labels_dir")
        self.concepts_list_path = self._resolve_required_path(canonical, "concepts_list")
        self.images_dir = self._resolve_required_path(canonical, "images_dir")
        self.labels_pattern = canonical.get("labels_pattern", "Labels_*.txt")

        self.output_paths: DatasetOutputPaths = build_dataset_output_paths(
            runtime=self.runtime,
            dataset_name="nuswide",
        )

    def run(self, dry_run: bool = False, limit: int | None = None) -> CleaningStats:
        id_counts, all_tags_total_rows = self._scan_all_tags_id_counts()
        image_index = self._build_image_index()
        concept_names = self._load_concept_names()
        label_file_paths = self._load_label_files(concept_names)
        duplicate_ids = {image_id for image_id, count in id_counts.items() if count > 1}

        LOGGER.info("NUS-WIDE rows in All_Tags: %s", all_tags_total_rows)
        LOGGER.info("Unique image ids in All_Tags: %s", len(id_counts))
        LOGGER.info("Duplicate image ids in metadata: %s", len(duplicate_ids))

        clean_fp: TextIO | None = None
        drop_fp: TextIO | None = None
        if not dry_run:
            ensure_output_dirs(self.output_paths)
            clean_fp = self.output_paths.clean_manifest_path.open("w", encoding="utf-8")
            drop_fp = self.output_paths.dropped_samples_path.open("w", encoding="utf-8")

        expected_total_rows = self.dataset.expected_total_rows
        expected_clean_rows = self.dataset.expected_clean_rows_after_preconfirmed_ambiguous_drop
        preconfirmed_ambiguous = set(self.dataset.preconfirmed_ambiguous_ids)

        actual_total_rows = 0
        actual_clean_rows = 0
        clean_records_written = 0
        dropped_records_written = 0
        parse_failure_count = 0
        missing_image_count = 0
        ambiguous_duplicate_drop_count = 0
        drop_reason_breakdown: Counter[str] = Counter()

        safe_duplicate_ids: set[str] = set()
        ambiguous_duplicate_ids: set[str] = set()
        duplicate_row_buffer: dict[str, list[_MetadataRow]] = defaultdict(list)

        def write_clean(sample: CleanSample) -> None:
            nonlocal clean_records_written
            if dry_run:
                return
            if limit is not None and clean_records_written >= limit:
                return
            write_jsonl(clean_fp, sample.to_dict())
            clean_records_written += 1

        def write_drop(sample: DroppedSample) -> None:
            nonlocal dropped_records_written
            if dry_run:
                return
            write_jsonl(drop_fp, sample.to_dict())
            dropped_records_written += 1

        def drop_row(row: _MetadataRow, reason: str, detail: str) -> None:
            nonlocal missing_image_count, ambiguous_duplicate_drop_count
            drop_reason_breakdown[reason] += 1
            if reason == "missing_image":
                missing_image_count += 1
            if reason == "ambiguous_duplicate_id":
                ambiguous_duplicate_drop_count += 1
            dropped = DroppedSample(
                id=f"nuswide:{row.row_index}",
                row_index=row.row_index,
                image_id=row.image_id,
                candidate_image_paths=image_index.get(row.image_id, []),
                drop_reason=reason,
                detail=detail,
            )
            write_drop(dropped)

        with ExitStack() as stack:
            all_tags_fp = stack.enter_context(self.all_tags_path.open("r", encoding="utf-8", errors="replace"))
            all_tags_81_fp = stack.enter_context(
                self.all_tags_81_path.open("r", encoding="utf-8", errors="replace")
            )
            all_tags_1k_fp = stack.enter_context(
                self.all_tags_1k_path.open("r", encoding="utf-8", errors="replace")
            )
            label_fps = [
                stack.enter_context(label_path.open("r", encoding="utf-8", errors="replace"))
                for label_path in label_file_paths
            ]

            while True:
                all_tags_line = all_tags_fp.readline()
                all_tags_81_line = all_tags_81_fp.readline()
                all_tags_1k_line = all_tags_1k_fp.readline()
                label_lines = [fp.readline() for fp in label_fps]

                eof_flags = [all_tags_line == "", all_tags_81_line == "", all_tags_1k_line == ""] + [
                    line == "" for line in label_lines
                ]
                if all(eof_flags):
                    break
                if any(eof_flags):
                    raise RuntimeError("Canonical source line counts are inconsistent (unexpected EOF).")

                actual_total_rows += 1
                row_index = actual_total_rows

                try:
                    row = self._build_metadata_row(
                        row_index=row_index,
                        all_tags_line=all_tags_line,
                        all_tags_81_line=all_tags_81_line,
                        all_tags_1k_line=all_tags_1k_line,
                        labels_lines=label_lines,
                    )
                except Exception as exc:  # noqa: BLE001
                    parse_failure_count += 1
                    if self.strict:
                        raise RuntimeError(f"Parse failure at row {row_index}: {exc}") from exc
                    drop_reason_breakdown["parse_failure"] += 1
                    dropped = DroppedSample(
                        id=f"nuswide:{row_index}",
                        row_index=row_index,
                        image_id="",
                        candidate_image_paths=[],
                        drop_reason="parse_failure",
                        detail=str(exc),
                    )
                    write_drop(dropped)
                    continue

                row_count_for_id = id_counts.get(row.image_id, 0)
                if row_count_for_id <= 1:
                    image_candidates = image_index.get(row.image_id, [])
                    if len(image_candidates) == 1:
                        actual_clean_rows += 1
                        write_clean(self._to_clean_sample(row, image_candidates[0]))
                    elif len(image_candidates) == 0:
                        if self.runtime.drop_policies.get("drop_missing_image", True):
                            drop_row(row, "missing_image", "No image file matched by image_id.")
                        else:
                            raise RuntimeError(f"Missing image for image_id={row.image_id} row={row.row_index}")
                    else:
                        ambiguous_duplicate_ids.add(row.image_id)
                        if self.dataset.drop_ambiguous_duplicate_ids:
                            drop_row(
                                row,
                                "ambiguous_duplicate_id",
                                "Multiple image candidates found for a non-duplicate metadata id.",
                            )
                        else:
                            raise RuntimeError(
                                f"Ambiguous image mapping for image_id={row.image_id} row={row.row_index}"
                            )
                    continue

                duplicate_row_buffer[row.image_id].append(row)
                if len(duplicate_row_buffer[row.image_id]) < row_count_for_id:
                    continue

                grouped_rows = sorted(duplicate_row_buffer.pop(row.image_id), key=lambda x: x.row_index)
                image_candidates = sorted(image_index.get(row.image_id, []))

                force_drop = row.image_id in preconfirmed_ambiguous
                if force_drop:
                    ambiguous_duplicate_ids.add(row.image_id)
                    for dup_row in grouped_rows:
                        drop_row(
                            dup_row,
                            "ambiguous_duplicate_id",
                            "Preconfirmed ambiguous duplicate id from dataset policy.",
                        )
                    continue

                tags_same = len({tuple(dup_row.text_raw_tokens) for dup_row in grouped_rows}) == 1
                labels_same = len({tuple(dup_row.labels_81) for dup_row in grouped_rows}) == 1
                candidate_count_match = len(image_candidates) == len(grouped_rows)

                if tags_same and labels_same and candidate_count_match:
                    safe_duplicate_ids.add(row.image_id)
                    for dup_row, image_path in zip(grouped_rows, image_candidates):
                        actual_clean_rows += 1
                        write_clean(self._to_clean_sample(dup_row, image_path))
                    continue

                ambiguous_duplicate_ids.add(row.image_id)
                for dup_row in grouped_rows:
                    if not candidate_count_match:
                        detail = (
                            "Duplicate id cannot be resolved because candidate image count does not match "
                            "metadata duplicate count."
                        )
                    elif not tags_same and not labels_same:
                        detail = "Duplicate id has mismatched tag text and labels_81 across rows."
                    elif not tags_same:
                        detail = "Duplicate id has mismatched tag text across rows."
                    else:
                        detail = "Duplicate id has mismatched labels_81 across rows."
                    drop_row(dup_row, "ambiguous_duplicate_id", detail)

        if clean_fp is not None:
            clean_fp.close()
        if drop_fp is not None:
            drop_fp.close()

        if duplicate_row_buffer:
            raise RuntimeError("Duplicate row buffering failed: unresolved metadata groups remain.")

        if self.strict:
            if actual_total_rows != expected_total_rows:
                raise RuntimeError(
                    f"Row count mismatch: expected {expected_total_rows}, got {actual_total_rows}"
                )
            if all_tags_total_rows != expected_total_rows:
                raise RuntimeError(
                    f"All_Tags scan mismatch: expected {expected_total_rows}, got {all_tags_total_rows}"
                )
            if parse_failure_count > 0:
                raise RuntimeError(f"parse_failure_count must be 0 in strict mode, got {parse_failure_count}")
            if actual_clean_rows != expected_clean_rows:
                raise RuntimeError(
                    "Clean row mismatch after canonical policy. "
                    f"Expected {expected_clean_rows}, got {actual_clean_rows}"
                )

        stats = CleaningStats(
            dataset_name="nuswide",
            expected_total_rows=expected_total_rows,
            actual_total_rows=actual_total_rows,
            expected_clean_rows_after_preconfirmed_ambiguous_drop=expected_clean_rows,
            actual_clean_rows=actual_clean_rows,
            clean_records_written=clean_records_written,
            dropped_records_written=dropped_records_written,
            safe_duplicate_ids=sorted(safe_duplicate_ids),
            ambiguous_duplicate_ids=sorted(ambiguous_duplicate_ids),
            safe_duplicate_resolution_policy=self.dataset.safe_duplicate_resolution_policy,
            ambiguous_duplicate_drop_count=ambiguous_duplicate_drop_count,
            missing_image_count=missing_image_count,
            parse_failure_count=parse_failure_count,
            drop_reason_breakdown=dict(sorted(drop_reason_breakdown.items())),
            strict_validation=self.strict,
            dry_run=dry_run,
            limit=limit,
        )

        if not dry_run:
            self._write_stats(stats)
            self._write_report(stats, concept_names)

        return stats

    def _resolve_required_path(self, canonical: dict[str, str], key: str) -> Path:
        if key not in canonical:
            raise KeyError(f"Missing canonical source key in dataset config: {key}")
        path = self.project_root / canonical[key]
        if not path.exists():
            raise FileNotFoundError(f"Canonical source path does not exist for {key}: {path}")
        return path

    def _scan_all_tags_id_counts(self) -> tuple[dict[str, int], int]:
        counts: Counter[str] = Counter()
        total_rows = 0
        with self.all_tags_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f, start=1):
                total_rows += 1
                tokens = line.strip().split()
                if not tokens:
                    raise RuntimeError(f"All_Tags parsing failed at line {line_idx}: empty line")
                image_id = tokens[0]
                if not image_id.isdigit():
                    raise RuntimeError(
                        f"All_Tags parsing failed at line {line_idx}: invalid image_id `{image_id}`"
                    )
                counts[image_id] += 1
        return dict(counts), total_rows

    def _build_image_index(self) -> dict[str, list[str]]:
        image_index: dict[str, list[str]] = defaultdict(list)
        invalid_image_name_count = 0
        for image_path in self.images_dir.glob("*.jpg"):
            match = IMAGE_NAME_PATTERN.match(image_path.name)
            if not match:
                invalid_image_name_count += 1
                continue
            image_id = match.group(1)
            image_index[image_id].append(self._to_rel(image_path))

        for image_id in image_index:
            image_index[image_id].sort()

        if invalid_image_name_count:
            LOGGER.warning("Ignored %s invalid jpg filenames in images/", invalid_image_name_count)
        LOGGER.info("Image index built for %s image ids", len(image_index))
        return dict(image_index)

    def _load_concept_names(self) -> list[str]:
        concepts = [line.strip() for line in self.concepts_list_path.read_text(encoding="utf-8").splitlines()]
        concepts = [concept for concept in concepts if concept]
        if not concepts:
            raise RuntimeError("Concepts list is empty.")
        if len(concepts) != 81 and self.strict:
            raise RuntimeError(f"Expected 81 concepts, got {len(concepts)}")
        return concepts

    def _load_label_files(self, concept_names: list[str]) -> list[Path]:
        label_files: list[Path] = []
        for concept in concept_names:
            file_path = self.all_labels_dir / f"Labels_{concept}.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing label file for concept `{concept}`: {file_path}")
            label_files.append(file_path)

        # Keep a basic directory-level consistency guard for easier debugging.
        matched_files = list(self.all_labels_dir.glob(self.labels_pattern))
        if self.strict and len(matched_files) != len(concept_names):
            raise RuntimeError(
                f"Label file count mismatch in {self.all_labels_dir}: "
                f"pattern={self.labels_pattern} found={len(matched_files)} expected={len(concept_names)}"
            )
        return label_files

    def _build_metadata_row(
        self,
        row_index: int,
        all_tags_line: str,
        all_tags_81_line: str,
        all_tags_1k_line: str,
        labels_lines: list[str],
    ) -> _MetadataRow:
        all_tag_tokens = all_tags_line.strip().split()
        if not all_tag_tokens:
            raise ValueError("All_Tags line is empty")

        image_id = all_tag_tokens[0]
        if not image_id.isdigit():
            raise ValueError(f"All_Tags image_id is not numeric: {image_id}")
        text_raw_tokens = all_tag_tokens[1:]
        text = " ".join(text_raw_tokens)

        tags_81 = self._parse_binary_vector(
            source_line=all_tags_81_line,
            expected_dim=81,
            source_name="AllTags81",
            row_index=row_index,
        )
        tags_1k = self._parse_binary_vector(
            source_line=all_tags_1k_line,
            expected_dim=1000,
            source_name="AllTags1k",
            row_index=row_index,
        )
        labels_81 = [
            self._parse_binary_scalar(source_line=line, source_name="Labels_*.txt", row_index=row_index)
            for line in labels_lines
        ]
        if len(labels_81) != 81:
            raise ValueError(f"labels_81 vector dimension mismatch: expected 81, got {len(labels_81)}")

        return _MetadataRow(
            row_index=row_index,
            image_id=image_id,
            text_raw_tokens=text_raw_tokens,
            text=text,
            tags_81=tags_81,
            tags_1k=tags_1k,
            labels_81=labels_81,
        )

    @staticmethod
    def _parse_binary_vector(source_line: str, expected_dim: int, source_name: str, row_index: int) -> list[int]:
        tokens = source_line.strip().split()
        if len(tokens) != expected_dim:
            raise ValueError(
                f"{source_name} row {row_index} dim mismatch: expected {expected_dim}, got {len(tokens)}"
            )
        values: list[int] = []
        for token in tokens:
            if token not in {"0", "1"}:
                raise ValueError(f"{source_name} row {row_index} contains invalid token: {token}")
            values.append(int(token))
        return values

    @staticmethod
    def _parse_binary_scalar(source_line: str, source_name: str, row_index: int) -> int:
        token = source_line.strip()
        if token not in {"0", "1"}:
            raise ValueError(f"{source_name} row {row_index} contains invalid scalar token: {token}")
        return int(token)

    def _to_clean_sample(self, row: _MetadataRow, image_path: str) -> CleanSample:
        return CleanSample(
            id=f"nuswide:{row.row_index}",
            dataset_name="nuswide",
            row_index=row.row_index,
            image_id=row.image_id,
            image_path=image_path,
            text_raw_tokens=row.text_raw_tokens,
            text=row.text,
            tags_1k=row.tags_1k,
            labels_81=row.labels_81,
        )

    def _write_stats(self, stats: CleaningStats) -> None:
        write_json(self.output_paths.clean_stats_path, stats.to_dict())

    def _write_report(self, stats: CleaningStats, concept_names: list[str]) -> None:
        canonical_sources = self.dataset.canonical_sources
        disabled_sources = self.dataset.disabled_sources
        notes = [
            f"Concepts order source: `{canonical_sources.get('concepts_list', '')}` (count={len(concept_names)})",
            "Groundtruth/TrainTestLabels/Labels_lake_Train.txt contains a confirmed source anomaly and is "
            "excluded from canonical raw alignment.",
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
