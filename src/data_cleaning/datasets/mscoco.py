from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TextIO

from ..base_cleaner import BaseCleaner
from ..config import DatasetConfig, RuntimeConfig
from ..exporters import (
    DatasetOutputPaths,
    build_dataset_output_paths,
    ensure_output_dirs,
    write_json,
    write_jsonl,
)
from ..records import CleaningStats
from ..reporting import build_cleaning_report_markdown

LOGGER = logging.getLogger(__name__)

COCO_IMAGE_NAME_PATTERN = re.compile(
    r"^COCO_(?P<split>train2014|val2014)_(?P<image_id>\d{12})\.jpg$"
)


class MSCOCOCleaner(BaseCleaner):
    """Canonical MSCOCO cleaner aligned to frozen pre-implementation decisions."""

    def __init__(
        self,
        project_root: Path,
        runtime_config: RuntimeConfig,
        dataset_config: DatasetConfig,
    ) -> None:
        self.project_root = project_root
        self.runtime = runtime_config
        self.dataset = dataset_config
        if self.dataset.dataset_name != "mscoco":
            raise ValueError(f"Unsupported dataset for this cleaner: {self.dataset.dataset_name}")

        self.strict = self.runtime.strict_validation
        self.allowed_drop_reasons = self._resolve_allowed_drop_reasons()

        canonical = self.dataset.canonical_sources
        self.images_train_dir = self._resolve_required_dir(canonical, "images_train_dir")
        self.images_val_dir = self._resolve_required_dir(canonical, "images_val_dir")
        self.captions_train_path = self._resolve_required_file(canonical, "captions_train_json")
        self.captions_val_path = self._resolve_required_file(canonical, "captions_val_json")
        self.instances_train_path = self._resolve_required_file(canonical, "instances_train_json")
        self.instances_val_path = self._resolve_required_file(canonical, "instances_val_json")

        self.output_paths: DatasetOutputPaths = build_dataset_output_paths(
            runtime=self.runtime,
            dataset_name="mscoco",
        )

    def run(self, dry_run: bool = False, limit: int | None = None) -> CleaningStats:
        expected_total_rows = (
            self.dataset.expected_total_rows
            if self.dataset.expected_total_rows is not None
            else 0
        )
        expected_clean_rows = self.dataset.expected_clean_rows_after_preconfirmed_ambiguous_drop

        actual_total_rows = 0
        actual_clean_rows = 0
        clean_records_written = 0
        dropped_records_written = 0
        parse_failure_count = 0
        missing_image_count = 0
        drop_reason_breakdown: Counter[str] = Counter()
        seen_row_ids: set[str] = set()
        images_without_instances_by_split: Counter[str] = Counter()
        category_id_sets: dict[str, set[int]] = {}

        clean_fp: TextIO | None = None
        drop_fp: TextIO | None = None
        if not dry_run:
            ensure_output_dirs(self.output_paths)
            clean_fp = self.output_paths.clean_manifest_path.open("w", encoding="utf-8")
            drop_fp = self.output_paths.dropped_samples_path.open("w", encoding="utf-8")

        def write_clean(sample: dict[str, object]) -> None:
            nonlocal clean_records_written
            if dry_run:
                return
            if limit is not None and clean_records_written >= limit:
                return
            write_jsonl(clean_fp, sample)
            clean_records_written += 1

        def write_drop(sample: dict[str, object]) -> None:
            nonlocal dropped_records_written, parse_failure_count, missing_image_count
            reason = str(sample["drop_reason"])
            if reason not in self.allowed_drop_reasons:
                raise RuntimeError(f"Unexpected drop reason `{reason}` outside closed enum.")
            drop_reason_breakdown[reason] += 1
            if reason == "missing_image":
                missing_image_count += 1
            if reason in {"caption_parse_failure", "instances_parse_failure"}:
                parse_failure_count += 1
            if dry_run:
                return
            write_jsonl(drop_fp, sample)
            dropped_records_written += 1

        split_specs = [
            (
                "train2014",
                self.images_train_dir,
                self.captions_train_path,
                self.instances_train_path,
            ),
            (
                "val2014",
                self.images_val_dir,
                self.captions_val_path,
                self.instances_val_path,
            ),
        ]

        for split, images_dir, captions_path, instances_path in split_specs:
            captions_obj = self._load_json(captions_path)
            instances_obj = self._load_json(instances_path)

            self._require_top_keys(
                captions_obj,
                required_keys={"info", "images", "licenses", "annotations"},
                source_name=captions_path.name,
            )
            self._require_top_keys(
                instances_obj,
                required_keys={"info", "images", "licenses", "annotations", "categories"},
                source_name=instances_path.name,
            )

            images_meta = self._build_images_meta_index(captions_obj, split=split)
            image_path_candidates = self._build_image_path_index(images_dir=images_dir, split=split)
            labels_by_image, instances_parse_failure_ids, category_ids = self._build_instances_index(
                instances_obj=instances_obj,
                split=split,
            )
            category_id_sets[split] = category_ids

            annotations = captions_obj.get("annotations")
            if not isinstance(annotations, list):
                raise RuntimeError(f"`annotations` must be an array in {captions_path.name}")

            for ann in annotations:
                actual_total_rows += 1
                row_index = actual_total_rows

                drop_candidate = self._build_base_drop_candidate(
                    ann=ann,
                    split=split,
                    row_index=row_index,
                    image_path_candidates=image_path_candidates,
                )
                row_id = str(drop_candidate["id"])
                image_id_text = str(drop_candidate["image_id"])
                caption_ann_id_text = str(drop_candidate["caption_ann_id"])
                candidate_paths = list(drop_candidate["candidate_image_paths"])

                image_id = self._parse_int_nullable(ann.get("image_id"))
                caption_ann_id = self._parse_int_nullable(ann.get("id"))
                if image_id is None or caption_ann_id is None:
                    write_drop(
                        self._make_drop_record(
                            row_id=row_id,
                            row_index=row_index,
                            split=split,
                            image_id=image_id_text,
                            caption_ann_id=caption_ann_id_text,
                            candidate_image_paths=candidate_paths,
                            drop_reason="caption_parse_failure",
                            detail="caption annotation id/image_id is missing or non-numeric",
                        )
                    )
                    continue

                row_id = f"mscoco:{split}:{image_id}:{caption_ann_id}"
                image_id_text = str(image_id)
                caption_ann_id_text = str(caption_ann_id)
                candidate_paths = list(image_path_candidates.get(image_id, []))

                if row_id in seen_row_ids:
                    write_drop(
                        self._make_drop_record(
                            row_id=row_id,
                            row_index=row_index,
                            split=split,
                            image_id=image_id_text,
                            caption_ann_id=caption_ann_id_text,
                            candidate_image_paths=candidate_paths,
                            drop_reason="alignment_key_conflict",
                            detail="duplicate row-level unique id generated from split/image_id/caption_ann_id",
                        )
                    )
                    continue
                seen_row_ids.add(row_id)

                image_meta = images_meta.get(image_id)
                if image_meta is None:
                    write_drop(
                        self._make_drop_record(
                            row_id=row_id,
                            row_index=row_index,
                            split=split,
                            image_id=image_id_text,
                            caption_ann_id=caption_ann_id_text,
                            candidate_image_paths=candidate_paths,
                            drop_reason="alignment_key_conflict",
                            detail="caption annotation image_id not found in captions.images",
                        )
                    )
                    continue

                image_file_name = image_meta.get("file_name")
                if not isinstance(image_file_name, str):
                    write_drop(
                        self._make_drop_record(
                            row_id=row_id,
                            row_index=row_index,
                            split=split,
                            image_id=image_id_text,
                            caption_ann_id=caption_ann_id_text,
                            candidate_image_paths=candidate_paths,
                            drop_reason="invalid_image_identity",
                            detail="image metadata file_name is missing or non-string",
                        )
                    )
                    continue

                image_name_match = COCO_IMAGE_NAME_PATTERN.match(image_file_name)
                if image_name_match is None:
                    write_drop(
                        self._make_drop_record(
                            row_id=row_id,
                            row_index=row_index,
                            split=split,
                            image_id=image_id_text,
                            caption_ann_id=caption_ann_id_text,
                            candidate_image_paths=candidate_paths,
                            drop_reason="invalid_image_identity",
                            detail=f"image filename does not match COCO naming: {image_file_name}",
                        )
                    )
                    continue

                file_split = image_name_match.group("split")
                file_image_id = int(image_name_match.group("image_id"))
                if file_split != split or file_image_id != image_id:
                    write_drop(
                        self._make_drop_record(
                            row_id=row_id,
                            row_index=row_index,
                            split=split,
                            image_id=image_id_text,
                            caption_ann_id=caption_ann_id_text,
                            candidate_image_paths=candidate_paths,
                            drop_reason="invalid_image_identity",
                            detail="image filename split/image_id does not match alignment key",
                        )
                    )
                    continue

                image_path = images_dir / image_file_name
                image_rel = self._to_rel(image_path)
                if not image_path.exists():
                    write_drop(
                        self._make_drop_record(
                            row_id=row_id,
                            row_index=row_index,
                            split=split,
                            image_id=image_id_text,
                            caption_ann_id=caption_ann_id_text,
                            candidate_image_paths=candidate_paths,
                            drop_reason="missing_image",
                            detail=f"canonical image file not found: {image_rel}",
                        )
                    )
                    continue

                caption_value = ann.get("caption")
                if caption_value is None:
                    write_drop(
                        self._make_drop_record(
                            row_id=row_id,
                            row_index=row_index,
                            split=split,
                            image_id=image_id_text,
                            caption_ann_id=caption_ann_id_text,
                            candidate_image_paths=[image_rel],
                            drop_reason="missing_caption",
                            detail="caption field is missing/null for caption annotation",
                        )
                    )
                    continue
                if not isinstance(caption_value, str):
                    write_drop(
                        self._make_drop_record(
                            row_id=row_id,
                            row_index=row_index,
                            split=split,
                            image_id=image_id_text,
                            caption_ann_id=caption_ann_id_text,
                            candidate_image_paths=[image_rel],
                            drop_reason="caption_parse_failure",
                            detail="caption field is not a string",
                        )
                    )
                    continue

                if image_id in instances_parse_failure_ids:
                    write_drop(
                        self._make_drop_record(
                            row_id=row_id,
                            row_index=row_index,
                            split=split,
                            image_id=image_id_text,
                            caption_ann_id=caption_ann_id_text,
                            candidate_image_paths=[image_rel],
                            drop_reason="instances_parse_failure",
                            detail="instances annotation parse failure flagged for this image_id",
                        )
                    )
                    continue

                label_indices = sorted(labels_by_image.get(image_id, set()))
                has_instances = len(label_indices) > 0
                labels_empty_by_source = not has_instances
                if labels_empty_by_source:
                    images_without_instances_by_split[split] += 1

                text = caption_value
                clean_row = {
                    "id": row_id,
                    "dataset_name": "mscoco",
                    "row_index": row_index,
                    "split": split,
                    "image_id": image_id_text,
                    "caption_ann_id": caption_ann_id_text,
                    "image_path": image_rel,
                    "text_raw_tokens": text.split(),
                    "text": text,
                    "text_empty": text == "",
                    # Keep compatibility with existing manifest style fields.
                    "tags_1k": [],
                    "labels_81": [],
                    # Frozen MSCOCO-specific payload.
                    "label_indices": label_indices,
                    "has_instances": has_instances,
                    "labels_empty_by_source": labels_empty_by_source,
                    "notes": [
                        "row_unit=per-image-caption",
                        "canonical_scope=train2014+val2014",
                    ],
                }
                actual_clean_rows += 1
                write_clean(clean_row)

        if category_id_sets:
            train_categories = category_id_sets.get("train2014", set())
            val_categories = category_id_sets.get("val2014", set())
            if self.strict and train_categories != val_categories:
                raise RuntimeError(
                    "Irrecoverable structural alignment failure: category id sets mismatch between train and val."
                )

        if clean_fp is not None:
            clean_fp.close()
        if drop_fp is not None:
            drop_fp.close()

        expected_total_rows = (
            self.dataset.expected_total_rows
            if self.dataset.expected_total_rows is not None
            else actual_total_rows
        )
        expected_clean_rows_resolved = (
            expected_clean_rows if expected_clean_rows is not None else actual_clean_rows
        )

        if self.strict:
            if self.dataset.expected_total_rows is not None and actual_total_rows != expected_total_rows:
                raise RuntimeError(
                    f"Row count mismatch: expected {expected_total_rows}, got {actual_total_rows}"
                )
            if (
                self.dataset.expected_clean_rows_after_preconfirmed_ambiguous_drop is not None
                and actual_clean_rows != expected_clean_rows_resolved
            ):
                raise RuntimeError(
                    "Clean row mismatch under strict mode: "
                    f"expected {expected_clean_rows_resolved}, got {actual_clean_rows}"
                )

        stats = CleaningStats(
            dataset_name="mscoco",
            expected_total_rows=expected_total_rows,
            actual_total_rows=actual_total_rows,
            expected_clean_rows_after_preconfirmed_ambiguous_drop=expected_clean_rows_resolved,
            actual_clean_rows=actual_clean_rows,
            clean_records_written=clean_records_written,
            dropped_records_written=dropped_records_written,
            safe_duplicate_ids=[],
            ambiguous_duplicate_ids=[],
            safe_duplicate_resolution_policy="primary_key=(split,image_id), row_unit=per-image-caption",
            ambiguous_duplicate_drop_count=0,
            missing_image_count=missing_image_count,
            parse_failure_count=parse_failure_count,
            drop_reason_breakdown=dict(sorted(drop_reason_breakdown.items())),
            strict_validation=self.strict,
            dry_run=dry_run,
            limit=limit,
        )

        if not dry_run:
            self._write_stats(stats)
            self._write_report(stats, images_without_instances_by_split)

        return stats

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

    def _resolve_required_file(self, canonical: dict[str, object], key: str) -> Path:
        if key not in canonical:
            raise KeyError(f"Missing canonical source key in dataset config: {key}")
        value = canonical[key]
        if not isinstance(value, str):
            raise ValueError(f"Canonical source `{key}` must be a string path")
        path = self.project_root / value
        if not path.exists():
            raise FileNotFoundError(f"Canonical source path does not exist for {key}: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Canonical source path for {key} is not a file: {path}")
        return path

    @staticmethod
    def _require_top_keys(obj: dict[str, Any], required_keys: set[str], source_name: str) -> None:
        missing = [key for key in required_keys if key not in obj]
        if missing:
            raise RuntimeError(
                f"Required top-level keys missing in {source_name}: {', '.join(sorted(missing))}"
            )

    def _load_json(self, path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Canonical JSON parse failure at {path}: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"Canonical JSON read failure at {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Canonical JSON root must be object at {path}")
        return data

    def _build_images_meta_index(self, captions_obj: dict[str, Any], split: str) -> dict[int, dict[str, Any]]:
        images = captions_obj.get("images")
        if not isinstance(images, list):
            raise RuntimeError("`images` must be an array in captions source")

        by_id: dict[int, dict[str, Any]] = {}
        for image in images:
            if not isinstance(image, dict):
                raise RuntimeError("`images[]` entry must be object in captions source")
            image_id = self._parse_int_nullable(image.get("id"))
            if image_id is None:
                raise RuntimeError("`images[]` entry has missing/non-numeric `id` in captions source")
            existing = by_id.get(image_id)
            if existing is None:
                by_id[image_id] = image
                continue
            if existing.get("file_name") != image.get("file_name"):
                raise RuntimeError(
                    "Irrecoverable structural alignment failure: duplicate image id with conflicting file_name "
                    f"in captions source for split={split}, image_id={image_id}"
                )
        return by_id

    def _build_image_path_index(self, images_dir: Path, split: str) -> dict[int, list[str]]:
        index: dict[int, list[str]] = defaultdict(list)
        for image_path in images_dir.glob("*.jpg"):
            match = COCO_IMAGE_NAME_PATTERN.match(image_path.name)
            if not match:
                continue
            if match.group("split") != split:
                continue
            image_id = int(match.group("image_id"))
            index[image_id].append(self._to_rel(image_path))
        for image_id in index:
            index[image_id].sort()
        return dict(index)

    def _build_instances_index(
        self,
        instances_obj: dict[str, Any],
        split: str,
    ) -> tuple[dict[int, set[int]], set[int], set[int]]:
        categories_raw = instances_obj.get("categories")
        if not isinstance(categories_raw, list):
            raise RuntimeError("`categories` must be an array in instances source")

        category_ids: set[int] = set()
        for item in categories_raw:
            if not isinstance(item, dict):
                raise RuntimeError("`categories[]` entry must be object in instances source")
            category_id = self._parse_int_nullable(item.get("id"))
            if category_id is None:
                raise RuntimeError("`categories[]` entry has missing/non-numeric `id` in instances source")
            category_ids.add(category_id)

        annotations_raw = instances_obj.get("annotations")
        if not isinstance(annotations_raw, list):
            raise RuntimeError("`annotations` must be an array in instances source")

        labels_by_image: dict[int, set[int]] = defaultdict(set)
        parse_failure_image_ids: set[int] = set()

        for ann in annotations_raw:
            if not isinstance(ann, dict):
                raise RuntimeError("`annotations[]` entry must be object in instances source")
            image_id = self._parse_int_nullable(ann.get("image_id"))
            if image_id is None:
                raise RuntimeError(
                    "Irrecoverable structural alignment failure: instances annotation has non-numeric image_id "
                    f"(split={split})"
                )
            category_id = self._parse_int_nullable(ann.get("category_id"))
            if category_id is None or category_id not in category_ids:
                parse_failure_image_ids.add(image_id)
                continue
            labels_by_image[image_id].add(category_id)

        return dict(labels_by_image), parse_failure_image_ids, category_ids

    @staticmethod
    def _parse_int_nullable(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    def _build_base_drop_candidate(
        self,
        ann: Any,
        split: str,
        row_index: int,
        image_path_candidates: dict[int, list[str]],
    ) -> dict[str, object]:
        image_id_text = ""
        caption_ann_id_text = ""
        candidate_paths: list[str] = []

        if isinstance(ann, dict):
            raw_image_id = ann.get("image_id")
            raw_caption_ann_id = ann.get("id")
            if raw_image_id is not None:
                image_id_text = str(raw_image_id)
            if raw_caption_ann_id is not None:
                caption_ann_id_text = str(raw_caption_ann_id)
            image_id_int = self._parse_int_nullable(raw_image_id)
            if image_id_int is not None:
                candidate_paths = list(image_path_candidates.get(image_id_int, []))

        row_id = f"mscoco:{split}:{image_id_text}:{caption_ann_id_text}"
        return {
            "id": row_id,
            "dataset_name": "mscoco",
            "row_index": row_index,
            "split": split,
            "image_id": image_id_text,
            "caption_ann_id": caption_ann_id_text,
            "candidate_image_paths": candidate_paths,
        }

    def _make_drop_record(
        self,
        *,
        row_id: str,
        row_index: int,
        split: str,
        image_id: str,
        caption_ann_id: str,
        candidate_image_paths: list[str],
        drop_reason: str,
        detail: str,
    ) -> dict[str, object]:
        return {
            "id": row_id,
            "dataset_name": "mscoco",
            "row_index": row_index,
            "split": split,
            "image_id": image_id,
            "caption_ann_id": caption_ann_id,
            "candidate_image_paths": candidate_image_paths,
            "drop_reason": drop_reason,
            "detail": detail,
        }

    def _resolve_allowed_drop_reasons(self) -> set[str]:
        raw = self.dataset.drop_reasons.get("allowed", [])
        reasons = {str(x) for x in raw}
        if not reasons:
            raise ValueError("MSCOCO dataset config must provide non-empty drop_reasons.allowed")
        return reasons

    def _write_stats(self, stats: CleaningStats) -> None:
        write_json(self.output_paths.clean_stats_path, stats.to_dict())

    def _write_report(self, stats: CleaningStats, images_without_instances_by_split: Counter[str]) -> None:
        canonical_sources = {str(k): str(v) for k, v in self.dataset.canonical_sources.items()}
        disabled_sources = [str(x) for x in self.dataset.non_canonical_sources]
        notes = [
            "MSCOCO canonical scope: train2014 + val2014; test2014 is reference-only.",
            "Row unit: per-image-caption.",
            "Primary alignment key: (split, image_id).",
            "Closed drop-reason enum: missing_image, invalid_image_identity, missing_caption, "
            "caption_parse_failure, instances_parse_failure, alignment_key_conflict.",
            f"images_without_instances_train={images_without_instances_by_split.get('train2014', 0)}",
            f"images_without_instances_val={images_without_instances_by_split.get('val2014', 0)}",
            "Rows without instances are kept with labels=[], has_instances=false, "
            "labels_empty_by_source=true.",
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
