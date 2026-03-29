from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class ManifestSample:
    sample_index: int
    manifest_line_number: int
    id: str
    dataset_name: str
    image_path: str
    text: str
    text_empty: bool
    row_index: int | None
    split: str | None
    caption_ann_id: str | None


def count_manifest_rows(path: Path, limit: int | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_manifest_obj(obj: dict[str, Any], sample_index: int, manifest_line_number: int) -> ManifestSample:
    sample_id = str(obj.get("id", ""))
    dataset_name = str(obj.get("dataset_name", ""))
    image_path = str(obj.get("image_path", ""))
    raw_text = obj.get("text", "")
    if raw_text is None:
        raw_text = ""
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    row_index = obj.get("row_index")
    parsed_row_index = row_index if isinstance(row_index, int) else None
    text_empty_flag = obj.get("text_empty")
    text_empty = bool(text_empty_flag) if isinstance(text_empty_flag, bool) else (raw_text == "")
    split = str(obj["split"]) if "split" in obj and obj["split"] is not None else None
    caption_ann_id = (
        str(obj["caption_ann_id"]) if "caption_ann_id" in obj and obj["caption_ann_id"] is not None else None
    )

    if sample_id == "":
        raise ValueError(f"Manifest line {manifest_line_number}: missing `id`")
    if dataset_name == "":
        raise ValueError(f"Manifest line {manifest_line_number}: missing `dataset_name`")
    if image_path == "":
        raise ValueError(f"Manifest line {manifest_line_number}: missing `image_path`")

    return ManifestSample(
        sample_index=sample_index,
        manifest_line_number=manifest_line_number,
        id=sample_id,
        dataset_name=dataset_name,
        image_path=image_path,
        text=raw_text,
        text_empty=text_empty,
        row_index=parsed_row_index,
        split=split,
        caption_ann_id=caption_ann_id,
    )


def iter_manifest_samples(path: Path, limit: int | None = None) -> Iterator[ManifestSample]:
    sample_index = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Manifest line {line_no}: expected json object")
            yield _parse_manifest_obj(obj=obj, sample_index=sample_index, manifest_line_number=line_no)
            sample_index += 1
            if limit is not None and sample_index >= limit:
                break


def sample_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
