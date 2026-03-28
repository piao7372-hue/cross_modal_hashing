from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .config import RuntimeConfig


@dataclass(frozen=True)
class DatasetOutputPaths:
    dataset_name: str
    processed_dataset_dir: Path
    outputs_dataset_dir: Path
    clean_manifest_path: Path
    dropped_samples_path: Path
    clean_stats_path: Path
    cleaning_report_path: Path


def build_dataset_output_paths(runtime: RuntimeConfig, dataset_name: str) -> DatasetOutputPaths:
    processed_dataset_dir = runtime.processed_root / dataset_name
    outputs_dataset_dir = runtime.outputs_root / dataset_name
    return DatasetOutputPaths(
        dataset_name=dataset_name,
        processed_dataset_dir=processed_dataset_dir,
        outputs_dataset_dir=outputs_dataset_dir,
        clean_manifest_path=processed_dataset_dir / "clean_manifest.jsonl",
        dropped_samples_path=processed_dataset_dir / "dropped_samples.jsonl",
        clean_stats_path=outputs_dataset_dir / "clean_stats.json",
        cleaning_report_path=outputs_dataset_dir / "cleaning_report.md",
    )


def ensure_output_dirs(paths: DatasetOutputPaths) -> None:
    paths.processed_dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.outputs_dataset_dir.mkdir(parents=True, exist_ok=True)


def write_jsonl(fp: TextIO | None, obj: dict[str, object]) -> None:
    if fp is None:
        return
    fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, obj: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
