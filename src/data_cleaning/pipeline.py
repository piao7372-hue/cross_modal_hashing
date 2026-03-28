from __future__ import annotations

from pathlib import Path
from typing import Type

from .base_cleaner import BaseCleaner
from .config import load_dataset_config, load_runtime_config
from .datasets.nuswide import NUSWIDECleaner
from .records import CleaningStats

CLEANER_REGISTRY: dict[str, Type[BaseCleaner]] = {
    "nuswide": NUSWIDECleaner,
}


def supported_datasets() -> list[str]:
    return sorted(CLEANER_REGISTRY.keys())


def run_pipeline(
    *,
    dataset_name: str,
    project_root: Path,
    runtime_config_path: Path,
    dataset_config_path: Path,
    dry_run: bool,
    limit: int | None,
) -> CleaningStats:
    runtime = load_runtime_config(project_root, runtime_config_path)
    dataset = load_dataset_config(dataset_config_path)
    if dataset_name != dataset.dataset_name:
        raise ValueError(
            f"CLI dataset `{dataset_name}` does not match dataset config `{dataset.dataset_name}`"
        )
    cleaner_cls = CLEANER_REGISTRY.get(dataset_name)
    if cleaner_cls is None:
        raise ValueError(f"Unsupported dataset `{dataset_name}`")

    cleaner = cleaner_cls(
        project_root=project_root,
        runtime_config=runtime,
        dataset_config=dataset,
    )
    return cleaner.run(dry_run=(dry_run or runtime.dry_run_default), limit=limit)
