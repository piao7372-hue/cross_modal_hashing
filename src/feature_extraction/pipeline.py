from __future__ import annotations

import gc
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache_writer import (
    build_cache_paths,
    create_feature_memmap,
    create_index_memmap,
    prepare_output_dir,
    write_jsonl,
    write_meta,
)
from .config import FeatureExtractionConfig
from .manifest_reader import (
    compute_file_sha256,
    count_manifest_rows,
    iter_manifest_samples,
    sample_text_hash,
)


@dataclass(frozen=True)
class ExtractionStats:
    dataset: str
    view: str
    feature_set_id: str
    manifest_path: str
    output_dir: str | None
    dry_run: bool
    limit: int | None
    manifest_rows: int
    unique_image_count: int
    empty_text_rows: int
    embedding_dim: int | None
    save_unique_image_cache: bool
    save_row_aligned_image_matrix: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text)


def _build_auto_feature_set_id(model_name: str, view: str, manifest_sha256: str, limit: int | None) -> str:
    model_part = _sanitize_name(model_name.split("/")[-1])
    view_part = _sanitize_name(view)
    sha_part = manifest_sha256[:8]
    if limit is not None:
        return f"{model_part}_{view_part}_{sha_part}_limit{limit}"
    return f"{model_part}_{view_part}_{sha_part}"


def _should_save_unique_image_cache(dataset: str, policy: str) -> bool:
    normalized = policy.lower()
    if normalized == "always":
        return True
    if normalized == "never":
        return False
    if normalized == "auto":
        return dataset == "mscoco"
    raise ValueError(f"Unsupported cache.save_unique_image_cache policy `{policy}`")


def run_feature_extraction(
    *,
    dataset: str,
    config: FeatureExtractionConfig,
    view: str | None,
    dry_run: bool,
    limit: int | None,
    feature_set_id: str | None,
    overwrite: bool,
) -> ExtractionStats:
    resolved_view, manifest_path = config.resolve_manifest_path(dataset=dataset, view=view)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest_rows = count_manifest_rows(manifest_path, limit=limit)
    if manifest_rows <= 0:
        raise RuntimeError(f"Manifest has no rows under current limit: {manifest_path}")

    manifest_sha256 = compute_file_sha256(manifest_path)
    resolved_feature_set_id = feature_set_id or _build_auto_feature_set_id(
        model_name=config.encoder.model_name,
        view=resolved_view,
        manifest_sha256=manifest_sha256,
        limit=limit,
    )
    output_dir = config.resolve_output_dir(dataset=dataset, feature_set_id=resolved_feature_set_id)
    cache_paths = build_cache_paths(output_dir=output_dir, cache_cfg=config.cache)

    image_path_to_idx: dict[str, int] = {}
    unique_image_paths: list[str] = []
    empty_text_rows = 0
    sample_count = 0

    if dry_run:
        for sample in iter_manifest_samples(manifest_path, limit=limit):
            sample_count += 1
            empty_text_rows += int(sample.text_empty)
            if sample.image_path not in image_path_to_idx:
                image_path_to_idx[sample.image_path] = len(unique_image_paths)
                unique_image_paths.append(sample.image_path)
        return ExtractionStats(
            dataset=dataset,
            view=resolved_view,
            feature_set_id=resolved_feature_set_id,
            manifest_path=manifest_path.as_posix(),
            output_dir=None,
            dry_run=True,
            limit=limit,
            manifest_rows=sample_count,
            unique_image_count=len(unique_image_paths),
            empty_text_rows=empty_text_rows,
            embedding_dim=None,
            save_unique_image_cache=_should_save_unique_image_cache(dataset, config.cache.save_unique_image_cache),
            save_row_aligned_image_matrix=config.cache.save_row_aligned_image_matrix,
        )

    prepare_output_dir(output_dir=output_dir, overwrite=overwrite)
    row_to_image_idx = create_index_memmap(cache_paths.row_to_image_idx_path, rows=manifest_rows)

    with cache_paths.sample_index_path.open("w", encoding="utf-8") as sample_fp:
        for sample in iter_manifest_samples(manifest_path, limit=limit):
            sample_count += 1
            empty_text_rows += int(sample.text_empty)
            image_idx = image_path_to_idx.get(sample.image_path)
            if image_idx is None:
                image_idx = len(unique_image_paths)
                image_path_to_idx[sample.image_path] = image_idx
                unique_image_paths.append(sample.image_path)
            row_to_image_idx[sample.sample_index] = image_idx
            write_jsonl(
                sample_fp,
                {
                    "sample_index": sample.sample_index,
                    "manifest_line_number": sample.manifest_line_number,
                    "id": sample.id,
                    "dataset_name": sample.dataset_name,
                    "row_index": sample.row_index,
                    "image_path": sample.image_path,
                    "image_index": image_idx,
                    "text_empty": sample.text_empty,
                    "text_sha256": sample_text_hash(sample.text),
                    "split": sample.split,
                    "caption_ann_id": sample.caption_ann_id,
                },
            )

    row_to_image_idx.flush()
    with cache_paths.image_index_path.open("w", encoding="utf-8") as image_fp:
        for image_index, rel_path in enumerate(unique_image_paths):
            write_jsonl(image_fp, {"image_index": image_index, "image_path": rel_path})

    from .extractors import ClipFeatureExtractor

    extractor = ClipFeatureExtractor(encoder_cfg=config.encoder, runtime_cfg=config.runtime)
    extractor_info = extractor.info()
    embedding_dim = extractor_info.embedding_dim

    save_unique = _should_save_unique_image_cache(dataset, config.cache.save_unique_image_cache)
    unique_feature_path = cache_paths.x_i_unique_path if save_unique else (output_dir / "_tmp_X_I_unique.npy")
    x_i_unique = create_feature_memmap(unique_feature_path, rows=len(unique_image_paths), dim=embedding_dim)

    image_batch = config.runtime.image_batch_size
    for start in range(0, len(unique_image_paths), image_batch):
        end = min(start + image_batch, len(unique_image_paths))
        abs_paths = []
        for rel_path in unique_image_paths[start:end]:
            absolute = config.project_root / rel_path
            if not absolute.exists():
                raise FileNotFoundError(f"Image path from manifest does not exist: {absolute}")
            abs_paths.append(absolute)
        features = extractor.encode_images(abs_paths)
        x_i_unique[start:end] = features
    x_i_unique.flush()

    if config.cache.save_row_aligned_image_matrix:
        x_i = create_feature_memmap(cache_paths.x_i_path, rows=manifest_rows, dim=embedding_dim)
        chunk = 65536
        for start in range(0, manifest_rows, chunk):
            end = min(start + chunk, manifest_rows)
            idx_chunk = np.asarray(row_to_image_idx[start:end], dtype=np.int64)
            x_i[start:end] = x_i_unique[idx_chunk]
        x_i.flush()

    x_t = create_feature_memmap(cache_paths.x_t_path, rows=manifest_rows, dim=embedding_dim)
    text_batch = config.runtime.text_batch_size
    batch_indices: list[int] = []
    batch_texts: list[str] = []

    def flush_text_batch() -> None:
        if not batch_indices:
            return
        features = extractor.encode_texts(batch_texts)
        x_t[np.asarray(batch_indices, dtype=np.int64)] = features
        batch_indices.clear()
        batch_texts.clear()

    for sample in iter_manifest_samples(manifest_path, limit=limit):
        batch_indices.append(sample.sample_index)
        batch_texts.append("" if sample.text_empty else sample.text)
        if len(batch_indices) >= text_batch:
            flush_text_batch()
    flush_text_batch()
    x_t.flush()

    if not save_unique:
        del x_i_unique
        gc.collect()
        if unique_feature_path.exists():
            unique_feature_path.unlink()

    meta_payload: dict[str, Any] = {
        "contract_version": "feature_cache_v1",
        "dataset": dataset,
        "view": resolved_view,
        "feature_set_id": resolved_feature_set_id,
        "manifest": {
            "path": manifest_path.as_posix(),
            "sha256": manifest_sha256,
            "rows": sample_count,
        },
        "sample_index_contract": {
            "basis": config.contracts.sample_index_basis,
            "row_index_used_as_position": False,
            "forbid_row_index_as_position": config.contracts.forbid_row_index_as_position,
        },
        "encoder": {
            "family": config.encoder.family,
            "model_name": config.encoder.model_name,
            "text_tokenizer": config.encoder.text_tokenizer,
            "text_max_length": config.encoder.text_max_length,
            "image_preprocess": config.encoder.image_preprocess,
            "runtime_device": extractor_info.device,
            "runtime_dtype": extractor_info.dtype,
            "runtime_backend": extractor_info.backend,
            "deterministic": config.runtime.deterministic,
            "seed": config.runtime.seed,
            "normalize": {"type": config.encoder.normalize_type, "epsilon": config.encoder.normalize_epsilon},
            "embedding_dim": embedding_dim,
        },
        "empty_text_policy": {
            "contract": config.contracts.empty_text_policy,
            "rows": empty_text_rows,
            "encoding_behavior": "empty_string_passed_to_same_tokenizer_and_encoder",
            "rows_dropped": 0,
        },
        "image_reuse": {
            "enabled": True,
            "unique_image_count": len(unique_image_paths),
            "row_to_image_idx_path": cache_paths.row_to_image_idx_path.name,
            "x_i_unique_saved": save_unique,
        },
        "outputs": {
            "X_I": cache_paths.x_i_path.name if config.cache.save_row_aligned_image_matrix else None,
            "X_T": cache_paths.x_t_path.name,
            "X_I_unique": cache_paths.x_i_unique_path.name if save_unique else None,
            "row_to_image_idx": cache_paths.row_to_image_idx_path.name,
            "sample_index": cache_paths.sample_index_path.name,
            "image_index": cache_paths.image_index_path.name,
        },
        "counts": {
            "manifest_rows": sample_count,
            "empty_text_rows": empty_text_rows,
            "unique_image_count": len(unique_image_paths),
        },
    }
    write_meta(cache_paths.meta_path, meta_payload)

    return ExtractionStats(
        dataset=dataset,
        view=resolved_view,
        feature_set_id=resolved_feature_set_id,
        manifest_path=manifest_path.as_posix(),
        output_dir=output_dir.as_posix(),
        dry_run=False,
        limit=limit,
        manifest_rows=sample_count,
        unique_image_count=len(unique_image_paths),
        empty_text_rows=empty_text_rows,
        embedding_dim=embedding_dim,
        save_unique_image_cache=save_unique,
        save_row_aligned_image_matrix=config.cache.save_row_aligned_image_matrix,
    )
