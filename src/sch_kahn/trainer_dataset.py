from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_reader import load_feature_cache_inputs


@dataclass(frozen=True)
class TrainerDatasetMeta:
    feature_cache_dir: Path
    feature_set_id: str
    rows: int
    dim: int
    sample_index_hash: str


class FeatureCacheTrainerDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, feature_cache_dir: Path) -> None:
        features = load_feature_cache_inputs(feature_cache_dir)
        self._x_i = features.x_i.astype(np.float32, copy=False)
        self._x_t = features.x_t.astype(np.float32, copy=False)
        self._sample_indices = np.arange(features.rows, dtype=np.int64)
        self.meta = TrainerDatasetMeta(
            feature_cache_dir=features.feature_cache_dir,
            feature_set_id=features.feature_set_id,
            rows=features.rows,
            dim=features.dim,
            sample_index_hash=features.sample_index_hash,
        )

    def __len__(self) -> int:
        return int(self.meta.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(index)
        if row < 0 or row >= len(self):
            raise IndexError(f"Dataset index out of range: {row}")
        return {
            "x_i": torch.from_numpy(self._x_i[row]),
            "x_t": torch.from_numpy(self._x_t[row]),
            "sample_index": torch.tensor(int(self._sample_indices[row]), dtype=torch.int64),
        }
