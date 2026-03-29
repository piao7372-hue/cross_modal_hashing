from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .config import PseudoConfig


@dataclass(frozen=True)
class PseudoBuildResult:
    s_pseudo: sparse.csr_matrix
    metadata: dict[str, Any]


class PseudoLabelBuilder:
    def __init__(self, cfg: PseudoConfig) -> None:
        self.cfg = cfg

    def ensure_with_pseudo_ready(self) -> None:
        if self.cfg.source_mode == "unconfigured":
            raise RuntimeError(
                "pipeline_mode=with_pseudo requires configured pseudo source; source_mode is `unconfigured`"
            )
        if self.cfg.source_mode != "external_matrix":
            raise RuntimeError(
                "Unsupported pseudo source_mode. Only `external_matrix` is allowed until low-dim Z source is locked"
            )
        if self.cfg.external_matrix_path is None:
            raise RuntimeError("pseudo.external_matrix_path is required for source_mode=external_matrix")
        if not self.cfg.external_matrix_path.exists():
            raise FileNotFoundError(f"Pseudo matrix file not found: {self.cfg.external_matrix_path}")

    def build(self, expected_rows: int) -> PseudoBuildResult:
        self.ensure_with_pseudo_ready()
        path = self.cfg.external_matrix_path
        assert path is not None

        suffix = path.suffix.lower()
        if suffix == ".npz":
            s_pseudo = sparse.load_npz(path).tocsr().astype(np.float32)
        elif suffix == ".npy":
            arr = np.load(path)
            if arr.ndim != 2:
                raise RuntimeError(f"Pseudo matrix must be 2D: {path}")
            s_pseudo = sparse.csr_matrix(np.asarray(arr, dtype=np.float32))
        else:
            raise RuntimeError("pseudo.external_matrix_path must be .npz or .npy")

        if s_pseudo.shape[0] != expected_rows or s_pseudo.shape[1] != expected_rows:
            raise RuntimeError(
                f"Pseudo matrix shape mismatch: expected ({expected_rows}, {expected_rows}), got {s_pseudo.shape}"
            )

        meta = {
            "pseudo_source_mode": self.cfg.source_mode,
            "z_source": self.cfg.z_source,
            "clustering_method": self.cfg.clustering_method,
            "n_clusters": self.cfg.n_clusters,
            "pseudo_seed": self.cfg.pseudo_seed,
            "external_matrix_path": path.as_posix(),
        }
        return PseudoBuildResult(s_pseudo=s_pseudo, metadata=meta)
