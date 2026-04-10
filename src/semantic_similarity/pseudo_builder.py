from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .config import PseudoConfig, RuntimeConfig
from .feature_reader import FeatureCacheInputs
from .lowdim_projection import build_shared_linear_tanh_projection
from .spectral_clustering import run_spectral_kmeans


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
        if self.cfg.source_mode == "spectral_clustering":
            if self.cfg.z_source != "fused_projection_from_x_i_x_t":
                raise RuntimeError("spectral_clustering mainline requires z_source=fused_projection_from_x_i_x_t")
            if self.cfg.projection_mode != "shared_linear_tanh":
                raise RuntimeError("spectral_clustering mainline requires projection_mode=shared_linear_tanh")
            if self.cfg.fusion_mode != "arithmetic_mean":
                raise RuntimeError("spectral_clustering mainline requires fusion_mode=arithmetic_mean")
            if self.cfg.z_dim is None or self.cfg.z_dim <= 0:
                raise RuntimeError("spectral_clustering mainline requires z_dim > 0")
            if self.cfg.affinity_builder != "sparse_knn_cosine":
                raise RuntimeError("spectral_clustering mainline requires affinity_builder=sparse_knn_cosine")
            if self.cfg.affinity_k is None or self.cfg.affinity_k <= 0:
                raise RuntimeError("spectral_clustering mainline requires affinity_k > 0")
            if self.cfg.laplacian_type != "symmetric_normalized":
                raise RuntimeError("spectral_clustering mainline requires laplacian_type=symmetric_normalized")
            if self.cfg.clustering_method != "spectral_kmeans":
                raise RuntimeError("spectral_clustering mainline requires clustering_method=spectral_kmeans")
            if self.cfg.n_clusters is None or self.cfg.n_clusters <= 1:
                raise RuntimeError("spectral_clustering mainline requires n_clusters > 1")
            return
        if self.cfg.source_mode == "external_matrix":
            if self.cfg.external_matrix_path is None:
                raise RuntimeError("pseudo.external_matrix_path is required for source_mode=external_matrix")
            if not self.cfg.external_matrix_path.exists():
                raise FileNotFoundError(f"Pseudo matrix file not found: {self.cfg.external_matrix_path}")
            return
        raise RuntimeError(f"Unsupported pseudo source_mode: {self.cfg.source_mode}")

    def _build_formal_mainline(
        self,
        *,
        features: FeatureCacheInputs,
        runtime_cfg: RuntimeConfig,
    ) -> PseudoBuildResult:
        assert self.cfg.z_dim is not None
        assert self.cfg.affinity_k is not None
        assert self.cfg.n_clusters is not None

        projection = build_shared_linear_tanh_projection(
            x_i=features.x_i,
            x_t=features.x_t,
            z_dim=self.cfg.z_dim,
            seed=self.cfg.pseudo_seed,
        )
        clustering = run_spectral_kmeans(
            z=projection.z,
            affinity_k=self.cfg.affinity_k,
            block_rows=runtime_cfg.block_rows,
            n_clusters=self.cfg.n_clusters,
            seed=self.cfg.pseudo_seed,
        )

        support = clustering.affinity_graph.copy().tocsr()
        row_ids = np.repeat(np.arange(support.shape[0], dtype=np.int32), np.diff(support.indptr))
        col_ids = support.indices.astype(np.int32, copy=False)
        same_cluster = clustering.labels[row_ids] == clustering.labels[col_ids]
        data = np.where(same_cluster, np.float32(1.0), np.float32(0.0)).astype(np.float32)
        s_pseudo = sparse.csr_matrix((data, support.indices, support.indptr), shape=support.shape, dtype=np.float32)
        s_pseudo.eliminate_zeros()
        s_pseudo = (s_pseudo + sparse.eye(support.shape[0], format="csr", dtype=np.float32)).tocsr()
        s_pseudo.data = np.ones_like(s_pseudo.data, dtype=np.float32)
        s_pseudo.sum_duplicates()
        s_pseudo.sort_indices()
        s_pseudo.eliminate_zeros()

        meta = {
            "pseudo_source_mode": "spectral_clustering",
            "z_source": self.cfg.z_source,
            "projection_mode": self.cfg.projection_mode,
            "fusion_mode": self.cfg.fusion_mode,
            "clustering_method": self.cfg.clustering_method,
            "n_clusters": self.cfg.n_clusters,
            "pseudo_seed": self.cfg.pseudo_seed,
            "z_dim": self.cfg.z_dim,
            "affinity_builder": self.cfg.affinity_builder,
            "affinity_k": self.cfg.affinity_k,
            "laplacian_type": self.cfg.laplacian_type,
            "pseudo_dims": projection.metadata["pseudo_dims"],
            "pseudo_label_count": clustering.metadata["pseudo_label_count"],
            "spectral_embedding_dim": clustering.metadata["spectral_embedding_dim"],
            "pseudo_support_nnz": clustering.metadata["pseudo_support_nnz"],
            "s_pseudo_storage": "csr_sparse_on_pseudo_support",
            "s_final_storage": "csr_sparse_on_union_support",
        }
        return PseudoBuildResult(s_pseudo=s_pseudo, metadata=meta)

    def _build_compat_external_matrix(self, *, expected_rows: int) -> PseudoBuildResult:
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
            "pseudo_source_mode": "external_matrix",
            "z_source": self.cfg.z_source,
            "clustering_method": self.cfg.clustering_method,
            "n_clusters": self.cfg.n_clusters,
            "pseudo_seed": self.cfg.pseudo_seed,
            "external_matrix_path": path.as_posix(),
            "s_pseudo_storage": "external_matrix_compat_input",
            "s_final_storage": "csr_sparse_on_union_support",
        }
        return PseudoBuildResult(s_pseudo=s_pseudo, metadata=meta)

    def build(
        self,
        *,
        features: FeatureCacheInputs,
        runtime_cfg: RuntimeConfig,
    ) -> PseudoBuildResult:
        self.ensure_with_pseudo_ready()
        if self.cfg.source_mode == "spectral_clustering":
            return self._build_formal_mainline(features=features, runtime_cfg=runtime_cfg)
        if self.cfg.source_mode == "external_matrix":
            return self._build_compat_external_matrix(expected_rows=features.rows)
        raise RuntimeError(f"Unsupported pseudo source_mode: {self.cfg.source_mode}")
