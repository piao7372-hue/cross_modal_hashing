from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .sparse_ops import EPS, cosine_topk_sparse, symmetrize_arithmetic_mean


@dataclass(frozen=True)
class SpectralClusteringResult:
    affinity_graph: sparse.csr_matrix
    embedding: np.ndarray
    labels: np.ndarray
    metadata: dict[str, Any]


def _row_l2_normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
    return (arr / np.maximum(norms, EPS)).astype(np.float32)


def _build_sparse_knn_affinity_graph(z: np.ndarray, affinity_k: int, block_rows: int) -> sparse.csr_matrix:
    graph = cosine_topk_sparse(np.asarray(z, dtype=np.float32), k=affinity_k, block_rows=block_rows).tocsr()
    graph.data = np.maximum(graph.data, np.float32(0.0)).astype(np.float32)
    graph.eliminate_zeros()

    graph = symmetrize_arithmetic_mean(graph)
    diag = graph.diagonal()
    missing = np.where(diag <= 0)[0]
    if missing.size > 0:
        filler = sparse.csr_matrix(
            (np.ones(missing.size, dtype=np.float32), (missing, missing)),
            shape=graph.shape,
            dtype=np.float32,
        )
        graph = (graph + filler).tocsr()
        graph.sum_duplicates()
    graph.sort_indices()
    graph.eliminate_zeros()
    return graph


def _symmetric_normalized_laplacian(w: sparse.csr_matrix) -> sparse.csr_matrix:
    deg = np.asarray(w.sum(axis=1)).reshape(-1)
    if np.any(deg <= 0):
        raise RuntimeError("Affinity graph contains zero-degree rows; spectral clustering requires positive degree")
    inv_sqrt = (1.0 / np.sqrt(np.maximum(deg, EPS))).astype(np.float32)
    d_inv = sparse.diags(inv_sqrt, format="csr")
    normalized_affinity = (d_inv @ w @ d_inv).tocsr()
    normalized_affinity.sum_duplicates()
    normalized_affinity.sort_indices()
    laplacian = (sparse.eye(w.shape[0], format="csr", dtype=np.float32) - normalized_affinity).tocsr()
    laplacian.sum_duplicates()
    laplacian.sort_indices()
    laplacian.eliminate_zeros()
    return laplacian


def _run_kmeans(x: np.ndarray, n_clusters: int, seed: int, max_iter: int = 50) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    n = int(arr.shape[0])
    if n_clusters <= 1 or n_clusters >= n:
        raise RuntimeError(f"n_clusters must be in [2, N-1], got n_clusters={n_clusters}, N={n}")

    rng = np.random.default_rng(seed)
    centers = arr[rng.choice(n, size=n_clusters, replace=False)].copy()
    labels = np.full(n, -1, dtype=np.int32)

    for _ in range(max_iter):
        diff = arr[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        new_labels = np.argmin(dist2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if np.any(mask):
                centers[cluster_id] = np.mean(arr[mask], axis=0, dtype=np.float32)
            else:
                centers[cluster_id] = arr[int(rng.integers(0, n))]
    return labels


def run_spectral_kmeans(
    *,
    z: np.ndarray,
    affinity_k: int,
    block_rows: int,
    n_clusters: int,
    seed: int,
) -> SpectralClusteringResult:
    if affinity_k <= 0:
        raise RuntimeError("affinity_k must be > 0")

    n = int(z.shape[0])
    if n_clusters >= n:
        raise RuntimeError(f"n_clusters must be < rows for spectral clustering: {n_clusters} >= {n}")

    affinity = _build_sparse_knn_affinity_graph(z, affinity_k=affinity_k, block_rows=block_rows)
    laplacian = _symmetric_normalized_laplacian(affinity)
    eigvals, eigvecs = eigsh(laplacian, k=n_clusters, which="SM")
    order = np.argsort(eigvals)
    embedding = _row_l2_normalize(np.asarray(eigvecs[:, order], dtype=np.float32))
    labels = _run_kmeans(embedding, n_clusters=n_clusters, seed=seed)

    return SpectralClusteringResult(
        affinity_graph=affinity,
        embedding=embedding,
        labels=labels,
        metadata={
            "affinity_builder": "sparse_knn_cosine",
            "laplacian_type": "symmetric_normalized",
            "clustering_method": "spectral_kmeans",
            "affinity_k": int(affinity_k),
            "n_clusters": int(n_clusters),
            "pseudo_label_count": int(np.unique(labels).shape[0]),
            "pseudo_support_nnz": int(affinity.nnz),
            "spectral_embedding_dim": int(embedding.shape[1]),
        },
    )
