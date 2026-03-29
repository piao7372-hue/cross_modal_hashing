from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy import sparse



def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Semantic cache output already exists: {output_dir}. Use --overwrite to replace."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)



def save_sparse_npz(path: Path, matrix: sparse.csr_matrix) -> None:
    sparse.save_npz(path, matrix.tocsr().astype("float32"), compressed=True)



def write_meta(path: Path, payload: dict[str, Any]) -> None:
    body = dict(payload)
    body["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)



def density_of(matrix: sparse.csr_matrix) -> float:
    n = matrix.shape[0]
    if n == 0:
        return 0.0
    return float(matrix.nnz) / float(n * n)



def avg_degree_of(matrix: sparse.csr_matrix) -> float:
    n = matrix.shape[0]
    if n == 0:
        return 0.0
    return float(matrix.nnz) / float(n)
