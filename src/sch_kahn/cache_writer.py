from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"SCH-KANH output already exists: {output_dir}. Use --overwrite to replace."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def save_array(path: Path, arr: np.ndarray) -> None:
    np.save(path, np.asarray(arr))


def write_meta(path: Path, payload: dict[str, Any]) -> None:
    body = dict(payload)
    body["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)
