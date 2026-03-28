from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_cleaning.config import load_dataset_config, load_runtime_config  # noqa: E402
from data_cleaning.datasets.nuswide_top10_filter import NUSWIDERATop10Filter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NUS-WIDE top-10 filtering for paper alignment.")
    parser.add_argument(
        "--config",
        default="configs/dataset_cleaning.yaml",
        help="Path to runtime cleaning config.",
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/datasets/nuswide.yaml",
        help="Path to dataset config (nuswide).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect stats only; do not write filtered output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of filtered records to write to filtered_manifest.jsonl.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_cfg_path = (PROJECT_ROOT / args.config).resolve()
    dataset_cfg_path = (PROJECT_ROOT / args.dataset_config).resolve()
    runtime = load_runtime_config(PROJECT_ROOT, runtime_cfg_path)
    dataset = load_dataset_config(dataset_cfg_path)
    logging.basicConfig(
        level=getattr(logging, runtime.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    runner = NUSWIDERATop10Filter(
        project_root=PROJECT_ROOT,
        runtime_config=runtime,
        dataset_config=dataset,
    )
    stats = runner.run(dry_run=(args.dry_run or runtime.dry_run_default), limit=args.limit)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
