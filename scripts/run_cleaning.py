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

from data_cleaning import load_runtime_config, run_pipeline, supported_datasets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset cleaning entrypoint.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=supported_datasets(),
        help="Dataset name.",
    )
    parser.add_argument(
        "--config",
        default="configs/dataset_cleaning.yaml",
        help="Path to runtime cleaning config.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional explicit dataset config path. Defaults to configs/datasets/<dataset>.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect stats only; do not write canonical output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of clean records to write to clean_manifest.jsonl.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_cfg_path = (PROJECT_ROOT / args.config).resolve()
    dataset_cfg_path = (
        (PROJECT_ROOT / args.dataset_config).resolve()
        if args.dataset_config
        else (PROJECT_ROOT / f"configs/datasets/{args.dataset}.yaml").resolve()
    )

    runtime = load_runtime_config(PROJECT_ROOT, runtime_cfg_path)
    logging.basicConfig(
        level=getattr(logging, runtime.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    stats = run_pipeline(
        dataset_name=args.dataset,
        project_root=PROJECT_ROOT,
        runtime_config_path=runtime_cfg_path,
        dataset_config_path=dataset_cfg_path,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
