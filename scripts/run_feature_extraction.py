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

from feature_extraction import load_feature_extraction_config, run_feature_extraction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature extraction and cache entrypoint.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["nuswide", "mirflickr25k", "mscoco"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--config",
        default="configs/feature_extraction.yaml",
        help="Path to feature extraction config.",
    )
    parser.add_argument(
        "--view",
        default=None,
        help="Optional dataset view override (e.g. canonical, ra_top10 for nuswide).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan manifest and report planned extraction stats without writing cache files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of manifest rows to process from the beginning.",
    )
    parser.add_argument(
        "--feature-set-id",
        default=None,
        help="Optional explicit feature set id. Defaults to auto-generated id.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory when feature_set_id already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    cfg = load_feature_extraction_config(PROJECT_ROOT, config_path)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    stats = run_feature_extraction(
        dataset=args.dataset,
        config=cfg,
        view=args.view,
        dry_run=args.dry_run,
        limit=args.limit,
        feature_set_id=args.feature_set_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
