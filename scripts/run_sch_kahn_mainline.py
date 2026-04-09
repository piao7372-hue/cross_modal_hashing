from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sch_kahn import load_sch_kahn_config, run_sch_kahn_mainline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCH-KANH mainline forward-only v2 pipeline (Zout->V->B).")
    parser.add_argument("--dataset", required=True, choices=["nuswide", "mirflickr25k", "mscoco"])
    parser.add_argument("--feature-set-id", required=True, help="Feature cache id under data/processed/<dataset>/feature_cache")
    parser.add_argument("--sch-set-id", required=True, help="SCH-KANH cache id under data/processed/<dataset>/sch_kahn_cache")
    parser.add_argument("--config", default="configs/sch_kahn_mainline.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_sch_kahn_config(PROJECT_ROOT, (PROJECT_ROOT / args.config).resolve())
    stats = run_sch_kahn_mainline(
        dataset=args.dataset,
        feature_set_id=args.feature_set_id,
        sch_set_id=args.sch_set_id,
        config=cfg,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
