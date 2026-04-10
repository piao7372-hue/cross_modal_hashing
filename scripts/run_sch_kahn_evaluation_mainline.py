from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sch_kahn.evaluation_mainline import load_evaluation_mainline_config, run_evaluation_mainline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCH-KANH formal evaluation mainline v1.")
    parser.add_argument("--config", default="configs/sch_kahn_evaluation_mainline.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to the training mainline checkpoint (`last.pt`).")
    parser.add_argument("--run-name", help="Override evaluation.run_name for this execution")
    parser.add_argument(
        "--max-image-groups",
        type=int,
        help="Optional validator/developer bound on the number of val2014 image groups, preserving sample order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_evaluation_mainline_config(PROJECT_ROOT, (PROJECT_ROOT / args.config).resolve())
    result = run_evaluation_mainline(
        cfg,
        checkpoint_path=(PROJECT_ROOT / args.checkpoint).resolve(),
        run_name=args.run_name,
        max_image_groups=args.max_image_groups,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
