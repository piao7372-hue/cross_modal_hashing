from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sch_kahn.trainer_config import load_training_mainline_config  # noqa: E402
from sch_kahn.trainer_step import run_training_mainline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCH-KANH formal training mainline v1.")
    parser.add_argument("--config", default="configs/sch_kahn_training_mainline.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from run_dir/last.pt")
    parser.add_argument("--run-name", help="Override train.run_name for this execution")
    parser.add_argument(
        "--stop-after-epochs",
        type=int,
        help="Stop after the given completed epoch count for this execution without changing config num_epochs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_training_mainline_config(PROJECT_ROOT, (PROJECT_ROOT / args.config).resolve())
    result = run_training_mainline(
        cfg,
        resume=args.resume,
        run_name=args.run_name,
        stop_after_epochs=args.stop_after_epochs,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
