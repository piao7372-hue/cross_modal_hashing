from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sch_kahn.trainer_config import load_trainer_smoke_config  # noqa: E402
from sch_kahn.trainer_step import run_single_trainer_smoke_step  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCH-KANH trainer smoke shadow path (single optimizer step).")
    parser.add_argument("--config", default="configs/sch_kahn_trainer_smoke.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_trainer_smoke_config(PROJECT_ROOT, (PROJECT_ROOT / args.config).resolve())
    result = run_single_trainer_smoke_step(cfg)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
