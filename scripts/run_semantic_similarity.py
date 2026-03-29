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

from semantic_similarity import load_semantic_similarity_config, run_semantic_similarity  # noqa: E402



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic similarity / graph cache pipeline.")
    parser.add_argument("--dataset", required=True, choices=["nuswide", "mirflickr25k", "mscoco"])
    parser.add_argument("--feature-set-id", required=True, help="Feature cache id under data/processed/<dataset>/feature_cache")
    parser.add_argument("--semantic-set-id", required=True, help="Semantic cache id under data/processed/<dataset>/semantic_cache")
    parser.add_argument("--config", default="configs/semantic_similarity.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    cfg = load_semantic_similarity_config(PROJECT_ROOT, (PROJECT_ROOT / args.config).resolve())

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    stats = run_semantic_similarity(
        dataset=args.dataset,
        feature_set_id=args.feature_set_id,
        semantic_set_id=args.semantic_set_id,
        config=cfg,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
