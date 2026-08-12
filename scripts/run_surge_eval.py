"""SurGE 평가 실행 — MMR λ 변형을 포함한 넓은 ablation.

    python scripts/run_surge_eval.py --limit 40 --out data/surge_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from survey_search.core.facets import load_dotenv
from survey_search.eval.surge import aggregate, load_gold, render, run

log = logging.getLogger(__name__)

CONFIGS = {
    "dense-only":    dict(lexical=False),
    "+bm25":         dict(),
    "+freshness":    dict(freshness=True),
    "+facets":       dict(facets=True, freshness=True),
    "+mmr λ=0.9":    dict(facets=True, freshness=True, diversity=True, mmr_lambda=0.9),
    "+mmr λ=0.7":    dict(facets=True, freshness=True, diversity=True, mmr_lambda=0.7),
    "+mmr λ=0.3":    dict(facets=True, freshness=True, diversity=True, mmr_lambda=0.3),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--n-papers", type=int, default=1500)
    ap.add_argument("--out", type=Path, default=Path("data/surge_eval.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(".env")

    topics = load_gold()[: args.limit]
    log.info("토픽 %d개, 설정 %d개 -> 검색 %d회",
             len(topics), len(CONFIGS), len(topics) * len(CONFIGS))

    from survey_search.backends.faiss_duckdb import FaissDuckDBBackend

    results = run(topics, backend=FaissDuckDBBackend(), configs=CONFIGS,
                  n_papers=args.n_papers)
    print("\n" + render(results))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {name: {"aggregate": aggregate(s), "per_topic": [asdict(x) for x in s]}
         for name, s in results.items()}, ensure_ascii=False, indent=2))
    log.info("결과 -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
