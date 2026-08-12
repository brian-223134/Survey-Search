"""검색 손실이 '깊이 부족' 인지 '표현 부족' 인지 가릅니다.

천장 측정은 검색 손실 18.1% 를 보여 주지만 원인은 말해 주지 않습니다. 둘 중 하나인데
처방이 완전히 다릅니다:

- **깊이 부족** — top_k 를 키우면 정답이 후보에 들어옴. `dense_top_k` 만 올리면 됩니다
- **표현 부족** — 아무리 깊게 파도 안 나옴. 초록이 그 논문을 대표하지 못한다는 뜻이고,
  **본문 도입이나 임베딩 교체가 의미를 갖는 유일한 경우**입니다

top_k 를 키워 가며 풀 recall 이 계속 오르는지 평평해지는지 봅니다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

from survey_search.core.dedup import strip_version
from survey_search.core.facets import load_dotenv
from survey_search.eval.surge import load_gold
from survey_search.types import SearchConfig

log = logging.getLogger(__name__)

DEPTHS = [500, 2000, 8000]
N_TOPICS = 8


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(".env")

    from survey_search.backends.faiss_duckdb import FaissDuckDBBackend
    from survey_search.search import search_topic

    backend = FaissDuckDBBackend()
    topics = load_gold()[:N_TOPICS]

    rows = []
    for depth in DEPTHS:
        recalls, pools = [], []
        for g in topics:
            cfg = SearchConfig(n_papers=200_000, facets=True, freshness=True,
                               dense_top_k=depth, lexical_top_k=depth)
            if g.date:
                cfg = replace(cfg, date_max=g.date)
            r = search_topic(g.topic, backend=backend, config=cfg)
            got = {strip_version(p) for p in r.ids()}
            gold = set(g.gold_ids)
            recalls.append(len(gold & got) / len(gold))
            pools.append(len(got))
        rows.append((depth, sum(recalls) / len(recalls), sum(pools) / len(pools)))
        log.info("depth=%d 완료", depth)

    print(f"\n{'쿼리당 top_k':>12}{'평균 후보 풀':>14}{'풀 recall':>12}{'증분':>9}")
    print("-" * 48)
    prev = None
    for depth, rec, pool in rows:
        delta = "" if prev is None else f"{(rec - prev) * 100:+.1f}%p"
        print(f"{depth:>12,}{pool:>14,.0f}{rec:>11.1%}{delta:>9}")
        prev = rec

    first, last = rows[0][1], rows[-1][1]
    span = rows[-1][0] / rows[0][0]
    print(f"\ntop_k 를 {span:.0f}배 키워 recall {(last - first) * 100:+.1f}%p")
    if last - first < 0.05:
        print("=> 깊이는 병목이 아닙니다. 아무리 파도 안 나오는 논문이 있다는 뜻이고,")
        print("   표현(초록 vs 본문, 임베딩 모델)이 남은 후보입니다.")
    else:
        print("=> 깊이가 아직 병목입니다. top_k 를 올리는 것이 가장 싼 개선입니다.")

    with open("data/retrieval_depth.json", "w") as f:
        json.dump([{"top_k": d, "pool_recall": r, "pool_size": p} for d, r, p in rows], f,
                  indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
