"""facet 쿼리 기반 재랭킹이 기준선을 넘는가.

TOPIC 모드는 이미 측정했고 기준선보다 나빴습니다(R@1500 56.3%, nDCG 0.190 vs
기준선 59.1% / 0.288). 원인은 "서베이 제목을 쿼리로 주면 다른 서베이·개론이
올라온다" 였으므로, 논문을 끌어올린 facet 의 쿼리로 물으면 달라져야 합니다.

**기준선을 못 넘으면 이 경로도 접습니다.**
"""
import json, logging
from survey_search.core.facets import load_dotenv
from survey_search.eval.surge import aggregate, load_gold, render, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
load_dotenv(".env")

from survey_search.backends.faiss_duckdb import FaissDuckDBBackend
from survey_search.core.rerank import CrossEncoderReranker, QueryMode, RerankConfig

# 모드별로 재랭커를 따로 두되 모델은 공유합니다 (2.2GB 를 세 번 읽지 않도록)
rr_facet = CrossEncoderReranker(RerankConfig(top_n=3000, query_mode=QueryMode.FACET))
rr_max = CrossEncoderReranker(RerankConfig(top_n=3000, query_mode=QueryMode.FACET_MAX,
                                           facet_max_n=3))
rr_max._model = rr_facet.model
rr_max._device = rr_facet._device

CONFIGS = {
    "facets (기준선)":   dict(facets=True, freshness=True),
    "+rerank FACET":     dict(facets=True, freshness=True, rerank=True, reranker=rr_facet),
    "+rerank FACET_MAX": dict(facets=True, freshness=True, rerank=True, reranker=rr_max),
}

topics = load_gold()[:16]
res = run(topics, backend=FaissDuckDBBackend(), configs=CONFIGS, n_papers=1500)
print("\n" + render(res))
print("\n참고 — 이전 측정: +rerank TOPIC  R@1500 56.3%  nDCG 0.190")
json.dump({k: aggregate(v) for k, v in res.items()},
          open("data/rerank_facet_eval.json", "w"), indent=2)
