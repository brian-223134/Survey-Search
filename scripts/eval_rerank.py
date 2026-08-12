"""재랭킹이 랭킹 손실 18.6%p 중 얼마를 되찾는가."""
import logging, json
from survey_search.core.facets import load_dotenv
from survey_search.eval.surge import load_gold, run, render, aggregate
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
load_dotenv(".env")
from survey_search.backends.faiss_duckdb import FaissDuckDBBackend
from survey_search.core.rerank import CrossEncoderReranker, RerankConfig

# 재랭커는 한 번만 만들어 모든 설정이 공유합니다 (2.2GB 재적재 방지)
rr = CrossEncoderReranker(RerankConfig(top_n=3000))

CONFIGS = {
  "facets (기준)":        dict(facets=True, freshness=True),
  "+rerank":              dict(facets=True, freshness=True, rerank=True, reranker=rr),
  "+rerank top_k=8000":   dict(facets=True, freshness=True, rerank=True, reranker=rr,
                               dense_top_k=8000, lexical_top_k=8000),
}
topics = load_gold()[:16]
res = run(topics, backend=FaissDuckDBBackend(), configs=CONFIGS, n_papers=1500)
print("\n" + render(res))
json.dump({k: aggregate(v) for k, v in res.items()},
          open("data/rerank_eval.json", "w"), indent=2)
