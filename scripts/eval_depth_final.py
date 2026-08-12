"""깊은 풀이 최종 recall 로 이어지는가 — 랭킹이 감당 못 하면 이득이 없습니다."""
import logging, json
from dataclasses import replace
from survey_search.core.facets import load_dotenv
from survey_search.eval.surge import load_gold, run, render, aggregate
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
load_dotenv(".env")
from survey_search.backends.faiss_duckdb import FaissDuckDBBackend

CONFIGS = {
  "facets top_k=2000": dict(facets=True, freshness=True, dense_top_k=2000, lexical_top_k=2000),
  "facets top_k=8000": dict(facets=True, freshness=True, dense_top_k=8000, lexical_top_k=8000),
}
topics = load_gold()[:16]
res = run(topics, backend=FaissDuckDBBackend(), configs=CONFIGS, n_papers=1500)
print("\n" + render(res))
json.dump({k: aggregate(v) for k, v in res.items()},
          open("data/depth_final.json", "w"), indent=2)
