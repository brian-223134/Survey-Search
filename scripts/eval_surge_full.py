"""SurGE 정답 토픽 170개 전체로 ablation.

**설정을 4개로 줄였습니다.** 16토픽 측정에서 다양성(MMR)과 재랭킹은 기준선 대비
격차가 −8~19%p 로 커서 결론이 이미 확실합니다. 표본을 늘려 확인이 필요한 것은
BM25(R@50 −3.0%p / R@1500 +3.4%p)와 freshness(nDCG +0.029)처럼 **격차가 작은
항목**입니다. 7설정 전체를 돌리면 6시간, 4설정이면 2시간입니다.

제외한 설정: `+mmr λ=0.9/0.7/0.3`, `+rerank *`.
필요하면 `scripts/run_surge_eval.py` 와 `scripts/eval_rerank*.py` 로 따로 돌리세요.
"""
import json
import logging
from pathlib import Path

from survey_search.core.facets import load_dotenv
from survey_search.eval.surge import aggregate, load_gold, render, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
load_dotenv(".env")

from survey_search.backends.faiss_duckdb import FaissDuckDBBackend

CONFIGS = {
    "dense only":   dict(lexical=False),
    "+bm25":        dict(),
    "+freshness":   dict(freshness=True),
    "+facets":      dict(facets=True, freshness=True),   # 권장 설정
}

topics = load_gold()
log = logging.getLogger(__name__)
log.info("토픽 %d개 × 설정 %d개 -> 검색 %d회", len(topics), len(CONFIGS),
         len(topics) * len(CONFIGS))

# 두 시간짜리 실행이라 (설정,토픽) 쌍마다 이어씁니다. 죽어도 같은 명령으로 다시 돌리면
# 그 지점부터 잇습니다. 조건을 바꾸려면 이 파일을 지우세요 — 섞이면 거부합니다.
res = run(topics, backend=FaissDuckDBBackend(), configs=CONFIGS, n_papers=1500,
          checkpoint=Path("data/surge_eval_170.ckpt.jsonl"))
print("\n" + render(res))
json.dump({k: aggregate(v) for k, v in res.items()},
          open("data/surge_eval_170.json", "w"), indent=2)
log.info("결과 -> data/surge_eval_170.json")
