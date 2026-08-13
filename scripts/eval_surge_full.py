"""SurGE 정답 토픽 170개 전체로 ablation.

**설정 5개.** 16토픽 측정에서 다양성(MMR)과 재랭킹은 기준선 대비 격차가 −8~19%p 로
커서 결론이 이미 확실합니다. 여기서 확인하는 것은 격차가 작아 표본이 필요한 항목들입니다:
BM25(R@50 −2.0%p / R@1500 +3.1%p), freshness(nDCG +0.040), 그리고 **facet 을 켠 상태에서
BM25 가 여전히 이득인가**(`+facets -bm25`).

제외한 설정: `+mmr λ=0.9/0.7/0.3`, `+rerank *`.
필요하면 `scripts/run_surge_eval.py` 와 `scripts/eval_rerank*.py` 로 따로 돌리세요.

**v2 (2026-08-13 05:45~)** — 1차 실행분(`surge_eval_170.json`)은 아래 두 문제가 있는
코드로 돌았습니다. 결론은 같지만 숫자를 재현할 수 없어 다시 돌립니다.

- BM25 정렬에 동점 규칙이 없어 **같은 토픽 재검색이 다른 결과**를 냈습니다
- `get_papers` 의 OR 조인 + 중복 조회로 검색 1회가 227초였습니다 (지금 14초)
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
    "dense only":     dict(lexical=False),
    "+bm25":          dict(),
    "+freshness":     dict(freshness=True),
    "+facets":        dict(facets=True, freshness=True),
    "+facets -bm25":  dict(facets=True, freshness=True, lexical=False),   # 권장 설정
}

OUT = Path("data/surge_eval_170_v2.json")
CKPT = Path("data/surge_eval_170_v2.ckpt.jsonl")

topics = load_gold()
log = logging.getLogger(__name__)
log.info("토픽 %d개 × 설정 %d개 -> 검색 %d회 (LLM 0회 — facet 캐시 사용)",
         len(topics), len(CONFIGS), len(topics) * len(CONFIGS))

# 한 시간 넘게 돌므로 (설정,토픽) 쌍마다 이어씁니다. 죽어도 같은 명령으로 이어집니다.
# 조건을 바꾸려면 이 파일을 지우세요 — 섞이면 거부합니다.
res = run(topics, backend=FaissDuckDBBackend(), configs=CONFIGS, n_papers=1500,
          checkpoint=CKPT)

print("\n" + render(res))
json.dump({k: aggregate(v) for k, v in res.items()}, open(OUT, "w"), indent=2)
log.info("결과 -> %s", OUT)
