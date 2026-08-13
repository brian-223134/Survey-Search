"""facet 을 켠 상태에서 BM25 가 여전히 이득인가.

170토픽 표에서 답이 안 나오는 질문이 하나 있었습니다. 그 표의 설정은 **누적**이라
`+facets` 행에도 BM25 가 켜져 있습니다. 그래서 아래 둘을 구분할 수 없습니다.

- BM25 가 facet 위에서도 꼬리 recall 을 더해 주는가
- 아니면 **facet fan-out 이 이미 BM25 가 주던 것을 덮어서** 이제는 nDCG 손해만 남는가

170토픽 기준 BM25 단독 효과는 R@1500 **+3.1%p** / nDCG **−0.023** 이었습니다.
facet 이 쿼리를 12~16개로 늘리면서 어휘 다양성을 이미 확보한다면, 두 번째일 수 있습니다.

**비교 대상은 이미 있는 `data/surge_eval_170.json` 의 `+facets` 행입니다.**
facet 캐시가 고정돼 있고 dense·BM25·RRF 가 전부 결정적이라, 같은 토픽·같은 컷오프로
새 설정 하나만 돌려서 비교해도 됩니다 — 4설정을 다시 돌릴 이유가 없습니다.
"""
import json
import logging
import math
import statistics
from pathlib import Path

from survey_search.core.facets import load_dotenv
from survey_search.eval.surge import aggregate, load_gold, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
load_dotenv(".env")

from survey_search.backends.faiss_duckdb import FaissDuckDBBackend

CONFIGS = {
    "+facets -bm25": dict(facets=True, freshness=True, lexical=False),
}

topics = load_gold()
log = logging.getLogger(__name__)
log.info("토픽 %d개 × 설정 %d개 -> 검색 %d회 (LLM 호출 0회 — facet 캐시 사용)",
         len(topics), len(CONFIGS), len(topics) * len(CONFIGS))

res = run(topics, backend=FaissDuckDBBackend(), configs=CONFIGS, n_papers=1500,
          checkpoint=Path("data/bm25_in_facets.ckpt.jsonl"))

new = {k: aggregate(v) for k, v in res.items()}
json.dump(new, open("data/bm25_in_facets.json", "w"), indent=2)

# 기존 170토픽 표의 해당 행과 나란히 찍습니다. 눈으로 비교할 수 있어야 결론이 섭니다.
old = json.load(open("data/surge_eval_170.json"))
cuts = [50, 100, 500, 1500]
rows = [("+facets (BM25 켬)", old["+facets"]), ("+facets -bm25 (BM25 끔)", new["+facets -bm25"])]


def rec(agg, c: int) -> float:
    """`aggregate()` 는 int 키를, JSON 에서 읽은 것은 문자열 키를 줍니다. 둘 다 받습니다."""
    r = agg["recall"]
    return r[str(c)] if str(c) in r else r[c]


print()
print(f"{'설정':24}" + "".join(f"{'R@' + str(c):>9}" for c in cuts) + f"{'nDCG':>9}")
print("-" * 72)
for name, a in rows:
    print(f"{name:24}" + "".join(f"{rec(a, c):>8.1%} " for c in cuts) + f"{a['ndcg']:>8.3f}")

d_on, d_off = rows[0][1], rows[1][1]
print()
print("BM25 를 켜서 얻는 것 (facet 이 이미 켜진 상태에서):")
for c in cuts:
    print(f"  R@{c:<5} {(rec(d_on, c) - rec(d_off, c)) * 100:+6.1f}%p")
print(f"  nDCG    {d_on['ndcg'] - d_off['ndcg']:+6.3f}")

# **평균 차이만 보면 안 됩니다.** 여기 나오는 격차는 0.5%p 수준이라 표본 잡음과
# 구분이 안 될 수 있습니다. 토픽별로 짝지어 승패와 표준오차를 같이 찍습니다.
print()
print("토픽별 짝지은 비교 (|평균차| < 2×SE 면 방향을 단정할 수 없습니다):")


def per_topic(path: str, cfg_name: str) -> dict[int, dict]:
    out = {}
    for ln in Path(path).read_text().splitlines()[1:]:
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if d["config"] == cfg_name:
            out[d["survey_id"]] = d
    return out


a_on = per_topic("data/surge_eval_170.ckpt.jsonl", "+facets")
a_off = per_topic("data/bm25_in_facets.ckpt.jsonl", "+facets -bm25")
ids = sorted(set(a_on) & set(a_off))
if ids:
    for c in cuts:
        diffs = [a_on[i]["recall"][str(c)] - a_off[i]["recall"][str(c)] for i in ids]
        win = sum(1 for x in diffs if x > 1e-9)
        loss = sum(1 for x in diffs if x < -1e-9)
        m = statistics.mean(diffs)
        se = statistics.stdev(diffs) / math.sqrt(len(diffs))
        print(f"  R@{c:<5} {m * 100:+6.2f}%p ± {se * 100:.2f}(SE)  BM25 승 {win} / 패 {loss}")
    diffs = [a_on[i]["ndcg"] - a_off[i]["ndcg"] for i in ids]
    m = statistics.mean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    print(f"  nDCG   {m:+7.4f} ± {se:.4f}(SE)  (토픽 {len(ids)}개)")

log.info("결과 -> data/bm25_in_facets.json")
