"""SurGE 정답 토픽 170개의 facet 캐시를 미리 채웁니다.

**왜 따로 채우나** — 평가 본체는 토픽마다 검색 4회를 도는데, 그 중 `+facets` 한 번이
OpenRouter 를 부릅니다. 호출 하나가 서면 평가 전체가 섭니다(실제로 2026-08-13 에
16분 정지). 캐시를 미리 채워두면 본체는 네트워크를 아예 안 타고, 실행 시간도
결정적이 됩니다.

캐시 키는 `(PROMPT_VERSION, model, n_facets, topic)` 이라 본체와 **같은 기본 설정**을
써야 맞습니다. 여기서 config 를 건드리면 캐시가 안 맞아 본체가 다시 호출합니다.

실패한 토픽은 조용히 넘기지 않고 끝에 목록으로 남깁니다 — 그 토픽들은 본체에서
규칙 기반 fallback 으로 내려가고, 그러면 `+facets` 설정의 측정이 오염됩니다.
"""
from __future__ import annotations

import logging
import os
import sys
import time

from survey_search.core.facets import FacetConfig, cache_key, decompose, load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prewarm")
load_dotenv(".env")

from survey_search.eval.surge import load_gold  # noqa: E402

cfg = FacetConfig().resolved()
topics = [t.topic for t in load_gold()]
todo = [t for t in topics
        if not (cfg.cache_dir / f"{cache_key(t, cfg)}.json").exists()]

# 호출 하나가 실측 40초라 순차로는 142개에 1.6시간입니다. 서로 다른 토픽만 맡도록
# 슬라이스를 나눠 여러 개 띄웁니다 — 겹치지 않으니 중복 호출도, 캐시 경합도 없습니다.
#   for s in 0 1 2; do SHARD=$s NSHARDS=3 python scripts/prewarm_facets.py & done
shard, nshards = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARDS", 1))
todo = todo[shard::nshards]

log.info("샤드 %d/%d — 맡은 토픽 %d개 (전체 %d, 캐시 없음 %d) model=%s deadline=%.0fs",
         shard, nshards, len(todo), len(topics),
         sum(1 for t in topics
             if not (cfg.cache_dir / f"{cache_key(t, cfg)}.json").exists()),
         cfg.model, cfg.deadline_s)

failed: list[tuple[str, str]] = []
t_start = time.perf_counter()
for i, topic in enumerate(todo, 1):
    _, st = decompose(topic)
    if st.source != "llm":
        failed.append((topic, "; ".join(st.warnings) or st.source))
        log.warning("[%d/%d] 실패(%s) %s", i, len(todo), st.source, topic[:60])
    if i % 10 == 0 or i == len(todo):
        rate = (time.perf_counter() - t_start) / i
        log.info("[%d/%d] 평균 %.1fs/토픽, 남은 예상 %.1f분, 실패 %d",
                 i, len(todo), rate, rate * (len(todo) - i) / 60, len(failed))

log.info("완료: %d개 중 %d개 성공, %d개 실패 (총 %.1f분)",
         len(todo), len(todo) - len(failed), len(failed),
         (time.perf_counter() - t_start) / 60)
for topic, why in failed:
    log.error("  실패: %s -> %s", topic[:70], why[:120])
sys.exit(1 if failed else 0)
