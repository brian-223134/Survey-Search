"""P3.2 — SurveyForge `GeneralRAG_langchain.retrieve_id` 드롭인 대체.

원본: [`../SurveyForge/code/src/rag.py:227`](../../../SurveyForge/code/src/rag.py).
SurveyForge 는 2단계 검색을 씁니다: 1단계에서 넓게 뽑아 서브셋을 만들고,
2단계에서 그 서브셋(`filter`)으로 좁혀 `rerank='citation'` 을 겁니다.
그 구조를 유지해야 통제 비교가 되므로 `filter` 와 `rerank` 를 그대로 받습니다.

`rerank` 인자 번역:

| 원본 | 이 어댑터 |
|---|---|
| `'raw'` | RRF 점수 그대로 (freshness off) |
| `'citation'` | **freshness 랭킹으로 대체** — 이게 이 프로젝트의 요점입니다 |
| `'citation_period'` | 동일 (원본의 `sort_by_citation_period` 자리) |

원본의 `sort_by_citation_period` 는 시간 윈도우 안에서 인용수 정렬이라 최근 6~12개월
논문이 구조적으로 탈락합니다. 여기서는 연령 정규화 인용률 백분위 + recency 로 대체합니다.
**대체했다는 사실이 `last_stats` 에 남습니다** — 조용히 바꾸면 A/B 비교가 무의미해집니다.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from survey_search.backends.faiss_duckdb import FaissDuckDBBackend
from survey_search.search import search_topic
from survey_search.types import SearchConfig

log = logging.getLogger(__name__)


def _filter_to_paper_ids(filter_obj, index_to_id: dict[int, str]) -> set[str] | None:
    """SurveyForge 의 `filter` 를 paper_id 집합으로 번역합니다.

    원본은 `{'id_selector': faiss.IDSelectorArray([...])}` 를 넘깁니다
    (`code/src/utils.py:206` 의 `get_index_filter`). 편의를 위해 다음도 받습니다:
    `{'paper_ids': [...]}`, 정수 인덱스 목록, arxiv id 목록.
    """
    if filter_obj is None:
        return None

    if isinstance(filter_obj, dict):
        if "paper_ids" in filter_obj:
            return {str(x) for x in filter_obj["paper_ids"]}
        sel = filter_obj.get("id_selector")
        if sel is not None:
            import faiss

            try:
                ids = faiss.rev_swig_ptr(sel.ids, sel.n)
            except AttributeError as e:
                raise TypeError(
                    f"id_selector 에서 id 를 읽지 못했습니다({e}). "
                    "IDSelectorArray 대신 {'paper_ids': [...]} 형태로 넘기세요."
                ) from e
            return {index_to_id[int(i)] for i in ids if int(i) in index_to_id}
        raise TypeError(f"알 수 없는 filter 형태: {list(filter_obj)}")

    items = list(filter_obj)
    if not items:
        return set()
    if isinstance(items[0], str):
        return {str(x) for x in items}
    return {index_to_id[int(i)] for i in items if int(i) in index_to_id}


class SurveySearchRAG:
    """`GeneralRAG_langchain` 호환. 검색만 survey-search 로 대체합니다."""

    def __init__(
        self,
        *args,
        backend: FaissDuckDBBackend | None = None,
        config: SearchConfig | None = None,
        **kwargs,
    ) -> None:
        # 원본 생성자는 args/kwargs 가 많습니다. 시그니처 호환만 하고 쓰지 않습니다 —
        # 어느 인덱스를 볼지는 survey_search.assets 가 정합니다.
        self.backend = backend or FaissDuckDBBackend()
        self.config = config or SearchConfig()
        self._last_result = None

    def retrieve_id(
        self,
        query,
        search_type: str = "similarity",
        rerank: str = "raw",
        top_k: int = 10,
        max_out: int = 10000,
        filter=None,
        fetch_k: int = 20,
        **kwargs,
    ) -> list[str]:
        """원본과 같은 시그니처. 쿼리 문자열 하나 또는 목록을 받습니다."""
        queries = [query] if isinstance(query, str) else list(query)
        _, index_to_id = self.backend._maps()
        allowed = _filter_to_paper_ids(filter, index_to_id)

        cfg = replace(
            self.config,
            n_papers=min(max_out, top_k * max(len(queries), 1)) if max_out else top_k,
            freshness=rerank in ("citation", "citation_period"),
        )

        # 원본은 쿼리별로 뽑아 union 합니다. 우리는 여러 쿼리를 하나의 facet 으로 묶어
        # RRF 로 융합합니다 — union 은 순위 정보를 버리지만 RRF 는 살립니다.
        topic = queries[0] if len(queries) == 1 else " ; ".join(queries)
        result = search_topic(topic, backend=self.backend, config=cfg)
        self._last_result = result

        ids = result.ids()
        if allowed is not None:
            before = len(ids)
            ids = [i for i in ids if i in allowed]
            log.info("filter 적용: %d -> %d (허용 집합 %d편)", before, len(ids), len(allowed))
            result.stats.warn(
                f"호스트가 넘긴 filter 로 {before - len(ids):,}편 제외 (허용 {len(allowed):,}편)"
            )

        if rerank in ("citation", "citation_period"):
            result.stats.warn(
                f"rerank={rerank!r} 를 freshness 랭킹으로 대체했습니다 — "
                "원본의 인용수 정렬이 아닙니다. A/B 비교 시 이 점을 명시하세요."
            )

        return ids[:max_out] if max_out else ids

    def retrieve_id4citation(self, query, **kwargs) -> list[str]:
        """원본의 인용 검증용 경로 — 재랭킹 없이 그대로."""
        return self.retrieve_id(query, rerank="raw", **kwargs)

    @property
    def last_stats(self):
        return self._last_result.stats if self._last_result else None
