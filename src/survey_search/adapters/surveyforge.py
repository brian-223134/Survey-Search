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

## 호스트가 `retrieve_id` 말고도 요구하는 것

코드 대조로 확인한 것들입니다. 없으면 `report_cutoffs_vs_database` 에서 즉시 터집니다
(LLM 호출 전이라 실비 손실은 0원).

| 호스트가 부르는 것 | 위치 |
|---|---|
| `rag.id_to_index` | `main.py:200`, `outline_writer.py:51`, `writer.py:39·50·365` |
| `rag.rag_data['doc_list']` | `main.py:201` — `.metadata['date']` 만 씁니다 |
| `rag.report_window_drops()` | `main.py:303` |

## 인스턴스마다 config 를 달리 주세요

`main.py:259·270` 이 RAG 를 둘 만듭니다. 성격이 달라서 같은 설정을 쓰면 안 됩니다.

| 인스턴스 | 권장 config | 왜 |
|---|---|---|
| `rag_abstract4outline` | `facets=True, freshness=True, lexical=False` | 토픽 수준 쿼리 — facet 이 +18%p 를 내는 곳 |
| `rag_title4citation` | `facets=False, dense_field="title"` | 쿼리가 **논문 제목**입니다. 제목을 하위 주제로 쪼개는 건 의미가 없고, 인용 1건마다 LLM 40초가 붙습니다 |

`retrieve_id4citation` 은 config 와 무관하게 facet 을 **강제로 끕니다** — 아래 참고.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from survey_search.backends.faiss_duckdb import FaissDuckDBBackend
from survey_search.search import search_topic
from survey_search.types import SearchConfig

log = logging.getLogger(__name__)


class _DateDoc:
    """`rag_data['doc_list']` 원소의 최소 대역폭 대체물.

    호스트(`main.py:201`)는 `d.metadata['date']` 하나만 읽어 `min`/`max` 를 냅니다.
    langchain `Document` 를 90만 개 만들 이유가 없습니다.
    """

    __slots__ = ("metadata",)

    def __init__(self, date: str) -> None:
        self.metadata = {"date": date}


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
            if not hasattr(sel, "ids"):
                # 호스트는 항상 IDSelectorArray 를 주지만, 테스트나 다른 호출부가
                # 평범한 목록/딕셔너리를 주는 것도 받습니다 — faiss 객체를 만들지
                # 않고 배선을 검사할 수 있어야 합니다.
                return _filter_to_paper_ids(sel, index_to_id)
            import faiss

            ids = faiss.rev_swig_ptr(sel.ids, sel.n)
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
        self._doc_list: list[_DateDoc] | None = None

    # --- 필터 배선 -------------------------------------------------------------

    def _allowed_ids(self, filter_obj, kwargs: dict, index_to_id, *, where: str):
        """호스트가 필터를 **`filter=` 가 아니라 `**` 로 펼쳐서** 넘깁니다.

        `utils.py:206` 의 `get_index_filter` 가 `{'id_selector': IDSelectorArray}` 를
        돌려주고, 호출부는 전부 `**index_filter` 로 펼칩니다
        (`writer.py:43·81·370`, `outline_writer.py:55`). 그래서 `filter=None` 시그니처만
        두면 `id_selector` 가 `**kwargs` 로 빨려 들어가 **조용히 버려집니다.**

        무시하면 둘을 잃습니다:
        1. `paper_id_cutoff` 시간 게이트 (`utils.py:288` 도 같은 모양을 돌려줍니다)
        2. writer 의 2단계 좁히기 (넓게 뽑고 → 서브셋으로 좁힘)

        둘 다 "예외 없이 결과만 틀리는" 종류라 여기서 명시적으로 집어냅니다.
        """
        sel = kwargs.pop("id_selector", None)
        if sel is not None and filter_obj is None:
            filter_obj = {"id_selector": sel}
        elif sel is not None:
            # 둘 다 온 경우. 원본에 그런 호출은 없지만 조용히 하나를 버리지 않습니다.
            log.warning("%s: filter= 와 id_selector= 가 같이 왔습니다 — 교집합을 씁니다", where)
            a = _filter_to_paper_ids(filter_obj, index_to_id) or set()
            b = _filter_to_paper_ids({"id_selector": sel}, index_to_id) or set()
            return a & b

        allowed = _filter_to_paper_ids(filter_obj, index_to_id)
        if allowed is None:
            # 호스트는 **항상** 넘깁니다. 안 왔다면 배선이 틀린 것이니 남깁니다.
            log.info("%s: 필터가 오지 않았습니다 (호스트 배선이라면 확인하세요)", where)
        return allowed

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
        allowed = self._allowed_ids(filter, kwargs, index_to_id, where="retrieve_id")

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

    def retrieve_id4citation(
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
        """인용 검증 경로. **쿼리 순서를 1:1 로 보존해야 합니다.**

        호스트가 이렇게 씁니다 (`writer.py:365`):

            ids = rag_title4citation.retrieve_id4citation(citations, top_k=1, **index_filter)
            citation_to_ids = {c: i for c, i in zip(citations, ids)}   # ← 순서 의존

        즉 계약은 **쿼리 N개 → id N×top_k 개, 입력 순서대로**입니다. 원본은 쿼리별로
        검색해 순서대로 flatten 합니다(`utils.py:109`).

        **이전 구현은 N개를 `" ; "` 로 이어 붙여 한 번 검색하고 랭킹 순 목록을
        돌려줬습니다.** 예외가 나지 않으면서 `zip` 이 엉뚱한 논문을 붙입니다 —
        본문의 `[1]` 이 다른 논문을 가리키는 서베이가 편당 $2 에 나오고, A/B 평가가
        전부 그 매핑 위에서 계산되므로 비교 자체가 무의미해집니다.

        **facet 은 강제로 끕니다.** 쿼리가 논문 **제목**이라 하위 주제로 쪼개는 것이
        의미가 없고, 인용 1건마다 LLM 40초·$0.0006 이 붙습니다(인용 100건이면 67분).
        끈 사실은 `stats` 에 남깁니다.
        """
        queries = [query] if isinstance(query, str) else list(query)
        if not queries:
            return []
        _, index_to_id = self.backend._maps()
        allowed = self._allowed_ids(filter, kwargs, index_to_id, where="retrieve_id4citation")

        base = replace(
            self.config,
            facets=False,
            freshness=rerank in ("citation", "citation_period"),
            # 필터를 통과한 것 중 top_k 를 채우려면 여유 있게 뽑아야 합니다. 우리는
            # FAISS 에 IDSelector 를 못 걸고 **사후 필터**를 하기 때문입니다.
            n_papers=max(top_k * 50, 1000) if allowed is not None else max(top_k, 1),
        )

        out: list[str] = []
        n_short = n_unfiltered = 0
        last = None
        for q in queries:
            r = search_topic(q, backend=self.backend, config=base)
            last = r
            ids = r.ids()
            if allowed is not None:
                hits = [i for i in ids if i in allowed]
                if not hits:
                    # 정렬을 지키려면 무엇이든 돌려줘야 합니다 — 빈손이면 뒤 인용이
                    # 전부 한 칸씩 밀립니다. 필터를 푼 최상위로 메우고 **셉니다.**
                    hits = ids[:top_k]
                    n_unfiltered += 1
                ids = hits
            if len(ids) < top_k:
                n_short += 1
            out.extend(ids[:top_k])

        self._last_result = last
        if last is not None:
            if self.config.facets:
                last.stats.warn(
                    "retrieve_id4citation: facet 분해를 강제로 껐습니다 — 쿼리가 논문 "
                    "제목이라 하위 주제 분해가 의미 없고 인용 1건마다 LLM 40초가 붙습니다"
                )
            if n_unfiltered:
                last.stats.warn(
                    f"retrieve_id4citation: {n_unfiltered}개 쿼리가 filter 안에서 후보를 "
                    "못 찾아 **필터 밖 최상위로 메웠습니다** — 순서 정렬을 지키기 위해서입니다. "
                    "그 인용은 호스트가 준 서브셋 밖 논문을 가리킵니다"
                )
            if n_short:
                last.stats.warn(
                    f"retrieve_id4citation: {n_short}개 쿼리가 top_k={top_k} 를 못 채웠습니다 "
                    "— 호스트의 zip 정렬이 어긋납니다"
                )

        # **길이 계약을 여기서 검사합니다.** 어긋난 채 넘어가면 호스트가 조용히 잘못된
        # 인용 매핑을 만듭니다. 이 버그는 조용한 것이 문제였습니다.
        want = len(queries) * top_k
        if len(out) != want:
            raise RuntimeError(
                f"retrieve_id4citation 이 순서 계약을 못 지켰습니다: "
                f"쿼리 {len(queries)}개 × top_k {top_k} = {want} 를 기대했는데 {len(out)}개. "
                "호스트가 zip 으로 인용-논문을 맞추므로 길이가 다르면 매핑이 어긋납니다."
            )
        return out[:max_out] if max_out else out

    # --- 호스트가 요구하는 나머지 표면 ------------------------------------------

    @property
    def id_to_index(self) -> dict[str, int]:
        """arxiv_id → FAISS id. `main.py:200` 이 `list(...)` 로 id 목록을 뽑고,
        `get_index_filter` 가 id → 인덱스 변환에 씁니다."""
        id_to_index, _ = self.backend._maps()
        return id_to_index

    @property
    def rag_data(self) -> dict:
        """`main.py:201` 이 `[d.metadata['date'] for d in rag_data['doc_list']]` 로
        **날짜만** 씁니다(`min`/`max`). 908,819편을 Document 로 만들 이유가 없어
        DuckDB 에서 날짜 컬럼만 읽어 가벼운 객체로 쌉니다."""
        if self._doc_list is None:
            rows = self.backend.con.execute(
                "SELECT CAST(date AS VARCHAR) FROM papers WHERE date IS NOT NULL"
            ).fetchall()
            self._doc_list = [_DateDoc(r[0]) for r in rows]
            log.info("rag_data['doc_list']: 날짜 %d건 (date 컬럼만)", len(self._doc_list))
        return {"doc_list": self._doc_list}

    def report_window_drops(self) -> None:
        """원본은 시간창 밖으로 버린 논문 수를 보고합니다(`rag.py:171`).

        이 어댑터에는 시간창 개념이 없습니다 — freshness 는 버리지 않고 순위만 바꿉니다.
        **조용한 no-op 으로 두지 않습니다.** 대신 직전 검색이 무엇을 버렸는지 찍습니다.
        """
        if self._last_result is None:
            log.info("[cutoff/rerank] survey-search: 아직 검색이 없습니다")
            return
        st = self._last_result.stats
        dropped = {s.name: s.dropped for s in st.stages if s.dropped}
        log.info(
            "[cutoff/rerank] survey-search 는 시간창으로 버리지 않습니다 "
            "(freshness 는 순위만 바꿉니다). 직전 검색의 단계별 폐기: %s",
            dropped or "없음",
        )
        for w in st.warnings:
            log.info("[cutoff/rerank]   경고: %s", w)

    @property
    def last_stats(self):
        return self._last_result.stats if self._last_result else None
