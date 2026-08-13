"""SurveyForge 어댑터의 **호스트 배선** — 조용히 틀리던 두 가지를 고정합니다.

호스트 코드(`../SurveyForge/code/`)와 대조해 찾은 결손입니다. 둘 다 예외를 내지 않고
결과만 틀리는 종류라, 테스트가 없으면 다음에 또 밟습니다.

- **filter 가 도착하지 않음** — 호스트는 `filter=` 가 아니라 `**{'id_selector': ...}`
  로 펼쳐 넘깁니다(`writer.py:43·81·370`, `outline_writer.py:55`). `filter=None`
  시그니처만 두면 `**kwargs` 로 빨려 들어가 버려지고, `paper_id_cutoff` 시간 게이트와
  2단계 좁히기가 통째로 무효가 됩니다
- **인용 순서 계약** — `zip(citations, ids)` 로 인용-논문을 맞추므로
  **쿼리 N개 → id N×top_k, 입력 순서**를 지켜야 합니다(`writer.py:365`)

백엔드는 가짜입니다 — 배선을 보는 테스트라 실제 인덱스가 필요 없습니다.
"""

from __future__ import annotations

import pytest

from survey_search.adapters.surveyforge import SurveySearchRAG
from survey_search.types import Paper, SearchConfig


class FakeBackend:
    """id 3개짜리 최소 백엔드. `search_topic` 이 부르는 것만 구현합니다."""

    name = "fake"

    def __init__(self, order=("p1", "p2", "p3")):
        self.order = list(order)
        self.queries: list[list[str]] = []

    def _maps(self):
        id_to_index = {p: i + 1 for i, p in enumerate(self.order)}
        return id_to_index, {v: k for k, v in id_to_index.items()}

    def dense_search(self, queries, top_k, field="title_abs"):
        self.queries.append(list(queries))
        # 쿼리마다 순서를 회전시켜 "쿼리별로 검색했는가"를 구분할 수 있게 합니다.
        out = []
        for q in queries:
            n = sum(ord(c) for c in q) % len(self.order)
            rot = self.order[n:] + self.order[:n]
            out.append([(pid, 1.0 / (i + 1)) for i, pid in enumerate(rot)][:top_k])
        return out

    def lexical_search(self, queries, top_k):
        return [[] for _ in queries]

    def get_papers(self, paper_ids):
        return [Paper(paper_id=p, base_id=p, title=p, abstract="", date="2024-01-01")
                for p in paper_ids if p in self.order]

    def filter_ids(self, **kw):
        return None


def rag(**cfg) -> SurveySearchRAG:
    return SurveySearchRAG(backend=FakeBackend(),
                           config=SearchConfig(lexical=False, **cfg))


# --- B: filter 배선 -------------------------------------------------------------

def test_id_selector_keyword_is_honoured():
    """호스트가 `**{'id_selector': ...}` 로 펼쳐 넘기는 형태를 받아야 합니다."""
    r = rag()
    allowed_all = r.retrieve_id("q", top_k=3, max_out=3)
    narrowed = r.retrieve_id("q", top_k=3, max_out=3, id_selector={"paper_ids": ["p2"]})
    assert allowed_all != narrowed, "id_selector 가 무시되고 있습니다"
    assert narrowed == ["p2"]


def test_filter_keyword_still_works():
    """`filter=` 로 오는 경우도 계속 받습니다 — 원본 시그니처입니다."""
    assert rag().retrieve_id("q", top_k=3, max_out=3,
                             filter={"paper_ids": ["p3"]}) == ["p3"]


def test_both_filter_forms_intersect_instead_of_dropping_one():
    r = rag()
    got = r.retrieve_id("q", top_k=3, max_out=3,
                        filter={"paper_ids": ["p1", "p2"]},
                        id_selector={"paper_ids": ["p2", "p3"]})
    assert got == ["p2"]


def test_id_selector_reaches_the_citation_path_too():
    """`writer.py:370` 이 인용 경로에도 같은 형태로 넘깁니다."""
    r = rag()
    got = r.retrieve_id4citation(["a", "b"], top_k=1,
                                 id_selector={"paper_ids": ["p2"]})
    assert got == ["p2", "p2"]


# --- C: 인용 순서 계약 -----------------------------------------------------------

def test_citation_path_searches_each_query_separately():
    """이전 구현은 N개를 `" ; "` 로 이어 붙여 **한 번** 검색했습니다."""
    be = FakeBackend()
    r = SurveySearchRAG(backend=be, config=SearchConfig(lexical=False))
    r.retrieve_id4citation(["alpha", "beta", "gamma"], top_k=1)
    flat = [q for batch in be.queries for q in batch]
    assert flat == ["alpha", "beta", "gamma"], f"쿼리가 합쳐졌습니다: {flat}"


def test_citation_path_returns_one_id_per_query_in_order():
    """`zip(citations, ids)` 가 성립하려면 길이와 순서가 정확해야 합니다."""
    be = FakeBackend()
    r = SurveySearchRAG(backend=be, config=SearchConfig(lexical=False))
    cits = ["alpha", "beta", "gamma", "delta"]
    ids = r.retrieve_id4citation(cits, top_k=1)
    assert len(ids) == len(cits)
    # 각 id 는 그 쿼리를 단독으로 검색했을 때의 1위여야 합니다.
    for c, got in zip(cits, ids):
        assert got == be.dense_search([c], 1)[0][0][0]


def test_citation_path_respects_top_k_greater_than_one():
    be = FakeBackend()
    r = SurveySearchRAG(backend=be, config=SearchConfig(lexical=False))
    ids = r.retrieve_id4citation(["a", "b"], top_k=2)
    assert len(ids) == 4          # 쿼리 2 × top_k 2, 순서대로


def test_citation_path_forces_facets_off():
    """쿼리가 논문 제목입니다. facet 을 켜면 인용 1건마다 LLM 40초가 붙습니다."""
    be = FakeBackend()
    r = SurveySearchRAG(backend=be, config=SearchConfig(lexical=False, facets=True))
    r.retrieve_id4citation(["some paper title"], top_k=1)
    # facet 을 켰다면 decompose 가 쿼리를 여러 개로 늘렸을 것입니다.
    assert be.queries == [["some paper title"]]
    assert any("facet" in w for w in r.last_stats.warnings)


def test_citation_path_reports_when_filter_starves_a_query():
    """정렬을 지키려고 필터 밖에서 메웠으면 **반드시** 남겨야 합니다."""
    be = FakeBackend()
    r = SurveySearchRAG(backend=be, config=SearchConfig(lexical=False))
    ids = r.retrieve_id4citation(["a", "b"], top_k=1,
                                 id_selector={"paper_ids": ["없는id"]})
    assert len(ids) == 2, "메우지 않으면 뒤 인용이 한 칸씩 밀립니다"
    assert any("필터 밖" in w for w in r.last_stats.warnings)


def test_citation_path_empty_input():
    assert rag().retrieve_id4citation([], top_k=1) == []


# --- A: 호스트가 요구하는 나머지 표면 ---------------------------------------------

def test_id_to_index_is_exposed():
    """`main.py:200` · `get_index_filter` 가 씁니다."""
    m = rag().id_to_index
    assert m["p1"] == 1 and set(m) == {"p1", "p2", "p3"}


def test_report_window_drops_is_not_a_silent_noop(caplog):
    """조용한 no-op 은 이 저장소 원칙에 어긋납니다."""
    r = rag()
    with caplog.at_level("INFO"):
        r.report_window_drops()          # 검색 전
        r.retrieve_id("q", top_k=1, max_out=1)
        r.report_window_drops()          # 검색 후
    assert any("cutoff/rerank" in m for m in caplog.messages)


def test_host_call_shape_end_to_end():
    """`writer.py:365` 가 실제로 쓰는 모양 그대로."""
    be = FakeBackend()
    r = SurveySearchRAG(backend=be, config=SearchConfig(lexical=False))
    citations = ["Attention Is All You Need", "BERT", "GPT-3"]
    index_filter = {"id_selector": {"paper_ids": ["p1", "p2", "p3"]}}
    ids = r.retrieve_id4citation(citations, search_type="similarity", top_k=1,
                                 **index_filter)
    mapping = dict(zip(citations, ids))
    assert len(mapping) == len(citations)
    assert all(v in ("p1", "p2", "p3") for v in mapping.values())


# --- 호스트가 실제로 보내는 형태 (진짜 faiss 객체) --------------------------------

def test_real_faiss_id_selector_array():
    """위 테스트들은 편의상 dict 를 씁니다. **호스트가 실제로 보내는 것**은
    `faiss.IDSelectorArray` 입니다 (`utils.py:212`). 그것도 읽을 수 있어야 합니다."""
    faiss = pytest.importorskip("faiss")
    import numpy as np

    be = FakeBackend()
    r = SurveySearchRAG(backend=be, config=SearchConfig(lexical=False))
    # FakeBackend 의 id_to_index 는 p1->1, p2->2, p3->3 입니다.
    sel = faiss.IDSelectorArray(np.array([2], dtype="int64"))
    got = r.retrieve_id("q", top_k=3, max_out=3, id_selector=sel)
    assert got == ["p2"], f"IDSelectorArray 를 못 읽었습니다: {got}"
