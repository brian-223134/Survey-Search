"""검색이 같은 입력에 같은 결과를 내는가.

**실측으로 아니었습니다.** 같은 토픽을 연속 두 번 검색하면 다른 논문 목록이 나왔고,
`facets=True` 일 때(쿼리 12~16개) 특히 그랬습니다. 원인은 DuckDB BM25 였습니다:

- 상위 300편의 **집합도 점수도 같은데**(최대차 8.9e-16 = 부동소수점 끝자리) 순서가 다름
- `ORDER BY score DESC` 에 동점 규칙이 없어 병렬 실행마다 순서가 갈림
- top_k 경계에서는 순서를 넘어 **어느 논문이 들어오느냐**까지 바뀜
  (실측: 2,000편 중 60등에서 다른 논문 유입)

`round(score, 9)` + `paper_id` tie-break 로 고쳤습니다. 이 파일은 그게 되돌아가는 것을
막습니다. 자산이 있어야 도는 테스트라 `-m assets` 로 분리했습니다.
"""

from __future__ import annotations

import pytest

from survey_search.types import SearchConfig

QUERIES = [
    "retrieval augmented generation",
    "graph neural network for molecules",
    "chain of thought prompting",
]


def _ids(hits_per_query) -> list[list[str]]:
    return [[pid for pid, _ in row] for row in hits_per_query]


@pytest.mark.assets
def test_lexical_search_returns_the_same_order_every_time(backend):
    """BM25 의 **id 순서**가 실행마다 같아야 합니다. 점수 float 은 끝자리가 흔들려도
    무해합니다 — 이 파이프라인은 융합에 순위만 쓰기 때문입니다."""
    runs = [_ids(backend.lexical_search(QUERIES, 2000)) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


@pytest.mark.assets
def test_lexical_search_is_stable_at_the_top_k_boundary(backend):
    """경계가 흔들리면 순서가 아니라 **집합**이 바뀝니다. 그게 더 나쁜 종류입니다."""
    a = set(_ids(backend.lexical_search(QUERIES, 2000))[0])
    b = set(_ids(backend.lexical_search(QUERIES, 2000))[0])
    assert a == b


@pytest.mark.assets
def test_dense_search_was_already_deterministic(backend):
    """FAISS 는 원래 결정적이었습니다. 여기가 깨지면 원인이 완전히 다른 곳입니다."""
    a = backend.dense_search(QUERIES, 500)
    b = backend.dense_search(QUERIES, 500)
    assert a == b


@pytest.mark.assets
@pytest.mark.parametrize("lexical", [True, False])
def test_search_topic_repeats_exactly(backend, lexical):
    """끝에서 끝까지. BM25 를 켜든 끄든 같은 토픽은 같은 목록을 내야 합니다."""
    from survey_search.search import search_topic

    cfg = SearchConfig(n_papers=300, lexical=lexical, freshness=True)
    first = search_topic("retrieval augmented generation", backend=backend, config=cfg).ids()
    second = search_topic("retrieval augmented generation", backend=backend, config=cfg).ids()
    assert first == second
    assert first, "결과가 비어 있으면 이 테스트는 아무것도 검증하지 못합니다"


def test_sort_precision_is_coarser_than_float_noise():
    """반올림 자리가 너무 촘촘하면(예: 15) 끝자리 흔들림을 못 걸러 냅니다.
    너무 성기면(예: 2) 진짜 점수 차이를 동점으로 뭉갭니다."""
    from survey_search.backends.faiss_duckdb import FaissDuckDBBackend

    p = FaissDuckDBBackend.BM25_SORT_PRECISION
    assert 6 <= p <= 12
