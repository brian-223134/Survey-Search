"""P3 어댑터 — 호스트가 기대하는 시그니처·반환형을 지키는지.

실제 검색은 자산이 있어야 하므로 assets 마크입니다. 시그니처 호환성 검사와
filter 번역은 자산 없이 돕니다.
"""

from __future__ import annotations

import inspect

import pytest

from survey_search.adapters.autosurvey import SurveySearchDatabase
from survey_search.adapters.surveyforge import SurveySearchRAG, _filter_to_paper_ids
from survey_search.assets import PAPERS_DUCKDB, SURVEYFORGE

INDEX_TO_ID = {1: "2401.00001v1", 2: "2401.00002v1", 3: "2401.00003v2"}


# --- filter 번역 (자산 불필요) -------------------------------------------------

def test_filter_none_means_no_restriction():
    assert _filter_to_paper_ids(None, INDEX_TO_ID) is None


def test_filter_accepts_explicit_paper_ids():
    out = _filter_to_paper_ids({"paper_ids": ["a", "b"]}, INDEX_TO_ID)
    assert out == {"a", "b"}


def test_filter_accepts_index_list():
    assert _filter_to_paper_ids([1, 3], INDEX_TO_ID) == {"2401.00001v1", "2401.00003v2"}


def test_filter_accepts_arxiv_id_list():
    assert _filter_to_paper_ids(["2401.00001v1"], INDEX_TO_ID) == {"2401.00001v1"}


def test_filter_empty_list_is_empty_set_not_none():
    """빈 집합과 None 은 뜻이 다릅니다 — 빈 집합은 '해당 없음'."""
    assert _filter_to_paper_ids([], INDEX_TO_ID) == set()


def test_filter_skips_unknown_indices_rather_than_crashing():
    assert _filter_to_paper_ids([1, 999], INDEX_TO_ID) == {"2401.00001v1"}


def test_filter_rejects_unknown_dict_shape():
    with pytest.raises(TypeError):
        _filter_to_paper_ids({"nope": 1}, INDEX_TO_ID)


def test_filter_translates_faiss_id_selector():
    faiss = pytest.importorskip("faiss")
    sel = faiss.IDSelectorArray([1, 2])
    out = _filter_to_paper_ids({"id_selector": sel}, INDEX_TO_ID)
    assert out == {"2401.00001v1", "2401.00002v1"}


# --- 시그니처 호환 (자산 불필요) ------------------------------------------------

def test_autosurvey_adapter_has_methods_the_host_calls():
    """AutoSurvey 에이전트가 실제로 부르는 이름들."""
    for name in ("get_ids_from_query", "get_ids_from_topic",
                 "get_titles_from_citations", "get_paper_info_from_ids"):
        assert callable(getattr(SurveySearchDatabase, name)), name


def test_autosurvey_get_ids_from_query_signature_matches_host():
    sig = inspect.signature(SurveySearchDatabase.get_ids_from_query)
    assert list(sig.parameters) == ["self", "query", "num", "shuffle"]


def test_autosurvey_body_access_fails_loudly_not_silently():
    """본문은 id 체계가 달라 못 줍니다. 빈 값을 조용히 주면 본문 없는 서베이가 나옵니다."""
    db = SurveySearchDatabase.__new__(SurveySearchDatabase)
    with pytest.raises(NotImplementedError, match="본문"):
        db.get_paper_from_ids(["x"])


def test_surveyforge_retrieve_id_signature_matches_host():
    sig = inspect.signature(SurveySearchRAG.retrieve_id)
    for p in ("query", "search_type", "rerank", "top_k", "max_out", "filter", "fetch_k"):
        assert p in sig.parameters, p


# --- 실제 검색 (자산 필요) -----------------------------------------------------

needs_assets = pytest.mark.skipif(
    bool(SURVEYFORGE.missing()) or not PAPERS_DUCKDB.exists(),
    reason="FAISS/DuckDB 자산 없음",
)


@needs_assets
@pytest.mark.assets
def test_autosurvey_adapter_returns_ids_and_matching_info():
    db = SurveySearchDatabase()
    ids = db.get_ids_from_query("retrieval augmented generation", 50)
    assert len(ids) == 50
    assert all(isinstance(i, str) for i in ids)

    infos = db.get_paper_info_from_ids(ids)
    assert len(infos) == 50
    # 호스트가 기대하는 키 (AutoSurvey main.py:104 는 p['id'] 를 씁니다)
    assert set(infos[0]) >= {"id", "title", "abs", "date", "cat", "url"}
    assert [i["id"] for i in infos] == ids
    assert db.last_stats is not None


@needs_assets
@pytest.mark.assets
def test_surveyforge_adapter_respects_filter():
    rag = SurveySearchRAG()
    wide = rag.retrieve_id("retrieval augmented generation", top_k=200, max_out=200)
    assert wide

    keep = set(wide[:20])
    narrowed = rag.retrieve_id("retrieval augmented generation", top_k=200, max_out=200,
                               filter={"paper_ids": list(keep)})
    assert narrowed, "filter 를 걸었더니 전부 사라지면 번역이 틀린 것입니다"
    assert set(narrowed) <= keep


@needs_assets
@pytest.mark.assets
def test_surveyforge_citation_rerank_is_flagged_as_substituted():
    """인용수 정렬을 freshness 로 바꿨다는 사실이 stats 에 남아야 합니다."""
    rag = SurveySearchRAG()
    rag.retrieve_id("retrieval augmented generation", top_k=100, max_out=100,
                    rerank="citation")
    warnings = " ".join(rag.last_stats.warnings)
    assert "freshness" in warnings
