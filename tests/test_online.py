"""온라인 백엔드 — 파싱과 캐시. **네트워크를 쓰지 않습니다.**

실제 API 호출은 `-m network` 로 분리했습니다. CI 나 오프라인에서도 나머지는 돌아야 합니다.
"""

from __future__ import annotations

import json

import pytest

from survey_search.backends.hybrid import HybridBackend
from survey_search.backends.online import OnlineBackend, _parse_arxiv_atom, _to_paper
from survey_search.types import Paper

ATOM_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2005.11401v4</id>
    <published>2020-05-22T21:16:47Z</published>
    <title>Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks</title>
    <summary>  We explore a general-purpose fine-tuning recipe.
    </summary>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2312.10997v5</id>
    <published>2023-12-18T00:00:00Z</published>
    <title>RAG for LLMs: A Survey</title>
    <summary>A survey.</summary>
    <category term="cs.CL"/>
  </entry>
</feed>"""


def test_parse_atom_extracts_ids_and_normalizes_whitespace():
    recs = _parse_arxiv_atom(ATOM_SAMPLE)
    assert [r["paper_id"] for r in recs] == ["2005.11401v4", "2312.10997v5"]
    assert recs[0]["base_id"] == "2005.11401"
    # 제목의 줄바꿈·중복 공백이 정리돼야 BM25/제목 매칭이 맞습니다
    assert recs[0]["title"] == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert recs[0]["date"] == "2020-05-22"
    assert recs[0]["categories"] == ["cs.CL", "cs.LG"]


def test_parse_atom_empty_feed():
    assert _parse_arxiv_atom(b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>') == []


def test_to_paper_uses_published_as_submitted_date():
    """arXiv API 의 published 는 v1 게시일입니다 — 로컬 DB 의 date 와 달리 신뢰할 수 있습니다."""
    p = _to_paper(_parse_arxiv_atom(ATOM_SAMPLE)[0])
    assert isinstance(p, Paper)
    assert p.submitted_date == "2020-05-22" == p.date


def test_offline_mode_returns_empty_without_network(tmp_path):
    be = OnlineBackend(cache_dir=tmp_path, offline=True)
    assert be.lexical_search(["anything"], 10) == [[]]
    assert be.references("2005.11401") == []
    assert be.cited_by("2005.11401") == []
    assert be.stats.arxiv_calls == 0 and be.stats.s2_calls == 0


def test_cache_hit_avoids_network(tmp_path):
    be = OnlineBackend(cache_dir=tmp_path, offline=True)
    be.cache.put("arxiv", "q|5", _parse_arxiv_atom(ATOM_SAMPLE))
    hits = be.lexical_search(["q"], 5)[0]
    assert [h[0] for h in hits] == ["2005.11401v4", "2312.10997v5"]
    assert be.stats.cache_hits == 1
    assert be.stats.arxiv_calls == 0


def test_lexical_scores_are_rank_based():
    """arXiv 는 점수를 안 줍니다. 순위를 점수로 쓰되 내림차순이어야 합니다."""
    be = OnlineBackend(offline=True)
    be.cache.put("arxiv", "q|5", _parse_arxiv_atom(ATOM_SAMPLE))
    hits = be.lexical_search(["q"], 5)[0]
    assert hits[0][1] > hits[1][1]


def test_filter_ids_returns_none_because_online_has_no_id_universe():
    be = OnlineBackend(offline=True)
    assert be.filter_ids(date_min="2024-01-01") is None


# --- 하이브리드 -----------------------------------------------------------------

class FakeLocal:
    name = "local"

    def __init__(self):
        self.papers = {"L1": Paper(paper_id="L1", base_id="L1", title="local one",
                                   abstract="", date="2024-01-01")}

    def dense_search(self, queries, top_k, field="title_abs"):
        return [[("L1", 0.9)] for _ in queries]

    def lexical_search(self, queries, top_k):
        return [[("L1", 5.0)] for _ in queries]

    def get_papers(self, ids):
        return [self.papers[i] for i in ids if i in self.papers]

    def filter_ids(self, **kw):
        return None

    def get_vectors(self, ids, field="title_abs"):
        return None


def _hybrid(tmp_path) -> HybridBackend:
    online = OnlineBackend(cache_dir=tmp_path, offline=True)
    online.cache.put("arxiv", "q|100", [
        {"paper_id": "2608.00001v1", "base_id": "2608.00001", "title": "beyond cutoff",
         "abstract": "", "date": "2026-08-20", "categories": ["cs.CL"]}])
    return HybridBackend(FakeLocal(), online)


def test_hybrid_dense_uses_local_only(tmp_path):
    h = _hybrid(tmp_path)
    assert h.dense_search(["q"], 10) == [[("L1", 0.9)]]


def test_hybrid_lexical_fuses_local_and_online(tmp_path):
    h = _hybrid(tmp_path)
    ids = [pid for pid, _ in h.lexical_search(["q"], 100)[0]]
    assert set(ids) == {"L1", "2608.00001v1"}


def test_hybrid_fills_metadata_the_local_index_lacks(tmp_path):
    """컷오프 이후 논문은 로컬에 없습니다 — 폴백이 없으면 결과에서 조용히 사라집니다."""
    h = _hybrid(tmp_path)
    got = h.get_papers(["L1", "2608.00001v1"])
    assert [p.paper_id for p in got] == ["L1", "2608.00001v1"]


def test_hybrid_preserves_requested_order(tmp_path):
    h = _hybrid(tmp_path)
    got = h.get_papers(["2608.00001v1", "L1"])
    assert [p.paper_id for p in got] == ["2608.00001v1", "L1"]


def test_hybrid_exposes_citation_edges(tmp_path):
    h = _hybrid(tmp_path)
    assert callable(h.references) and callable(h.cited_by)
    from survey_search.backends.base import CitationBackend
    assert isinstance(h, CitationBackend)


def test_hybrid_vectors_none_when_local_cannot_supply(tmp_path):
    """온라인 논문은 인덱스에 벡터가 없습니다 — 일부만 채우면 MMR 이 조용히 틀립니다."""
    assert _hybrid(tmp_path).get_vectors(["L1", "2608.00001v1"]) is None


# --- 실제 네트워크 (선택) --------------------------------------------------------

@pytest.mark.network
def test_live_arxiv_search(tmp_path):
    be = OnlineBackend(cache_dir=tmp_path)
    hits = be.lexical_search(["retrieval augmented generation"], 5)[0]
    assert hits, "arXiv API 가 결과를 못 냈습니다"
    assert be.stats.arxiv_calls == 1
