"""S7 다양성 단위 테스트 — 합성 벡터, 자산 불필요."""

import numpy as np

from survey_search.core.diversity import diversify, facet_quota, mmr
from survey_search.types import Paper


def mk(pid: str, score: float, facets: tuple[str, ...] = ("f",)) -> Paper:
    return Paper(paper_id=pid, base_id=pid, title=pid, abstract="", date="2025-01-01",
                 score=score, facets=facets)


def unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def test_mmr_lambda_one_is_pure_relevance():
    """lambda=1.0 이면 원래 점수 순서와 같아야 합니다 — 껐을 때의 기준선."""
    papers = [mk("a", 3.0), mk("b", 2.0), mk("c", 1.0)]
    vecs = np.stack([unit([1, 0]), unit([1, 0.01]), unit([0, 1])])
    order = mmr(papers, vecs, k=3, lambda_=1.0)
    assert [papers[i].paper_id for i in order] == ["a", "b", "c"]


def test_mmr_skips_near_duplicate_in_favor_of_different_one():
    """b 가 a 와 거의 같으면, 점수가 낮아도 다른 c 를 먼저 골라야 합니다."""
    papers = [mk("a", 3.0), mk("b", 2.9), mk("c", 1.0)]
    vecs = np.stack([unit([1, 0]), unit([1, 0.001]), unit([0, 1])])
    order = mmr(papers, vecs, k=2, lambda_=0.3)
    assert [papers[i].paper_id for i in order] == ["a", "c"]


def test_mmr_returns_all_when_k_exceeds_n():
    papers = [mk("a", 1.0), mk("b", 2.0)]
    vecs = np.stack([unit([1, 0]), unit([0, 1])])
    assert len(mmr(papers, vecs, k=99)) == 2


def test_mmr_empty_input():
    assert mmr([], np.zeros((0, 2), dtype="float32"), k=5) == []


def test_mmr_handles_identical_scores():
    """점수가 전부 같으면 정규화에서 0으로 나누면 안 됩니다."""
    papers = [mk("a", 1.0), mk("b", 1.0), mk("c", 1.0)]
    vecs = np.stack([unit([1, 0]), unit([0, 1]), unit([1, 1])])
    assert len(mmr(papers, vecs, k=3, lambda_=0.5)) == 3


def test_facet_quota_is_noop_with_single_facet():
    """S1 이 꺼져 있으면 facet 이 하나뿐 — 아무것도 안 하고 그 사실을 남깁니다."""
    papers = [mk(f"p{i}", 1.0) for i in range(5)]
    out, stats = facet_quota(papers, n=3)
    assert out == [0, 1, 2]
    assert stats.facet_quota_applied is False
    assert stats.n_facets == 1
    assert "S1" in stats.note


def test_facet_quota_guarantees_minimum_per_facet():
    """점수가 낮아도 각 facet 이 최소 배정을 받아야 합니다."""
    papers = (
        [mk(f"big{i}", 10.0 - i, ("big",)) for i in range(10)]
        + [mk(f"small{i}", 0.1 - i * 0.01, ("small",)) for i in range(3)]
    )
    out, stats = facet_quota(papers, n=6, min_per_facet=2)
    picked = [papers[i].facets[0] for i in out]
    assert picked.count("small") >= 2, f"small facet 이 밀려남: {picked}"
    assert stats.facet_quota_applied is True
    assert stats.n_facets == 2


def test_facet_quota_fills_remainder_by_score():
    papers = (
        [mk(f"a{i}", 10.0 - i, ("A",)) for i in range(5)]
        + [mk(f"b{i}", 1.0 - i * 0.1, ("B",)) for i in range(5)]
    )
    out, _ = facet_quota(papers, n=6, min_per_facet=1)
    assert len(out) == 6


def test_diversify_without_vectors_keeps_score_order_and_says_so():
    """벡터가 없으면 조용히 통과시키지 말고 stats 에 남겨야 합니다."""
    papers = [mk(f"p{i}", 5.0 - i) for i in range(5)]
    out, stats = diversify(papers, n=3, vectors=None)
    assert [p.paper_id for p in out] == ["p0", "p1", "p2"]
    assert stats.mmr_applied is False
    assert "MMR" in stats.note


def test_diversify_with_vectors_applies_mmr():
    papers = [mk(f"p{i}", 5.0 - i) for i in range(4)]
    vecs = np.stack([unit([1, 0]), unit([1, 0.001]), unit([0, 1]), unit([0.5, 0.5])])
    out, stats = diversify(papers, n=2, vectors=vecs, lambda_=0.3)
    assert stats.mmr_applied is True
    assert len(out) == 2


def test_diversify_respects_n():
    papers = [mk(f"p{i}", 5.0 - i) for i in range(10)]
    vecs = np.stack([unit([np.cos(i), np.sin(i)]) for i in range(10)])
    out, stats = diversify(papers, n=4, vectors=vecs)
    assert len(out) == 4 == stats.n_out
