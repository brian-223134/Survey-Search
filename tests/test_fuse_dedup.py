"""S4(RRF)·S5(dedup) 단위 테스트 — 합성 데이터, 자산 불필요."""

import pytest

from survey_search.core.dedup import dedup, normalize_title, strip_version, version_of
from survey_search.core.fuse import provenance_of, rrf


def test_rrf_ignores_scores_uses_only_rank():
    """점수 스케일이 달라도 결과가 같아야 합니다 — 이 인덱스에서의 필수 조건."""
    small = [[("a", 0.01), ("b", 0.009)]]
    huge = [[("a", 9999.0), ("b", 1.0)]]
    assert [p for p, _ in rrf(small)] == [p for p, _ in rrf(huge)]


def test_rrf_rewards_appearing_in_multiple_lists():
    dense = [("a", 1.0), ("b", 1.0)]
    bm25 = [("b", 1.0), ("c", 1.0)]
    out = dict(rrf([dense, bm25]))
    # b 는 양쪽 모두에 있으므로 어느 한쪽 1위보다 높아야 합니다
    assert out["b"] > out["a"]
    assert out["b"] > out["c"]


def test_rrf_accepts_plain_id_lists():
    assert [p for p, _ in rrf([["a", "b"], ["b", "a"]])] == ["a", "b"]


def test_rrf_k_changes_sharpness():
    lists = [["a", "b", "c"]]
    tight = dict(rrf(lists, k=1))
    loose = dict(rrf(lists, k=1000))
    assert tight["a"] / tight["c"] > loose["a"] / loose["c"]


def test_rrf_weights_can_disable_a_list():
    out = dict(rrf([["a"], ["b"]], weights=[1.0, 0.0]))
    assert "a" in out and "b" not in out


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], weights=[1.0])


def test_version_helpers():
    assert strip_version("2401.12345v3") == "2401.12345"
    assert version_of("2401.12345v3") == 3
    assert version_of("2401.12345") == 0
    assert normalize_title("Prophet Inequalities for\n  I.I.D.") == "prophetinequalitiesforiid"


def test_dedup_merges_versions_keeping_latest_and_max_score():
    out, dropped, merged = dedup([("2401.1v1", 0.9), ("2401.1v2", 0.4)])
    assert dropped["version"] == 1
    assert out == [("2401.1v2", 0.9)]          # 대표는 최신 버전, 점수는 최대값
    assert merged["2401.1v2"] == ["2401.1v1"]


def test_dedup_merges_by_normalized_title():
    scored = [("a v1", 0.9), ("b v1", 0.5)]
    titles = {"a v1": "Deep  Learning!", "b v1": "deep learning"}
    out, dropped, merged = dedup(scored, titles=titles)
    assert dropped["title"] == 1
    assert [p for p, _ in out] == ["a v1"]
    assert merged["a v1"] == ["b v1"]


def test_dedup_without_titles_skips_title_merge():
    scored = [("a v1", 0.9), ("b v1", 0.5)]
    out, dropped, _ = dedup(scored)
    assert dropped["title"] == 0
    assert len(out) == 2


def test_dedup_keeps_papers_with_unknown_title():
    """제목을 모른다고 조용히 버리면 안 됩니다."""
    out, dropped, _ = dedup([("a v1", 0.9), ("b v1", 0.5)], titles={"a v1": "x"})
    assert len(out) == 2
    assert dropped["title"] == 0


def test_provenance_labels_paths():
    assert provenance_of("a", {"dense": [("a", 1.0)], "bm25": [("b", 1.0)]}) == ("dense",)
    assert set(provenance_of("a", {"dense": ["a"], "bm25": ["a"]})) == {"dense", "bm25"}
