"""채점기 단위 테스트 — 자산 없이 돕니다."""

from survey_search.metrics.paper_set import (
    calculate_ndcg,
    calculate_recall,
    normalize_ids,
    score_paper_set,
)


def test_normalize_strips_version_and_dedups():
    assert normalize_ids(["2401.12345v1", "2401.12345v2", "1811.06122"]) == [
        "2401.12345",
        "1811.06122",
    ]


def test_recall_and_ndcg_basic():
    assert calculate_recall(["a", "b"], ["a", "c"]) == 0.5
    assert calculate_recall([], ["a"]) == 0.0
    assert calculate_recall(["a"], []) == 0.0
    # 정답이 1위면 nDCG 만점
    assert calculate_ndcg(["a", "x"], ["a"]) == 1.0
    assert calculate_ndcg(["x", "a"], ["a"]) == 0.5


def test_perfect_submission_scores_one():
    rel = {"a": 3, "b": 3}
    out = score_paper_set(["a", "b"], rel)
    assert out["recall_at_est"] == 1.0
    # 등급이 균일하면 순서를 매길 수 없으므로 1.0 (PFB 원본의 0 을 고친 부분)
    assert out["rank"] == 1.0
    assert out["reward"] == 1.0


def test_only_grade3_counts_toward_recall():
    # 등급 1~2 는 recall 에 안 들어갑니다 — PFB 정의
    rel = {"a": 3, "b": 2, "c": 1}
    out = score_paper_set(["b", "c"], rel)
    assert out["n_gold"] == 1
    assert out["recall_at_est"] == 0.0
    assert out["reward"] == 0.0  # recall 0 이면 조화평균도 0


def test_order_matters_for_rank():
    rel = {"a": 3, "b": 1}
    good = score_paper_set(["a", "b"], rel)
    bad = score_paper_set(["b", "a"], rel)
    assert good["rank"] > bad["rank"]
    assert good["reward"] > bad["reward"]


def test_recall_truncates_at_est_so_padding_does_not_help():
    rel = {"a": 3, "b": 3}
    # 정답 2편인데 쓰레기로 앞을 채우면 상위 2개 안에 정답이 안 들어옵니다
    padded = score_paper_set(["x", "y", "a", "b"], rel)
    assert padded["recall_at_est"] == 0.0


def test_versioned_ids_match_gold():
    # 검색은 v2 를 돌려주고 정답은 v1 로 적혀 있어도 같은 논문으로 채점돼야 합니다
    out = score_paper_set(["2401.12345v2"], {"2401.12345v1": 3})
    assert out["recall_at_est"] == 1.0


def test_empty_prediction():
    out = score_paper_set([], {"a": 3})
    assert out["reward"] == 0.0
    assert out["n_pred"] == 0
