"""S6 freshness 랭킹 단위 테스트 — 합성 데이터, 자산 불필요."""

from datetime import date

from survey_search.core.rank import (
    COHORT_MONTHS,
    FreshnessConfig,
    RecencyMode,
    citation_rate,
    cohort_percentiles,
    recency_weight,
    rerank,
)
from survey_search.types import Paper

TODAY = date(2026, 8, 12)


def mk(pid: str, months_ago: float, cc: int | None, score: float = 1.0) -> Paper:
    day = date.fromordinal(TODAY.toordinal() - int(months_ago * 30.44))
    return Paper(
        paper_id=pid, base_id=pid, title=pid, abstract="",
        date=day.isoformat(), citation_count=cc, score=score,
    )


def test_citation_rate_normalizes_by_age():
    old = mk("old", 60, 60)     # 60개월에 60회 -> 1.0/월
    new = mk("new", 3, 30)      # 3개월에 30회 -> 10.0/월
    assert citation_rate(new, TODAY) > citation_rate(old, TODAY)


def test_citation_rate_floors_age_so_brand_new_does_not_explode():
    """갓 나온 논문의 인용률이 무한대로 튀면 안 됩니다."""
    fresh = mk("fresh", 0, 10)
    assert citation_rate(fresh, TODAY) == 10 / 3.0   # MIN_AGE_MONTHS


def test_citation_rate_none_when_unknown():
    assert citation_rate(mk("x", 12, None), TODAY) is None


def test_recency_weight_halves_each_half_life():
    p_now, p_half = mk("a", 0, 0), mk("b", 18, 0)
    assert recency_weight(p_now, TODAY, 18.0) == 1.0
    assert abs(recency_weight(p_half, TODAY, 18.0) - 0.5) < 0.02


def test_percentile_is_within_cohort_not_global():
    """오래된 고인용 논문이 최신 논문의 백분위를 깎으면 안 됩니다."""
    papers = [
        mk("old_star", 60, 600),    # 10/월
        mk("old_dud", 60, 6),       # 0.1/월
        mk("new_star", 2, 30),      # 10/월
        mk("new_dud", 2, 0),        # 0
    ]
    pct, stats = cohort_percentiles(papers, TODAY)
    assert stats.n_cohorts == 2
    # 각 코호트의 1등은 둘 다 백분위 1.0 — 절대 인용수가 100배 차이나도
    assert pct["old_star"] == 1.0
    assert pct["new_star"] == 1.0
    assert pct["old_dud"] == 0.0
    assert pct["new_dud"] == 0.0


def test_unknown_citation_gets_neutral_percentile():
    """정보 없음이 페널티도 보상도 되면 안 됩니다."""
    pct, stats = cohort_percentiles([mk("a", 12, None)], TODAY)
    assert pct["a"] == 0.5
    assert stats.n_missing_citation == 1


def test_rerank_promotes_recent_over_equally_relevant_old():
    old = mk("old", 48, 100, score=1.0)
    new = mk("new", 2, 5, score=1.0)
    out, _ = rerank([old, new], config=FreshnessConfig(alpha=0.5, beta=0.5), today=TODAY)
    assert out[0].paper_id == "new"


def test_rerank_is_multiplicative_so_relevance_still_dominates():
    """관련성이 크게 높은 옛 논문이 최신 논문에 밀리면 안 됩니다."""
    strong_old = mk("old", 48, 100, score=10.0)
    weak_new = mk("new", 1, 0, score=1.0)
    out, _ = rerank([strong_old, weak_new], today=TODAY)
    assert out[0].paper_id == "old"


def test_rerank_preserves_count_and_reports_missing():
    papers = [mk("a", 5, None), mk("b", 5, 3)]
    out, stats = rerank(papers, today=TODAY)
    assert len(out) == 2
    assert stats.n_scored == 2
    assert stats.n_missing_citation == 1


def test_quota_mode_guarantees_recent_share_in_every_prefix():
    """쿼터는 모든 접두구간에서 성립해야 합니다 — 컷 위치가 어디든 통하도록."""
    old = [mk(f"old{i}", 40, 100, score=10.0 - i * 0.1) for i in range(20)]
    new = [mk(f"new{i}", 3, 0, score=1.0 - i * 0.01) for i in range(20)]
    cfg = FreshnessConfig(alpha=0.0, beta=0.0, recency_mode=RecencyMode.QUOTA,
                          quota_months=12, quota_ratio=0.30)
    out, stats = rerank(old + new, config=cfg, today=TODAY)

    for prefix in (10, 20, 30):
        recent = sum(1 for p in out[:prefix] if p.paper_id.startswith("new"))
        assert recent >= int(prefix * 0.30), f"상위 {prefix}에 최신 {recent}편뿐"
    assert stats.quota_promoted > 0


def test_quota_does_nothing_when_already_satisfied():
    """쿼터는 하한이지 상한이 아닙니다."""
    new = [mk(f"new{i}", 3, 0, score=10.0 - i) for i in range(10)]
    cfg = FreshnessConfig(recency_mode=RecencyMode.QUOTA, quota_ratio=0.30)
    out, stats = rerank(new, config=cfg, today=TODAY)
    assert stats.quota_promoted == 0
    assert len(out) == 10


def test_missing_date_does_not_crash():
    p = Paper(paper_id="x", base_id="x", title="x", abstract="", date="", citation_count=5)
    out, stats = rerank([p], today=TODAY)
    assert len(out) == 1
    assert stats.n_missing_date == 1
