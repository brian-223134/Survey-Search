"""P4 진단 하네스 — 합성 결과, 자산 불필요."""

from datetime import date

from survey_search.metrics.diagnostics import (
    compare,
    coverage_report,
    diff_snapshot,
    freshness_report,
    snapshot,
    stage_report,
)
from survey_search.types import Facet, Paper, SearchResult, SearchStats, StageStat

TODAY = date(2026, 8, 12)


def mk(pid: str, months_ago: int, cc: int | None = 10, facets=("f",)) -> Paper:
    d = date.fromordinal(TODAY.toordinal() - int(months_ago * 30.44))
    return Paper(paper_id=pid, base_id=pid.rstrip("v123456789").rstrip("v") or pid,
                 title=pid, abstract="", date=d.isoformat(),
                 citation_count=cc, facets=facets)


def mkresult(papers, facets=None, stats=None) -> SearchResult:
    st = stats or SearchStats(topic="t", backend="b")
    st.n_final = len(papers)
    return SearchResult(topic="t", papers=tuple(papers),
                        facets=tuple(facets or ()), stats=st)


def test_freshness_report_buckets_and_histogram():
    r = mkresult([mk("a", 3), mk("b", 10), mk("c", 20), mk("d", 40)])
    f = freshness_report(r, TODAY)
    assert f.n == 4
    assert f.recent_6m == 0.25
    assert f.recent_12m == 0.5
    assert f.recent_24m == 0.75
    assert sum(f.by_year.values()) == 4
    assert "연도별 분포" in f.render()


def test_freshness_report_separates_recent_citation_median():
    """최근 논문의 인용수 중앙값이 따로 나와야 freshness 의 대가를 볼 수 있습니다."""
    r = mkresult([mk("old", 40, 100), mk("new1", 3, 1), mk("new2", 4, 3)])
    f = freshness_report(r, TODAY)
    assert f.median_citation == 3
    assert f.median_citation_recent_12m == 3


def test_freshness_report_handles_missing_dates():
    p = Paper(paper_id="x", base_id="x", title="x", abstract="", date="", citation_count=1)
    f = freshness_report(mkresult([p]), TODAY)
    assert f.n_missing_date == 1
    assert f.recent_12m == 0.0


def test_freshness_report_empty_result():
    f = freshness_report(mkresult([]), TODAY)
    assert f.n == 0 and f.recent_12m == 0.0


def test_coverage_flags_empty_and_underfilled_facets():
    papers = [mk(f"p{i}", 5) for i in range(100)]
    facets = [
        Facet(name="big", paper_ids=tuple(f"p{i}" for i in range(60))),
        Facet(name="small", paper_ids=("p60",)),
        Facet(name="none", paper_ids=()),
    ]
    c = coverage_report(mkresult(papers, facets))
    assert c.n_facets == 3
    assert c.empty_facets == ["none"]
    assert ("small", 1) in c.underfilled
    assert "미달" in c.render()


def test_coverage_expected_accounts_for_overlapping_facets():
    """facet 은 서로 겹칩니다. 기대치를 n_papers/n_facets 로 잡으면 너무 작아져
    '미달' 판정이 발동하지 않습니다 — 총 소속 수를 분자로 써야 합니다."""
    papers = [mk(f"p{i}", 5) for i in range(100)]
    # 논문 100편, facet 2개, 각각 90편씩 잡음 -> 소속 180건, 논문당 1.8개
    facets = [
        Facet(name="a", paper_ids=tuple(f"p{i}" for i in range(90))),
        Facet(name="b", paper_ids=tuple(f"p{i}" for i in range(10, 100))),
    ]
    c = coverage_report(mkresult(papers, facets))
    assert c.memberships == 180
    assert c.expected_per_facet == 90        # 100//2 = 50 이 아니라 180//2 = 90
    assert abs(c.facets_per_paper - 1.8) < 1e-6
    assert c.underfilled == []               # 둘 다 기대치와 같으므로 미달 아님


def test_coverage_with_single_facet_is_computed_not_hidden():
    """facet 이 하나여도 계산은 합니다 — 판단은 호출부가."""
    c = coverage_report(mkresult([mk("a", 1)], [Facet(name="(topic)", paper_ids=("a",))]))
    assert c.n_facets == 1
    assert c.empty_facets == []


def test_compare_counts_new_and_shared():
    a = mkresult([mk("x", 30), mk("y", 30)])
    b = mkresult([mk("y", 30), mk("z", 3)])
    c = compare(a, b, label_a="base", label_b="ours", today=TODAY)
    assert c.n_shared == 1
    assert c.only_b == ["z"]
    assert c.only_a == ["x"]
    assert 0 < c.jaccard < 1
    assert c.recent_12m_b > c.recent_12m_a


def test_compare_uses_base_id_so_versions_are_not_double_counted():
    """같은 논문의 v1/v2 를 신규 유입으로 세면 효과가 부풀려집니다."""
    a = mkresult([Paper(paper_id="2401.1v1", base_id="2401.1", title="t",
                        abstract="", date="2024-01-01")])
    b = mkresult([Paper(paper_id="2401.1v2", base_id="2401.1", title="t",
                        abstract="", date="2024-01-01")])
    c = compare(a, b, today=TODAY)
    assert c.n_shared == 1
    assert c.only_b == []


def test_compare_disjoint_sets():
    c = compare(mkresult([mk("a", 1)]), mkresult([mk("b", 1)]), today=TODAY)
    assert c.jaccard == 0.0
    assert c.n_shared == 0


def test_snapshot_roundtrip_and_diff(tmp_path):
    r = mkresult([mk("a", 1), mk("b", 2)])
    p = tmp_path / "snap.json"

    assert diff_snapshot(r, p)["status"] == "no_baseline"

    snapshot(r, p)
    assert diff_snapshot(r, p)["status"] == "same"

    r2 = mkresult([mk("a", 1), mk("c", 3)])
    d = diff_snapshot(r2, p)
    assert d["status"] == "changed"
    assert d["n_added"] == 1 and d["n_removed"] == 1
    assert d["added_sample"] == ["c"]


def test_snapshot_excludes_timing_so_it_does_not_churn(tmp_path):
    st = SearchStats(topic="t", backend="b", total_s=1.23)
    data = snapshot(mkresult([mk("a", 1)], stats=st), tmp_path / "s.json")
    assert "total_s" not in data
    assert "elapsed" not in json_keys(data)


def json_keys(d, out=None):
    out = out if out is not None else []
    if isinstance(d, dict):
        for k, v in d.items():
            out.append(k)
            json_keys(v, out)
    return out


def test_stage_report_totals_drops_and_lists_skips():
    st = SearchStats(topic="t", backend="b")
    st.add(StageStat("S2 dense", 1, 100))
    st.add(StageStat("S5 dedup", 100, 90, dropped=10, reason="version"))
    st.add(StageStat("S7 diversity", 90, 90, skipped=True, note="disabled"))
    text = stage_report(mkresult([mk("a", 1)], stats=st))
    assert "총 폐기: 10편" in text
    assert "S7 diversity" in text
