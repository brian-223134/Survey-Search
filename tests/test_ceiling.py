"""천장 측정의 해석 로직 — 검색 병목과 랭킹 병목을 옳게 갈라내는지."""

from survey_search.eval.ceiling import CeilingReport, CeilingRow


def row(**kw) -> CeilingRow:
    base = dict(topic="t", n_gold=100, n_in_corpus=100, n_after_cutoff=100,
                n_in_pool=100, n_in_final=100, pool_size=5000, final_size=1500)
    base.update(kw)
    return CeilingRow(**base)


def test_rates_are_fractions_of_gold():
    r = row(n_in_corpus=90, n_after_cutoff=80, n_in_pool=60, n_in_final=30)
    assert r.corpus_rate == 0.9
    assert r.cutoff_rate == 0.8
    assert r.pool_rate == 0.6
    assert r.final_rate == 0.3


def test_gaps_split_retrieval_from_ranking():
    r = row(n_after_cutoff=80, n_in_pool=60, n_in_final=30)
    assert abs(r.retrieval_gap - 0.2) < 1e-9    # 80 -> 60
    assert abs(r.ranking_gap - 0.3) < 1e-9      # 60 -> 30


def test_zero_gold_does_not_divide_by_zero():
    r = row(n_gold=0, n_in_corpus=0, n_after_cutoff=0, n_in_pool=0, n_in_final=0)
    assert r.corpus_rate == 0.0 and r.final_rate == 0.0


def test_verdict_points_at_retrieval_when_pool_misses():
    """풀에 못 들어온 게 크면 '검색 문제' 로 읽어야 합니다 — 본문이 의미를 갖는 경우."""
    rep = CeilingReport(rows=[row(n_after_cutoff=90, n_in_pool=30, n_in_final=28)])
    text = rep.render()
    assert "검색이 병목" in text


def test_verdict_points_at_ranking_when_pool_has_them():
    """풀에는 다 있는데 최종에 못 들면 '랭킹 문제' — 본문은 recall 을 못 올립니다."""
    rep = CeilingReport(rows=[row(n_after_cutoff=90, n_in_pool=88, n_in_final=20)])
    text = rep.render()
    assert "랭킹이 병목" in text
    assert "본문 도입은" in text


def test_verdict_balanced():
    rep = CeilingReport(rows=[row(n_after_cutoff=90, n_in_pool=60, n_in_final=35)])
    assert "비슷하게 기여" in rep.render()


def test_render_reports_all_four_levels():
    rep = CeilingReport(rows=[row()], config_note="facets=True")
    text = rep.render()
    for label in ("코퍼스 상한", "컷오프 상한", "풀 recall", "최종 recall"):
        assert label in text


def test_empty_report_does_not_crash():
    assert "측정된 토픽 없음" in CeilingReport().render()
