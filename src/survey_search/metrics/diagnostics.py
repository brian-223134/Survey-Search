"""P4 — 진단 하네스.

정량 벤치마크(SurGE)가 막혀 있으므로 **자체 진단 지표로 개발을 굴립니다.**

여기 있는 지표들은 전부 **정답 집합 없이** 계산됩니다. 그게 장점이자 한계입니다:
언제든 돌릴 수 있지만, "얼마나 최신인가"·"얼마나 다양한가"·"원본과 얼마나 다른가"만
말해 줄 뿐 **"얼마나 맞는가"는 말해 주지 않습니다.** 표를 읽을 때 이 선을 넘지 마세요.

- `stage_report` (4.1) — 단계별 in/out, 폐기 건수와 사유, 소요 시간
- `freshness_report` (4.2) — 최근 6/12/24개월 비율 + 연도별 히스토그램
- `coverage_report` (4.3) — facet별 논문 수, 미충족 facet
- `compare` (4.4) — 두 결과의 교집합/신규 유입. **베이스라인 대조의 핵심**
- `snapshot` / `diff_snapshot` (4.5) — 회귀 고정
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

from survey_search.types import SearchResult


# --- 4.1 단계별 리포트 ---------------------------------------------------------

def stage_report(result: SearchResult) -> str:
    """`stats.report()` 에 폐기 합계를 덧붙인 것."""
    dropped = sum(s.dropped for s in result.stats.stages)
    skipped = [s.name for s in result.stats.stages if s.skipped]
    lines = [result.stats.report(), ""]
    lines.append(f"총 폐기: {dropped:,}편")
    if skipped:
        lines.append(f"건너뛴 단계: {', '.join(skipped)}")
    return "\n".join(lines)


# --- 4.2 최신성 ---------------------------------------------------------------

@dataclass
class FreshnessReport:
    n: int
    recent_6m: float
    recent_12m: float
    recent_24m: float
    by_year: dict[str, int]
    n_missing_date: int
    median_citation: int | None
    median_citation_recent_12m: int | None

    def render(self) -> str:
        lines = [
            f"논문 {self.n:,}편  |  6m={self.recent_6m:.1%}  12m={self.recent_12m:.1%}  "
            f"24m={self.recent_24m:.1%}",
        ]
        if self.median_citation is not None:
            lines.append(
                f"인용수 중앙값: 전체 {self.median_citation:,}"
                + (f"  최근12m {self.median_citation_recent_12m:,}"
                   if self.median_citation_recent_12m is not None else "")
            )
        if self.n_missing_date:
            lines.append(f"날짜 없음: {self.n_missing_date:,}편")
        if self.by_year:
            peak = max(self.by_year.values())
            lines.append("\n연도별 분포:")
            for year in sorted(self.by_year):
                n = self.by_year[year]
                bar = "#" * max(1, round(40 * n / peak))
                lines.append(f"  {year}  {n:>5,}  {bar}")
        return "\n".join(lines)


def _median(xs: list[int]) -> int | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def freshness_report(result: SearchResult, today: _date | None = None) -> FreshnessReport:
    today = today or _date.today()
    papers = result.papers
    ages = [(p, p.months_since(today)) for p in papers]
    known = [(p, a) for p, a in ages if a is not None]
    n = len(papers) or 1

    recent12 = [p for p, a in known if a <= 12]
    return FreshnessReport(
        n=len(papers),
        recent_6m=sum(1 for _, a in known if a <= 6) / n,
        recent_12m=len(recent12) / n,
        recent_24m=sum(1 for _, a in known if a <= 24) / n,
        by_year=dict(sorted(Counter(p.date[:4] for p in papers if p.date).items())),
        n_missing_date=len(ages) - len(known),
        median_citation=_median([p.citation_count for p in papers
                                 if p.citation_count is not None]),
        median_citation_recent_12m=_median([p.citation_count for p in recent12
                                            if p.citation_count is not None]),
    )


# --- 4.3 커버리지 -------------------------------------------------------------

@dataclass
class CoverageReport:
    n_facets: int
    per_facet: dict[str, int]
    empty_facets: list[str]
    underfilled: list[tuple[str, int]]   # (facet, n) — 기대치 미달
    expected_per_facet: int
    memberships: int                     # facet 소속의 총합 (논문 1편이 여러 facet 에 속함)
    facets_per_paper: float

    def render(self) -> str:
        lines = [
            f"facet {self.n_facets}개  |  논문당 평균 {self.facets_per_paper:.1f}개 facet 소속",
            f"기대 배정 {self.expected_per_facet:,}편/개 (총 소속 {self.memberships:,}건 ÷ facet 수)",
        ]
        for name in sorted(self.per_facet, key=lambda k: -self.per_facet[k]):
            mark = "  <- 미달" if any(name == f for f, _ in self.underfilled) else ""
            share = self.per_facet[name] / self.expected_per_facet if self.expected_per_facet else 0
            lines.append(f"  {self.per_facet[name]:>5,}편  ({share:>4.0%} of 기대)  {name}{mark}")
        if self.empty_facets:
            lines.append(f"\n논문 0편인 facet {len(self.empty_facets)}개: {self.empty_facets}")
        return "\n".join(lines)


def coverage_report(result: SearchResult, *, min_ratio: float = 0.5) -> CoverageReport:
    """facet 별 기여. 기대치의 `min_ratio` 미만이면 '미달'로 표시합니다.

    **기대치는 `n_papers / n_facets` 가 아닙니다.** facet 들은 서로 겹칩니다 — 한 논문이
    여러 facet 에 잡히는 것이 정상이고 (실측: 논문당 평균 8개), 그래서 소속의 총합이
    논문 수의 몇 배가 됩니다. 분모를 논문 수로 잡으면 기대치가 실제보다 훨씬 작아져
    '미달' 판정이 사실상 발동하지 않습니다. 총 소속 수를 facet 수로 나눈 값이
    올바른 귀무 기대치입니다.

    facet 이 하나뿐이면(S1 이 꺼져 있으면) 의미가 없습니다 — 그래도 계산은 하고,
    호출부가 `n_facets == 1` 을 보고 판단하게 둡니다.
    """
    per = {f.name: len(f.paper_ids) for f in result.facets}
    n_facets = max(len(per), 1)
    memberships = sum(per.values())
    expected = memberships // n_facets
    threshold = expected * min_ratio
    return CoverageReport(
        n_facets=len(per),
        per_facet=per,
        empty_facets=[k for k, v in per.items() if v == 0],
        underfilled=[(k, v) for k, v in per.items() if 0 < v < threshold],
        expected_per_facet=expected,
        memberships=memberships,
        facets_per_paper=memberships / len(result.papers) if result.papers else 0.0,
    )


# --- 4.4 베이스라인 대조 -------------------------------------------------------

@dataclass
class ComparisonReport:
    label_a: str
    label_b: str
    n_a: int
    n_b: int
    n_shared: int
    only_a: list[str]
    only_b: list[str]
    recent_12m_a: float
    recent_12m_b: float
    median_citation_a: int | None
    median_citation_b: int | None

    @property
    def jaccard(self) -> float:
        union = self.n_a + self.n_b - self.n_shared
        return self.n_shared / union if union else 0.0

    def render(self, sample: int = 5) -> str:
        lines = [
            f"{self.label_a} ({self.n_a:,}편)  vs  {self.label_b} ({self.n_b:,}편)",
            f"  교집합 {self.n_shared:,}  |  Jaccard {self.jaccard:.3f}",
            f"  {self.label_b} 만: {len(self.only_b):,}편   {self.label_a} 만: {len(self.only_a):,}편",
            f"  최근 12개월: {self.recent_12m_a:.1%} -> {self.recent_12m_b:.1%}",
        ]
        if self.median_citation_a is not None and self.median_citation_b is not None:
            lines.append(
                f"  인용수 중앙값: {self.median_citation_a:,} -> {self.median_citation_b:,}"
            )
        if self.only_b[:sample]:
            lines.append(f"  {self.label_b} 만 있는 예시: {self.only_b[:sample]}")
        return "\n".join(lines)


def compare(
    a: SearchResult, b: SearchResult, *, label_a: str = "A", label_b: str = "B",
    today: _date | None = None,
) -> ComparisonReport:
    """**"우리 검색이 원본이 못 찾던 무엇을 찾는가"를 정답 집합 없이 정량화하는 유일한 방법.**

    비교는 `base_id`(버전 제거) 기준입니다 — 같은 논문의 v1/v2 를 다른 논문으로 세면
    신규 유입이 부풀려집니다.
    """
    today = today or _date.today()
    ids_a = {p.base_id: p for p in a.papers}
    ids_b = {p.base_id: p for p in b.papers}
    shared = set(ids_a) & set(ids_b)

    fa, fb = freshness_report(a, today), freshness_report(b, today)
    return ComparisonReport(
        label_a=label_a, label_b=label_b,
        n_a=len(a.papers), n_b=len(b.papers), n_shared=len(shared),
        only_a=sorted(set(ids_a) - shared),
        only_b=sorted(set(ids_b) - shared),
        recent_12m_a=fa.recent_12m, recent_12m_b=fb.recent_12m,
        median_citation_a=fa.median_citation, median_citation_b=fb.median_citation,
    )


# --- 4.5 회귀 고정 ------------------------------------------------------------

def snapshot(result: SearchResult, path: Path) -> dict:
    """결과를 스냅샷으로 저장. 논문 목록과 핵심 지표만 남깁니다 — 소요 시간처럼
    실행마다 달라지는 값은 **일부러 뺍니다**. 넣으면 매번 diff 가 납니다."""
    data = {
        "topic": result.topic,
        "n_papers": len(result.papers),
        "recent_12m_ratio": round(result.stats.recent_12m_ratio, 4),
        "paper_ids": [p.paper_id for p in result.papers],
        "facets": {f.name: len(f.paper_ids) for f in result.facets},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def diff_snapshot(result: SearchResult, path: Path) -> dict:
    """저장된 스냅샷과 현재 결과의 차이. 파일이 없으면 `{"status": "no_baseline"}`."""
    if not path.exists():
        return {"status": "no_baseline", "path": str(path)}
    old = json.loads(path.read_text())
    old_ids, new_ids = set(old["paper_ids"]), {p.paper_id for p in result.papers}
    added, removed = new_ids - old_ids, old_ids - new_ids
    return {
        "status": "same" if not added and not removed else "changed",
        "n_added": len(added),
        "n_removed": len(removed),
        "added_sample": sorted(added)[:10],
        "removed_sample": sorted(removed)[:10],
        "recent_12m_ratio": {
            "old": old.get("recent_12m_ratio"),
            "new": round(result.stats.recent_12m_ratio, 4),
        },
    }
