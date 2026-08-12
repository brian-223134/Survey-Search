"""S6 — freshness-aware 랭킹. DESIGN §S6.

**이 프로젝트의 핵심 가설이 들어가는 자리입니다.**

문제: 인용수 정렬은 최근 논문을 구조적으로 배제합니다. 2026-06 논문은 아무리 중요해도
인용수가 0에 가깝습니다. 이 코퍼스에서 2025~2026년 논문이 전체의 30.6%인데,
`sort_by_citation_period` 같은 인용수 정렬은 그 30%를 통째로 뒤로 보냅니다.

제안하는 점수:

    final = rrf × (1 + α · citation_rate_percentile) × (1 + β · recency_weight)
    citation_rate = citation_count / max(months_since_pub, 3)      # 연령 정규화

핵심은 **절대 인용수가 아니라 또래 대비 인용 속도**를 보는 것입니다. 2026년 논문의
인용수 3은 2019년 논문의 인용수 3과 전혀 다른 의미입니다. 그래서 같은 연령 코호트
안에서의 백분위로 바꿔 씁니다.

DESIGN이 요구한 대로 recency 처리 방식 **두 가지를 다 구현하고 비교**합니다:

- `RecencyMode.WEIGHT` — 최근 논문에 완만한 가산 (연속적, 부드러움)
- `RecencyMode.QUOTA`  — 최종 목록의 x%를 최근 논문에 강제 배정 (이산적, 보장됨)

가산은 "밀어주되 보장은 없음", 쿼터는 "보장하되 순위를 왜곡함"입니다. 어느 쪽이 나은지는
실측으로 정합니다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date as _date
from enum import Enum

from survey_search.types import Paper

#: 인용수를 연령으로 나눌 때의 하한(개월). 갓 나온 논문의 인용률이 폭발하는 것을 막습니다.
MIN_AGE_MONTHS = 3.0

#: 같은 코호트로 묶는 연령 구간(개월). 6개월이면 arXiv 회전 속도에 맞습니다.
COHORT_MONTHS = 6.0


class RecencyMode(str, Enum):
    NONE = "none"
    WEIGHT = "weight"
    QUOTA = "quota"


@dataclass(frozen=True)
class FreshnessConfig:
    alpha: float = 0.5              # 인용률 백분위 가중
    beta: float = 0.5               # recency 가중
    recency_mode: RecencyMode = RecencyMode.WEIGHT
    half_life_months: float = 18.0  # WEIGHT 모드의 감쇠 반감기
    quota_months: int = 12          # QUOTA 모드에서 "최근"의 정의
    quota_ratio: float = 0.30       # 최종 목록에서 최근 논문에 보장할 비율


@dataclass
class RankStats:
    """무음 폐기 금지 — 인용 정보가 없어 항이 죽은 건수까지 남깁니다."""

    n_scored: int = 0
    n_missing_citation: int = 0     # citation_count 가 None -> 인용 항이 1.0
    n_missing_date: int = 0         # 날짜 없음 -> recency 항이 1.0
    n_cohorts: int = 0
    quota_promoted: int = 0         # QUOTA 모드가 끌어올린 논문 수
    note: str = ""


def _age_months(paper: Paper, today: _date) -> float | None:
    return paper.months_since(today)


def citation_rate(paper: Paper, today: _date) -> float | None:
    """연령 정규화 인용률. 인용수나 날짜를 모르면 None."""
    if paper.citation_count is None:
        return None
    age = _age_months(paper, today)
    if age is None:
        return None
    return paper.citation_count / max(age, MIN_AGE_MONTHS)


def cohort_percentiles(
    papers: Sequence[Paper], today: _date
) -> tuple[dict[str, float], RankStats]:
    """**같은 연령대 안에서의** 인용률 백분위를 계산합니다.

    전체를 한 줄로 세우면 오래된 논문이 상위 백분위를 독식합니다. 6개월 코호트로
    나눠 각 코호트 안에서 순위를 매기면 "또래 대비 얼마나 빨리 인용되는가"가 됩니다.

    인용률을 모르는 논문은 백분위 0.5(중립)를 줍니다 — 0을 주면 정보 없음이
    페널티가 되고, 1을 주면 보상이 됩니다. 둘 다 틀립니다.
    """
    stats = RankStats(n_scored=len(papers))
    cohorts: dict[int, list[tuple[str, float]]] = {}
    unknown: list[str] = []

    for p in papers:
        rate = citation_rate(p, today)
        if rate is None:
            unknown.append(p.paper_id)
            if p.citation_count is None:
                stats.n_missing_citation += 1
            if _age_months(p, today) is None:
                stats.n_missing_date += 1
            continue
        age = _age_months(p, today) or 0.0
        cohorts.setdefault(int(age // COHORT_MONTHS), []).append((p.paper_id, rate))

    stats.n_cohorts = len(cohorts)
    pct: dict[str, float] = {pid: 0.5 for pid in unknown}

    for members in cohorts.values():
        members.sort(key=lambda kv: kv[1])
        n = len(members)
        if n == 1:
            pct[members[0][0]] = 0.5
            continue
        for i, (pid, _) in enumerate(members):
            pct[pid] = i / (n - 1)

    return pct, stats


def recency_weight(paper: Paper, today: _date, half_life_months: float) -> float:
    """0~1. 오늘 나온 논문이 1.0, 반감기마다 절반. 날짜를 모르면 0."""
    age = _age_months(paper, today)
    if age is None:
        return 0.0
    return math.pow(0.5, max(age, 0.0) / half_life_months)


def rerank(
    papers: Sequence[Paper],
    *,
    config: FreshnessConfig | None = None,
    today: _date | None = None,
) -> tuple[list[Paper], RankStats]:
    """freshness 점수로 재랭킹. `Paper.score`(=RRF 점수)를 기반으로 곱셈 보정합니다.

    곱셈인 이유: RRF 점수가 이미 "여러 검색이 얼마나 동의하는가"를 담고 있으므로,
    freshness 는 그걸 **대체**하는 게 아니라 **조정**해야 합니다. 덧셈이면 스케일이
    다른 두 신호를 섞게 되고, 관련성 없는 최신 논문이 올라옵니다.
    """
    cfg = config or FreshnessConfig()
    today = today or _date.today()

    pct, stats = cohort_percentiles(papers, today)

    scored: list[tuple[Paper, float]] = []
    for p in papers:
        cite_term = 1.0 + cfg.alpha * pct.get(p.paper_id, 0.5)
        if cfg.recency_mode is RecencyMode.WEIGHT:
            rec_term = 1.0 + cfg.beta * recency_weight(p, today, cfg.half_life_months)
        else:
            rec_term = 1.0
        scored.append((p, p.score * cite_term * rec_term))

    scored.sort(key=lambda kv: (-kv[1], kv[0].paper_id))

    if cfg.recency_mode is RecencyMode.QUOTA:
        scored, promoted = _apply_quota(scored, cfg, today)
        stats.quota_promoted = promoted

    stats.note = f"mode={cfg.recency_mode.value} alpha={cfg.alpha} beta={cfg.beta}"
    out = [
        Paper(
            paper_id=p.paper_id, base_id=p.base_id, title=p.title, abstract=p.abstract,
            date=p.date, submitted_date=p.submitted_date,
            categories=p.categories, citation_count=p.citation_count,
            score=s, facets=p.facets, provenance=p.provenance,
        )
        for p, s in scored
    ]
    return out, stats


def _apply_quota(
    scored: list[tuple[Paper, float]], cfg: FreshnessConfig, today: _date
) -> tuple[list[tuple[Paper, float]], int]:
    """최근 논문 쿼터를 **모든 접두구간에서** 보장하도록 재배치합니다.

    최종 컷(`n_papers`)은 이 목록의 접두구간이므로, 접두구간마다 비율이 지켜지면
    컷 위치가 어디든 쿼터가 성립합니다. rank 단계가 `n_papers` 를 몰라도 되는 이유입니다.

    규칙: 위치 `i`(0-based)까지 최근 논문이 `ceil(quota_ratio × (i+1))` 편 미만이면
    다음 최근 논문을 먼저 뽑고, 아니면 남은 것 중 점수 최고를 뽑습니다.

    **쿼터는 하한이지 상한이 아닙니다** — 이미 충분하면 원래 순위를 그대로 둡니다.
    `promoted` 는 점수 순서를 거스르고 최근 논문을 당겨온 횟수입니다. 0이면
    "쿼터가 필요 없었다"는 뜻이지 고장이 아닙니다.
    """

    def is_recent(p: Paper) -> bool:
        age = p.months_since(today)
        return age is not None and age <= cfg.quota_months

    recent = [(p, s) for p, s in scored if is_recent(p)]
    others = [(p, s) for p, s in scored if not is_recent(p)]
    if not recent or not others:
        return scored, 0

    out: list[tuple[Paper, float]] = []
    ri = oi = 0
    promoted = 0
    n = len(scored)

    for i in range(n):
        need = math.ceil(cfg.quota_ratio * (i + 1))
        take_recent = ri < need and ri < len(recent)

        if take_recent and oi < len(others):
            # 점수만 보면 others 가 먼저인데 쿼터 때문에 당겨온 경우만 셉니다
            if others[oi][1] > recent[ri][1]:
                promoted += 1
        if take_recent:
            out.append(recent[ri]); ri += 1
        elif oi < len(others):
            # 쿼터가 이미 충족됐으면 남은 것 중 점수 높은 쪽
            if ri < len(recent) and recent[ri][1] >= others[oi][1]:
                out.append(recent[ri]); ri += 1
            else:
                out.append(others[oi]); oi += 1
        elif ri < len(recent):
            out.append(recent[ri]); ri += 1

    return out, promoted
