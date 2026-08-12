"""S7 — 다양성. DESIGN §S7.

서베이의 목적 함수는 정확도가 아니라 **커버리지**입니다. 상위권이 한 연구 그룹·한 계열로
쏠리면 서베이의 섹션 하나가 통째로 비어 버립니다. 관련성만 최적화하면 정확히 그 일이
일어납니다 — 가장 관련 있는 논문 1500편은 서로 매우 비슷하기 때문입니다.

두 장치를 씁니다:

- **MMR**: `λ · relevance − (1−λ) · max_sim(이미 고른 것들)`.
  유사도는 인덱스에 **저장된 초록 임베딩**을 그대로 씁니다 (재임베딩 0).
- **facet 쿼터**: 최종 N편을 facet 수로 나눠 최소 배정 보장. facet 크기가 균등하지
  않을 수 있으므로 **최소 보장만** 하고 나머지는 점수순입니다.

주의: 저장 벡터를 꺼낼 때 `faiss_id - 1` 을 행 번호로 쓰면 안 됩니다 (id_map 이 순열).
`index/inspect_faiss.py` 의 `build_id_to_row()` 를 거쳐야 합니다 — SETTING.md §6-A.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from survey_search.types import Paper


@dataclass
class DiversityStats:
    """무음 스킵 금지 — 벡터가 없어 MMR 을 못 돌렸으면 그 사실이 남습니다."""

    n_in: int = 0
    n_out: int = 0
    mmr_applied: bool = False
    n_missing_vectors: int = 0
    facet_quota_applied: bool = False
    n_facets: int = 0
    quota_per_facet: int = 0
    n_promoted_by_quota: int = 0
    note: str = ""


def mmr(
    papers: Sequence[Paper],
    vectors: np.ndarray,
    *,
    k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance. 고른 논문의 **인덱스** 목록을 순서대로 돌려줍니다.

    `vectors[i]` 가 `papers[i]` 의 임베딩이어야 합니다. 단위 norm 을 가정합니다
    (이 인덱스의 저장 벡터가 그렇습니다) — 그래서 내적이 곧 코사인입니다.

    `lambda_=1.0` 이면 순수 관련성(= 원래 순위), `0.0` 이면 순수 다양성입니다.

    구현 노트: 이미 고른 것들과의 최대 유사도를 매번 전부 다시 재면 O(k²n) 입니다.
    새로 고른 것 하나와의 유사도만 갱신하면 O(kn) 이 됩니다. k=1500, n=3000 에서
    그 차이가 분 단위와 초 단위를 가릅니다.
    """
    n = len(papers)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)

    relevance = np.array([p.score for p in papers], dtype="float32")
    # 점수 스케일이 제각각이라(RRF vs freshness 보정) 0~1 로 정규화해야
    # lambda_ 가 의도한 균형을 갖습니다.
    rng = relevance.max() - relevance.min()
    relevance = (relevance - relevance.min()) / rng if rng > 0 else np.ones(n, dtype="float32")

    selected: list[int] = []
    remaining = np.ones(n, dtype=bool)
    max_sim = np.zeros(n, dtype="float32")

    first = int(np.argmax(relevance))
    selected.append(first)
    remaining[first] = False
    max_sim = vectors @ vectors[first]

    while len(selected) < k:
        score = lambda_ * relevance - (1.0 - lambda_) * max_sim
        score[~remaining] = -np.inf
        pick = int(np.argmax(score))
        if not np.isfinite(score[pick]):
            break
        selected.append(pick)
        remaining[pick] = False
        # 새로 고른 것과의 유사도만 반영해서 갱신 — 전체 재계산 안 함
        np.maximum(max_sim, vectors @ vectors[pick], out=max_sim)

    return selected


def facet_quota(
    papers: Sequence[Paper],
    *,
    n: int,
    min_per_facet: int | None = None,
) -> tuple[list[int], DiversityStats]:
    """facet 별 **최소 배정**을 보장하고 나머지는 점수순으로 채웁니다.

    facet 이 하나뿐이면(= S1 이 꺼져 있으면) 아무것도 하지 않습니다. 그 사실을
    `facet_quota_applied=False` 로 남깁니다 — 조용히 통과시키지 않습니다.
    """
    stats = DiversityStats(n_in=len(papers))
    by_facet: dict[str, list[int]] = {}
    for i, p in enumerate(papers):
        for f in (p.facets or ("(none)",)):
            by_facet.setdefault(f, []).append(i)

    stats.n_facets = len(by_facet)
    if len(by_facet) <= 1:
        out = list(range(min(n, len(papers))))
        stats.n_out = len(out)
        stats.note = "facet 1개 -> 쿼터 무의미 (S1 이 꺼져 있으면 정상)"
        return out, stats

    quota = min_per_facet if min_per_facet is not None else max(1, n // len(by_facet))
    stats.quota_per_facet = quota
    stats.facet_quota_applied = True

    chosen: list[int] = []
    seen: set[int] = set()
    for members in by_facet.values():
        for i in members[:quota]:
            if i not in seen:
                seen.add(i)
                chosen.append(i)
    stats.n_promoted_by_quota = sum(1 for i in chosen if i >= n)

    for i in range(len(papers)):
        if len(chosen) >= n:
            break
        if i not in seen:
            seen.add(i)
            chosen.append(i)

    out = chosen[:n]
    stats.n_out = len(out)
    return out, stats


def diversify(
    papers: Sequence[Paper],
    *,
    n: int,
    vectors: np.ndarray | None = None,
    lambda_: float = 0.7,
    min_per_facet: int | None = None,
) -> tuple[list[Paper], DiversityStats]:
    """S7 전체 — MMR 로 뽑고, facet 쿼터로 보정합니다.

    `vectors` 가 None 이면 MMR 을 건너뛰고 **그 사실을 stats 에 남깁니다.**
    벡터를 못 구했는데 조용히 점수순으로 돌려주면, 껐을 때와 구분이 안 됩니다.
    """
    if not papers:
        return [], DiversityStats()

    if vectors is None:
        order = list(range(len(papers)))
        stats = DiversityStats(n_in=len(papers), mmr_applied=False,
                               note="벡터 없음 -> MMR 건너뜀, 점수순 유지")
    else:
        # MMR 은 최종 n 보다 넉넉히 뽑아 둡니다 — facet 쿼터가 뒤에서 재배치하므로
        order = mmr(papers, vectors, k=min(len(papers), max(n * 2, n)), lambda_=lambda_)
        stats = DiversityStats(n_in=len(papers), mmr_applied=True,
                               note=f"MMR lambda={lambda_}")

    reordered = [papers[i] for i in order]
    picked, qstats = facet_quota(reordered, n=n, min_per_facet=min_per_facet)

    stats.n_out = len(picked)
    stats.n_facets = qstats.n_facets
    stats.facet_quota_applied = qstats.facet_quota_applied
    stats.quota_per_facet = qstats.quota_per_facet
    stats.n_promoted_by_quota = qstats.n_promoted_by_quota
    if qstats.note:
        stats.note = f"{stats.note}; {qstats.note}"

    return [reordered[i] for i in picked], stats
