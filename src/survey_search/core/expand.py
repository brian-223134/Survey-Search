"""S8 — 인용 스노우볼링. DESIGN §S8.

상위 시드 논문의 참고문헌(후방)과 피인용(전방)을 타고 후보를 넓힙니다.
**dense 유사도도 BM25 도 못 하는 신호입니다.** 임베딩은 "말이 비슷한 논문"을 찾고
BM25 는 "단어가 겹치는 논문"을 찾지만, 인용은 **저자가 직접 관련 있다고 선언한 관계**입니다.
사람이 문헌조사하는 방식이기도 합니다.

두 방향의 성격이 다릅니다:

- **후방(references)** — 시드가 딛고 선 토대. 오래됐지만 확실히 관련 있습니다
- **전방(cited_by)** — 시드 이후의 후속 연구. **컷오프 이후 논문이 여기서 나옵니다.**
  실측: RAG 원논문의 피인용 309편 중 76편(25%)이 우리 로컬 코퍼스에 없습니다

백엔드가 `references`/`cited_by` 를 지원하지 않으면 이 단계는 no-op 이 되고
**그 사실이 stats 에 남습니다** — 조용히 건너뛰면 켠 실험과 끈 실험이 구분되지 않습니다.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from survey_search.types import Paper

log = logging.getLogger(__name__)


@dataclass
class SnowballConfig:
    n_seeds: int = 20            # 상위 몇 편에서 출발할지
    hops: int = 1                # 1 = 시드의 이웃까지. 2 는 폭발적으로 늘어납니다
    backward: bool = True        # references
    forward: bool = True         # cited_by
    max_new: int = 2000          # 유입 상한 — 없으면 후보가 통제 불능이 됩니다
    #: 이만큼 이상의 시드가 가리킨 논문만 채택. 1이면 전부, 2면 "두 시드가 공통으로
    #: 인용/피인용한 논문"만 — 후자가 훨씬 정밀합니다.
    min_seed_support: int = 1


@dataclass
class SnowballStats:
    supported: bool = False      # 백엔드가 인용 엣지를 아는가
    n_seeds: int = 0
    n_backward: int = 0
    n_forward: int = 0
    n_new: int = 0               # 기존 후보에 없던 논문 수
    n_dropped_by_support: int = 0
    n_dropped_by_cap: int = 0
    n_meta_missing: int = 0      # 엣지로는 나왔는데 메타데이터를 못 구한 수
    elapsed_s: float = 0.0
    note: str = ""
    errors: list[str] = field(default_factory=list)


def snowball(
    candidates: Sequence[Paper],
    *,
    backend,
    config: SnowballConfig | None = None,
) -> tuple[list[Paper], SnowballStats]:
    """시드의 인용 이웃을 후보에 추가합니다.

    Returns:
        (추가된 논문 목록, 통계). 추가분만 돌려주므로 호출부가 융합 방식을 정합니다.
    """
    import time

    cfg = config or SnowballConfig()
    stats = SnowballStats()
    t0 = time.perf_counter()

    has_refs = callable(getattr(backend, "references", None))
    has_cits = callable(getattr(backend, "cited_by", None))
    if not (has_refs or has_cits):
        stats.note = "백엔드가 인용 엣지를 모릅니다 — 단계 건너뜀"
        stats.elapsed_s = time.perf_counter() - t0
        return [], stats

    stats.supported = True
    seeds = list(candidates[: cfg.n_seeds])
    stats.n_seeds = len(seeds)
    existing = {p.base_id for p in candidates}

    # 논문 -> 이 논문을 가리킨 시드 수. min_seed_support 판정에 씁니다.
    support: dict[str, int] = {}
    frontier = [p.paper_id for p in seeds]
    seen_seeds: set[str] = set()

    for _hop in range(max(cfg.hops, 0)):
        next_frontier: list[str] = []
        for pid in frontier:
            if pid in seen_seeds:
                continue
            seen_seeds.add(pid)
            if cfg.backward and has_refs:
                try:
                    refs = backend.references(pid)
                except Exception as e:              # noqa: BLE001 — 한 시드 실패로 전체를 죽이지 않습니다
                    stats.errors.append(f"references({pid}): {e}")
                    refs = []
                stats.n_backward += len(refs)
                for r in refs:
                    support[r] = support.get(r, 0) + 1
                    next_frontier.append(r)
            if cfg.forward and has_cits:
                try:
                    cits = backend.cited_by(pid)
                except Exception as e:              # noqa: BLE001
                    stats.errors.append(f"cited_by({pid}): {e}")
                    cits = []
                stats.n_forward += len(cits)
                for c in cits:
                    support[c] = support.get(c, 0) + 1
                    next_frontier.append(c)
        frontier = next_frontier

    from survey_search.core.dedup import strip_version

    fresh = {}
    for pid, n in support.items():
        base = strip_version(pid)
        if base in existing:
            continue
        if n < cfg.min_seed_support:
            stats.n_dropped_by_support += 1
            continue
        fresh[base] = max(fresh.get(base, 0), n)

    # 지지도 높은 순으로 자릅니다 — 여러 시드가 공통으로 가리킨 논문이 먼저입니다
    ranked = sorted(fresh.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > cfg.max_new:
        stats.n_dropped_by_cap = len(ranked) - cfg.max_new
        ranked = ranked[: cfg.max_new]

    ids = [pid for pid, _ in ranked]
    papers = backend.get_papers(ids) if ids else []
    stats.n_meta_missing = len(ids) - len(papers)
    stats.n_new = len(papers)

    # 지지도를 점수로 실어 보냅니다. 호출부가 RRF 로 융합할 때 순위로 쓰입니다.
    support_of = dict(ranked)
    out = [
        Paper(
            paper_id=p.paper_id, base_id=p.base_id, title=p.title, abstract=p.abstract,
            date=p.date, submitted_date=p.submitted_date, categories=p.categories,
            citation_count=p.citation_count,
            score=float(support_of.get(p.base_id, 1)),
            facets=p.facets, provenance=("snowball",),
        )
        for p in papers
    ]
    out.sort(key=lambda p: -p.score)

    stats.elapsed_s = time.perf_counter() - t0
    stats.note = (f"seeds={stats.n_seeds} back={stats.n_backward} fwd={stats.n_forward} "
                  f"-> new={stats.n_new}")
    if stats.n_meta_missing:
        log.warning("스노우볼링: 엣지로 나온 %d편의 메타데이터를 못 구했습니다",
                    stats.n_meta_missing)
    return out, stats
