"""P1.6 — 오케스트레이터. DESIGN §4·§5.

    topic -> S2 dense -> S3 lexical -> S4 RRF -> S5 dedup -> (S6 freshness) -> (S7 diversity)

`SearchConfig` 의 불리언이 곧 ablation 축입니다. 전부 끄면 "토픽 1쿼리 → dense top-k" 가
되어 AutoSurvey 베이스라인과 (임베딩 모델 차이를 빼면) 같아집니다. 그게 비교의 원점입니다.

**모든 단계는 자기가 버린 건수를 `stats` 에 남깁니다.** 껐거나 백엔드가 지원하지 않아
건너뛴 단계도 `skipped=True` 로 남습니다 — 무음 스킵 금지.
"""

from __future__ import annotations

import logging
import time
from datetime import date as _date

from survey_search.backends.base import Backend
from survey_search.core.dedup import dedup
from survey_search.core.fuse import provenance_of, rrf
from survey_search.types import Facet, Paper, SearchConfig, SearchResult, SearchStats, StageStat

log = logging.getLogger(__name__)


def _elapsed(t0: float) -> float:
    return time.perf_counter() - t0


def search_topic(
    topic: str,
    *,
    backend: Backend,
    config: SearchConfig | None = None,
    today: _date | None = None,
) -> SearchResult:
    cfg = config or SearchConfig()
    today = today or _date.today()
    stats = SearchStats(topic=topic, backend=getattr(backend, "name", type(backend).__name__))
    t_all = time.perf_counter()

    # --- S1 facet 분해 ------------------------------------------------------
    # P2 에서 구현합니다. 끈 상태의 기본 동작은 "토픽 문자열이 유일한 쿼리".
    if cfg.facets:
        raise NotImplementedError("S1 facet 분해는 P2.1 — 아직 구현되지 않았습니다")
    queries = [topic]
    facet_names = {topic: "(topic)"}
    stats.add(StageStat("S1 facets", 1, 1, skipped=True, note="disabled -> 토픽 1쿼리"))
    stats.n_queries = len(queries)

    # --- 사전 필터 (날짜·카테고리) --------------------------------------------
    t = time.perf_counter()
    allowed = backend.filter_ids(
        date_min=cfg.date_min, date_max=cfg.date_max, categories=cfg.categories
    )
    if allowed is None:
        stats.add(StageStat("filter", 0, 0, skipped=True, note="조건 없음 -> 전체 대상",
                            elapsed_s=_elapsed(t)))
    else:
        stats.add(StageStat("filter", 0, len(allowed), elapsed_s=_elapsed(t),
                            note=f"date=[{cfg.date_min},{cfg.date_max}] cat={cfg.categories}"))

    # --- S2 dense -----------------------------------------------------------
    t = time.perf_counter()
    dense_lists = backend.dense_search(queries, cfg.dense_top_k, field=cfg.dense_field)
    n_dense = sum(len(l) for l in dense_lists)
    stats.n_dense_hits = n_dense
    stats.add(StageStat("S2 dense", len(queries), n_dense, elapsed_s=_elapsed(t),
                        note=f"field={cfg.dense_field} top_k={cfg.dense_top_k}"))

    # --- S3 lexical ---------------------------------------------------------
    if cfg.lexical:
        t = time.perf_counter()
        lex_lists = backend.lexical_search(queries, cfg.lexical_top_k)
        n_lex = sum(len(l) for l in lex_lists)
        stats.n_lexical_hits = n_lex
        stats.add(StageStat("S3 lexical", len(queries), n_lex, elapsed_s=_elapsed(t),
                            note=f"BM25 top_k={cfg.lexical_top_k}"))
    else:
        lex_lists = [[] for _ in queries]
        stats.add(StageStat("S3 lexical", len(queries), 0, skipped=True, note="disabled"))

    # --- S4 RRF -------------------------------------------------------------
    t = time.perf_counter()
    all_lists = [l for l in dense_lists if l] + [l for l in lex_lists if l]
    fused = rrf(all_lists, k=cfg.rrf_k)
    stats.add(StageStat("S4 rrf", sum(len(l) for l in all_lists), len(fused),
                        elapsed_s=_elapsed(t), note=f"k={cfg.rrf_k}, {len(all_lists)} lists"))

    # 날짜·카테고리 필터를 융합 뒤에 적용합니다 — 그래야 "필터가 몇 편 버렸나"를
    # 검색 품질과 분리해서 셀 수 있습니다.
    if allowed is not None:
        before = len(fused)
        fused = [(pid, s) for pid, s in fused if pid in allowed]
        stats.add(StageStat("filter-apply", before, len(fused), dropped=before - len(fused),
                            reason="date/category 범위 밖"))

    # --- S5 dedup -----------------------------------------------------------
    t = time.perf_counter()
    # 제목 병합을 하려면 제목이 필요합니다. 후보 전체를 조회하면 비싸므로
    # 최종 목록의 3배까지만 가져와 병합하고, 그 사실을 stats 에 남깁니다.
    title_window = min(len(fused), max(cfg.n_papers * 3, 1000))
    head = fused[:title_window]
    head_papers = backend.get_papers([pid for pid, _ in head])
    titles = {p.paper_id: p.title for p in head_papers}
    n_meta_missing = len(head) - len(head_papers)

    deduped_head, dropped, merged = dedup(head, titles=titles)
    deduped = deduped_head + fused[title_window:]
    stats.add(StageStat(
        "S5 dedup", len(fused), len(deduped),
        dropped=dropped["version"] + dropped["title"],
        reason=f"version={dropped['version']}, title={dropped['title']}",
        elapsed_s=_elapsed(t),
        note=f"제목 병합은 상위 {title_window:,}편에만 적용",
    ))
    if n_meta_missing:
        stats.warn(f"메타데이터를 못 찾은 id {n_meta_missing:,}건 — 인덱스와 DuckDB 불일치 가능")
    if len(fused) > title_window:
        stats.warn(
            f"{len(fused) - title_window:,}편은 제목 병합 대상에서 제외됨 (윈도우 {title_window:,})"
        )

    # --- 메타데이터 조회 (S6 부터는 논문 객체가 필요합니다) -----------------------
    # freshness 는 인용수·날짜를 봐야 하므로, 컷 전에 후보 메타를 가져옵니다.
    # 후보 전체가 아니라 최종의 2배까지만 — 그 이하는 어차피 컷됩니다.
    t = time.perf_counter()
    rank_window = min(len(deduped), max(cfg.n_papers * 2, 2000))
    window = deduped[:rank_window]
    score_of = dict(window)
    fetched = backend.get_papers([pid for pid, _ in window])
    if len(fetched) != len(window):
        stats.warn(f"후보 {len(window):,}편 중 {len(window) - len(fetched):,}편의 메타데이터 없음")
    if len(deduped) > rank_window:
        stats.warn(f"{len(deduped) - rank_window:,}편은 랭킹 대상에서 제외됨 (윈도우 {rank_window:,})")

    dense_ids = {pid for l in dense_lists for pid, _ in l}
    lex_ids = {pid for l in lex_lists for pid, _ in l}
    candidates = [
        Paper(
            paper_id=p.paper_id,
            base_id=p.base_id,
            title=p.title,
            abstract=p.abstract,
            date=p.date,
            categories=p.categories,
            citation_count=p.citation_count,
            score=score_of.get(p.paper_id, 0.0),
            facets=(facet_names.get(topic, "(topic)"),),
            provenance=provenance_of(p.paper_id, {"dense": dense_ids, "bm25": lex_ids}),
        )
        for p in fetched
    ]
    stats.add(StageStat("meta", len(window), len(candidates), elapsed_s=_elapsed(t)))

    # --- S6 freshness -------------------------------------------------------
    if cfg.freshness:
        from survey_search.core.rank import FreshnessConfig, rerank

        t = time.perf_counter()
        fcfg = cfg.freshness_config or FreshnessConfig()
        candidates, rank_stats = rerank(candidates, config=fcfg, today=today)
        stats.add(StageStat("S6 freshness", len(candidates), len(candidates),
                            elapsed_s=_elapsed(t),
                            note=f"{rank_stats.note} cohorts={rank_stats.n_cohorts} "
                                 f"promoted={rank_stats.quota_promoted}"))
        if rank_stats.n_missing_citation:
            stats.warn(
                f"인용수 없는 논문 {rank_stats.n_missing_citation:,}편 — 인용 항이 중립(0.5)으로 처리됨"
            )
        if rank_stats.n_missing_date:
            stats.warn(f"날짜 없는 논문 {rank_stats.n_missing_date:,}편 — recency 항이 0으로 처리됨")
    else:
        stats.add(StageStat("S6 freshness", len(candidates), len(candidates), skipped=True,
                            note="disabled -> RRF 점수 그대로"))

    # --- S7 diversity -------------------------------------------------------
    if cfg.diversity:
        raise NotImplementedError("S7 다양성은 P2.4 — 아직 구현되지 않았습니다")
    stats.add(StageStat("S7 diversity", len(candidates), len(candidates), skipped=True,
                        note="disabled -> 점수순 상위 N"))

    # --- 최종 컷 ------------------------------------------------------------
    if len(candidates) > cfg.n_papers:
        stats.add(StageStat("cut", len(candidates), cfg.n_papers,
                            dropped=len(candidates) - cfg.n_papers,
                            reason=f"n_papers={cfg.n_papers}"))
    papers = tuple(candidates[: cfg.n_papers])

    _fill_recency(stats, papers, today)
    _fill_provenance(stats, papers)
    stats.n_final = len(papers)
    stats.total_s = _elapsed(t_all)

    return SearchResult(topic=topic, papers=papers, facets=(Facet(name="(topic)",
                        queries=tuple(queries), paper_ids=tuple(p.paper_id for p in papers)),),
                        stats=stats)


def _fill_provenance(stats: SearchStats, papers: tuple[Paper, ...]) -> None:
    """경로별 기여 집계. `bm25_only` 가 0에 가까우면 S3 는 값을 못 내고 있는 것입니다."""
    for p in papers:
        prov = set(p.provenance)
        if prov == {"dense"}:
            stats.n_dense_only += 1
        elif prov == {"bm25"}:
            stats.n_bm25_only += 1
        elif {"dense", "bm25"} <= prov:
            stats.n_both += 1


def _fill_recency(stats: SearchStats, papers: tuple[Paper, ...], today: _date) -> None:
    """최근 6/12/24개월 비율 — 이 프로젝트의 주 관심 지표."""
    if not papers:
        return
    ages = [(p, p.months_since(today)) for p in papers]
    known = [a for _, a in ages if a is not None]
    stats.n_missing_date = len(ages) - len(known)
    stats.n_missing_citation = sum(1 for p in papers if p.citation_count is None)

    n = len(papers)
    stats.recent_6m_ratio = sum(1 for a in known if a <= 6) / n
    stats.recent_12m_ratio = sum(1 for a in known if a <= 12) / n
    stats.recent_24m_ratio = sum(1 for a in known if a <= 24) / n

    dates = sorted(p.date for p in papers if p.date)
    if dates:
        stats.date_min, stats.date_max = dates[0], dates[-1]
    if stats.n_missing_date:
        stats.warn(f"날짜 없는 논문 {stats.n_missing_date:,}편 — 최신성 비율의 분모에는 포함됨")
