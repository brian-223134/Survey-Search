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
from survey_search.core.fuse import as_id_set, provenance_of, rrf
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
    if cfg.facets:
        from survey_search.core.facets import decompose

        t = time.perf_counter()
        facet_list, fstats = decompose(topic, config=cfg.facet_config)
        stats.add(StageStat("S1 facets", 1, fstats.n_facets, elapsed_s=_elapsed(t),
                            note=f"source={fstats.source} model={fstats.model} "
                                 f"queries={fstats.n_queries} llm_calls={fstats.llm_calls}"))
        for w in fstats.warnings:
            stats.warn(w)
    else:
        facet_list = [Facet(name="(topic)", queries=(topic,))]
        stats.add(StageStat("S1 facets", 1, 1, skipped=True, note="disabled -> 토픽 1쿼리"))

    # 쿼리를 평평하게 펴되, 어느 facet 소속인지 유지합니다 — facet 쿼터(S7)가
    # 걸리려면 이 정보가 끝까지 살아 있어야 합니다.
    queries: list[str] = []
    query_facet: list[str] = []
    for f in facet_list:
        for q in f.queries:
            queries.append(q)
            query_facet.append(f.name)
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

    # --- S4 RRF (2단) --------------------------------------------------------
    # ① facet 내부에서 dense+BM25 융합 → ② facet 간 융합.
    # 나누는 이유는 facet 소속 정보를 살려 S7 의 facet 쿼터를 걸기 위해서입니다.
    # 한 번에 다 섞으면 쿼리를 많이 가진 facet 이 결과를 지배하기도 합니다.
    t = time.perf_counter()
    per_facet: dict[str, list[tuple[str, float]]] = {}
    for i, fname in enumerate(query_facet):
        lists = [l for l in (dense_lists[i], lex_lists[i]) if l]
        if not lists:
            continue
        inner = rrf(lists, k=cfg.rrf_k)
        prev = per_facet.get(fname)
        per_facet[fname] = rrf([prev, inner], k=cfg.rrf_k) if prev else inner

    facet_ids = {name: [pid for pid, _ in lst] for name, lst in per_facet.items()}
    fused = rrf(list(per_facet.values()), k=cfg.rrf_k)
    n_in = sum(len(l) for l in dense_lists) + sum(len(l) for l in lex_lists)
    stats.add(StageStat("S4 rrf", n_in, len(fused), elapsed_s=_elapsed(t),
                        note=f"k={cfg.rrf_k}, 2단 ({len(queries)} 쿼리 -> "
                             f"{len(per_facet)} facet -> 1)"))

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
    title_window = len(fused) if cfg.title_window is None else min(len(fused), cfg.title_window)
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
    if n_meta_missing > 0:
        stats.warn(f"메타데이터를 못 찾은 id {n_meta_missing:,}건 — 인덱스와 DuckDB 불일치 가능")
    if len(fused) > title_window:
        stats.warn(
            f"{len(fused) - title_window:,}편은 제목 병합 대상에서 제외됨 (윈도우 {title_window:,})"
        )

    # --- 메타데이터 조회 (S6 부터는 논문 객체가 필요합니다) -----------------------
    # freshness 는 인용수·날짜를 봐야 하므로, 컷 전에 후보 메타를 가져옵니다.
    # 후보 전체가 아니라 최종의 2배까지만 — 그 이하는 어차피 컷됩니다.
    t = time.perf_counter()
    rank_window = len(deduped) if cfg.rank_window is None else min(len(deduped), cfg.rank_window)
    window = deduped[:rank_window]
    score_of = dict(window)
    fetched = backend.get_papers([pid for pid, _ in window])
    # 하이브리드 백엔드는 로컬에 없는 논문을 온라인에서 채우므로 **요청보다 많이**
    # 돌아올 수 있습니다. 그 경우는 결손이 아니라 보강입니다 — 구분해서 남깁니다.
    delta = len(fetched) - len(window)
    if delta < 0:
        stats.warn(f"후보 {len(window):,}편 중 {-delta:,}편의 메타데이터 없음")
    elif delta > 0:
        stats.warn(f"로컬에 없는 논문 {delta:,}편을 온라인에서 보강했습니다")
    if len(deduped) > rank_window:
        stats.warn(f"{len(deduped) - rank_window:,}편은 랭킹 대상에서 제외됨 (윈도우 {rank_window:,})")

    # 집합은 여기서 한 번만 만듭니다 — provenance_of 안에서 매번 만들면
    # 후보 48,000편 × 원본 72,000개에서 5분이 걸립니다.
    dense_ids = as_id_set([h for l in dense_lists for h in l])
    lex_ids = as_id_set([h for l in lex_lists for h in l])

    # 논문 → 그 논문을 끌어올린 facet 들. S7 의 facet 쿼터가 이걸 봅니다.
    # **순위 순으로 정렬합니다** — 첫 원소가 "그 논문을 가장 높게 본 facet" 이어야
    # S8b 의 facet 쿼리 재랭킹이 대표 쿼리를 옳게 고릅니다. 삽입 순서로 두면
    # facet 사전 순서라는 무의미한 기준으로 쿼리가 정해집니다.
    facet_rank: dict[str, list[tuple[int, str]]] = {}
    for fname, ids in facet_ids.items():
        for rank, pid in enumerate(ids):
            facet_rank.setdefault(pid, []).append((rank, fname))
    facet_of = {
        pid: [f for _, f in sorted(pairs)] for pid, pairs in facet_rank.items()
    }

    candidates = [
        Paper(
            paper_id=p.paper_id,
            base_id=p.base_id,
            title=p.title,
            abstract=p.abstract,
            date=p.date,
            submitted_date=p.submitted_date,
            categories=p.categories,
            citation_count=p.citation_count,
            score=score_of.get(p.paper_id, 0.0),
            facets=tuple(facet_of.get(p.paper_id, ("(none)",))),
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

    # --- S8 스노우볼링 --------------------------------------------------------
    # freshness 뒤, diversity 앞에 둡니다: 랭킹된 상위권을 시드로 삼아야 좋은 이웃이
    # 나오고, 유입된 논문도 다양성 선택의 후보가 되어야 하기 때문입니다.
    if cfg.snowball:
        from survey_search.core.expand import snowball as run_snowball

        t = time.perf_counter()
        added, sstats = run_snowball(candidates, backend=backend,
                                     config=cfg.snowball_config)
        if not sstats.supported:
            stats.add(StageStat("S8 snowball", len(candidates), len(candidates),
                                skipped=True, note=sstats.note))
            stats.warn("백엔드가 인용 엣지를 몰라 스노우볼링을 건너뛰었습니다 "
                       "(OnlineBackend/HybridBackend 가 필요합니다)")
        elif added:
            # 유입 논문의 점수(시드 지지도)는 RRF 점수와 스케일이 아예 다릅니다.
            # 그래서 S4 와 같은 이유로 **순위 기반 RRF** 로 융합합니다. 점수를 임의로
            # 깎아 바닥에 깔면 유입 논문이 최종 컷에 영원히 못 들어옵니다(실제로 겪었습니다).
            before = len(candidates)
            existing_rank = [p.paper_id for p in candidates]
            snow_rank = [p.paper_id for p in added]
            fused_scores = dict(rrf([existing_rank, snow_rank], k=cfg.rrf_k))

            pool = {p.paper_id: p for p in candidates}
            pool.update({p.paper_id: p for p in added if p.paper_id not in pool})
            candidates = sorted(
                (Paper(**{**p.__dict__, "score": fused_scores.get(pid, 0.0)})
                 for pid, p in pool.items()),
                key=lambda p: (-p.score, p.paper_id),
            )
            stats.add(StageStat("S8 snowball", before, len(candidates),
                                elapsed_s=_elapsed(t), note=sstats.note))
        else:
            stats.add(StageStat("S8 snowball", len(candidates), len(candidates),
                                elapsed_s=_elapsed(t),
                                note=f"{sstats.note} (유입 0편)"))
            if sstats.n_dropped_by_cap:
                stats.warn(f"스노우볼링 유입을 상한으로 잘라 {sstats.n_dropped_by_cap:,}편 제외")
            if sstats.n_meta_missing:
                stats.warn(f"스노우볼링: 엣지로 나온 {sstats.n_meta_missing:,}편의 "
                           f"메타데이터를 못 구해 제외")
            for e in sstats.errors[:3]:
                stats.warn(f"스노우볼링 오류: {e}")
    else:
        stats.add(StageStat("S8 snowball", len(candidates), len(candidates),
                            skipped=True, note="disabled"))

    # --- S8b cross-encoder 재랭킹 ---------------------------------------------
    # 다양성(S7) 앞에 둡니다: MMR 은 관련성 순서를 입력으로 받아 다양성과 저울질하므로,
    # 더 정확한 순서를 먼저 만들어 주는 편이 맞습니다.
    if cfg.rerank:
        from survey_search.core.rerank import CrossEncoderReranker

        t = time.perf_counter()
        reranker = cfg.reranker or CrossEncoderReranker(cfg.rerank_config)
        if cfg.reranker is None:
            stats.warn("재랭커를 매 검색마다 새로 만들고 있습니다 — 배치 실행에서는 "
                       "SearchConfig(reranker=...) 로 인스턴스를 재사용하세요")
        # **top_n <= n_papers 면 재랭킹이 최종 목록을 못 바꿉니다.** 상위 top_n 안에서만
        # 순서가 바뀌는데 컷이 그보다 아래에 있으면, 재랭킹 전후의 최종 집합이 같습니다
        # (순서만 다름). 끄고 켠 실험이 recall 로는 구분되지 않게 됩니다.
        eff_top_n = getattr(reranker.config, "top_n", 0)
        if eff_top_n <= cfg.n_papers:
            stats.warn(
                f"재랭킹 범위(top_n={eff_top_n:,})가 최종 편수(n_papers={cfg.n_papers:,})보다 "
                f"작거나 같습니다 — 최종 '집합'은 그대로이고 순서만 바뀝니다. "
                f"recall 을 올리려면 top_n 을 n_papers 보다 크게 잡으세요"
            )

        # facet 이름 → 대표 쿼리. facet 은 쿼리를 1~3개 갖는데 첫 번째가 LLM 이
        # 그 하위 주제를 가장 직접적으로 표현한 것입니다.
        facet_queries = {f.name: f.queries[0] for f in facet_list if f.queries}
        candidates, rstats = reranker.rerank(topic, candidates, facet_queries)
        stats.add(StageStat("S8b rerank", len(candidates), len(candidates),
                            skipped=not rstats.applied, elapsed_s=_elapsed(t),
                            note=f"{rstats.note} device={rstats.device}"))
        for e in rstats.errors:
            stats.warn(f"재랭킹 실패(원래 순서 유지): {e}")
        if rstats.applied and rstats.n_untouched:
            stats.warn(f"재랭킹은 상위 {rstats.n_scored:,}편에만 적용 — "
                       f"{rstats.n_untouched:,}편은 원래 순서 유지")
    else:
        stats.add(StageStat("S8b rerank", len(candidates), len(candidates),
                            skipped=True, note="disabled"))

    # --- S7 diversity -------------------------------------------------------
    if cfg.diversity:
        from survey_search.core.diversity import diversify

        t = time.perf_counter()
        # MMR 은 풀이 커질수록 다양성 쪽으로 쏠립니다(관련성을 풀 안에서 정규화하므로).
        # 그래서 S6 과 달리 풀을 묶습니다. 잘라낸 건수는 아래 stats 에 남습니다.
        pool_n = cfg.mmr_pool if cfg.mmr_pool is not None else max(cfg.n_papers * 2, 3000)
        pool = candidates[:pool_n]

        vectors = None
        getter = getattr(backend, "get_vectors", None)
        if getter is None:
            stats.warn("백엔드에 get_vectors 가 없어 MMR 을 건너뜁니다 (facet 쿼터만 적용)")
        else:
            vectors = getter([p.paper_id for p in pool], cfg.dense_field)
            if vectors is None:
                stats.warn("저장 벡터를 못 구해 MMR 을 건너뜁니다 (facet 쿼터만 적용)")

        selected, dstats = diversify(
            pool, n=cfg.n_papers, vectors=vectors,
            lambda_=cfg.mmr_lambda, min_per_facet=cfg.min_per_facet,
        )
        stats.add(StageStat("S7 diversity", len(pool), len(selected),
                            dropped=len(pool) - len(selected),
                            reason=f"n_papers={cfg.n_papers}",
                            elapsed_s=_elapsed(t),
                            note=f"pool={len(pool):,}/{len(candidates):,} "
                                 f"mmr={dstats.mmr_applied} facets={dstats.n_facets} "
                                 f"quota={dstats.facet_quota_applied} | {dstats.note}"))
        if len(candidates) > len(pool):
            stats.warn(
                f"MMR 풀을 {len(pool):,}편으로 제한 — {len(candidates) - len(pool):,}편은 "
                f"S7 대상에서 제외 (풀이 커지면 다양성이 관련성을 압도합니다)"
            )
        papers = tuple(selected)
    else:
        stats.add(StageStat("S7 diversity", len(candidates), len(candidates), skipped=True,
                            note="disabled -> 점수순 상위 N"))
        if len(candidates) > cfg.n_papers:
            stats.add(StageStat("cut", len(candidates), cfg.n_papers,
                                dropped=len(candidates) - cfg.n_papers,
                                reason=f"n_papers={cfg.n_papers}"))
        papers = tuple(candidates[: cfg.n_papers])

    _fill_recency(stats, papers, today)
    _fill_provenance(stats, papers)
    stats.n_final = len(papers)
    stats.total_s = _elapsed(t_all)

    # facet 별 최종 기여를 채웁니다 (P4.3 커버리지 지표의 입력).
    final_ids = {p.paper_id for p in papers}
    out_facets = tuple(
        Facet(
            name=f.name,
            queries=f.queries,
            paper_ids=tuple(pid for pid in facet_ids.get(f.name, []) if pid in final_ids),
        )
        for f in facet_list
    )
    empty = [f.name for f in out_facets if not f.paper_ids]
    if empty:
        stats.warn(f"최종 목록에 논문이 하나도 없는 facet {len(empty)}개: {empty[:5]}")

    return SearchResult(topic=topic, papers=papers, facets=out_facets, stats=stats)


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
