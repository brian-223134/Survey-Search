"""P3.3 — CLI. 토픽 → JSON 덤프 + stats 리포트.

    python -m survey_search.cli --topic "Retrieval-Augmented Generation for LLMs"
    python -m survey_search.cli --topics-file topics.txt --out results/ --all
    python -m survey_search.cli --topic "..." --ablate            # 설정별 비교표

`--ablate` 는 P4.4(베이스라인 대조)를 CLI 에서 바로 돌리는 경로입니다. 같은 백엔드를
재사용하므로 인덱스를 한 번만 읽습니다(cold 12초 × N 을 피합니다).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

from survey_search.core.facets import load_dotenv
from survey_search.types import SearchConfig

log = logging.getLogger(__name__)

#: `--ablate` 가 도는 설정들. 이름이 곧 리포트의 행 레이블입니다.
ABLATIONS: dict[str, dict] = {
    "dense-only":  dict(lexical=False),
    "+bm25":       dict(),
    "+freshness":  dict(freshness=True),
    "+diversity":  dict(freshness=True, diversity=True, mmr_lambda=0.3),
    "+facets":     dict(facets=True, freshness=True),
    "all-on":      dict(facets=True, freshness=True, diversity=True, mmr_lambda=0.3),
}


def result_to_dict(result) -> dict:
    return {
        "topic": result.topic,
        "n_papers": len(result.papers),
        "stats": {
            **{k: v for k, v in asdict(result.stats).items() if k != "stages"},
            "stages": [asdict(s) for s in result.stats.stages],
        },
        "facets": [
            {"name": f.name, "queries": list(f.queries), "n_papers": len(f.paper_ids)}
            for f in result.facets
        ],
        "papers": [
            {
                "paper_id": p.paper_id,
                "base_id": p.base_id,
                "title": p.title,
                "date": p.date,
                "categories": list(p.categories),
                "citation_count": p.citation_count,
                "score": p.score,
                "facets": list(p.facets),
                "provenance": list(p.provenance),
            }
            for p in result.papers
        ],
    }


def _build_config(args) -> SearchConfig:
    return SearchConfig(
        n_papers=args.n_papers,
        facets=args.facets,
        lexical=not args.no_lexical,
        freshness=args.freshness,
        diversity=args.diversity,
        mmr_lambda=args.mmr_lambda,
        date_min=args.date_min,
        date_max=args.date_max,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--topic", help="검색할 토픽 하나")
    src.add_argument("--topics-file", type=Path, help="한 줄에 토픽 하나씩")

    ap.add_argument("--out", type=Path, help="결과 JSON 을 쓸 디렉터리 (없으면 stdout 요약만)")
    ap.add_argument("--n-papers", type=int, default=1500)
    ap.add_argument("--facets", action="store_true", help="S1 LLM facet 분해 (OpenRouter)")
    ap.add_argument("--no-lexical", action="store_true", help="S3 BM25 끄기")
    ap.add_argument("--freshness", action="store_true", help="S6 freshness 랭킹")
    ap.add_argument("--diversity", action="store_true", help="S7 MMR")
    ap.add_argument("--mmr-lambda", type=float, default=0.3)
    ap.add_argument("--all", action="store_true", help="S1·S3·S6·S7 전부 켜기")
    ap.add_argument("--ablate", action="store_true", help="설정별 비교표 (P4.4)")
    ap.add_argument("--date-min"), ap.add_argument("--date-max")
    ap.add_argument("--env", type=Path, default=Path(".env"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    n_env = load_dotenv(args.env)
    if n_env:
        log.info("%s 에서 환경변수 %d개 로드", args.env, n_env)

    if args.all:
        args.facets = args.freshness = args.diversity = True

    topics = (
        [args.topic]
        if args.topic
        else [t.strip() for t in args.topics_file.read_text().splitlines() if t.strip()]
    )

    # 백엔드는 한 번만 만듭니다 — 인덱스 cold 로드가 토픽·설정마다 12초씩 붙는 것을 피합니다.
    from survey_search.backends.faiss_duckdb import FaissDuckDBBackend
    from survey_search.search import search_topic

    backend = FaissDuckDBBackend()

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    if args.ablate:
        return _run_ablation(topics, backend, search_topic, args)

    base_cfg = _build_config(args)
    for topic in topics:
        result = search_topic(topic, backend=backend, config=base_cfg)
        print(f"\n{'=' * 72}\n{result.stats.report()}")
        if args.out:
            path = args.out / f"{_slug(topic)}.json"
            path.write_text(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2))
            print(f"-> {path}")
    return 0


def _run_ablation(topics, backend, search_topic, args) -> int:
    rows = []
    for topic in topics:
        base_ids: set[str] | None = None
        print(f"\n{'=' * 92}\n토픽: {topic}\n{'=' * 92}")
        print(f"{'설정':14}{'쿼리':>5}{'후보':>9}{'6m':>7}{'12m':>7}{'24m':>7}"
              f"{'인용중앙':>9}{'신규':>8}{'초':>7}")
        print("-" * 92)
        for name, overrides in ABLATIONS.items():
            cfg = replace(SearchConfig(n_papers=args.n_papers), **overrides)
            r = search_topic(topic, backend=backend, config=cfg)
            ids = set(r.ids())
            if base_ids is None:
                base_ids = ids
            s = r.stats
            ccs = sorted(p.citation_count for p in r.papers if p.citation_count is not None)
            med = ccs[len(ccs) // 2] if ccs else 0
            dedup_stage = s.stage("S5 dedup")
            print(f"{name:14}{s.n_queries:>5}{(dedup_stage.out_n if dedup_stage else 0):>9,}"
                  f"{s.recent_6m_ratio:>6.1%}{s.recent_12m_ratio:>6.1%}{s.recent_24m_ratio:>6.1%}"
                  f"{med:>9,}{len(ids - base_ids):>8,}{s.total_s:>7.1f}")
            rows.append({"topic": topic, "config": name,
                         "recent_12m": s.recent_12m_ratio,
                         "median_citation": med,
                         "new_vs_baseline": len(ids - base_ids),
                         "n_queries": s.n_queries, "total_s": s.total_s})
            if args.out:
                (args.out / f"{_slug(topic)}__{name}.json").write_text(
                    json.dumps(result_to_dict(r), ensure_ascii=False, indent=2))
    if args.out:
        (args.out / "ablation_summary.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"\n-> {args.out / 'ablation_summary.json'}")
    return 0


def _slug(text: str, max_len: int = 60) -> str:
    keep = [c if c.isalnum() else "-" for c in text.lower()]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:max_len]


if __name__ == "__main__":
    sys.exit(main())
