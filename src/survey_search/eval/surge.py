"""P5.5 — SurGE 정답 집합으로 검색을 평가합니다.

**정답이 어디서 오는가**: `../SurGE/data/surveys.json` 에 GT 서베이 205편이 있고,
각각 `survey_title`(토픽)과 `all_cites`(그 서베이가 실제로 인용한 논문 목록)를 가집니다.
서베이가 인용한 논문 집합이 곧 그 토픽의 정답입니다. SurGE README 가 언급하는
`data/queries.json` 은 배포본에 없지만, 없어도 이 경로로 평가가 됩니다.

**연결 방법**: SurGE 코퍼스에는 arXiv id 가 없고 제목만 있습니다. 그래서 **정규화 제목**으로
우리 코퍼스와 잇습니다. 실측 연결률 91.4% (인용 13,485편 중 12,324편).

**날짜 컷오프가 필수입니다.** GT 서베이는 2019~2023년입니다. 2020년 서베이는 2021년 논문을
인용할 수 없습니다. 컷오프 없이 평가하면 우리 파이프라인이 최신 논문을 밀어주는 만큼
**자동으로 깎입니다** — 가설을 검증하는 게 아니라 구조적으로 지는 실험이 됩니다.
그래서 서베이마다 `date_max=<게시일>` 을 겁니다.

컷오프를 걸어도 freshness 가설은 그대로 시험됩니다: 서베이는 당대 기준 최신 논문을 많이
인용하므로, 인용수 정렬보다 연령 정규화 랭킹이 더 잘 맞아야 합니다.

사용:

    python -m survey_search.eval.surge --build-gold      # 정답 캐시 생성 (1회, ~3분)
    python -m survey_search.eval.surge --limit 30        # 30개 토픽으로 ablation
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from survey_search.assets import DATA_DIR
from survey_search.core.dedup import normalize_title, strip_version
from survey_search.metrics.paper_set import calculate_ndcg, calculate_recall

log = logging.getLogger(__name__)

SURGE_DIR = Path("/data2/chanjoong/survey-agent/SurGE/data")
GOLD_CACHE = DATA_DIR / "surge_gold.json"

#: 정답 중앙값이 51편이라 1500 은 과합니다. 이 지점들에서 recall 을 잽니다.
CUTOFFS = (50, 100, 500, 1500)

#: 이보다 정답이 적은 서베이는 제외합니다. 3편짜리 토픽에서의 recall 은 잡음입니다.
MIN_GOLD = 10


@dataclass
class GoldTopic:
    survey_id: int
    topic: str
    year: int
    date: str                    # 서베이 게시일 — 검색의 date_max 가 됩니다
    gold_ids: list[str]          # 우리 코퍼스의 base_id
    n_cites_total: int           # 원본 all_cites 개수
    n_matched: int               # 그중 우리 코퍼스에 연결된 수

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_cites_total if self.n_cites_total else 0.0


@dataclass
class BuildGoldStats:
    """무음 폐기 금지 — 제외한 서베이는 사유별로 셉니다."""

    n_surveys: int = 0
    n_kept: int = 0
    dropped_no_cites: int = 0        # all_cites 가 비어 있음
    dropped_too_few_gold: int = 0    # 연결 후 정답이 MIN_GOLD 미만
    n_cites_total: int = 0
    n_cites_matched: int = 0
    dropped_survey_ids: list[int] = field(default_factory=list)

    def log(self) -> None:
        log.info("서베이 %d개 중 %d개 사용", self.n_surveys, self.n_kept)
        log.info("  제외: all_cites 없음 %d, 정답 %d편 미만 %d",
                 self.dropped_no_cites, MIN_GOLD, self.dropped_too_few_gold)
        rate = self.n_cites_matched / self.n_cites_total if self.n_cites_total else 0
        log.info("  인용 %s편 중 %s편 연결 (%.1f%%)",
                 f"{self.n_cites_total:,}", f"{self.n_cites_matched:,}", rate * 100)


def build_gold(
    *, surge_dir: Path = SURGE_DIR, out: Path = GOLD_CACHE, duckdb_path: Path | None = None
) -> tuple[list[GoldTopic], BuildGoldStats]:
    """SurGE 서베이 → 우리 코퍼스 id 기준 정답 집합. 결과를 캐시합니다.

    1.6GB 코퍼스를 파싱하므로 몇 분 걸립니다. 그래서 한 번 만들고 캐시를 씁니다.
    """
    import duckdb

    from survey_search.assets import PAPERS_DUCKDB

    stats = BuildGoldStats()

    t = time.perf_counter()
    surveys = json.loads((surge_dir / "surveys.json").read_text())
    corpus = json.loads((surge_dir / "corpus.json").read_text())
    log.info("SurGE 로드: 서베이 %d, 코퍼스 %s (%.0fs)",
             len(surveys), f"{len(corpus):,}", time.perf_counter() - t)

    doc_title = {c["doc_id"]: c.get("Title") or "" for c in corpus}
    del corpus

    con = duckdb.connect(str(duckdb_path or PAPERS_DUCKDB), read_only=True)
    rows = con.execute("SELECT base_id, title FROM papers WHERE title IS NOT NULL").fetchall()
    # 같은 정규화 제목이 여러 편일 수 있습니다(806개 그룹). 정답은 집합이므로
    # 하나만 남겨도 recall 계산에는 영향이 없습니다.
    ours = {normalize_title(t): b for b, t in rows}
    log.info("우리 코퍼스 정규화 제목 %s개", f"{len(ours):,}")

    out_topics: list[GoldTopic] = []
    stats.n_surveys = len(surveys)
    for s in surveys:
        cites = s.get("all_cites") or []
        if not cites:
            stats.dropped_no_cites += 1
            stats.dropped_survey_ids.append(s.get("survey_id", -1))
            continue

        gold, matched = [], 0
        for cid in cites:
            title = doc_title.get(cid)
            if not title:
                continue
            base = ours.get(normalize_title(title))
            if base:
                matched += 1
                gold.append(base)
        gold = list(dict.fromkeys(gold))

        stats.n_cites_total += len(cites)
        stats.n_cites_matched += matched

        if len(gold) < MIN_GOLD:
            stats.dropped_too_few_gold += 1
            stats.dropped_survey_ids.append(s.get("survey_id", -1))
            continue

        out_topics.append(GoldTopic(
            survey_id=s.get("survey_id", -1),
            topic=s["survey_title"].replace("\\\\", " ").strip(),
            year=int(s.get("year", 0)),
            date=str(s.get("date", ""))[:10],
            gold_ids=gold,
            n_cites_total=len(cites),
            n_matched=matched,
        ))

    stats.n_kept = len(out_topics)
    stats.log()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"stats": asdict(stats), "topics": [asdict(t) for t in out_topics]},
        ensure_ascii=False))
    log.info("정답 캐시 -> %s", out)
    return out_topics, stats


def load_gold(path: Path = GOLD_CACHE) -> list[GoldTopic]:
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 — --build-gold 를 먼저 실행하세요")
    data = json.loads(path.read_text())
    return [GoldTopic(**t) for t in data["topics"]]


# --- 채점 -------------------------------------------------------------------

@dataclass
class TopicScore:
    survey_id: int
    topic: str
    n_gold: int
    recall: dict[int, float]     # cutoff -> recall
    ndcg: float
    n_returned: int


def score_topic(predicted_ids: list[str], gold: GoldTopic) -> TopicScore:
    """예측(랭킹 순 paper_id) 대 정답. **base_id 기준**으로 맞춥니다.

    검색은 `2401.12345v2` 를 돌려주는데 정답은 버전 없는 id 입니다. 버전을 안 떼면
    전부 불일치가 나서 recall 이 0 이 됩니다.
    """
    pred = list(dict.fromkeys(strip_version(p) for p in predicted_ids))
    gold_set = list(dict.fromkeys(gold.gold_ids))
    return TopicScore(
        survey_id=gold.survey_id,
        topic=gold.topic,
        n_gold=len(gold_set),
        recall={k: calculate_recall(pred[:k], gold_set) for k in CUTOFFS},
        ndcg=calculate_ndcg(pred, gold_set),
        n_returned=len(pred),
    )


def aggregate(scores: list[TopicScore]) -> dict:
    if not scores:
        return {}
    n = len(scores)
    return {
        "n_topics": n,
        "recall": {k: sum(s.recall[k] for s in scores) / n for k in CUTOFFS},
        "ndcg": sum(s.ndcg for s in scores) / n,
        "mean_gold": sum(s.n_gold for s in scores) / n,
    }


# --- 실행 -------------------------------------------------------------------

#: 평가할 설정들. 이름이 리포트의 행 레이블입니다.
ABLATIONS: dict[str, dict] = {
    "dense-only":   dict(lexical=False),
    "+bm25":        dict(),
    "+freshness":   dict(freshness=True),
    "+diversity":   dict(freshness=True, diversity=True, mmr_lambda=0.3),
    "+facets":      dict(facets=True, freshness=True),
    "all-on":       dict(facets=True, freshness=True, diversity=True, mmr_lambda=0.3),
}


def run(
    topics: list[GoldTopic],
    *,
    backend,
    configs: dict[str, dict] | None = None,
    n_papers: int = 1500,
    respect_cutoff: bool = True,
) -> dict[str, list[TopicScore]]:
    from dataclasses import replace as dc_replace

    from survey_search.search import search_topic
    from survey_search.types import SearchConfig

    configs = configs or ABLATIONS
    out: dict[str, list[TopicScore]] = {name: [] for name in configs}

    for i, g in enumerate(topics, 1):
        for name, overrides in configs.items():
            cfg = dc_replace(SearchConfig(n_papers=n_papers), **overrides)
            if respect_cutoff and g.date:
                # 서베이가 못 봤을 논문을 우리가 찾아 오면 오답으로 세집니다.
                cfg = dc_replace(cfg, date_max=g.date)
            r = search_topic(g.topic, backend=backend, config=cfg)
            out[name].append(score_topic(r.ids(), g))
        if i % 5 == 0 or i == len(topics):
            log.info("%d/%d 토픽 완료", i, len(topics))
    return out


def render(results: dict[str, list[TopicScore]]) -> str:
    header = (f"{'설정':14}" + "".join(f"{'R@'+str(k):>9}" for k in CUTOFFS)
              + f"{'nDCG':>9}{'정답수':>8}")
    lines = [header, "-" * len(header)]
    for name, scores in results.items():
        a = aggregate(scores)
        if not a:
            continue
        lines.append(f"{name:14}"
                     + "".join(f"{a['recall'][k]:>8.1%} " for k in CUTOFFS)
                     + f"{a['ndcg']:>8.3f} {a['mean_gold']:>7.0f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-gold", action="store_true", help="정답 캐시 생성 (1회)")
    ap.add_argument("--limit", type=int, help="토픽 수 상한 (빠른 확인용)")
    ap.add_argument("--n-papers", type=int, default=1500)
    ap.add_argument("--no-cutoff", action="store_true",
                    help="날짜 컷오프 없이 평가 (불공정 — 비교용으로만)")
    ap.add_argument("--out", type=Path, help="결과 JSON 경로")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.build_gold:
        build_gold()
        return 0

    topics = load_gold()
    if args.limit:
        topics = topics[: args.limit]
    log.info("평가 토픽 %d개 (정답 중앙값 %d편)", len(topics),
             sorted(len(t.gold_ids) for t in topics)[len(topics) // 2])

    from survey_search.backends.faiss_duckdb import FaissDuckDBBackend
    from survey_search.core.facets import load_dotenv

    load_dotenv(".env")
    backend = FaissDuckDBBackend()

    results = run(topics, backend=backend, n_papers=args.n_papers,
                  respect_cutoff=not args.no_cutoff)
    print("\n" + render(results))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {name: {"aggregate": aggregate(s), "per_topic": [asdict(x) for x in s]}
             for name, s in results.items()}, ensure_ascii=False, indent=2))
        log.info("결과 -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
