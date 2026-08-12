"""초록 기반 검색의 **천장**을 측정합니다.

랭킹 실패와 검색 실패를 구분하는 것이 목적입니다. 정답을 못 맞히는 이유는 둘 중 하나인데,
처방이 정반대입니다:

- **랭킹 실패** — 후보 풀에는 들어왔는데 순위가 낮아 컷에 못 듦
  → 재랭킹·융합·쿼리 확장으로 고칩니다. 본문은 도움이 안 됩니다
- **검색 실패** — 후보 풀에 아예 안 들어옴
  → 표현(임베딩)이나 데이터가 부족한 것입니다. **본문 도입이 의미를 갖는 유일한 경우**

그래서 네 단계로 나눠 잽니다:

    ① 코퍼스 상한   정답 중 우리 코퍼스에 존재하는 비율        (데이터의 한계)
    ② 컷오프 상한   그중 날짜 컷오프를 통과하는 비율            (평가 설계의 한계)
    ③ 풀 recall     후보 풀(컷 이전)에 들어온 비율              (검색의 한계)
    ④ 최종 recall   최종 N편에 들어온 비율                      (랭킹의 한계)

②→③ 격차가 크면 검색 문제, ③→④ 격차가 크면 랭킹 문제입니다.

    python -m survey_search.eval.ceiling --limit 25
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from survey_search.core.dedup import strip_version
from survey_search.eval.surge import GoldTopic, load_gold

log = logging.getLogger(__name__)


@dataclass
class CeilingRow:
    topic: str
    n_gold: int                 # 정답 (우리 코퍼스에 연결된 것)
    n_in_corpus: int            # 그중 DuckDB 에 실제로 있는 것
    n_after_cutoff: int         # 그중 날짜 컷오프를 통과하는 것
    n_in_pool: int              # 후보 풀에 들어온 것
    n_in_final: int             # 최종 N편에 들어온 것
    pool_size: int
    final_size: int

    @property
    def corpus_rate(self) -> float:
        return self.n_in_corpus / self.n_gold if self.n_gold else 0.0

    @property
    def cutoff_rate(self) -> float:
        return self.n_after_cutoff / self.n_gold if self.n_gold else 0.0

    @property
    def pool_rate(self) -> float:
        return self.n_in_pool / self.n_gold if self.n_gold else 0.0

    @property
    def final_rate(self) -> float:
        return self.n_in_final / self.n_gold if self.n_gold else 0.0

    @property
    def retrieval_gap(self) -> float:
        """②→③ — 컷오프는 통과하는데 검색이 못 찾은 비율."""
        return self.cutoff_rate - self.pool_rate

    @property
    def ranking_gap(self) -> float:
        """③→④ — 찾긴 했는데 순위가 낮아 잘린 비율."""
        return self.pool_rate - self.final_rate


@dataclass
class CeilingReport:
    rows: list[CeilingRow] = field(default_factory=list)
    n_papers: int = 1500
    config_note: str = ""

    def _mean(self, attr: str) -> float:
        return sum(getattr(r, attr) for r in self.rows) / len(self.rows) if self.rows else 0.0

    def render(self) -> str:
        if not self.rows:
            return "(측정된 토픽 없음)"
        m = self._mean
        pool = sum(r.pool_size for r in self.rows) / len(self.rows)
        lines = [
            f"토픽 {len(self.rows)}개 · 최종 {self.n_papers:,}편 · 평균 후보 풀 {pool:,.0f}편",
            f"설정: {self.config_note}",
            "",
            f"  ① 코퍼스 상한   {m('corpus_rate'):>6.1%}   정답이 우리 DB 에 있는 비율",
            f"  ② 컷오프 상한   {m('cutoff_rate'):>6.1%}   + 날짜 컷오프 통과",
            f"  ③ 풀 recall     {m('pool_rate'):>6.1%}   + 검색이 후보로 데려옴",
            f"  ④ 최종 recall   {m('final_rate'):>6.1%}   + 최종 목록에 살아남음",
            "",
            f"  검색 손실 (②→③)  {m('retrieval_gap'):>6.1%}",
            f"  랭킹 손실 (③→④)  {m('ranking_gap'):>6.1%}",
        ]
        r_gap, k_gap = m("retrieval_gap"), m("ranking_gap")
        lines.append("")
        if r_gap > k_gap * 1.5:
            lines.append("  => 검색이 병목입니다. 표현(임베딩)·쿼리 확장·데이터를 늘려야 합니다.")
        elif k_gap > r_gap * 1.5:
            lines.append("  => 랭킹이 병목입니다. 후보에는 들어와 있으니 재랭킹·융합을 손보세요.")
            lines.append("     이 경우 본문 도입은 검색 recall 을 못 올립니다.")
        else:
            lines.append("  => 검색과 랭킹이 비슷하게 기여합니다.")
        return "\n".join(lines)


def measure(
    topics: list[GoldTopic],
    *,
    backend,
    config,
    n_papers: int = 1500,
    pool_size: int = 100_000,
    respect_cutoff: bool = True,
) -> CeilingReport:
    """풀 recall 은 `n_papers` 를 아주 크게 잡아 컷이 안 물리게 해서 잽니다.

    다양성(S7)은 끄고 잽니다 — MMR 은 관련성을 일부러 희생하므로 '검색이 데려왔는가'를
    묻는 이 측정에서는 잡음입니다.
    """
    import duckdb
    from dataclasses import replace as dc_replace

    from survey_search.assets import PAPERS_DUCKDB
    from survey_search.search import search_topic

    con = duckdb.connect(str(PAPERS_DUCKDB), read_only=True)
    corpus = {
        b: (sub or dt or "")
        for b, dt, sub in con.execute(
            "SELECT base_id, CAST(date AS VARCHAR), CAST(submitted_date AS VARCHAR) FROM papers"
        ).fetchall()
    }

    report = CeilingReport(n_papers=n_papers,
                           config_note=f"facets={config.facets} lexical={config.lexical} "
                                       f"freshness={config.freshness} diversity=False")

    for i, g in enumerate(topics, 1):
        gold = list(dict.fromkeys(g.gold_ids))
        in_corpus = [x for x in gold if x in corpus]
        after_cut = ([x for x in in_corpus if not g.date or corpus[x] <= g.date]
                     if respect_cutoff else in_corpus)

        base = dc_replace(config, diversity=False)
        if respect_cutoff and g.date:
            base = dc_replace(base, date_max=g.date)

        pool_cfg = dc_replace(base, n_papers=pool_size)
        pool = search_topic(g.topic, backend=backend, config=pool_cfg)
        pool_ids = {strip_version(p) for p in pool.ids()}

        final_ids = set(list(dict.fromkeys(strip_version(p) for p in pool.ids()))[:n_papers])

        gold_set = set(gold)
        report.rows.append(CeilingRow(
            topic=g.topic,
            n_gold=len(gold),
            n_in_corpus=len(in_corpus),
            n_after_cutoff=len(after_cut),
            n_in_pool=len(gold_set & pool_ids),
            n_in_final=len(gold_set & final_ids),
            pool_size=len(pool_ids),
            final_size=len(final_ids),
        ))
        if i % 5 == 0 or i == len(topics):
            log.info("%d/%d 토픽", i, len(topics))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--n-papers", type=int, default=1500)
    ap.add_argument("--no-facets", action="store_true")
    ap.add_argument("--no-cutoff", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from survey_search.backends.faiss_duckdb import FaissDuckDBBackend
    from survey_search.core.facets import load_dotenv
    from survey_search.types import SearchConfig

    load_dotenv(".env")
    topics = load_gold()[: args.limit]
    cfg = SearchConfig(facets=not args.no_facets, lexical=True, freshness=True)

    report = measure(topics, backend=FaissDuckDBBackend(), config=cfg,
                     n_papers=args.n_papers, respect_cutoff=not args.no_cutoff)
    print("\n" + report.render())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"n_papers": report.n_papers, "config": report.config_note,
             "rows": [asdict(r) for r in report.rows]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
