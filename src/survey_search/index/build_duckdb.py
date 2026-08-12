"""P0.4 / P0.5 — TinyDB JSON → DuckDB `papers` 테이블 + BM25(FTS) 인덱스.

DuckDB FTS 사용법(`PRAGMA create_fts_index` → `fts_main_<table>.match_bm25`)은
SimScholarSearch의 `env/etl/build_fts.py` · `env/reader.py` 에서 가져왔습니다
(Apache-2.0, trillion-labs/scholar-search-rl). 스키마는 공유하지 않습니다 —
그쪽은 S2ORC 정수 `corpus_id` + 본문 전체이고, 이쪽은 arXiv 문자열 id + 초록뿐입니다.

**실측으로 확인된 함정 두 가지가 여기서 처리됩니다 (SETTING.md §6):**

- `citation_count` 가 `'17'` 같은 **문자열**입니다. BIGINT 캐스팅을 안 하면
  `'9' > '175503'` 이 되어 랭킹이 조용히 뒤집힙니다.
- `faiss_id` 는 **1-based**입니다. 0-based로 넣으면 한 칸 밀린 논문이 반환됩니다.

무음 폐기 금지 원칙에 따라, 캐스팅 실패·매핑 누락 건수는 세어서 보고합니다.

사용:

    python -m survey_search.index.build_duckdb           # data/papers.duckdb 생성
    python -m survey_search.index.build_duckdb --limit 5000   # 빠른 스모크
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from survey_search.assets import PAPERS_DUCKDB, SURVEYFORGE

log = logging.getLogger(__name__)

_VERSION_SUFFIX = re.compile(r"v\d+$")

SCHEMA = """
CREATE TABLE papers (
    paper_id        VARCHAR PRIMARY KEY,   -- 백엔드 native id, 버전 포함 ("2401.12345v2")
    base_id         VARCHAR NOT NULL,      -- 버전 제거 — 교차 코퍼스 정합 키
    faiss_id        BIGINT NOT NULL,       -- 1-based. FAISS search() 가 돌려주는 값
    title           VARCHAR,
    abstract        VARCHAR,
    date            DATE,
    categories      VARCHAR,               -- 원본이 공백 구분 문자열 ("cs.IT cs.LG")
    primary_category VARCHAR,
    authors         VARCHAR,
    url             VARCHAR,
    citation_count  BIGINT                 -- 원본은 문자열. 여기서 정수로 고정
)
"""


def strip_version(paper_id: str) -> str:
    """`2401.12345v2` -> `2401.12345`. 버전 접미사가 없으면 그대로."""
    return _VERSION_SUFFIX.sub("", paper_id)


@dataclass
class BuildStats:
    """무음 폐기 금지 — 버린 건 전부 여기 남습니다."""

    tinydb_rows: int = 0
    id_map_entries: int = 0
    inserted: int = 0
    dropped_no_faiss_id: int = 0     # id_map 에 없는 논문
    dropped_duplicate_id: int = 0    # paper_id 중복
    citation_unparseable: int = 0    # 숫자로 못 읽어 0 처리한 건
    date_unparseable: int = 0
    authors_coerced: int = 0         # authors 는 원래 리스트 — 전건 해당이 정상
    text_coerced_unexpected: int = 0  # title/abs/url/cat 이 문자열이 아니었던 건 (0이어야 정상)
    parse_s: float = 0.0
    insert_s: float = 0.0
    fts_s: float = 0.0

    def log(self) -> None:
        for k, v in asdict(self).items():
            log.info("  %-24s %s", k, f"{v:,}" if isinstance(v, int) else f"{v:.1f}")


def _to_int(value) -> tuple[int, bool]:
    """('17', True) 같은 문자열 인용수를 정수로. 실패하면 (0, False)."""
    if value is None:
        return 0, False
    try:
        return int(str(value).strip()), True
    except (ValueError, TypeError):
        return 0, False


def _to_text(value) -> tuple[str | None, bool]:
    """텍스트 컬럼 값을 문자열로 통일. 두 번째 값은 "원래 문자열이었나".

    `authors` 는 레코드마다 문자열(`"['A', 'B']"`)이거나 진짜 리스트(`['A','B']`)입니다.
    원본 수집 경로가 섞여 있어서 그렇습니다. 조용히 바꾸지 않고 건수를 셉니다.
    """
    if value is None:
        return None, True
    if isinstance(value, str):
        return value, True
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False), False
    return str(value), False


def _to_date(value) -> tuple[str | None, bool]:
    text = str(value or "").strip()[:10]
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return text, True
    return None, False


def build(
    *,
    tinydb_path: Path,
    id_map_path: Path,
    out_path: Path,
    limit: int | None = None,
    threads: int = 16,
) -> BuildStats:
    stats = BuildStats()

    t = time.perf_counter()
    with open(id_map_path) as f:
        id_to_faiss = {k: int(v) for k, v in json.load(f).items()}
    stats.id_map_entries = len(id_to_faiss)
    log.info("id_map: %s entries", f"{len(id_to_faiss):,}")

    with open(tinydb_path) as f:
        table = json.load(f)["cs_paper_info"]
    stats.tinydb_rows = len(table)
    stats.parse_s = time.perf_counter() - t
    log.info("tinydb: %s rows in %.1fs", f"{len(table):,}", stats.parse_s)

    rows = []
    seen: set[str] = set()
    for rec in table.values():
        if limit is not None and len(rows) >= limit:
            break
        paper_id = rec["id"]
        if paper_id in seen:
            stats.dropped_duplicate_id += 1
            continue
        faiss_id = id_to_faiss.get(paper_id)
        if faiss_id is None:
            stats.dropped_no_faiss_id += 1
            continue
        seen.add(paper_id)

        cc, cc_ok = _to_int(rec.get("citation_count"))
        if not cc_ok:
            stats.citation_unparseable += 1
        date, date_ok = _to_date(rec.get("date"))
        if not date_ok:
            stats.date_unparseable += 1

        title, t_ok = _to_text(rec.get("title"))
        abstract, a_ok = _to_text(rec.get("abs"))
        authors, au_ok = _to_text(rec.get("authors"))
        url, u_ok = _to_text(rec.get("url"))
        cat_raw, c_ok = _to_text(rec.get("cat"))
        if not au_ok:
            stats.authors_coerced += 1
        if not (t_ok and a_ok and u_ok and c_ok):
            stats.text_coerced_unexpected += 1

        cats = (cat_raw or "").strip()
        rows.append((
            paper_id,
            strip_version(paper_id),
            faiss_id,
            title,
            abstract,
            date,
            cats,
            cats.split()[0] if cats else None,
            authors,
            url,
            cc,
        ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    wal = out_path.with_suffix(out_path.suffix + ".wal")
    if wal.exists():
        wal.unlink()

    t = time.perf_counter()
    con = duckdb.connect(str(out_path))
    con.execute(f"SET threads = {threads}")
    con.execute(SCHEMA)

    # 행 단위 executemany 는 908k행 × 긴 초록에서 3시간 넘게 걸립니다(실측).
    # Arrow 테이블로 넘기면 DuckDB 가 컬럼 단위로 받아 수십 초에 끝납니다.
    import pyarrow as pa

    cols = list(zip(*rows)) if rows else [()] * 11
    arrow = pa.table(
        {
            "paper_id": pa.array(cols[0], pa.string()),
            "base_id": pa.array(cols[1], pa.string()),
            "faiss_id": pa.array(cols[2], pa.int64()),
            "title": pa.array(cols[3], pa.string()),
            "abstract": pa.array(cols[4], pa.string()),
            "date": pa.array(cols[5], pa.string()).cast(pa.date32()),
            "categories": pa.array(cols[6], pa.string()),
            "primary_category": pa.array(cols[7], pa.string()),
            "authors": pa.array(cols[8], pa.string()),
            "url": pa.array(cols[9], pa.string()),
            "citation_count": pa.array(cols[10], pa.int64()),
        }
    )
    con.register("incoming", arrow)
    con.execute("INSERT INTO papers SELECT * FROM incoming")
    con.unregister("incoming")
    stats.inserted = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    stats.insert_s = time.perf_counter() - t
    log.info("inserted %s rows in %.1fs", f"{stats.inserted:,}", stats.insert_s)

    con.execute("CREATE INDEX papers_faiss_id_idx ON papers(faiss_id)")
    con.execute("CREATE INDEX papers_base_id_idx ON papers(base_id)")
    con.execute("CREATE INDEX papers_date_idx ON papers(date)")

    t = time.perf_counter()
    con.execute("INSTALL fts; LOAD fts;")
    # stemmer/stopwords 기본값(porter/english)을 그대로 씁니다. 논문 제목·초록은
    # 영어이고, 약어·모델명을 살리려면 stemming 을 약하게 두는 편이 유리합니다.
    con.execute("PRAGMA create_fts_index('papers', 'paper_id', 'title', 'abstract')")
    stats.fts_s = time.perf_counter() - t
    log.info("FTS index built in %.1fs", stats.fts_s)

    con.close()
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=SURVEYFORGE.tinydb)
    ap.add_argument("--id-map", type=Path, default=SURVEYFORGE.id_map)
    ap.add_argument("--out", type=Path, default=PAPERS_DUCKDB)
    ap.add_argument("--limit", type=int, help="스모크용 상한")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stats = build(
        tinydb_path=args.db,
        id_map_path=args.id_map,
        out_path=args.out,
        limit=args.limit,
        threads=args.threads,
    )
    log.info("build stats:")
    stats.log()

    dropped = stats.dropped_no_faiss_id + stats.dropped_duplicate_id
    if dropped and args.limit is None:
        log.warning("총 %s편이 적재되지 않았습니다 — 위 사유별 건수를 확인하세요", f"{dropped:,}")
    if stats.text_coerced_unexpected:
        log.warning(
            "title/abs/url/cat 이 문자열이 아닌 레코드 %s건 — 원본 스키마가 바뀌었는지 확인하세요",
            f"{stats.text_coerced_unexpected:,}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
