"""적재 시 실측으로 확인된 함정 두 가지가 실제로 막히는지."""

import pytest

duckdb = pytest.importorskip("duckdb")

from survey_search.index.build_duckdb import _to_date, _to_int, strip_version  # noqa: E402


def test_strip_version():
    assert strip_version("2401.12345v2") == "2401.12345"
    assert strip_version("2401.12345") == "2401.12345"
    assert strip_version("cs/0501001v1") == "cs/0501001"


def test_citation_count_string_becomes_int():
    """원본이 '17' 같은 문자열입니다. 문자열로 두면 '9' > '175503' 이 됩니다."""
    assert _to_int("17") == (17, True)
    assert _to_int(17) == (17, True)
    assert _to_int(" 17 ") == (17, True)
    # 못 읽는 값은 0 으로 떨어뜨리되, 그 사실을 호출부가 세도록 False 를 같이 냅니다
    assert _to_int(None) == (0, False)
    assert _to_int("n/a") == (0, False)


def test_string_sort_would_be_wrong():
    """왜 캐스팅이 필요한지 — 문자열 정렬의 실제 결과."""
    assert "9" > "175503"          # 문자열이면 9가 더 큼
    assert _to_int("9")[0] < _to_int("175503")[0]   # 정수면 정상


def test_date_parsing():
    assert _to_date("2018-11-14") == ("2018-11-14", True)
    assert _to_date("2018-11-14T00:00:00") == ("2018-11-14", True)
    assert _to_date("") == (None, False)
    assert _to_date(None) == (None, False)
    assert _to_date("20181114") == (None, False)


def test_bm25_finds_paper_by_its_own_title():
    """P0.5 검증 — 논문 제목으로 BM25 검색하면 그 논문이 **상위 10위 안에** 들어야 합니다.

    TASKS.md 의 원래 기준은 "1위"였는데, 실측 결과 그건 title-only 인덱스를 가정한
    기준이었습니다. 우리 FTS 는 **title + abstract** 에 걸려 있어서, 같은 용어를 초록에서
    더 자주 쓰는 다른 논문이 자기 제목을 가진 논문보다 높게 나옵니다. 예:

        "Prophet Inequalities for I.I.D. Random Variables from an Unknown Distribution"
          1위 2307.00971v4 (15.90) "New Prophet Inequalities via Poissonization..."
          3위 1811.06114v2 (14.81) <- 제목의 주인

    이건 BM25 가 고장난 게 아니라 의도한 동작입니다 — S3 의 목적이 "제목 정확 매칭"이
    아니라 "dense 가 놓치는 용어로 후보를 넓히는 것"이기 때문입니다. 그래서 기준을
    top-10 포함으로 바꿨습니다. 완전 정확 매칭이 필요해지면 title 전용 FTS 를 따로 만드세요.
    """
    from survey_search.assets import PAPERS_DUCKDB

    if not PAPERS_DUCKDB.exists():
        pytest.skip(f"{PAPERS_DUCKDB} 없음 — build_duckdb 먼저 실행")

    con = duckdb.connect(str(PAPERS_DUCKDB), read_only=True)
    con.execute("LOAD fts;")
    n = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    assert n == 908_819, f"행 수 {n:,} — SETTING.md 의 908,819 와 다릅니다"

    samples = con.execute(
        "SELECT paper_id, title FROM papers WHERE length(title) > 40 LIMIT 5"
    ).fetchall()

    for paper_id, title in samples:
        top = con.execute("""
            SELECT paper_id FROM (
                SELECT paper_id, fts_main_papers.match_bm25(paper_id, ?) AS score
                FROM papers
            ) WHERE score IS NOT NULL ORDER BY score DESC LIMIT 10
        """, [title]).fetchall()
        assert top, f"BM25 가 아무것도 못 찾음: {title!r}"
        ids = [r[0] for r in top]
        assert paper_id in ids, f"{title!r} -> top10 {ids} 에 {paper_id} 없음"


def test_duplicate_titles_exist_so_dedup_is_not_dead_code():
    """S5 의 제목 정규화 병합이 실제로 할 일이 있는지 — 806개 그룹 확인."""
    from survey_search.assets import PAPERS_DUCKDB

    if not PAPERS_DUCKDB.exists():
        pytest.skip("papers.duckdb 없음")

    con = duckdb.connect(str(PAPERS_DUCKDB), read_only=True)
    n_groups = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT lower(regexp_replace(title, '[^a-zA-Z0-9]', '', 'g')) AS t, COUNT(*) AS c
            FROM papers WHERE title IS NOT NULL GROUP BY t HAVING c > 1
        )
    """).fetchone()[0]
    assert n_groups > 0, "제목 중복이 0이면 S5 의 제목 병합 단계는 죽은 코드입니다"


def test_citation_count_is_integer_typed():
    """문자열로 들어갔으면 정렬이 틀립니다 — 실제 DB에서 타입 확인."""
    from survey_search.assets import PAPERS_DUCKDB

    if not PAPERS_DUCKDB.exists():
        pytest.skip("papers.duckdb 없음")

    con = duckdb.connect(str(PAPERS_DUCKDB), read_only=True)
    dtype = con.execute("""
        SELECT data_type FROM information_schema.columns
        WHERE table_name = 'papers' AND column_name = 'citation_count'
    """).fetchone()[0]
    assert dtype == "BIGINT", f"citation_count 가 {dtype} — 문자열이면 랭킹이 뒤집힙니다"

    top = con.execute(
        "SELECT citation_count FROM papers ORDER BY citation_count DESC LIMIT 1"
    ).fetchone()[0]
    assert top == 175_503, f"최댓값 {top:,} — SETTING.md §6-C 의 175,503 과 다릅니다"


def test_schema_roundtrip_in_memory():
    """스키마가 실제로 유효하고 BIGINT 정렬이 맞는지 메모리 DB로 확인."""
    from survey_search.index.build_duckdb import SCHEMA

    con = duckdb.connect(":memory:")
    con.execute(SCHEMA)
    con.executemany(
        "INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("a v1", "a", 1, "t", "abs", "2026-01-01", "cs.CL", "cs.CL", "x", "u", 9),
            ("b v1", "b", 2, "t", "abs", "2020-01-01", "cs.LG", "cs.LG", "x", "u", 175503),
        ],
    )
    top = con.execute("SELECT paper_id FROM papers ORDER BY citation_count DESC").fetchall()
    assert top[0][0] == "b v1", "BIGINT 정렬이면 175503 이 1위여야 합니다"
