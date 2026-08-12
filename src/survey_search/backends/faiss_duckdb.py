"""P1.3 — FAISS(dense) + DuckDB(meta·BM25) 백엔드.

SurveyForge 2026-08 스냅샷을 **재임베딩 없이** 그대로 씁니다. 쿼리 임베딩만
gte-large-en-v1.5 로 계산합니다.

실측으로 확인된 제약 세 가지가 여기서 처리됩니다 (SETTING.md §6-A):

1. FAISS id 는 1-based. `index_to_id` 를 한 번만 적용합니다 (이중 매핑 아님)
2. 인덱스는 `IndexFlatIP` — 점수는 클수록 좋습니다. 다만 쿼리 벡터가 비정규화라
   **쿼리 간 점수 비교는 무의미**합니다. 그래서 융합은 순위 기반(RRF)으로만 합니다
3. `faiss_id - 1` 은 행 번호가 아닙니다 — 저장 벡터가 필요하면 `reconstruct_by_id`
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from survey_search.assets import GTE_MODEL, PAPERS_DUCKDB, SURVEYFORGE
from survey_search.backends.base import Hit
from survey_search.types import Paper

log = logging.getLogger(__name__)


class FaissDuckDBBackend:
    """dense = FAISS, lexical/meta = DuckDB.

    임베딩 모델과 FAISS 인덱스는 **처음 쓸 때 로드**합니다 (lazy). BM25만 쓰는
    ablation 에서 3.7GB 인덱스를 읽을 이유가 없기 때문입니다.
    """

    def __init__(
        self,
        *,
        name: str = "surveyforge-2026-08",
        duckdb_path: Path = PAPERS_DUCKDB,
        faiss_title_abs: Path = SURVEYFORGE.faiss_title_abs,
        faiss_title: Path = SURVEYFORGE.faiss_title,
        id_map_path: Path = SURVEYFORGE.id_map,
        model_name: str = GTE_MODEL,
        device: str = "auto",
        embed_batch_size: int = 32,
    ) -> None:
        self.name = name
        self.duckdb_path = Path(duckdb_path)
        self._faiss_paths = {"title_abs": Path(faiss_title_abs), "title": Path(faiss_title)}
        self._id_map_path = Path(id_map_path)
        self._model_name = model_name
        self._device = device
        self._embed_batch_size = embed_batch_size

        self._con = None
        self._model = None
        self._active_device: str | None = None   # 실제로 올라간 곳 (cuda/cpu)
        self._indexes: dict[str, object] = {}
        self._index_to_id: dict[int, str] | None = None
        self._id_to_index: dict[str, int] | None = None
        self._id_to_row: dict[int, int] | None = None
        self._id_to_row_field: str | None = None
        self._lock = threading.Lock()

    # --- lazy 자원 -----------------------------------------------------------

    @property
    def con(self):
        if self._con is None:
            import duckdb

            if not self.duckdb_path.exists():
                raise FileNotFoundError(
                    f"{self.duckdb_path} 없음 — python -m survey_search.index.build_duckdb 먼저 실행"
                )
            self._con = duckdb.connect(str(self.duckdb_path), read_only=True)
            self._con.execute("LOAD fts;")
        return self._con

    @property
    def model(self):
        """쿼리 임베딩 모델. **GPU 가 막혀 있으면 CPU 로 물러섭니다.**

        이 머신은 GPU 를 여러 사람이 공유합니다 — 8장이 전부 40GB 넘게 차 있는 시점이
        실제로 관측됩니다. 쿼리 임베딩은 한 번에 수십 건이라 CPU 로도 돌아가므로,
        여기서 죽는 것보다 느리게라도 도는 편이 낫습니다. **어디로 갔는지는 로그에 남깁니다.**
        """
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            t = time.perf_counter()
            m = SentenceTransformer(self._model_name, trust_remote_code=True)

            want = self._device
            if want == "auto":
                want = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                m.to(torch.device(want))
                self._active_device = want
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if want == "cpu":
                    raise
                log.warning("%s 로 올리지 못해 CPU 로 폴백합니다: %s", want, e)
                m.to(torch.device("cpu"))
                self._active_device = "cpu"

            log.info("embedding model loaded in %.1fs (%s)",
                     time.perf_counter() - t, self._active_device)
            self._model = m
        return self._model

    def _index(self, field: str):
        if field not in self._faiss_paths:
            raise ValueError(f"unknown field {field!r}; expected one of {list(self._faiss_paths)}")
        with self._lock:
            if field not in self._indexes:
                import faiss

                t = time.perf_counter()
                self._indexes[field] = faiss.read_index(str(self._faiss_paths[field]))
                log.info("faiss[%s] loaded in %.1fs", field, time.perf_counter() - t)
        return self._indexes[field]

    def _maps(self) -> tuple[dict[str, int], dict[int, str]]:
        if self._index_to_id is None:
            from survey_search.index.inspect_faiss import load_id_map

            self._id_to_index = load_id_map(self._id_map_path)
            self._index_to_id = {v: k for k, v in self._id_to_index.items()}
        return self._id_to_index, self._index_to_id  # type: ignore[return-value]

    # --- Backend 프로토콜 ------------------------------------------------------

    def dense_search(
        self, queries: list[str], top_k: int, field: str = "title_abs"
    ) -> list[list[Hit]]:
        if not queries:
            return []
        index = self._index(field)
        _, index_to_id = self._maps()

        qv = np.asarray(
            self.model.encode(queries, batch_size=self._embed_batch_size), dtype="float32"
        )
        scores, ids = index.search(qv, top_k)

        out: list[list[Hit]] = []
        for row_ids, row_scores in zip(ids, scores):
            hits: list[Hit] = []
            for fid, sc in zip(row_ids, row_scores):
                if fid == -1:  # FAISS 가 후보를 못 채웠을 때의 표식
                    continue
                pid = index_to_id.get(int(fid))
                if pid is not None:
                    hits.append((pid, float(sc)))
            out.append(hits)
        return out

    def lexical_search(self, queries: list[str], top_k: int) -> list[list[Hit]]:
        """DuckDB FTS BM25. `match_bm25` 는 매칭 안 되면 NULL 이라 걸러냅니다."""
        out: list[list[Hit]] = []
        for q in queries:
            rows = self.con.execute(
                """
                SELECT paper_id, score FROM (
                    SELECT paper_id, fts_main_papers.match_bm25(paper_id, ?) AS score
                    FROM papers
                ) WHERE score IS NOT NULL
                ORDER BY score DESC LIMIT ?
                """,
                [q, top_k],
            ).fetchall()
            out.append([(str(pid), float(sc)) for pid, sc in rows])
        return out

    def get_papers(self, paper_ids: list[str]) -> list[Paper]:
        """**입력 순서를 보존**합니다. 찾지 못한 id 는 결과에서 빠지므로,
        호출부가 `len(in) - len(out)` 으로 결측을 셀 수 있습니다."""
        if not paper_ids:
            return []

        # `IN (?, ?, ... )` 는 id 가 수천 개가 되면 급격히 느려집니다 (3,000개에 19.7초
        # 실측). id 목록을 Arrow 테이블로 넘겨 조인하면 같은 일이 0.1초입니다.
        # facet 을 켜면 후보가 수만 편이 되므로 이 차이가 파이프라인 전체를 좌우합니다.
        import pyarrow as pa

        # **버전 없는 id 도 받아야 합니다.** S2/arXiv 같은 외부 소스는 `2401.12345`
        # 형태로 주는데 우리 DB 의 키는 `2401.12345v2` 입니다. base_id 로도 맞춰 보지
        # 않으면 스노우볼링이 통째로 0편을 내면서도 조용히 지나갑니다 (실제로 겪었습니다).
        from survey_search.core.dedup import strip_version

        ids_tbl = pa.table({
            "want": pa.array(paper_ids, pa.string()),
            "want_base": pa.array([strip_version(p) for p in paper_ids], pa.string()),
        })
        self.con.register("wanted", ids_tbl)
        try:
            rows = self.con.execute(
                """
                SELECT w.want, p.paper_id, p.base_id, p.title, p.abstract,
                       CAST(p.date AS VARCHAR), CAST(p.submitted_date AS VARCHAR),
                       p.categories, p.citation_count
                FROM wanted w JOIN papers p
                  ON p.paper_id = w.want OR p.base_id = w.want_base
                """
            ).fetchall()
        finally:
            self.con.unregister("wanted")

        # r[0] 은 호출부가 물어본 id, r[1] 은 DB 의 native id 입니다.
        # base_id 로 매칭되면 한 want 에 여러 버전이 걸릴 수 있으므로 최신 버전을 남깁니다.
        by_want: dict[str, Paper] = {}
        for r in rows:
            want, native = r[0], r[1]
            prev = by_want.get(want)
            if prev is not None and prev.paper_id >= native:
                continue
            by_want[want] = Paper(
                paper_id=native,
                base_id=r[2],
                title=r[3] or "",
                abstract=r[4] or "",
                date=r[5] or "",
                submitted_date=r[6] or "",
                categories=tuple((r[7] or "").split()),
                citation_count=int(r[8]) if r[8] is not None else None,
            )
        return [by_want[pid] for pid in paper_ids if pid in by_want]

    def filter_ids(
        self,
        *,
        date_min: str | None = None,
        date_max: str | None = None,
        categories: tuple[str, ...] | None = None,
    ) -> set[str] | None:
        """조건이 하나도 없으면 None(=제한 없음). 빈 집합과 뜻이 다릅니다."""
        # **갱신일(`date`)이 아니라 제출일 기준입니다.** "시점 T 에 존재하던 논문"은
        # T 이전에 제출된 논문이지, T 이전에 마지막 개정된 논문이 아닙니다. `date` 로
        # 거르면 2019년 논문이 2024년에 개정됐다는 이유로 2020년 시점 검색에서
        # 탈락합니다. 실측: SurGE 정답 기준 탈락률 7.07% -> 6.75%.
        # 구식 id(992편)는 submitted_date 가 없어 date 로 폴백합니다.
        age = "COALESCE(submitted_date, date)"
        clauses, params = [], []
        if date_min:
            clauses.append(f"{age} >= CAST(? AS DATE)")
            params.append(date_min)
        if date_max:
            clauses.append(f"{age} <= CAST(? AS DATE)")
            params.append(date_max)
        if categories:
            clauses.append(
                "(" + " OR ".join(["categories LIKE ?"] * len(categories)) + ")"
            )
            params.extend(f"%{c}%" for c in categories)

        if not clauses:
            return None

        rows = self.con.execute(
            f"SELECT paper_id FROM papers WHERE {' AND '.join(clauses)}", params
        ).fetchall()
        return {str(r[0]) for r in rows}

    def get_vectors(self, paper_ids: list[str], field: str = "title_abs") -> np.ndarray | None:
        """저장된 문서 임베딩을 꺼냅니다 (S7 MMR 용). **재임베딩하지 않습니다.**

        `faiss_id - 1` 을 행 번호로 쓰면 안 됩니다 — id_map 이 순열이라 441,842개가
        어긋납니다(SETTING.md §6-A). 반드시 `build_id_to_row()` 를 거칩니다.

        찾지 못한 id 가 하나라도 있으면 **None 을 돌려줍니다** — 일부만 채운 행렬은
        MMR 에서 조용히 틀린 유사도를 만들기 때문입니다. 부분 성공보다 명시적 실패가 낫습니다.
        """
        if not paper_ids:
            return None
        import faiss

        from survey_search.index.inspect_faiss import build_id_to_row

        index = self._index(field)
        id_to_index, _ = self._maps()

        with self._lock:
            if self._id_to_row is None or self._id_to_row_field != field:
                self._id_to_row = build_id_to_row(index)
                self._id_to_row_field = field

        inner = faiss.downcast_index(index.index)
        out = np.empty((len(paper_ids), index.d), dtype="float32")
        for i, pid in enumerate(paper_ids):
            fid = id_to_index.get(pid)
            if fid is None:
                log.warning("get_vectors: %s 의 faiss_id 를 못 찾음 -> None 반환", pid)
                return None
            row = self._id_to_row.get(int(fid))
            if row is None:
                log.warning("get_vectors: faiss_id %d 의 행 번호를 못 찾음 -> None 반환", fid)
                return None
            out[i] = inner.reconstruct(row)
        return out

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
