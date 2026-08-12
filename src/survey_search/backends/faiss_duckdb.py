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
        device: str = "cuda",
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
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            t = time.perf_counter()
            m = SentenceTransformer(self._model_name, trust_remote_code=True)
            m.to(torch.device(self._device))
            log.info("embedding model loaded in %.1fs (%s)", time.perf_counter() - t, self._device)
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
        placeholders = ",".join("?" * len(paper_ids))
        rows = self.con.execute(
            f"""
            SELECT paper_id, base_id, title, abstract, CAST(date AS VARCHAR),
                   categories, citation_count
            FROM papers WHERE paper_id IN ({placeholders})
            """,
            paper_ids,
        ).fetchall()

        by_id = {
            r[0]: Paper(
                paper_id=r[0],
                base_id=r[1],
                title=r[2] or "",
                abstract=r[3] or "",
                date=r[4] or "",
                categories=tuple((r[5] or "").split()),
                citation_count=int(r[6]) if r[6] is not None else None,
            )
            for r in rows
        }
        return [by_id[pid] for pid in paper_ids if pid in by_id]

    def filter_ids(
        self,
        *,
        date_min: str | None = None,
        date_max: str | None = None,
        categories: tuple[str, ...] | None = None,
    ) -> set[str] | None:
        """조건이 하나도 없으면 None(=제한 없음). 빈 집합과 뜻이 다릅니다."""
        clauses, params = [], []
        if date_min:
            clauses.append("date >= CAST(? AS DATE)")
            params.append(date_min)
        if date_max:
            clauses.append("date <= CAST(? AS DATE)")
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
