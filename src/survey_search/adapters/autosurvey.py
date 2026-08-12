"""P3.1 — AutoSurvey `database` 드롭인 대체.

원본: [`../AutoSurvey/src/database.py:26`](../../../AutoSurvey/src/database.py) 의 `database` 클래스.
호스트 코드는 한 줄만 바꾸면 됩니다:

    # from src.database import database
    from survey_search.adapters.autosurvey import SurveySearchDatabase as database

    db = database(db_path, embedding_model)     # 시그니처 동일

AutoSurvey 에이전트가 실제로 부르는 메서드는 4개뿐입니다(호출 횟수 실측):

    get_paper_info_from_ids (6회) · get_ids_from_query (3회)
    get_titles_from_citations (1회) · get_ids_from_topic (1회)

> **`get_ids_from_topic` 은 원본 `database` 에 정의가 없습니다.**
> `src/agents/outline_writer.py:88` 이 부르는데 클래스에 그 이름이 없어서, 그 경로를
> 타면 `AttributeError` 가 납니다. 형제 레포는 읽기 전용이라 고치지 않고, 이 어댑터가
> 그 이름을 제공합니다 — 결과적으로 이 어댑터를 끼우면 그 경로가 살아납니다.

## ⚠ 코퍼스가 바뀝니다

원본 AutoSurvey 는 자기 스냅샷(nomic 768d, 909,293편)을 봅니다. 이 어댑터는
**SurveyForge 스냅샷(gte 1024d, 908,819편)** 을 봅니다. 반환되는 id 는 그쪽 체계이므로,
**모든 조회가 이 어댑터를 거쳐야 합니다.** 원본 DB의 `paper_content.h5` 를 직접 읽는
`get_paper_from_ids` 는 id 가 맞지 않아 깨집니다 — 그래서 여기서는 명시적으로
`NotImplementedError` 를 냅니다. 조용히 빈 값을 돌려주면 본문 없는 서베이가 나옵니다.
"""

from __future__ import annotations

import logging
from pathlib import Path

from survey_search.backends.faiss_duckdb import FaissDuckDBBackend
from survey_search.search import search_topic
from survey_search.types import SearchConfig

log = logging.getLogger(__name__)


class SurveySearchDatabase:
    """`AutoSurvey.src.database.database` 호환. 검색만 survey-search 로 대체합니다."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        embedding_model: str | None = None,
        *,
        backend: FaissDuckDBBackend | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        # db_path / embedding_model 은 시그니처 호환을 위해 받되 쓰지 않습니다.
        # 어느 인덱스를 볼지는 survey_search.assets 가 정합니다 — 경로가 두 군데서
        # 정해지면 어느 코퍼스를 쟀는지 나중에 알 수 없게 됩니다.
        if db_path is not None:
            log.info("db_path=%s 는 무시됩니다 — survey_search.assets 의 경로를 씁니다", db_path)
        self.backend = backend or FaissDuckDBBackend()
        self.config = config or SearchConfig()
        self._last_result = None

    # --- 검색 ---------------------------------------------------------------

    def get_ids_from_query(self, query: str, num: int, shuffle: bool = False) -> list[str]:
        cfg = _with_n(self.config, num)
        result = search_topic(query, backend=self.backend, config=cfg)
        self._last_result = result
        if shuffle:
            import random

            ids = result.ids()
            random.shuffle(ids)
            return ids
        return result.ids()

    def get_ids_from_topic(self, topic: str, num: int, shuffle: bool = False) -> list[str]:
        """원본에 없는 메서드 — 모듈 docstring 참조."""
        return self.get_ids_from_query(topic, num, shuffle)

    def get_ids_from_queries(self, queries: list[str], num: int,
                             shuffle: bool = False) -> list[list[str]]:
        return [self.get_ids_from_query(q, num, shuffle) for q in queries]

    def get_titles_from_citations(self, citations: list[str]) -> list[str]:
        """인용 문자열 → 가장 가까운 논문의 제목 1편씩. 원본과 동작을 맞춥니다."""
        if not citations:
            return []
        hits = self.backend.dense_search(citations, top_k=1)
        ids = [h[0][0] for h in hits if h]
        by_id = {p.paper_id: p.title for p in self.backend.get_papers(ids)}
        return [by_id.get(i, "") for i in ids]

    # --- 메타데이터 ----------------------------------------------------------

    def get_paper_info_from_ids(self, ids: list[str]) -> list[dict]:
        """원본과 같은 dict 모양으로 돌려줍니다 (`id`/`title`/`abs`/`date`/`cat`/`url`)."""
        papers = self.backend.get_papers(list(ids))
        if len(papers) != len(ids):
            log.warning("메타데이터를 못 찾은 id %d건", len(ids) - len(papers))
        return [
            {
                "id": p.paper_id,
                "title": p.title,
                "abs": p.abstract,
                "date": p.date,
                "cat": " ".join(p.categories),
                "url": f"http://arxiv.org/pdf/{p.paper_id}",
            }
            for p in papers
        ]

    def get_date_from_ids(self, ids: list[str]) -> list[str]:
        return [r["date"] for r in self.get_paper_info_from_ids(ids)]

    def get_title_from_ids(self, ids: list[str]) -> list[str]:
        return [r["title"] for r in self.get_paper_info_from_ids(ids)]

    def get_abs_from_ids(self, ids: list[str]) -> list[str]:
        return [r["abs"] for r in self.get_paper_info_from_ids(ids)]

    def get_paper_from_ids(self, ids: list[str], max_len: int = 1500):
        raise NotImplementedError(
            "본문(paper_content.h5)은 AutoSurvey 자기 코퍼스의 id 체계를 씁니다. "
            "이 어댑터는 SurveyForge 스냅샷의 id 를 돌려주므로 본문 조회가 맞지 않습니다. "
            "본문이 필요한 실험이면 어댑터를 쓰지 말고 코어 API 로 두 코퍼스를 "
            "base_id 로 정합시키세요."
        )

    # --- 진단 ---------------------------------------------------------------

    @property
    def last_stats(self):
        """직전 검색의 `SearchStats`. 호스트를 안 고치고도 단계별 숫자를 볼 수 있게."""
        return self._last_result.stats if self._last_result else None


def _with_n(cfg: SearchConfig, n: int) -> SearchConfig:
    from dataclasses import replace

    return replace(cfg, n_papers=n)
