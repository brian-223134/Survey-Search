"""로컬 + 온라인 하이브리드 백엔드.

**역할 분담이 핵심입니다.** 온라인이 로컬을 대체하는 게 아닙니다:

| | 담당 |
|---|---|
| 로컬 (FAISS + DuckDB) | recall 의 본체, dense 검색, 결정성 |
| 온라인 (arXiv + S2) | **컷오프 이후 논문**, **인용 엣지** |

로컬 인덱스는 2026-08-04 에서 멈춰 있고 인용 엣지가 없습니다. 그 두 구멍만 온라인이
메웁니다. 나머지를 온라인으로 돌리면 느려지고 비결정적이 될 뿐 얻는 게 없습니다.

두 결과는 **RRF 로 융합**합니다 — 로컬 점수(IP)와 arXiv 순위는 스케일이 아예 다르므로
순위 기반 융합이 유일하게 맞는 방법입니다.
"""

from __future__ import annotations

import logging

from survey_search.backends.base import Hit
from survey_search.core.fuse import rrf
from survey_search.types import Paper

log = logging.getLogger(__name__)


class HybridBackend:
    """로컬 백엔드를 감싸고 온라인 결과를 덧댑니다."""

    def __init__(self, local, online, *, rrf_k: int = 60, online_top_k: int = 100) -> None:
        self.local = local
        self.online = online
        self.rrf_k = rrf_k
        #: 온라인에서 가져올 편수. arXiv API 상한(200)과 지연을 고려한 값입니다.
        self.online_top_k = online_top_k
        self.name = f"hybrid({local.name}+{online.name})"

    def dense_search(self, queries: list[str], top_k: int,
                     field: str = "title_abs") -> list[list[Hit]]:
        """dense 는 로컬만 씁니다 — arXiv API 에는 임베딩 검색이 없습니다."""
        return self.local.dense_search(queries, top_k, field=field)

    def lexical_search(self, queries: list[str], top_k: int) -> list[list[Hit]]:
        """로컬 BM25 + arXiv 어휘 검색을 쿼리별로 RRF 융합합니다."""
        local = self.local.lexical_search(queries, top_k)
        remote = self.online.lexical_search(queries, self.online_top_k)

        out: list[list[Hit]] = []
        for l, r in zip(local, remote):
            if not r:
                out.append(l)
                continue
            fused = rrf([l, r], k=self.rrf_k)
            out.append(fused[:top_k])
        return out

    def get_papers(self, paper_ids: list[str]) -> list[Paper]:
        """로컬 우선, 못 찾은 것만 온라인 캐시에서 채웁니다.

        컷오프 이후 논문은 로컬에 없으므로 이 폴백이 없으면 결과에서 조용히 사라집니다.
        """
        found = self.local.get_papers(paper_ids)
        have = {p.paper_id for p in found}
        missing = [p for p in paper_ids if p not in have]
        if missing:
            extra = self.online.get_papers(missing)
            if extra:
                log.info("로컬에 없는 논문 %d편 중 %d편을 온라인에서 채움",
                         len(missing), len(extra))
            found.extend(extra)

        order = {pid: i for i, pid in enumerate(paper_ids)}
        found.sort(key=lambda p: order.get(p.paper_id, 1 << 30))
        return found

    def filter_ids(self, **kwargs):
        """로컬 기준으로만 필터합니다. **온라인 논문은 이 집합에 없습니다** —
        사전 필터를 걸면 컷오프 이후 논문이 전부 탈락하므로, 하이브리드에서는
        날짜 필터를 쓰지 마세요. 쓸 거면 그 사실이 stats 에 남습니다."""
        result = self.local.filter_ids(**kwargs)
        if result is not None:
            log.warning("하이브리드에서 사전 필터를 걸면 온라인(컷오프 이후) 논문이 "
                        "전부 탈락합니다 — 의도한 것인지 확인하세요")
        return result

    def get_vectors(self, paper_ids: list[str], field: str = "title_abs"):
        """S7 MMR 용 저장 벡터. **로컬에 없는 논문이 하나라도 있으면 None** 입니다 —
        온라인 논문은 인덱스에 벡터가 없기 때문입니다. 일부만 채운 행렬은
        MMR 에서 조용히 틀린 유사도를 만듭니다."""
        getter = getattr(self.local, "get_vectors", None)
        return getter(paper_ids, field) if getter else None

    # --- 인용 그래프는 온라인만 압니다 --------------------------------------------

    def references(self, paper_id: str) -> list[str]:
        return self.online.references(paper_id)

    def cited_by(self, paper_id: str) -> list[str]:
        return self.online.cited_by(paper_id)
