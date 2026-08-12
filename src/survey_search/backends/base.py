"""P1.2 — Backend 프로토콜. DESIGN §3 의 구현.

백엔드가 구현해야 하는 것은 이게 전부입니다. 검색 결과는 **백엔드의 native id 를
그대로** 돌려줍니다 — 호스트 에이전트가 자기 DB에서 바로 조회해야 하기 때문입니다.

`dense_search` / `lexical_search` 가 **배치**인 이유: facet fan-out 이 기본 사용
패턴이고, 실측상 배치가 쿼리당 9배 빠릅니다(1쿼리 790ms vs 32쿼리 배치 85ms/쿼리,
SETTING.md §6-B). 쿼리 하나씩 도는 API는 처음부터 만들지 않습니다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from survey_search.types import Paper

#: (paper_id, score) — score 의 방향은 백엔드마다 다릅니다(L2는 작을수록, IP는 클수록).
#: **그래서 융합 전에 순위로 바꿔 씁니다.** 절대 점수를 그대로 비교하지 마세요.
Hit = tuple[str, float]


@runtime_checkable
class Backend(Protocol):
    """검색 백엔드. 필수 4개 + 선택 2개."""

    name: str

    def dense_search(
        self, queries: list[str], top_k: int, field: str = "title_abs"
    ) -> list[list[Hit]]:
        """쿼리 배치 → 쿼리별 (paper_id, score) 목록. **점수 내림차순**으로 돌려주세요."""
        ...

    def lexical_search(self, queries: list[str], top_k: int) -> list[list[Hit]]:
        """BM25 등 어휘 검색. dense 가 구조적으로 못 잡는 것(새 방법론 이름, 모델명,
        데이터셋명, 약어)을 담당합니다 — 최신 논문을 식별하는 바로 그 토큰들입니다."""
        ...

    def get_papers(self, paper_ids: list[str]) -> list[Paper]:
        """메타데이터 조회. **입력 순서를 보존**하고, 없는 id 는 조용히 빼지 말고
        호출부가 셀 수 있게 결과에서 빠진 사실이 드러나야 합니다."""
        ...

    def filter_ids(
        self,
        *,
        date_min: str | None = None,
        date_max: str | None = None,
        categories: tuple[str, ...] | None = None,
    ) -> set[str] | None:
        """조건에 맞는 id 집합. 조건이 하나도 없으면 `None`(=제한 없음)을 돌려줍니다.
        빈 집합과 None 은 뜻이 다릅니다 — 빈 집합은 "해당 없음"입니다."""
        ...


@runtime_checkable
class CitationBackend(Protocol):
    """인용 엣지를 아는 백엔드 (선택). 없으면 스노우볼링 단계가 no-op 이 되고
    **그 사실이 stats 에 남습니다** — 무음 스킵 금지."""

    def references(self, paper_id: str) -> list[str]: ...
    def cited_by(self, paper_id: str) -> list[str]: ...


def supports_citations(backend: object) -> bool:
    return isinstance(backend, CitationBackend)
