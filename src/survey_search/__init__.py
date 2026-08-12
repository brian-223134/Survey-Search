"""survey-search — 토픽 하나를 받아 관련 논문 집합을 찾는 검색 레이어.

설계는 DESIGN.md, 확인된 환경 사실은 SETTING.md, 작업 순서는 TASKS.md에 있습니다.
"""

__version__ = "0.1.0"

from survey_search.types import (
    Facet,
    Paper,
    SearchConfig,
    SearchResult,
    SearchStats,
)

__all__ = [
    "Facet",
    "Paper",
    "SearchConfig",
    "SearchResult",
    "SearchStats",
    "search_topic",
]


def search_topic(*args, **kwargs):
    """지연 import — `survey_search` 를 읽는 것만으로 faiss/torch 를 끌어오지 않도록."""
    from survey_search.search import search_topic as _impl

    return _impl(*args, **kwargs)
