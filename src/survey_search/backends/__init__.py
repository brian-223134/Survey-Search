"""백엔드 — 코퍼스를 아는 유일한 계층.

코어 파이프라인은 어느 코퍼스도 모르고 `Backend` 프로토콜만 압니다.
그래야 AutoSurvey·SurveyForge·SurveyX가 각자 다른 임베딩·id 체계를 써도
같은 검색 로직을 공유할 수 있습니다 (DESIGN §1-①).
"""

from survey_search.backends.base import Backend

__all__ = ["Backend"]
