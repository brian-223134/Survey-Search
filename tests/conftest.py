"""공용 픽스처.

**백엔드는 세션당 하나만 만듭니다.** 테스트마다 새로 만들면 매번 gte(약 1.7GB)를
GPU 에 올리고 3.7GB FAISS 인덱스를 다시 읽습니다. 이 머신은 GPU 를 여러 사람이
공유하므로 사본이 서너 개 쌓이면 OOM 이 납니다 — 실제로 전체 테스트를 돌릴 때
발생했고, 개별 테스트만 돌리면 통과해서 원인이 잘 안 보였습니다.
"""

from __future__ import annotations

import pytest

from survey_search.assets import PAPERS_DUCKDB, SURVEYFORGE


def assets_available() -> bool:
    return not SURVEYFORGE.missing() and PAPERS_DUCKDB.exists()


@pytest.fixture(scope="session")
def backend():
    """세션 공유 FaissDuckDBBackend. 자산이 없으면 skip."""
    if not assets_available():
        pytest.skip("FAISS/DuckDB 자산 없음")
    from survey_search.backends.faiss_duckdb import FaissDuckDBBackend

    be = FaissDuckDBBackend()
    yield be
    be.close()
