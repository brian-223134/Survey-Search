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


def pytest_addoption(parser):
    parser.addoption("--network", action="store_true", default=False,
                     help="실제 arXiv/S2 API 를 부르는 테스트도 돌립니다 (기본: 건너뜀)")


def pytest_collection_modifyitems(config, items):
    """`network` 표시된 테스트는 명시적으로 켤 때만 돕니다.

    `pyproject.toml` 의 마커 설명에 "기본 제외"라고 적혀 있었는데 **그걸 구현한 코드가
    없었습니다.** 그래서 arXiv 가 느린 날이면 기본 실행이 30초를 매달렸다가 실패했습니다
    (실측: 같은 쿼리를 curl 로 던져도 45초 타임아웃 — arXiv 쪽 문제인데 우리 테스트가
    빨간불이 됩니다).

    외부 서비스의 가용성은 우리 코드의 정확성이 아닙니다. 켜서 확인하는 건 유용하니
    지우지 않고 `--network` 뒤에 둡니다.
    """
    if config.getoption("--network"):
        return
    skip = pytest.mark.skip(reason="네트워크 테스트 — `--network` 로 켜세요")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
