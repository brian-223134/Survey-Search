"""P0.2 회귀 고정 — FAISS id 매핑이 계속 1-based 단일 매핑인지.

SETTING.md §6-A 의 "왕복 10/10"은 이 테스트가 있어야 주장이 됩니다.
자산이 없는 환경에서는 skip 합니다.

    pytest tests/test_faiss_mapping.py                    # 구조 검증만 (~10초)
    pytest tests/test_faiss_mapping.py -m assets          # 의미 왕복 포함 (~2분, GPU)
"""

from __future__ import annotations

import pytest

from survey_search.assets import EMBED_DIM, FAISS_ID_BASE, N_PAPERS, SURVEYFORGE

pytestmark = pytest.mark.skipif(
    bool(SURVEYFORGE.missing()),
    reason=f"SurveyForge 자산 없음: {SURVEYFORGE.missing()}",
)


@pytest.fixture(scope="module")
def index():
    import faiss

    return faiss.read_index(str(SURVEYFORGE.faiss_title_abs))


@pytest.fixture(scope="module")
def id_to_index():
    from survey_search.index.inspect_faiss import load_id_map

    return load_id_map(SURVEYFORGE.id_map)


def test_mapping_structure(index, id_to_index):
    """id_map 이 1-based 연속이고 json 값 집합과 일치하는가."""
    from survey_search.index.inspect_faiss import inspect_mapping

    r = inspect_mapping(index, id_to_index)

    assert r.ntotal == N_PAPERS
    assert r.dim == EMBED_DIM
    assert r.metric == "IP", "IndexFlatIP 여야 합니다 — L2 로 바뀌면 점수 방향이 반대"
    assert r.id_base == FAISS_ID_BASE, "0-based 로 바뀌면 한 칸 밀린 논문이 반환됩니다"
    assert r.id_is_permutation, "id 집합이 {1..N} 전단사여야 조회가 성립합니다"
    assert r.json_matches_index
    assert r.ok

    # id_map 은 행 순서와 일치하지 **않습니다**. 이 사실 자체를 고정해 둡니다 —
    # 나중에 누가 faiss_id-1 을 행 번호로 쓰는 코드를 넣으면 여기서 걸려야 합니다.
    assert not r.id_row_aligned
    assert r.n_row_misaligned > 0


def test_id_minus_one_is_not_a_row_number(index):
    """`faiss_id - 1 == row` 라는 흔한 가정이 실제로 틀리다는 것을 보입니다.

    이게 통과하면(=가정이 틀림) S7 MMR 등에서 반드시 build_id_to_row 를 써야 합니다.
    """
    import faiss
    import numpy as np

    from survey_search.index.inspect_faiss import build_id_to_row, reconstruct_by_id

    id_to_row = build_id_to_row(index)
    # 어긋나는 id 하나를 찾습니다
    bad = next(fid for fid, row in id_to_row.items() if row != fid - 1)

    correct = reconstruct_by_id(index, [bad], id_to_row)[0]
    inner = faiss.downcast_index(index.index)
    naive = inner.reconstruct(bad - 1)

    assert not np.allclose(correct, naive), "행 정렬이 복구됐다면 이 테스트를 지우세요"


def test_no_zero_id(id_to_index):
    """0 이 id 로 쓰이면 1-based 가정이 깨진 것입니다."""
    assert 0 not in set(id_to_index.values())
    assert min(id_to_index.values()) == FAISS_ID_BASE


@pytest.mark.assets
def test_semantic_roundtrip(index, id_to_index):
    """논문 → 임베딩 → 검색 → 같은 논문. 매핑 검증의 본체입니다.

    구조 검증만 통과하고 이게 실패하면, 매핑은 맞는데 인덱스가 다른 텍스트로
    만들어졌다는 뜻입니다 (예: title 인덱스를 title+abs 로 착각).
    """
    from survey_search.index.inspect_faiss import load_tinydb, probe_roundtrip

    by_arxiv = load_tinydb(SURVEYFORGE.tinydb)
    detail = probe_roundtrip(index, id_to_index, by_arxiv, n_probe=10, device="auto")

    failed = [d for d in detail if not d["ok"]]
    assert not failed, f"왕복 실패 {len(failed)}/{len(detail)}: {failed}"
