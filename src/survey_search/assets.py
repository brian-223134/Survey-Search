"""자산 경로 한 곳 모음.

형제 레포의 인덱스를 **읽기 전용으로** 참조합니다. 경로가 여기저기 하드코딩되면
통제 비교가 깨진 채로도 눈치채기 어려우므로 전부 이 모듈을 거치게 합니다.

환경변수로 덮어쓸 수 있습니다: SURVEY_SEARCH_SF_DB, SURVEY_SEARCH_DATA_DIR.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: SurveyForge 2026-08 스냅샷 — 1차 인덱스 (gte-large-en-v1.5, 1024d, 908,819편)
SF_DB_DIR = Path(
    os.environ.get(
        "SURVEY_SEARCH_SF_DB",
        REPO_ROOT.parent / "SurveyForge_data" / "database_2026-08",
    )
)

#: 우리가 만들어 쓰는 산출물 (DuckDB, facet 캐시 등). .gitignore 대상
DATA_DIR = Path(os.environ.get("SURVEY_SEARCH_DATA_DIR", REPO_ROOT / "data"))

#: 쿼리 임베딩 모델. 1차 인덱스가 이 모델로 만들어졌으므로 바꾸면 안 됩니다.
GTE_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
HF_CACHE = Path(
    os.environ.get("HF_HOME", "/data2/chanjoong/.cache/huggingface")
) / "hub"


@dataclass(frozen=True)
class SurveyForgeAssets:
    """SurveyForge 2026-08 스냅샷의 파일들.

    SETTING.md §3-A / §6 의 실측값이 이 자산에 대한 것입니다.
    """

    root: Path = SF_DB_DIR

    @property
    def faiss_title_abs(self) -> Path:
        return self.root / "faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_260804.bin"

    @property
    def faiss_title(self) -> Path:
        return self.root / "faiss_paper_title_embeddings_FROM_2012_0101_TO_260804.bin"

    @property
    def id_map(self) -> Path:
        """arxiv_id -> FAISS id. **값이 1-based입니다** (SETTING.md §6-A)."""
        return self.root / "arxivid_to_index_abs.json"

    @property
    def tinydb(self) -> Path:
        """TinyDB JSON. 테이블 `cs_paper_info`, 908,819행."""
        return self.root / "arxiv_paper_db_with_cc.json"

    def missing(self) -> list[Path]:
        """존재하지 않는 자산 목록. 비어 있으면 전부 준비된 것."""
        return [
            p
            for p in (self.faiss_title_abs, self.faiss_title, self.id_map, self.tinydb)
            if not p.exists()
        ]


SURVEYFORGE = SurveyForgeAssets()

#: P0.4 산출물
PAPERS_DUCKDB = DATA_DIR / "papers.duckdb"

# --- 실측으로 확정된 상수 (SETTING.md §6). 코드가 이 값을 가정합니다. -------------
N_PAPERS = 908_819
EMBED_DIM = 1024
#: FAISS id 는 1-based 연속입니다. 0-based로 가정하면 한 칸 밀린 논문이 조용히 반환됩니다.
FAISS_ID_BASE = 1
