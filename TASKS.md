# TASKS — survey-search 구현 계획

설계는 [`DESIGN.md`](DESIGN.md), 환경 사실은 [`SETTING.md`](SETTING.md).
소요 시간은 추정치이며, GPU가 필요한 항목은 명시했습니다.

**진행 규칙** — 각 단계는 *검증*을 통과해야 다음으로 넘어갑니다. 검증은 전부
"돌려보고 숫자를 확인한다"로 정의돼 있고, 눈으로 훑는 것은 검증이 아닙니다.

---

## P0 — 기반 확인 및 인덱스 준비

기존 자산이 실제로 쓸 수 있는지 확인하는 단계입니다. **여기서 막히면 이후 설계가 바뀝니다.**

| # | 작업 | 산출물 | 검증 | 상태 |
|---|---|---|---|---|
| 0.1 | 프로젝트 스캐폴딩 — `pyproject.toml`, 자체 venv, `git init` | 설치 가능한 패키지 | `pip install -e .` 성공 | ✅ **완료** |
| 0.2 | **FAISS id 매핑 실측** (`index/inspect_faiss.py`) | 매핑 규칙 문서화 | 알려진 논문 10편 arxiv_id → 검색 → 같은 id 왕복 | ✅ **완료** — 10/10, `tests/test_faiss_mapping.py` 로 고정 |
| 0.3 | **Flat 검색 지연 실측** | 쿼리당 ms, 32쿼리 배치 총 시간 | 실측값을 `SETTING.md`에 기록 | ✅ **완료** — 32쿼리 2.73초 |
| 0.4 | TinyDB JSON → DuckDB (`index/build_duckdb.py`) | `papers.duckdb` | 행 수 = 908,819, id 전단사 | ✅ **완료** |
| 0.5 | DuckDB FTS 인덱스 (title, abstract) | BM25 검색 가능 | ~~1위~~ → **top-10 포함** (사유 아래) | ✅ **완료** |
| 0.6 | `citation_count` 분포 · `date` 의미 확인 | 분포표 | 연도별 결측률·중앙값 표 | 🟡 분포 완료, `date` 의미(v1 vs 최신본)만 잔여 |

**0.2는 해소됐습니다** ([SETTING.md](SETTING.md) §6-A). id 집합이 `{1..908819}` 전단사이고
`arxivid_to_index_abs.json` 값과 일치해, **이중 매핑이 아니라 단일 매핑**입니다.
의미 수준 왕복(논문 → 임베딩 → 검색 → 같은 논문) 10/10 통과했습니다.

남은 함정 4가지 — 구현 시 반드시 지킬 것 (전부 회귀 테스트로 고정해 뒀습니다):

1. **1-based** — 0-based로 가정하면 한 칸 밀린 논문이 조용히 반환됩니다
2. **`faiss_id - 1` 은 행 번호가 아님** — id_map이 순열이라 441,842개가 어긋납니다.
   저장된 벡터를 꺼낼 때(S7 MMR)는 `build_id_to_row()` 필수
3. **`citation_count`가 문자열** (`'17'`) — `BIGINT` 캐스팅 필수. 아니면 `'9' > '175503'`
4. **쿼리 벡터가 비정규화** (norm ≈ 24) — 쿼리 간 점수 비교 불가. 순위 기반 RRF가 필수 조건

**0.5의 검증 기준을 바꿨습니다** — 원래 "제목으로 검색 시 1위"였으나 실측에서 3위로 나왔습니다.
FTS가 title **+ abstract** 에 걸려 있어 같은 용어를 초록에서 더 자주 쓰는 논문이 위로 옵니다
(BM25 정상 동작). S3의 목적이 제목 정확 매칭이 아니라 후보 확장이므로 **top-10 포함**으로
바꿨습니다. 정확 매칭이 필요해지면 title 전용 FTS를 따로 만드세요.

**적재 성능** — 행 단위 `executemany` 는 908k행에 3시간 이상 걸립니다. pyarrow 컬럼 단위로
넘겨 **총 90초**(파싱 19s + 적재 16s + FTS 41s), 산출 1.1GB. P1 반복 중 재빌드가 잦으므로 중요합니다.

**부수 확인** — 정규화 제목이 겹치는 그룹이 **806개** 있습니다. S5의 제목 병합은 죽은 코드가 아닙니다.

## P1 — 최소 동작 파이프라인

facet 없이, 단일 쿼리로 dense+BM25+RRF+dedup까지. **여기까지가 "돌아가는 검색"입니다.**

| # | 작업 | 산출물 | 검증 |
|---|---|---|---|
| 1.1 | `types.py` — Paper / Facet / SearchResult / SearchStats | 자료형 | — |
| 1.2 | `backends/base.py` — Backend 프로토콜 | 인터페이스 | mock 백엔드로 테스트 통과 |
| 1.3 | `backends/faiss_duckdb.py` — dense + lexical + meta | 동작하는 백엔드 | 토픽 1개로 top-100 반환 |
| 1.4 | `core/fuse.py` — RRF | 융합 함수 | 합성 순위로 단위 테스트 |
| 1.5 | `core/dedup.py` — 버전 병합 + 제목 정규화 | 중복 제거 | 알려진 중복 쌍이 병합됨 |
| 1.6 | `search.py` — 오케스트레이터 (facet 없이) | `search_topic()` | RAG 토픽으로 1500편, stats 출력 |

**P1 검증 기준**: `search_topic("Retrieval-Augmented Generation for Large Language Models")`가
1500편을 반환하고, `stats`에 단계별 건수와 최근 12개월 비율이 찍힐 것.

## P2 — facet + freshness + 다양성

여기서부터 베이스라인과 달라집니다.

| # | 작업 | 산출물 | 검증 |
|---|---|---|---|
| 2.1 | `core/facets.py` — LLM facet 분해 + 디스크 캐시 | facet 8~16개 | 같은 토픽 재실행 시 LLM 호출 0회 |
| 2.2 | facet fan-out 배선 (S2·S3 배치화) | 멀티쿼리 검색 | 단일 쿼리 대비 신규 논문 유입 수 |
| 2.3 | `core/rank.py` — 연령 정규화 인용률 + recency | freshness 랭킹 | 최근 12개월 비율이 P1 대비 상승 |
| 2.4 | `core/diversity.py` — MMR + facet 쿼터 | 다양성 제어 | facet별 최소 배정 충족 |
| 2.5 | `SearchConfig` ablation 스위치 배선 | 단계 on/off | 전부 off = 베이스라인 재현 |

**P2 검증 기준**: 같은 토픽에서 `facets=False` 대비 `facets=True`의 신규 논문 유입 수와
최근 12개월 비율 변화를 표로 낼 것. **이 표가 이 프로젝트의 첫 결과물입니다.**

## P3 — 어댑터 · CLI

호스트 에이전트에 실제로 꽂아보는 단계.

| # | 작업 | 산출물 | 검증 |
|---|---|---|---|
| 3.1 | `adapters/autosurvey.py` | 드롭인 `database` 대체 | AutoSurvey 스모크 1편 생성 |
| 3.2 | `adapters/surveyforge.py` (`filter`·`rerank` 인자 호환) | 드롭인 RAG 대체 | SurveyForge 스모크 1편 생성 |
| 3.3 | `cli.py` — 토픽 → JSON 덤프 | CLI | 토픽 3개 배치 실행 |

> 서베이 생성은 편당 실비가 듭니다($0.3~2). 스모크는 사용자 승인 후에만 돌립니다.

## P4 — 진단 하네스

정량 벤치마크(SurGE)가 막혀 있으므로, **자체 진단 지표로 개발을 굴립니다.**

| # | 작업 | 내용 |
|---|---|---|
| 4.1 | stats 리포트 | 단계별 in/out, 폐기 건수와 사유, 소요 시간 |
| 4.2 | 최신성 지표 | 최근 6/12/24개월 논문 비율, 결과의 날짜 분포 히스토그램 |
| 4.3 | 커버리지 지표 | facet별 논문 수, 미충족 facet 목록 |
| 4.4 | 베이스라인 대조 | 같은 토픽에서 AutoSurvey·SurveyForge 원본 검색 결과와의 교집합/신규 유입 |
| 4.5 | 회귀 고정 | 토픽 3~5개의 결과를 스냅샷으로 저장, 변경 시 diff |

**4.4가 특히 중요합니다** — "우리 검색이 원본이 못 찾던 무엇을 찾는가"를 논문 없이 정량화하는
유일한 방법입니다.

## P5 — 후속 (별도 판단 후 착수)

| # | 작업 | 선행 조건 |
|---|---|---|
| 5.1 | 인용 스노우볼링 | Semantic Scholar / OpenAlex로 인용 엣지 수집 |
| 5.2 | cross-encoder 재랭킹 | `bge-reranker-v2-m3` 다운로드, GPU |
| 5.3 | BGE-M3 하이브리드 인덱스 재구축 | GPU 2~4시간, 디스크 +8GB |
| 5.4 | ~~HNSW 전환~~ | ❌ **불필요 확정** — 0.3 실측에서 32쿼리 2.73초. 착수 사유 소멸 |
| 5.5 | SurGE 평가 | 벤치마크 구현 확보 또는 자체 구현 |
| 5.6 | ReAct 에이전트 루프 | 규칙 기반 파이프라인이 안정된 뒤 |

---

## 지금 당장의 다음 한 걸음

**P0.1 (스캐폴딩) → P0.4/0.5 (DuckDB + FTS)**. P0.2·0.3은 실측으로 끝났고
그 결과가 P1 백엔드 설계를 이미 확정해 줬습니다 — 남은 P0는 BM25 쪽 자산 준비뿐입니다.

```bash
cd /data2/chanjoong/survey-agent/survey-search
python3 -m venv .venv && .venv/bin/pip install -e .   # faiss-cpu, duckdb, sentence-transformers
.venv/bin/python -m survey_search.index.build_duckdb \
    --db ../SurveyForge_data/database_2026-08/arxiv_paper_db_with_cc.json \
    --id-map ../SurveyForge_data/database_2026-08/arxivid_to_index_abs.json \
    --out data/papers.duckdb
```

P0.2의 왕복 검증은 `index/inspect_faiss.py`로 정식화해 회귀 테스트에 넣습니다
(측정 자체는 끝났으므로 재실행이 아니라 **고정**이 목적입니다).
