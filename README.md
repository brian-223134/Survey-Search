# survey-search

토픽 하나를 받아 관련 논문 집합을 찾아 주는 **검색 레이어**.
AutoSurvey · SurveyForge · SurveyX 등 여러 서베이 에이전트가 **공유해서 쓰는 독립 패키지**입니다.

```
topic: str  →  ranked · deduped · facet-grouped papers
```

**최종 갱신**: 2026-08-12 (P0~P4 구현·검증 완료. 정량 평가는 착수 전)

---

## 왜 만드는가

기존 서베이 에이전트의 검색은 전부 **dense 벡터 1방 + top-k**입니다.

| | AutoSurvey | SurveyForge |
| --- | --- | --- |
| 쿼리 | 토픽 문자열 1개 → top-1200 | 멀티쿼리 union + 2단계(1500편 서브셋) |
| 임베딩 | nomic-embed-text-v1 (768d) | gte-large-en-v1.5 (1024d) |
| 재랭킹 | 없음 | 2년 윈도우별 **인용수 정렬** |
| BM25 | 없음 | `NotImplementedError` 스텁 |
| 인용 그래프 | 없음 | 없음 |

여기서 두 가지 문제가 생깁니다.

**1. 데이터 신선도** — 배포본 DB의 컷오프가 2024-04(AutoSurvey) / 2024-09(SurveyForge)였습니다.
2026-08 스냅샷으로 최신화하는 작업은 이미 끝났습니다(각 레포 `HANDOFF.md`).

**2. 랭킹이 최신 논문에 구조적으로 불리함** — DB만 최신화해서는 해결되지 않는 부분입니다.

- SurveyForge의 `sort_by_citation_period`([code/src/utils.py:161](../SurveyForge/code/src/utils.py#L161))는
  시간 윈도우 안에서 **인용수로 정렬**합니다. 최근 6~12개월 논문은 인용수가 구조적으로 0에 가까워
  검색에 걸려도 랭킹에서 다시 탈락합니다.
- AutoSurvey는 재랭킹이 없는 대신 토픽 임베딩의 최근접 이웃이 전부입니다. 새로 등장한 하위 분야는
  **용어 자체가 새로워서** dense 이웃에 잡히지 않습니다.

이 패키지는 (2)를 정면으로 다룹니다. 멀티쿼리 fan-out, BM25 하이브리드, freshness-aware 랭킹,
중복·다양성 제어, (후속) 인용 스노우볼링을 **교체 가능한 단계로** 쌓아 각각의 기여를 분리 측정합니다.

---

## 현재 상태

| 항목 | 상태 |
|---|---|
| 설계 문서 | ✅ `DESIGN.md` (비공개) |
| 환경·자산 조사 | ✅ `SETTING.md` (비공개) |
| 작업 목록 | ✅ `TASKS.md` (비공개) |
| **P0** 기반·인덱스 | ✅ 완료 — id 매핑 왕복 10/10, DuckDB 908,819행 + BM25 |
| **P1** 최소 파이프라인 | ✅ 완료 — dense+BM25+RRF+dedup, 1500편 반환 |
| **P2** facet · freshness · 다양성 | ✅ 완료 — S1(OpenRouter) · S6 · S7 전부 동작 |
| **P3** 어댑터 · CLI | 🟡 어댑터 시그니처·CLI 완료. **서베이 생성 스모크는 승인 대기** |
| **P4** 진단 하네스 | ✅ 완료 — stats · 최신성 · 커버리지 · 베이스라인 대조 · 회귀 스냅샷 |
| 평가 | 🟡 **가능한 것으로 확인됨** — SurGE GT 서베이 205편의 인용 목록이 정답 집합. 우리 코퍼스와 91.4% 연결 (`TASKS.md` §5.5, 비공개) |

테스트 103개 통과. 단계별 실측 결과표는 비공개 `TASKS.md` 에 있습니다.

## 지금까지 나온 결과

토픽 "RAG for LLMs", 1500편 기준. 각 단계를 누적으로 켜 가며 잰 값입니다.
**논문 나이는 arXiv id 에서 유도한 제출일 기준**입니다 (아래 주의 참조).

| 설정 | 쿼리 | 최근 6m | 최근 12m | 상위200 유사도 | 카테고리 | 인용 중앙값 | 신규 |
|---|---|---|---|---|---|---|---|
| dense only (베이스라인) | 1 | 10.8% | 25.1% | 0.7969 | 23 | 4 | — |
| + BM25 | 1 | 10.9% | 25.6% | 0.7545 | 27 | 3 | 351 |
| + freshness | 1 | 12.7% | 28.7% | 0.7574 | 29 | 4 | 361 |
| + 다양성 (MMR λ=0.3) | 1 | 14.7% | 30.8% | 0.6310 | 32 | 2 | 862 |
| + facets | 36 | 14.7% | 33.7% | 0.7653 | 25 | 4 | 744 |
| **전부 켜기** | 36 | **15.5%** | **34.9%** | **0.6302** | 31 | 3 | **1,022** |

**최근 12개월 비율 25.1% → 34.9%(+9.8%p), 6개월 10.8% → 15.5%, 유사도 0.797 → 0.630,
최종 1,500편 중 1,022편(68%) 교체.** 기여도가 가장 큰 단계는 **facet 분해**입니다 —
LLM이 내놓은 쿼리에 Self-RAG · IRCoT · DPR · FEVER 같은 방법론·데이터셋 이름이 들어가는데,
이게 dense 임베딩이 구조적으로 못 잡는 바로 그 토큰들이기 때문입니다.

**전제도 데이터로 확인됐습니다** — 코퍼스 908,819편 중 2025~2026년 논문이 277,804편(**30.6%**)인데
인용수 중앙값은 6입니다. 인용수 정렬은 코퍼스의 3할을 구조적으로 탈락시킵니다.

> ⚠ **날짜 필드 주의** — 원본 DB의 `date` 는 v1 게시일이 아니라 **최신 버전 갱신일**입니다
> (v1 논문은 96%가 제출월과 일치하는데 v2+ 는 70%). 2007년 논문이 2025년에 개정된 사례도
> 있습니다. 그래서 arXiv id 의 `YYMM` 에서 제출월을 유도해 씁니다 — `date` 기준으로 재면
> 검색 결과의 최신성이 약 10%p 부풀려집니다(최종 1,500편의 31%가 제출월 ≠ 갱신월).

> ⚠ **이 표는 방향 지표이지 성능 지표가 아닙니다.** 최신성 비율과 유사도는 "얼마나 최신인가",
> "얼마나 다양한가"만 재고 **"얼마나 맞는가"는 재지 않습니다.** 새로 들어온 1,022편이 *좋은*
> 논문인지는 정답 집합(SurGE 또는 대체)이 붙어야 답할 수 있습니다.

## 문서

> ⚠ 아래 세 문서는 **이 저장소에 포함되지 않습니다** (`.gitignore` 처리). 개발 머신의
> 환경 사실·실측 수치·내부 작업 계획이 들어 있어 공개 대상이 아닙니다.
> 클론한 사본에는 없으니, 필요하면 저장소 관리자에게 요청하세요.

| 문서 | 내용 |
|---|---|
| `DESIGN.md` | 아키텍처, API 계약, 파이프라인 8단계 사양, 어댑터 규약 |
| `SETTING.md` | GPU·디스크·인덱스 지문·모델 캐시 등 확인된 환경 사실 |
| `TASKS.md` | P0~P5 단계별 작업, 산출물, 검증 방법, 실측 결과표 |

## 확정된 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 위치 | `survey-agent/survey-search` 독립 패키지 | 여러 서베이 에이전트가 공유. 베이스라인 레포를 오염시키지 않아야 통제 비교가 유지됨 |
| 1차 인덱스 | **SurveyForge gte FAISS 재사용** + DuckDB BM25 신규 | GPU 재임베딩 0. `citation_count`가 이미 있어 freshness 랭킹 실험이 바로 가능 |
| 1차 범위 | 툴 + **규칙 기반** 오케스트레이터 | 결정적이라 단계별 ablation·디버깅이 쉬움. ReAct 루프는 후속 |
| 벡터 DB | FAISS + DuckDB (Milvus 아님) | 이 머신에서 docker 소켓 권한이 없어 Milvus를 띄울 수 없음 |

## SimScholarSearch에서 실제로 가져온 것

파생 프로젝트지만 **코드 수준으로 가져올 수 있는 건 많지 않습니다.** 그쪽은 Milvus + BGE-M3 +
S2ORC 본문 + verl RL 스택이고, 우리는 FAISS + gte + arXiv 초록 + 규칙 기반이라 하부가 다릅니다.
가져온 것과 안 가져온 것을 명시해 둡니다 (라이선스 Apache-2.0, 출처는 각 파일 상단에 표기).

**가져온 것 — 코드**

| 우리 쪽 | 원본 | 내용 |
|---|---|---|
| [`metrics/paper_set.py`](src/survey_search/metrics/paper_set.py) | `synthesis/paper_set.py`, `eval/litsearch.py` | `score_paper_set`(PFB adjusted_f1 = harmonic(recall@est, nDCG-rank)), LitSearch `calculate_recall`/`calculate_ndcg`. **id 타입만 int→str(arXiv)로 변경** |
| [`index/build_duckdb.py`](src/survey_search/index/build_duckdb.py) | `env/etl/build_fts.py`, `env/reader.py` | DuckDB FTS 사용법 — `PRAGMA create_fts_index` → `fts_main_<table>.match_bm25`. 스키마는 공유 안 함 |

**가져온 것 — 설계 패턴만**

- `env/tools/registry.py`의 `build_registry()` + `compose()` — DESIGN의 `tools.py` 팩토리가 이 형태
- `search_papers.py`의 `RRFRanker(60)` — RRF k=60 선택의 선례. 구현은 우리가 직접 (`core/fuse.py`)
- 툴 9종의 시그니처 — 후속 ReAct 루프(P5.6) 착수 시 참고

**안 가져온 것과 이유**

| 원본 | 이유 |
|---|---|
| `search_papers`, `find_similar`, `milvus_client`, `etl/ingest_milvus` | 전부 `pymilvus` 의존. 이 머신에 Milvus 없음 |
| `encoder.py` (BGE-M3 dense+sparse 하이브리드) | 1차는 기존 gte 인덱스 재사용이 목적. BGE-M3는 P5.3의 별도 축 |
| `graph.py`, `list_citations`, `list_references` | S2 인용 엣지가 있어야 함 — P5.1의 선행 조건 |
| `read_paper`, `find_in_paper` | 본문 전체가 필요. 우리 코퍼스는 title+abstract뿐 |
| `trainer/`, `synthesis/`, `agent/` | verl RL 학습 스택. 이 프로젝트 범위 밖 |

> 두 프로젝트의 id 체계가 다릅니다 — 그쪽은 S2ORC 정수 `corpus_id`, 이쪽은 버전 접미사가 붙은
> arXiv 문자열 id(`2401.12345v2`). 이식한 코드에서 이 부분은 전부 바꿨습니다.

## 참고한 레포

- [`../SimScholarSearch`](../SimScholarSearch) — 이 프로젝트의 모태. 위 표 참조
- [`../AutoSurvey`](../AutoSurvey), [`../SurveyForge`](../SurveyForge) — 1차 소비자
- [`../SurGE`](../SurGE) — 평가 대상 (SIGIR 2026). **`data/queries.json` 은 없지만
  `data/surveys.json` 의 205편 × `all_cites` 가 그대로 정답 집합입니다.** 제목 정규화로
  우리 코퍼스와 91.4% 연결됩니다 — 비공개 `TASKS.md` §5.5 참조
