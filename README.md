# survey-search

토픽 하나를 받아 관련 논문 집합을 찾아 주는 **검색 레이어**.
AutoSurvey · SurveyForge · SurveyX 등 여러 서베이 에이전트가 **공유해서 쓰는 독립 패키지**입니다.

```
topic: str  →  ranked · deduped · facet-grouped papers
```

**최종 갱신**: 2026-08-12 (설계 문서 + P0 실측 검증 완료, 코드 구현 전)

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
| 설계 문서 | ✅ [`DESIGN.md`](DESIGN.md) |
| 환경·자산 조사 | ✅ [`SETTING.md`](SETTING.md) |
| 작업 목록 | ✅ [`TASKS.md`](TASKS.md) |
| **P0** 기반·인덱스 | ✅ 완료 — id 매핑 왕복 10/10, DuckDB 908,819행 + BM25 |
| **P1** 최소 파이프라인 | ✅ 완료 — dense+BM25+RRF+dedup, 1500편 반환 |
| **P2.3** freshness 랭킹 | ✅ 완료 — 최근 12개월 비율 35.5% → 38.5% |
| **P2.4** 다양성 (MMR) | ✅ 완료 — 상위200 유사도 0.758 → 0.629, 카테고리 27 → 32 |
| P2.1/2.2 facet | ⬜ 미착수 — **LLM(OpenRouter) 필요** |
| P3 어댑터 · P4 진단 | ⬜ 미착수 |
| 평가 | ❌ SurGE 평가 코드가 미공개라 후속으로 미룸 |

테스트 55개 통과. 실측 결과표는 [`TASKS.md`](TASKS.md) 의 P1·P2.3·P2.4 절에 있습니다.

**전제가 데이터로 확인됐습니다** — 코퍼스 908,819편 중 2025~2026년 논문이 277,804편(**30.6%**)이고,
인용수 중앙값은 6에 불과합니다. 즉 인용수 정렬은 코퍼스의 3할을 구조적으로 탈락시킵니다.

**첫 실측이 가리키는 방향** — BM25는 dense가 못 찾은 논문 234편을 데려오지만 최신성은
1.1%p밖에 못 올립니다. 최신성은 **검색이 아니라 랭킹의 문제**이고, S6(freshness)가
그걸 확인해 줍니다: 인용수 중앙값을 떨어뜨리지 않으면서 12개월 비율을 +1.9%p 올립니다.

## 문서

| 문서 | 내용 |
|---|---|
| [`DESIGN.md`](DESIGN.md) | 아키텍처, API 계약, 파이프라인 8단계 사양, 어댑터 규약 |
| [`SETTING.md`](SETTING.md) | GPU·디스크·인덱스 지문·모델 캐시 등 확인된 환경 사실 |
| [`TASKS.md`](TASKS.md) | P0~P5 단계별 작업, 산출물, 검증 방법, 예상 소요 |

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
- [`../SurGE`](../SurGE) — 후속 평가 대상 (SIGIR 2026, GT 서베이 205편 + topic→publication 매핑).
  벤치마크 구현 코드가 전부 공개되지 않아 지금 단계에서는 보류
