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
| 1.1 | `types.py` — Paper / Facet / SearchResult / SearchStats | 자료형 | ✅ |
| 1.2 | `backends/base.py` — Backend 프로토콜 | 인터페이스 | ✅ |
| 1.3 | `backends/faiss_duckdb.py` — dense + lexical + meta | 동작하는 백엔드 | ✅ 1500편 반환 |
| 1.4 | `core/fuse.py` — RRF | 융합 함수 | ✅ 단위 테스트 6개 |
| 1.5 | `core/dedup.py` — 버전 병합 + 제목 정규화 | 중복 제거 | ✅ 단위 테스트 6개 |
| 1.6 | `search.py` — 오케스트레이터 (facet 없이) | `search_topic()` | ✅ 아래 표 참조 |

**P1 검증 기준**: `search_topic("Retrieval-Augmented Generation for Large Language Models")`가
1500편을 반환하고, `stats`에 단계별 건수와 최근 12개월 비율이 찍힐 것. → ✅ **통과**

### P1 실측 결과 (2026-08-12, 토픽 = RAG for LLMs, n_papers=1500)

| | dense only (베이스라인) | dense + BM25 |
|---|---|---|
| 최근 6개월 비율 | 16.9% | 16.9% |
| 최근 12개월 비율 | 35.5% | **36.6%** |
| 최근 24개월 비율 | 69.7% | **71.7%** |
| dense만 | 1,500 | 199 |
| **BM25만 (dense가 못 찾음)** | — | **234** |
| 양쪽 | 0 | 1,067 |
| 최종 목록 교체 | — | **351편 (23.4%)** |
| 소요 (warm) | 1.4 s | 1.5 s |

**읽는 법 두 가지:**

1. **BM25는 값을 냅니다** — dense가 전혀 못 찾은 논문 234편을 데려왔고, 최종 1500편의
   23.4%가 교체됐습니다. SurveyForge가 `NotImplementedError`로 비워 둔 자리가 놀고 있던 게
   맞습니다.
2. **하지만 최신성 문제는 BM25로 안 풀립니다** — 12개월 비율이 35.5% → 36.6%로 1.1%p
   움직였을 뿐입니다. **최신성은 검색 단계가 아니라 랭킹 단계(S6)의 문제**라는 뜻이고,
   이 프로젝트의 가설이 맞는 방향을 가리킵니다. P2.3이 진짜 시험대입니다.

## P2 — facet + freshness + 다양성

여기서부터 베이스라인과 달라집니다.

| # | 작업 | 산출물 | 검증 |
|---|---|---|---|
| 2.1 | `core/facets.py` — LLM facet 분해 + 디스크 캐시 | facet 8~16개 | ✅ **완료** — OpenRouter, 12 facet / 36 쿼리, 캐시 히트 시 호출 0회 |
| 2.2 | facet fan-out 배선 (S2·S3 배치화) | 멀티쿼리 검색 | ✅ **완료** — 2단 RRF, 신규 745편 |
| 2.3 | `core/rank.py` — 연령 정규화 인용률 + recency | freshness 랭킹 | ✅ **완료** — 12개월 비율 상승 확인 |
| 2.4 | `core/diversity.py` — MMR + facet 쿼터 | 다양성 제어 | ✅ **완료** — 쿼터도 활성 (facet 12개) |
| 2.5 | `SearchConfig` ablation 스위치 배선 | 단계 on/off | ✅ **완료** — `cli.py --ablate` |

**P2 검증 기준**: 같은 토픽에서 `facets=False` 대비 `facets=True`의 신규 논문 유입 수와
최근 12개월 비율 변화를 표로 낼 것. **이 표가 이 프로젝트의 첫 결과물입니다.**

### P2.3 실측 결과 (2026-08-12, 토픽 = RAG for LLMs, n_papers=1500)

| 설정 | 최근 6m | 최근 12m | 최근 24m | 인용수 중앙값 | 베이스라인 대비 신규 |
|---|---|---|---|---|---|
| ① dense only (베이스라인) | 16.9% | 35.5% | 69.7% | 4 | — |
| ② + BM25 | 16.9% | 36.6% | 71.7% | 3 | 351 |
| ③ + freshness (weight) | **18.5%** | **38.5%** | **73.1%** | **4** | 353 |
| ④ + freshness (quota 0.30) | 17.0% | 36.3% | 71.2% | 4 | 352 |
| ⑤ + freshness (quota 0.50) | **23.7%** | **50.0%** | **77.5%** | 3 | 382 |

**읽는 법 세 가지:**

1. **weight 모드는 공짜로 최신성을 올립니다** — 12개월 비율 36.6% → 38.5%(+1.9%p)인데
   인용수 중앙값은 3 → 4로 오히려 올랐습니다. 최신 논문을 밀어주면 품질이 떨어질 것이라는
   우려가 이 데이터에서는 나타나지 않습니다.
2. **quota 0.30이 무효과인 것은 정상입니다** — 후보 풀에 이미 최근 12개월 논문이 30% 넘게
   있어서 하한이 이미 충족돼 있습니다. 쿼터는 하한이지 상한이 아닙니다.
   `quota_promoted=0`은 고장이 아니라 "필요 없었다"는 뜻입니다.
3. **quota 0.50은 정확히 50.0%를 맞춥니다** — 다만 인용수 중앙값이 4 → 3으로 내려갑니다.
   즉 **weight는 부드럽게 밀어주고, quota는 정확히 보장하되 대가가 있습니다.** 둘 다 남겨
   두고 골라 쓰는 게 맞습니다.

> ⚠ **아직 답하지 못한 것**: 새로 올라온 논문이 *좋은* 논문인지는 이 숫자로 알 수 없습니다.
> 최신성 비율은 "얼마나 최신인가"만 재지 "얼마나 맞는가"를 재지 않습니다. 정답 집합
> (SurGE 또는 대체)이 붙기 전까지 이 표는 **방향 지표이지 성능 지표가 아닙니다.**

### P2.4 실측 결과 (같은 토픽·조건, ③ 위에 S7 추가)

다양성 측정은 **상위 200편의 평균 쌍별 코사인 유사도**(저장 벡터 기준)입니다. 낮을수록 다양합니다.

| 설정 | 최근 12m | 상위200 평균 유사도 | 카테고리 수 | ③ 대비 신규 |
|---|---|---|---|---|
| ③ freshness (weight) | 38.5% | 0.7578 | 27 | — |
| ⑥ + MMR λ=0.7 | 38.1% | 0.7474 | 30 | 291 |
| ⑦ + MMR λ=0.3 | 38.4% | **0.6287** | **32** | **700** |

**읽는 법:**

1. **다양성은 최신성을 깎지 않습니다** — λ=0.3에서 유사도가 17% 내려가고 카테고리가
   27 → 32개로 늘었는데 12개월 비율은 38.5% → 38.4%로 그대로입니다. S6과 S7은
   서로 간섭하지 않습니다.
2. **λ가 실질적인 손잡이입니다** — 0.7은 완만(신규 291편), 0.3은 공격적(신규 700편).
   서베이 커버리지 목적이면 낮은 λ가 맞는 방향으로 보입니다.
3. **MMR이 BM25 고유 논문을 끌어올립니다** — `bm25_only` 가 233편 → **607편**으로
   늘었습니다. dense 이웃에서 벗어난 논문일수록 서로 다르기 때문입니다.
   S3와 S7이 같은 방향으로 작동한다는 뜻입니다.

> 위 표는 S1이 꺼진 상태(facet 1개)에서 잰 것이라 **facet 쿼터가 비활성**이었습니다.
> P2.1을 붙인 뒤의 최종 수치는 아래 통합 표를 보세요.

### P2 통합 실측 — 전 단계 누적 (2026-08-12, RAG for LLMs, n_papers=1500)

| 설정 | 쿼리 | 후보 | 6m | 12m | 24m | 유사도 | 카테고리 | 베이스라인 대비 신규 |
|---|---|---|---|---|---|---|---|---|
| ① dense only | 1 | 1,996 | 16.9% | 35.5% | 69.7% | 0.7969 | 23 | — |
| ② + BM25 | 1 | 2,903 | 16.9% | 36.6% | 71.7% | 0.7545 | 27 | 351 |
| ③ + freshness | 1 | 2,903 | 19.5% | 39.8% | 74.0% | 0.7576 | 28 | 351 |
| ④ + MMR λ=0.3 | 1 | 2,903 | 20.4% | 39.0% | 69.1% | 0.6327 | 31 | 852 |
| ⑤ + facets | 36 | 48,214 | 23.1% | 45.1% | 75.7% | 0.7633 | 26 | 745 |
| ⑥ **전부 켜기** | 36 | 48,214 | **25.1%** | **45.8%** | 70.7% | **0.6245** | **32** | **1,029** |

**베이스라인 → 전부 켜기: 최근 12개월 비율 35.5% → 45.8%(+10.3%p), 유사도 0.797 → 0.625,
카테고리 23 → 32개, 최종 1,500편 중 1,029편(69%)이 교체.** Jaccard 0.186입니다.

기여도를 분리하면 **facet(S1)이 가장 큽니다** — 12개월 비율을 39.8% → 45.1%로 올립니다.
LLM이 내놓은 쿼리에 Self-RAG · IRCoT · DPR · FEVER · CodeT5 같은 방법론·데이터셋 이름이
들어가는데, 이게 dense 임베딩이 구조적으로 못 잡는 바로 그 토큰들입니다.

> ⚠ **대가**: 인용수 중앙값이 4 → 3으로, 최근 12개월 논문만 보면 **1**입니다.
> 최신 논문이 실제로 덜 인용됐기 때문이지만, 그 논문들이 *좋은* 논문인지는
> 이 숫자로 알 수 없습니다. 정답 집합이 붙기 전까지 이 표는 방향 지표입니다.

### P2 구현 중 잡은 문제 3가지

전부 "무음 폐기 금지" 원칙이 잡아낸 것들입니다. stats 경고가 없었으면 조용히 넘어갔습니다.

1. **랭킹 윈도우가 후보를 굶기고 있었습니다** — facet을 켜니 후보가 48,214편인데
   S6·S7이 상위 3,000편만 보고 있었습니다(윈도우가 단일 쿼리 기준으로 잡혀 있었음).
   `rank_window` / `title_window` 를 설정 가능하게 하고 기본을 전량으로 바꿨습니다.
2. **`provenance_of` 가 호출마다 집합을 재구성했습니다** — 후보 48,000편 × 원본 72,000개에서
   **315초**. 집합을 호출부에서 한 번만 만들도록 계약을 바꿔 **1.4초**가 됐습니다.
3. **`get_papers` 의 `IN (?,?,...)` 이 id 수천 개에서 급격히 느려집니다** — 3,000개에 19.7초.
   pyarrow 테이블 조인으로 바꿔 5,000개에 **0.32초**.

### MMR 풀은 일부러 묶습니다

`mmr_pool` 기본값은 `max(n_papers × 2, 3000)` 입니다. **풀이 커지면 다양성이 관련성을
압도합니다** — 관련성을 풀 안에서 min-max 정규화하므로, 48,214편을 넣으면 대부분의
relevance가 0 근처가 되고 λ가 의도한 균형이 깨집니다. 실측: 풀 3,000 → 12개월 45.8%,
풀 48,214 → **18.3%**. 잘라낸 건수는 `stats.warnings` 에 남습니다.

## P3 — 어댑터 · CLI

호스트 에이전트에 실제로 꽂아보는 단계.

| # | 작업 | 산출물 | 검증 |
|---|---|---|---|
| 3.1 | `adapters/autosurvey.py` | 드롭인 `database` 대체 | 🟡 시그니처·반환형 검증 완료. **서베이 생성 스모크는 승인 대기** |
| 3.2 | `adapters/surveyforge.py` (`filter`·`rerank` 인자 호환) | 드롭인 RAG 대체 | 🟡 `filter` 번역·`rerank` 대체 검증 완료. **스모크는 승인 대기** |
| 3.3 | `cli.py` — 토픽 → JSON 덤프 | CLI | ✅ **완료** — `--ablate` 로 설정별 비교표까지 |

> 서베이 생성은 편당 실비가 듭니다($0.3~2). 스모크는 사용자 승인 후에만 돌립니다.

## P4 — 진단 하네스

정량 벤치마크(SurGE)가 막혀 있으므로, **자체 진단 지표로 개발을 굴립니다.**

| # | 작업 | 내용 |
|---|---|---|
| 4.1 | stats 리포트 | ✅ `diagnostics.stage_report` |
| 4.2 | 최신성 지표 | ✅ `freshness_report` — 비율 + 연도별 히스토그램 + 최근 논문 인용 중앙값 |
| 4.3 | 커버리지 지표 | ✅ `coverage_report` — facet 중첩을 반영한 기대치 |
| 4.4 | 베이스라인 대조 | ✅ `compare` — base_id 기준 교집합/Jaccard/신규 유입 |
| 4.5 | 회귀 고정 | ✅ `snapshot` / `diff_snapshot` |

**4.4가 특히 중요합니다** — "우리 검색이 원본이 못 찾던 무엇을 찾는가"를 논문 없이 정량화하는
유일한 방법입니다. 실측 결과:

```
dense-only(베이스라인) (1,500편)  vs  survey-search(all-on) (1,500편)
  교집합 471  |  Jaccard 0.186
  all-on 만: 1,029편   베이스라인 만: 1,029편
  최근 12개월: 35.5% -> 45.8%
  인용수 중앙값: 4 -> 3
```

연도별 분포(all-on)는 2025년 491편 · 2026년 458편으로, 최근 2년이 전체의 63%입니다.

> `coverage_report` 의 기대치는 `n_papers / n_facets` 가 **아닙니다.** facet 들은 서로
> 겹칩니다(실측: 논문당 평균 7.9개 facet). 논문 수를 분모로 잡으면 기대치가 실제보다
> 훨씬 작아져 '미달' 판정이 발동하지 않습니다. 총 소속 수를 facet 수로 나눈 값을 씁니다.

## P5 — 후속 (별도 판단 후 착수)

| # | 작업 | 선행 조건 |
|---|---|---|
| 5.1 | 인용 스노우볼링 | Semantic Scholar / OpenAlex로 인용 엣지 수집 |
| 5.2 | cross-encoder 재랭킹 | `bge-reranker-v2-m3` 다운로드, GPU |
| 5.3 | BGE-M3 하이브리드 인덱스 재구축 | GPU 2~4시간, 디스크 +8GB |
| 5.4 | ~~HNSW 전환~~ | ❌ **불필요 확정** — 0.3 실측에서 32쿼리 2.73초. 착수 사유 소멸 |
| 5.5 | **SurGE 평가 — 막혀 있지 않습니다** | ✅ 선행 조건 충족 확인 (2026-08-12). 아래 참조 |
| 5.6 | ReAct 에이전트 루프 | 규칙 기반 파이프라인이 안정된 뒤 |

### 5.5 — SurGE 평가가 가능하다는 실측 근거 (2026-08-12)

이 레포는 지금까지 "SurGE 벤치마크 구현이 미공개라 정량 평가 보류"를 전제로 써 왔습니다.
**그 전제는 틀렸습니다.** `../SurGE/data/` 를 직접 확인한 결과:

| 자산 | 상태 |
|---|---|
| `data/surveys.json` | ✅ 있음 — GT 서베이 **205편**, 각각 `survey_title` + `all_cites`(인용 논문 id 목록) |
| `data/corpus.json` | ✅ 있음 — 1,086,992편, `Title`/`Date`/`Abstract`/`doc_id` |
| `data/queries.json` | ❌ 없음 (README 가 언급하는 topic→article 매핑) |
| `src/evaluator.py` | ✅ 있음 |

**`queries.json` 이 없어도 됩니다.** `surveys.json` 의 `survey_title` → `all_cites` 자체가
토픽→논문집합 정답입니다. 서베이가 실제로 인용한 논문 목록이 곧 그 토픽의 정답 집합입니다.

우리 코퍼스와의 연결도 실측했습니다 (SurGE 코퍼스에는 arXiv id 가 없어 **정규화 제목**으로 연결):

```
인용 논문 총 13,485편 중 우리 코퍼스에 매칭: 12,324편 (91.4%)
서베이별 매칭률 중앙값 92.1%
서베이당 인용 수 중앙값 51편
GT 서베이 연도: 2019 ~ 2023 (중앙값 2020)
```

**설계상 반드시 지켜야 할 것 — 날짜 컷오프**

GT 서베이는 2019~2023년입니다. 2020년 서베이는 2021년 논문을 인용할 수 없습니다.
날짜 제한 없이 평가하면 **우리 파이프라인이 최신 논문을 밀어주는 만큼 손해를 봅니다** —
가설을 검증하는 게 아니라 자동으로 깎이는 구조가 됩니다. 서베이마다
`SearchConfig(date_max=<서베이 게시일>)` 를 걸어야 공정합니다. `filter` 단계가 이미 있습니다.

날짜를 맞춘 뒤에도 freshness 가설은 그대로 시험됩니다: 서베이는 당대 기준으로 최신 논문을
많이 인용하므로, 인용수 정렬보다 연령 정규화 랭킹이 더 잘 맞아야 합니다.

**주의할 점 2가지**

- 인용 0편인 서베이가 섞여 있습니다(예: `MAC Protocols for Terahertz Communication`).
  제외하되 **몇 편을 왜 제외했는지 세어서 남길 것** — 무음 폐기 금지
- 정답 집합 중앙값이 51편이라 `n_papers=1500` 은 과합니다. recall@50/@100/@500 으로
  평가하세요. 채점기는 [`metrics/paper_set.py`](src/survey_search/metrics/paper_set.py) 에
  이미 있습니다(LitSearch recall/nDCG + PFB adjusted_f1)

---

## 지금 당장의 다음 한 걸음

**P3 스모크 — 서베이 생성 1편** (사용자 승인 필요, 편당 $0.3~2). 어댑터는 시그니처와
반환형까지 검증했지만, 호스트 에이전트가 실제로 끝까지 도는지는 돌려봐야 압니다.
그 전까지 P3는 🟡입니다.

돈이 안 드는 것 중에서는 **P0.6 잔여**(`date` 가 v1 게시일인지 최신본 갱신일인지)가
남아 있습니다. recency 가중의 기준이라 확인해 두는 게 좋습니다.

### 환경 재현

```bash
cd /data2/chanjoong/survey-agent/survey-search
virtualenv -p python3.10 .venv                     # python3 -m venv 는 ensurepip 없어서 실패
.venv/bin/pip install -e .
.venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
cp .env.example .env && chmod 600 .env             # OPENROUTER_API_KEY 채우기
.venv/bin/python -m survey_search.index.build_duckdb    # 90초, 1.1GB
.venv/bin/python -m pytest tests/ -q                    # 101 passed
```

### 돌려보기

```bash
# 단일 토픽, 전 단계 켜기
.venv/bin/python -m survey_search.cli --topic "Retrieval-Augmented Generation for LLMs" --all

# 설정별 비교표 (P4.4) — 백엔드를 재사용하므로 인덱스를 한 번만 읽습니다
.venv/bin/python -m survey_search.cli --topic "..." --ablate --out results/
```

```python
from survey_search.core.facets import load_dotenv; load_dotenv(".env")
from survey_search.backends.faiss_duckdb import FaissDuckDBBackend
from survey_search.search import search_topic
from survey_search.types import SearchConfig
from survey_search.metrics.diagnostics import compare, freshness_report, coverage_report

be = FaissDuckDBBackend()          # 첫 검색은 cold 로 12초, 이후 0.1초
base = search_topic(T, backend=be, config=SearchConfig(lexical=False))
ours = search_topic(T, backend=be, config=SearchConfig(facets=True, freshness=True,
                                                       diversity=True, mmr_lambda=0.3))
print(compare(base, ours).render())
print(freshness_report(ours).render())
print(coverage_report(ours).render())
```
