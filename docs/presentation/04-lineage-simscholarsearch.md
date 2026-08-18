# 4. SimScholarSearch(S3) 에서 무엇을 가져왔나

교수님이 명시적으로 물으실 부분이라, **가져온 것 / 못 가져온 것 / 왜** 를 파일 단위로
확인해서 정리했습니다. (라이선스 Apache-2.0, `trillion-labs/scholar-search-rl`.
출처는 이식한 각 파일 상단에 주석으로 박아 두었습니다.)

## 먼저 — 두 프로젝트의 목적이 다릅니다

| | **SimScholarSearch (S3)** | **survey-search** |
|---|---|---|
| 무엇 | 검색 **에이전트를 학습·평가하는 환경** | 서베이 에이전트가 쓰는 **검색 레이어** |
| 에이전트 | 멀티턴 ReAct, 툴 9종 | 없음 — **규칙 기반 결정적 파이프라인** |
| 학습 | verl RL, outcome reward | 없음 |
| 코퍼스 | S2ORC CS **약 112만편, 본문 포함** | arXiv **908,819편, 초록만** |
| 검색 | Milvus 하이브리드 (BGE-M3 dense + sparse) | FAISS(gte 1024d) + DuckDB BM25 |
| 결과물 | 학습된 검색 정책 | 토픽 → 논문 집합 |

**S3 는 "검색을 잘하는 정책을 학습시키는 문제"** 이고,
**여기는 "검색 파이프라인을 고정하고 각 단계의 기여를 측정하는 문제"** 입니다.

> 이 차이가 "코드가 왜 많이 안 넘어왔는가"의 답이기도 합니다.
> 하부 스택(Milvus / BGE-M3 / 본문 / verl)이 통째로 다릅니다.

## 가져온 것 — 코드 수준

| 우리 쪽 | 원본 | 무엇 |
|---|---|---|
| `metrics/paper_set.py` | `synthesis/paper_set.py` | `score_paper_set` — PaperFindingBench 의 **adjusted_f1 = recall@est + nDCG**, `_dcg` / `_ndcg_rank` / `_harmonic` |
| `metrics/paper_set.py` | `eval/litsearch.py` | `calculate_recall` / `calculate_ndcg` — princeton-nlp/**LitSearch** 원본을 S3 가 그대로 옮겨 둔 것 |
| `index/build_duckdb.py` | `env/etl/build_fts.py`, `env/reader.py` | **DuckDB FTS 사용법** (`fts_main_papers.match_bm25`). 스키마는 공유하지 않습니다 |

**바꾼 것은 사실상 id 타입 하나입니다.**
S3 는 S2ORC 정수 `corpus_id`, 여기는 버전 접미사가 붙은 arXiv 문자열 id
(`"2401.12345v2"`). 그래서 채점 전에 `base_id` 로 정규화하는 `normalize_ids` 를
앞에 붙였습니다 — 안 그러면 `v1` 과 `v2` 가 **다른 논문으로 채점**됩니다.

### 이게 왜 "적지만 가장 중요한 이식"인가

발표에서 이 문장을 꼭 넣으면 좋겠습니다:

> **지표는 실험보다 먼저 고정돼 있어야 합니다.**
> 벤치마크가 열린 뒤에 채점기를 새로 짜면, 그 시점의 구현이 곧 결과를 좌우합니다.
> 그래서 쓸 정답 집합이 아직 없던 시점(SurGE 미확인)에 **채점기부터 먼저 옮겨
> 놓았습니다.**

실제로 그 판단이 맞았습니다. 나중에 SurGE 를 붙였을 때 recall/nDCG 정의를 **손대지
않고** 바로 쟀고, "우리한테 유리한 지표를 골랐다"는 의심이 원천적으로 안 생깁니다.
LitSearch → S3 → survey-search 로 **정의가 그대로 이어집니다.**

## 가져온 것 — 패턴·설계 수준

코드는 안 옮겼지만 **참고했다고 말해야 정직한 것들**입니다.

| 무엇 | 원본 | 여기서 |
|---|---|---|
| **RRF k=60** | `search_papers.py` 의 Milvus `RRFRanker(60)` | 같은 k 로 직접 구현. **선례를 따랐습니다** |
| **툴 팩토리** | `registry.py` 의 `build_registry()` / `compose()` | `Backend` 프로토콜 + 단계 on/off 구성 |
| **하이브리드 검색 구조** | dense + sparse 를 rank 로 융합 | dense + BM25 를 RRF 로 융합 (2단으로 확장) |
| **툴 9종 시그니처** | `search_papers` `search_snippets` `read_paper` `find_in_paper` `list_references` `list_citations` `find_similar` `paper_info` `submit_answer` | 지금은 안 씀. **후속 ReAct 루프에 착수하면 이 인터페이스를 따를 것** |
| **로컬 봉인 코퍼스** | 롤아웃마다 API 를 안 부르는 재현 가능 환경 | 로컬 우선 + 온라인은 보강만. 같은 이유 |

> **RRF k=60 은 "그냥 논문 기본값"이 아니라 S3 에서 확인한 선례**라고 말할 수 있습니다.
> 다만 여기서는 **필수 조건**이라는 근거가 하나 더 있습니다 — 쿼리 벡터가 비정규화라
> 점수 융합이 애초에 불가능합니다 ([02](02-module.md) ②).

## 안 가져온 것 — 그리고 이유

| 안 가져온 것 | 이유 |
|---|---|
| `pymilvus` 의존 모듈 전부 | **이 머신에 docker 소켓 권한이 없어 Milvus 를 못 씁니다.** FAISS + DuckDB 로 갔습니다 |
| `encoder.py` (BGE-M3 하이브리드 인코더) | 1차는 **재임베딩 0** 원칙 — SurveyForge 의 기존 gte 인덱스를 재사용합니다. BGE-M3 재구축은 별도 축 |
| `read_paper` / `find_in_paper` | **본문이 없습니다.** 우리 코퍼스는 초록만입니다 |
| `synthesis/` (질문 합성) | 정답 집합을 **합성하지 않습니다** — SurGE 의 실제 서베이 인용 목록을 씁니다 |
| `agent/` · `trainer/` (ReAct 루프, verl RL) | 1차 범위가 **규칙 기반 오케스트레이터**입니다. 결정적이라 단계별 ablation·디버깅이 쉽습니다 |

## "그럼 RL 로 안 가나?" — 예상 질문

S3 처럼 학습으로 가는 선택지는 열려 있고, 지금 안 가는 이유는 이렇습니다.

1. **지금은 각 단계의 기여를 측정하는 단계입니다.** 정책을 학습시키면 무엇이
   기여했는지 분해할 수 없습니다. 규칙 기반 ablation 이 먼저입니다
2. **오프라인 코퍼스는 한계가 아니라 전제입니다.** S3 가 코퍼스를 봉인한 것도
   롤아웃마다 API 를 부르면 RL 실험이 재현되지 않기 때문입니다.
   즉 "온라인으로 가자"와 "S3 처럼 RL 로 가자"는 **서로 반대 방향의 요구**입니다
3. **본문이 없습니다.** "읽고 판단하는 에이전트"가 목표라면 본문은 선택이 아니라
   전제인데, 지금 코퍼스는 초록만입니다

> 다만 측정이 말하는 것도 있습니다 — 검색 손실의 대부분은 **표현(초록 vs 본문)이
> 아니라 깊이 부족**이었습니다 ([05](05-results.md) 천장 분석). 본문 도입은
> 35GB·수 주 작업인데 상한이 약 5%p 이고, `top_k` 설정 한 줄이 17.8%p 를 줍니다.

## 계보 한 줄로

```
princeton-nlp/LitSearch  ──(recall/nDCG 정의)──┐
                                              ├─→ survey-search/metrics/paper_set.py
AstaBench PaperFindingBench ─(adjusted_f1)─┐  │
                                           ├──┘
trillion-labs/SimScholarSearch (S3) ───────┘
        │  RRF k=60 선례 · DuckDB FTS 사용법 · 툴 레지스트리 패턴
        │  로컬 봉인 코퍼스 · 하이브리드 검색 구조
        └────────────────────────────────────→ survey-search 설계
```
