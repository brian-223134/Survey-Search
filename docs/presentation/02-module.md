# 2. 이 모듈이 하는 일

## 인터페이스는 한 줄입니다

```python
from survey_search import search_topic, SearchConfig

result = search_topic("Retrieval-Augmented Generation for LLMs",
                      backend=backend,
                      config=SearchConfig(n_papers=1500, facets=True, freshness=True))

result.papers   # 랭킹된 · 중복 제거된 논문 1500편
result.facets   # 어느 하위 주제에서 나왔는지
result.stats    # 각 단계가 무엇을 몇 편 버렸는지
```

```
topic: str  →  ranked · deduped · facet-grouped papers
```

**독립 패키지**입니다. AutoSurvey · SurveyForge · SurveyX 가 공유해서 씁니다.
코어는 어느 코퍼스도 모르고 `Backend` 프로토콜만 압니다.

## 파이프라인 8단계

```
topic
  ├─ S1  facet 분해 ──── LLM 1회(캐시) → 하위 주제 8~16개, 각 1~3 쿼리
  ├─ S2  dense 검색 ──── FAISS (gte-large-en-v1.5, 1024d, 908,819편)
  ├─ S3  어휘 검색 ────── DuckDB BM25
  ├─ S4  RRF 융합 ────── facet 내부 → facet 간, 2단
  ├─ S5  중복 제거 ────── arXiv 버전 병합 + 제목 정규화
  ├─ S6  freshness ───── 연령 정규화 인용률 백분위 + recency
  ├─ S8  스노우볼링 ───── S2 API 인용 그래프 (전방·후방)
  ├─ S8b cross-encoder ─ 재랭킹      ← 측정 결과 **끕니다**
  ├─ S7  다양성 ──────── MMR + facet 쿼터  ← 측정 결과 **끕니다**
  └─ SearchResult (papers + facets + stats)
```

**전부 개별적으로 끌 수 있습니다.** 이게 설계의 핵심입니다 — 설정 스위치가 곧
실험 축이라, 같은 코드로 ablation 을 돌립니다.

## 각 단계를 한 줄로

| 단계 | 무엇을 하나 | 왜 필요한가 |
|---|---|---|
| **S1** facet | LLM 이 토픽을 하위 주제 8~16개로 쪼개고, 각각 표면형이 다른 쿼리 1~3개 | 토픽 문자열 하나는 임베딩 공간의 **점 하나**. 그 근처만 봅니다 |
| **S2** dense | FAISS 배치 검색 | "말이 비슷한 논문" |
| **S3** BM25 | DuckDB FTS | "단어가 겹치는 논문" — 새 방법론명·모델명·약어의 영역 |
| **S4** RRF | `Σ 1/(k+rank)`, k=60, **2단** | 이 인덱스에서는 **선택이 아니라 필수** (아래) |
| **S5** dedup | `2401.12345v1`/`v2` 병합 + 제목 정규화 | 버전이 갈리면 같은 논문의 순위가 반토막 |
| **S6** freshness | 인용률을 **6개월 코호트 안 백분위**로 | 2026년의 인용 3과 2019년의 인용 3은 다른 의미 |
| **S8** 스노우볼링 | S2 API 로 전방·후방 인용 | 임베딩도 어휘도 아닌 **제3의 신호** — 저자가 직접 선언한 관계 |

## 설계에서 설명할 가치가 있는 것 넷

### ① facet 분해가 노리는 것은 dense 의 사각지대입니다

LLM 에게 **방법론·모델·데이터셋 이름을 넣도록** 프롬프트에서 요구합니다.
실제로 Self-RAG · IRCoT · DPR · FEVER · CodeT5 같은 이름이 나오는데,
**이게 dense 임베딩이 구조적으로 못 잡는 바로 그 토큰들입니다.**

- 결과는 토픽 해시로 캐시 → 재실행은 LLM 호출 0회 (**파이프라인은 결정적**)
- 비용 호출당 **$0.0006** (실측)
- 한계: LLM 의 사전 지식에 의존하므로 **모델 컷오프 이후 신조어는 못 냅니다.**
  그 구멍을 S3(BM25)와 S8(스노우볼링)이 메우는 구조입니다

### ② RRF 는 이 인덱스에서 필수 조건입니다

```
score(d) = Σ_q  1 / (k + rank_q(d)),   k = 60
```

저장된 문서 벡터는 단위 norm 인데 **gte 가 내놓는 쿼리 벡터는 정규화돼 있지
않습니다(norm ≈ 24).** 한 쿼리 안의 순위는 멀쩡하지만 **쿼리끼리 점수를 비교할 수
없습니다.** BM25 점수는 단위가 아예 다릅니다.

**순위만 쓰면 이 문제가 사라집니다.** — 이건 나중에 재랭킹 실패의 원인이기도 합니다
([06](06-negative-results.md)).

2단인 이유: ① facet 내부에서 dense+BM25 융합 → ② facet 간 융합.
한 번에 다 섞으면 **쿼리를 많이 가진 facet 이 결과를 지배**합니다.

### ③ freshness 는 대체가 아니라 조정입니다

```
final = rrf × (1 + α·citation_rate_percentile) × (1 + β·recency_weight)
citation_rate = citation_count / max(months_since_pub, 3)
```

곱셈인 이유: RRF 점수가 이미 "여러 검색이 얼마나 동의하는가"를 담고 있으므로
freshness 는 그걸 **조정**해야 합니다. **덧셈이면 관련 없는 최신 논문이 올라옵니다.**

백분위를 **6개월 코호트 안에서** 매기는 이유: 전체를 한 줄로 세우면 오래된 논문이
상위 백분위를 독식합니다.

### ④ 무음 폐기 금지 — 모든 단계가 자기가 버린 것을 신고합니다

```
S2 dense          36 ->  72,000  17.07s
S4 rrf       144,000 ->  48,257  0.21s
S5 dedup      48,257 ->  48,214 -43 (version=28, title=15)  1.06s
S7 diversity   3,000 ->   1,500 -1,500 (n_papers=1500)  0.14s
warnings:
  ! MMR 풀을 3,000편으로 제한 — 45,214편은 S7 대상에서 제외
```

`dropped` 는 `in − out` 을 계산한 값이 **아니라 단계가 스스로 센 값**입니다.
둘이 다르면 그 자체가 버그 신호입니다.

**이 원칙이 실제로 버그를 다섯 번 잡았습니다** → [07](07-engineering.md)

## 구조 — 의존 방향이 한쪽입니다

```
src/survey_search/
├── search.py      오케스트레이터           ← core 와 backend 를 아는 유일한 곳
├── backends/      코퍼스를 아는 유일한 계층  (faiss_duckdb / online / hybrid)
├── core/          파이프라인 단계          (facets fuse dedup rank expand rerank diversity)
├── adapters/      호스트 호환 계층         (autosurvey / surveyforge)
├── metrics/       지표                    (paper_set ← SimScholarSearch 이식)
└── eval/          정량 평가               (surge / ceiling)
```

**`core/` 는 백엔드를 모르고, 백엔드는 `core/` 를 모릅니다.**
그래서 백엔드를 갈아 끼워도 파이프라인이 그대로이고, 단계를 추가해도 백엔드를 안 건드립니다.

### Backend 프로토콜 — 구현할 게 이것뿐입니다

```python
def dense_search(queries: list[str], top_k: int, field: str) -> list[list[Hit]]
def lexical_search(queries: list[str], top_k: int)             -> list[list[Hit]]
def get_papers(paper_ids: list[str])                           -> list[Paper]
def filter_ids(*, date_min, date_max, categories)              -> set[str] | None

# 선택 — 없으면 해당 단계가 no-op 이 되고 그 사실이 stats 에 남습니다
def references(paper_id) -> list[str]
def cited_by(paper_id)   -> list[str]
```

두 가지가 의도적이고, 질문받으면 답할 거리입니다.

- **검색이 배치입니다.** facet fan-out 이 기본 사용 패턴이고, 실측상 배치가 쿼리당
  **9배** 빠릅니다 (1쿼리 790ms vs 32쿼리 배치 85ms/쿼리). 쿼리 하나씩 도는 API 는
  아예 만들지 않았습니다
- **`filter_ids` 의 `None` 과 빈 집합은 뜻이 다릅니다.** `None` = 제한 없음,
  `set()` = 조건에 맞는 논문이 없음. 섞으면 **필터가 조용히 무력화**됩니다

## 백엔드 둘

| | 로컬 (`faiss_duckdb`) | 온라인 (`online`) |
|---|---|---|
| 무엇 | FAISS 1024d + DuckDB BM25/메타 | arXiv API + Semantic Scholar |
| 강점 | recall · 결정성 · 속도 | **컷오프 이후 논문** + 인용 엣지 |
| 역할 | 주력 | 로컬을 **대체하지 않고 보강** |

온라인이 필요한 근거도 실측입니다 — RAG 원논문의 피인용 309편 중 **76편(25%)이
로컬 스냅샷에 없습니다.**
